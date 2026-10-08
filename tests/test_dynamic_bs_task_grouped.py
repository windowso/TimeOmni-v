"""Tests for task-grouped batching in TokenBudgetBatchSampler.

Cover the contract that motivated this feature:
  1. Every emitted batch is task-pure (all indices share a task_id).
  2. One epoch covers every input index exactly once across ranks.
  3. Same `seed + epoch` produces identical batches on every rank.
  4. Falling back to `task_ids=None` reproduces the legacy global packing.
"""

from __future__ import annotations

import json

from timeomni_v.data.dataset import TimeOmniVDataset
from timeomni_v.data.dynamic_bs import TokenBudgetBatchSampler


def _flatten(batches):
    return [i for b in batches for i in b]


def _build_lengths_and_tasks(sizes):
    """sizes: dict[task_id -> n_samples]. Returns (lengths, task_ids).

    Lengths are a deterministic mix so the bucket-sort produces non-trivial
    packing, but stay small enough that several samples fit per batch."""
    lengths: list[int] = []
    task_ids: list[int] = []
    for tid, n in sizes.items():
        for k in range(n):
            # alternating short/long inside each task
            lengths.append(50 if k % 3 else 200)
            task_ids.append(tid)
    return lengths, task_ids


def test_task_grouped_batches_are_task_pure():
    sizes = {0: 8, 1: 50, 2: 200}
    lengths, task_ids = _build_lengths_and_tasks(sizes)
    s = TokenBudgetBatchSampler(
        lengths=lengths,
        max_tokens=2000,
        max_bs=8,
        bucket_size=32,
        task_ids=task_ids,
    )
    batches = list(iter(s))
    # Every batch's indices come from a single task.
    for b in batches:
        tids = {task_ids[i] for i in b}
        assert len(tids) == 1, f"mixed-task batch: {b} -> {tids}"
    # Total coverage equals every input row exactly once (single-rank run).
    flat = _flatten(batches)
    assert sorted(flat) == list(range(len(lengths)))


def test_task_grouped_determinism_across_ranks():
    """Two samplers with same seed + epoch agree on the master batch list,
    so the world_size>1 path produces compatible per-rank slices."""
    sizes = {0: 17, 1: 80, 2: 30}
    lengths, task_ids = _build_lengths_and_tasks(sizes)
    common = dict(
        lengths=lengths,
        max_tokens=1500,
        max_bs=4,
        bucket_size=24,
        task_ids=task_ids,
        seed=123,
    )
    s_a = TokenBudgetBatchSampler(rank=0, world_size=1, **common)
    s_b = TokenBudgetBatchSampler(rank=0, world_size=1, **common)
    s_a.set_epoch(7)
    s_b.set_epoch(7)
    assert list(iter(s_a)) == list(iter(s_b))


def test_task_grouped_ddp_shard_covers_all_kept_batches():
    sizes = {0: 50, 1: 50, 2: 50, 3: 50}
    lengths, task_ids = _build_lengths_and_tasks(sizes)
    common = dict(
        lengths=lengths,
        max_tokens=1000,
        max_bs=4,
        bucket_size=20,
        task_ids=task_ids,
        seed=11,
    )
    rank0 = TokenBudgetBatchSampler(rank=0, world_size=2, **common)
    rank1 = TokenBudgetBatchSampler(rank=1, world_size=2, **common)
    rank0.set_epoch(0)
    rank1.set_epoch(0)
    b0 = list(iter(rank0))
    b1 = list(iter(rank1))
    # Per-rank batches stay task-pure.
    for b in b0 + b1:
        assert len({task_ids[i] for i in b}) == 1
    # Each rank gets the same number of batches (truncate_to_world_size=True).
    assert len(b0) == len(b1)
    # CRITICAL invariant: at every step k, both ranks consume the same task.
    # Without per-task chunking before the rank shard, the master shuffle
    # could put differently-tasked batches at adjacent positions and the
    # `batches[rank::world_size]` slice would split them across ranks.
    for step, (rb0, rb1) in enumerate(zip(b0, b1)):
        t0 = task_ids[rb0[0]]
        t1 = task_ids[rb1[0]]
        assert t0 == t1, f"step {step}: rank0 task={t0} != rank1 task={t1}"
    # No overlap in the indices each rank sees within an epoch.
    inter = set(_flatten(b0)) & set(_flatten(b1))
    assert not inter


def test_task_ids_none_matches_legacy_packing():
    """Sanity: task_ids=None must reproduce the pre-feature behavior."""
    lengths = [50, 200, 50, 200, 50, 200, 50, 200]
    s_legacy = TokenBudgetBatchSampler(
        lengths=lengths,
        max_tokens=500,
        max_bs=4,
        bucket_size=8,
        seed=42,
    )
    s_legacy.set_epoch(0)
    legacy = list(iter(s_legacy))
    s_grouped = TokenBudgetBatchSampler(
        lengths=lengths,
        max_tokens=500,
        max_bs=4,
        bucket_size=8,
        seed=42,
        task_ids=None,
    )
    s_grouped.set_epoch(0)
    grouped = list(iter(s_grouped))
    assert legacy == grouped


def test_dataset_multi_source_concat(tmp_path):
    """TimeOmniVDataset accepting a list of jsonls concatenates rows in order
    and exposes parallel task_of_index for the sampler to consume."""
    rows_a = [
        {"id": f"a{i}", "task": "classification",
         "video_path": "/v.mp4", "prompt": "P", "answer": "A"}
        for i in range(3)
    ]
    rows_b = [
        {"id": f"b{i}", "task": "classification",
         "image_path": ["/x.png"], "prompt": "P", "answer": "A"}
        for i in range(2)
    ]
    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    pa.write_text("\n".join(json.dumps(r) for r in rows_a) + "\n")
    pb.write_text("\n".join(json.dumps(r) for r in rows_b) + "\n")

    ds = TimeOmniVDataset([pa, pb])
    assert len(ds) == 5
    assert ds.source_paths == [pa, pb]
    assert ds.task_names == ["a", "b"]
    assert ds.task_of_index == [0, 0, 0, 1, 1]
    assert ds[0]["id"] == "a0"
    assert ds[3]["id"] == "b0"

    # Single-path call path retains task_of_index = [0, ...] for compat.
    ds_single = TimeOmniVDataset(pa)
    assert ds_single.task_of_index == [0, 0, 0]
    assert ds_single.source_paths == [pa]
