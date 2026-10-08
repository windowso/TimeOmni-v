"""Token-budget batch sampler: short samples batched together, long samples
kept at bs=1.

Why:
  Step cost for multimodal LLM training is dominated by attention (O(n²)) and
  activations (O(n)). With a fixed per_device_batch_size=1 every step has the
  same *count* but wildly different *token counts* (vis_tok + text_tok can
  vary 10x across CoVLA). Packing samples until total tokens hits a budget
  evens out step cost — a 1k-token short sample barely moves the needle so
  we batch 4 of them; a 10k-token long sample still runs at bs=1.

What the budget is measured against:
  The collator right-pads every sample in a batch to the longest one, so the
  actual tensor handed to the GPU is `max_len * bs` tokens (attention is
  O((max_len*bs)²), activations O(max_len*bs)). We therefore cost a batch as
  `max_len_in_batch * len(batch)` rather than the naive sum of per-sample
  lengths — the naive sum under-counts padding and lets mixed-length batches
  blow past the real budget. Since chunks are length-sorted (step 2 below)
  the in-batch max grows monotonically as we append, so the cost is just
  `candidate_len * (len(batch) + 1)` at admission time.

What this gives up:
  HF Trainer assumes a fixed global batch size for loss averaging and lr
  scaling. With variable bs the *effective* gradient batch fluctuates. This
  is a correctness-vs-speed tradeoff: for SFT with answer-only loss it's
  negligible (loss is already averaged over answer tokens, which vary a lot
  already); for carefully-tuned pretraining runs it matters more.

Usage:
  1. Precompute a `lengths: list[int]` for the dataset (see `estimate_length`
     in `timeomni_v.data.length_estimate`).
  2. Construct `TokenBudgetBatchSampler(lengths, max_tokens=8192, max_bs=4)`.
  3. Pass as `batch_sampler=` to a DataLoader.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Yield variable-size batches whose total token count ≤ `max_tokens`.

    Algorithm:
      1. Shuffle all indices with per-epoch seed.
      2. Partition shuffled indices into chunks of `bucket_size`; sort each
         chunk by length. This keeps shuffling (different epochs see
         different orderings) while giving the greedy packer same-ish-length
         neighbors so it can fill a batch without early cutoff.
      3. Greedy-pack each sorted chunk into batches: add the next sample as
         long as (a) total token count stays under `max_tokens` AND (b)
         sample count stays under `max_bs`. Overflow triggers a flush.
      4. Shuffle the order of emitted batches so training order is not
         monotonically short-to-long inside a chunk.
      5. Round-robin across ranks for distributed training; each rank sees a
         disjoint slice and the same total step count (truncated to the
         shortest per-rank slice to keep collectives synchronized).
    """

    def __init__(
        self,
        lengths: list[int] | np.ndarray,
        max_tokens: int,
        max_bs: int = 8,
        bucket_size: int = 200,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        drop_last: bool = False,
        truncate_to_world_size: bool = True,
        task_ids: list[int] | np.ndarray | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.max_tokens = max_tokens
        self.max_bs = max_bs
        self.bucket_size = bucket_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.drop_last = drop_last
        # Train / eval need every rank to run the same number of steps so
        # NCCL collectives (allreduce in backward, gather_for_metrics in
        # eval) don't deadlock. Inference only does a single all_gather at
        # the end and can tolerate uneven step counts — pass False there to
        # avoid losing tail samples.
        self.truncate_to_world_size = truncate_to_world_size
        # Per-sample source/task assignment for task-grouped packing. When
        # supplied, every emitted batch contains indices from a single task —
        # packing runs independently inside each task's index range, then all
        # per-task batches are concatenated and globally shuffled.
        if task_ids is not None:
            tids = np.asarray(task_ids, dtype=np.int64)
            if tids.shape[0] != self.lengths.shape[0]:
                raise ValueError(
                    f"task_ids size {tids.shape[0]} != lengths size {self.lengths.shape[0]}"
                )
            self.task_ids: np.ndarray | None = tids
        else:
            self.task_ids = None
        self.epoch = 0
        self._len_cache: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Called by HF Trainer before each epoch to reshuffle. Invalidates
        the cached step count since the new shuffle produces new batches."""
        self.epoch = int(epoch)
        self._len_cache = None

    def _pack_indices(self, indices: np.ndarray, rng: np.random.Generator) -> list[list[int]]:
        """Shuffle, bucket, sort-by-length, greedy-pack a flat array of indices.

        Same algorithm as the global path — pulled out so the task-grouped
        path can call it once per task's index range with the existing budget
        knobs, while the global path calls it once on every index.
        """
        order = indices.copy()
        rng.shuffle(order)
        batches: list[list[int]] = []
        for start in range(0, len(order), self.bucket_size):
            chunk = order[start : start + self.bucket_size]
            # Sort ascending so the in-batch max equals the most recently
            # appended sample — the padded-cost formula `max_len * bs` then
            # reduces to `li * (len(batch)+1)` at admission time.
            chunk_sorted = sorted(chunk.tolist(), key=lambda i: int(self.lengths[i]))
            batch: list[int] = []
            for i in chunk_sorted:
                li = int(self.lengths[i])
                padded_cost = li * (len(batch) + 1)
                # A single sample over the budget still gets emitted alone —
                # we'd rather OOM-risk one step than silently drop samples.
                if batch and (padded_cost > self.max_tokens or len(batch) >= self.max_bs):
                    batches.append(batch)
                    batch = []
                batch.append(i)
            if batch:
                batches.append(batch)
        return batches

    def _build_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.task_ids is None:
            batches = self._pack_indices(np.arange(len(self.lengths), dtype=np.int64), rng)
            rng.shuffle(batches)
            return batches

        # Task-grouped path. Pack each task's index range independently so
        # every emitted batch is single-task. Then group `world_size`
        # consecutive same-task batches into a "step block" so the per-rank
        # shard ``batches[rank::world_size]`` produces the same task on
        # every rank at every step. The tail (fewer than world_size batches)
        # of each task is dropped so the master batch list is a clean
        # multiple of world_size — at most ``world_size - 1`` batches per
        # task per epoch are lost (negligible vs. epoch length).
        chunk = max(1, self.world_size)
        per_task_batches: list[list[list[int]]] = []
        unique_tasks = np.unique(self.task_ids)
        for t in unique_tasks:
            indices = np.flatnonzero(self.task_ids == t)
            if indices.size == 0:
                continue
            task_batches = self._pack_indices(indices, rng)
            # Shuffle inside each task before chunking so the per-step
            # world_size sibling batches are randomly drawn from the task's
            # pool, not the length-sorted order _pack_indices emits.
            rng.shuffle(task_batches)
            per_task_batches.append(task_batches)

        step_blocks: list[list[list[int]]] = []
        for task_batches in per_task_batches:
            n = (len(task_batches) // chunk) * chunk
            for start in range(0, n, chunk):
                step_blocks.append(task_batches[start : start + chunk])
        rng.shuffle(step_blocks)
        # Flatten so block i occupies master positions i*chunk .. i*chunk+chunk-1.
        # `batches[rank::world_size]` then yields rank's batch at every step.
        return [b for block in step_blocks for b in block]

    def _shard(self, batches: list[list[int]]) -> list[list[int]]:
        """Per-rank slicing. With truncation (train / eval) each rank gets
        exactly ``len(batches) // world_size`` batches; without (inference)
        ranks may differ by 1, but no batch is dropped."""
        if self.world_size <= 1:
            return batches
        if self.truncate_to_world_size:
            trunc = (len(batches) // self.world_size) * self.world_size
            batches = batches[:trunc]
        return batches[self.rank :: self.world_size]

    def __iter__(self):
        batches = self._shard(self._build_batches())
        self._len_cache = len(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        # HF Trainer asks for length before any iteration to compute
        # max_steps / lr schedule length. Build once and cache; clears on
        # set_epoch so the next epoch recomputes with its own shuffle.
        if self._len_cache is None:
            self._len_cache = len(self._shard(self._build_batches()))
        return self._len_cache
