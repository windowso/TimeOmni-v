"""Run inference on an TimeOmni-v / baseline / zero-shot checkpoint and report
per-sample answer + aggregate accuracy.

Backend dispatch: ``--backend {qwen2_5_omni,timeomni_v,qwen3_omni,qwen3_vl_*,
internvl_*,gpt,gemini}`` (default: ``timeomni_v``). The orchestration layer
(resume, dynamic batching, DDP, ablations, parser auto-tuning, metrics
file) is shared; each backend encapsulates the model-family-specific
load / collator / generate path.

Single-process (model-parallel across all visible GPUs via device_map="auto")::

    python -m timeomni_v.inference.infer \
        --backend timeomni_v --model_path ckpts/Qwen2.5-Omni-7B \
        --chronos_path ckpts/chronos-2 \
        --adapter_path runs/timeomni_v-covla-v1 \
        --test_jsonl data/video_ts/jsonl/covla_test.jsonl \
        --fusion_mode block_adjacent \
        --out_jsonl runs/timeomni_v-covla-v1/predictions.jsonl

Data-parallel (one rank per GPU; only for local backends — API backends
must run single-process)::

    torchrun --standalone --nproc-per-node=N -m timeomni_v.inference.infer <args>

Output: a jsonl where each line has {id, ground_truth, prediction, raw,
failed[, error]}. A sidecar ``<out_jsonl>.metrics.json`` carries success_rate,
accuracy, and ablation provenance.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from timeomni_v.data.collator import (
    DEFAULT_FPS,
    DEFAULT_IMAGE_MAX_PIXELS,
    DEFAULT_IMAGE_MIN_PIXELS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_VIDEO_MAX_PIXELS,
    DEFAULT_VIDEO_MIN_PIXELS,
)
from timeomni_v.data.dataset import TimeOmniVDataset
from timeomni_v.data.dynamic_bs import TokenBudgetBatchSampler
from timeomni_v.inference import backends
from timeomni_v.inference.backends.base import ApiBackend, LocalHFBackend
from timeomni_v.inference.forecast_parse import extract_forecast_block
from timeomni_v.inference.parse import build_parser
from timeomni_v.utils.warnings import silence_rope_scaling_warning

silence_rope_scaling_warning()


def load_existing_predictions(path: Path) -> tuple[list[dict], set]:
    """Read a prior predictions.jsonl (if any) so we can resume.

    Returns (rows_kept, done_ids). Truncated/corrupt trailing lines are
    silently dropped — a SIGKILL'd run picks up cleanly without hand-editing.
    Rows without an ``id`` are kept but excluded from the skip set (they
    can't be matched against the dataset and so get re-inferred).
    """
    if not path.exists():
        return [], set()
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    done = {r["id"] for r in rows if r.get("id") is not None}
    return rows, done


def _is_oom(exc: BaseException) -> bool:
    """Detect a CUDA OOM. Newer torch raises ``torch.cuda.OutOfMemoryError``;
    older versions raise plain ``RuntimeError`` with "out of memory" in the
    message. Both forms are caught."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    if oom_cls and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def run_inference_local(
    backend: LocalHFBackend, loader: DataLoader,
    *, max_new_tokens: int, parser: Callable[[str | None], str | None],
    collate_fn: Callable[[list[dict]], dict] | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """Iterate the DataLoader, run a batched generate per step.

    Failure handling:

    * On any non-OOM exception → mark every sample in the batch
      ``failed=True`` (one bad sample shouldn't tank the run, but we also
      can't tell which sample is bad without re-running, and most non-OOM
      errors are batch-level — bug, dtype mismatch, etc.).
    * On CUDA OOM with bs>1 → free memory, **re-collate each row solo and
      retry**, so non-pathological samples in the batch still get a real
      prediction. Only the genuinely-too-big rows get marked ``failed``.

    The per-row retry needs ``collate_fn`` (the backend's row→batch builder)
    so it can produce a fresh single-sample batch from the original row dict.
    The wrapped DataLoader's collate_fn stashes the raw rows under
    ``batch["_rows"]`` so we can pass them back through.
    """
    results: list[dict] = []
    iterator = tqdm(
        loader, total=len(loader), desc="infer", disable=not show_progress,
    )

    def _ok(raw: str, gt, vid) -> dict:
        return {
            "id": vid, "ground_truth": gt,
            "prediction": parser(raw), "raw": raw, "failed": False,
        }

    def _fail(err: str, gt, vid) -> dict:
        return {
            "id": vid, "ground_truth": gt,
            "prediction": None, "raw": None,
            "failed": True, "error": err,
        }

    for batch in iterator:
        answers = batch["answers"]
        ids = batch["ids"]
        rows = batch.get("_rows")
        try:
            decoded = backend.generate(
                batch["inputs"], max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Per-row retry path: only on real OOM, when we have >1 sample,
            # and only when the orchestrator handed us the raw rows.
            if _is_oom(e) and len(answers) > 1 and rows is not None and collate_fn is not None:
                tqdm.write(
                    f"[INFER] batch OOM (n={len(answers)}); "
                    "re-running each row solo…",
                )
                for row, gt, vid in zip(rows, answers, ids):
                    try:
                        single = collate_fn([row])
                        decoded_one = backend.generate(
                            single["inputs"], max_new_tokens=max_new_tokens,
                        )
                        results.append(_ok(decoded_one[0], gt, vid))
                    except Exception as e2:
                        results.append(_fail(f"{type(e2).__name__}: {e2}", gt, vid))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                continue

            # Whole-batch fail (non-OOM, or bs==1, or no rows captured).
            tqdm.write(
                f"[INFER] batch failed (n={len(answers)}): {err[:200]}",
            )
            for gt, vid in zip(answers, ids):
                results.append(_fail(err, gt, vid))
            continue

        # Happy path — full batch decoded.
        for raw, gt, vid in zip(decoded, answers, ids):
            results.append(_ok(raw, gt, vid))
    return results


def run_inference_api(
    backend: ApiBackend, rows: list[dict],
    *, max_new_tokens: int, parser: Callable[[str | None], str | None],
    concurrency: int, show_progress: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    existing_results: list[dict] | None = None,
) -> list[dict]:
    """Per-row HTTP inference under a ThreadPoolExecutor. Each row's exception
    is caught and turned into a ``failed=True`` marker so a transient API
    error doesn't sink the whole sweep.

    Crash safety: when ``checkpoint_every > 0`` and ``checkpoint_path`` is
    given, every ``checkpoint_every`` completions get appended to the file
    on disk. ``existing_results`` are rewritten first (in 'w' mode) so the
    file stays consistent with what ``load_existing_predictions`` would
    re-read on resume — corrupt trailing lines from a prior run get cleaned
    up. ``_write_outputs`` rewrites the file canonically at the very end.
    """
    results: list[dict] = []
    do_checkpoint = checkpoint_every > 0 and checkpoint_path is not None
    pending_flush: list[dict] = []

    if do_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite existing rows so the on-disk file is clean (drops any
        # corrupt trailing line that load_existing_predictions tolerated)
        # before we start appending new ones.
        with checkpoint_path.open("w") as f:
            for r in existing_results or []:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _flush() -> None:
        if not pending_flush:
            return
        with checkpoint_path.open("a") as f:
            for r in pending_flush:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pending_flush.clear()

    def _one(row: dict) -> str:
        return backend.infer_one(row, max_new_tokens=max_new_tokens)

    # Soft-fail per row: real datasets carry the occasional corrupt video,
    # oversized request, or transient API hiccup. Halting on the first
    # exception makes a single bad row poison the whole sweep. Instead,
    # mark the row failed (with the full traceback in `error`) and keep
    # going — the user can grep the predictions.jsonl / log for systemic
    # issues. Tracebacks are also printed to stderr so they surface live.
    ex = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futs = {ex.submit(_one, r): r for r in rows}
    try:
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="infer",
            disable=not show_progress,
        ):
            row = futs[fut]
            vid = row.get("id")
            gt = row.get("answer")
            try:
                raw = fut.result()
                rec = {
                    "id": vid,
                    "ground_truth": gt,
                    "prediction": parser(raw),
                    "raw": raw,
                    "failed": False,
                }
            except Exception as e:
                tb_str = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__),
                )
                tqdm.write(
                    f"[INFER] api row failed (vid={vid}): "
                    f"{type(e).__name__}: {e}",
                )
                tqdm.write(tb_str)
                rec = {
                    "id": vid,
                    "ground_truth": gt,
                    "prediction": None,
                    "raw": None,
                    "failed": True,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": tb_str,
                }
            results.append(rec)
            if do_checkpoint:
                pending_flush.append(rec)
                if len(pending_flush) >= checkpoint_every:
                    _flush()
    finally:
        if do_checkpoint:
            _flush()
    ex.shutdown(wait=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    # Backend selection.
    ap.add_argument(
        "--backend", default="timeomni_v",
        choices=backends.list_names(),
        help="Inference backend.",
    )
    ap.add_argument("--model_path", required=True)
    # TimeOmni-v-only.
    ap.add_argument("--chronos_path", default=None)
    ap.add_argument("--adapter_path", default=None,
                    help="LoRA adapter; omit for zero-shot.")
    ap.add_argument("--fusion_mode", default="block_adjacent")
    # Test data.
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True, type=Path)
    ap.add_argument("--metrics_path", default=None, type=Path,
                    help="Where to write the metrics file. Defaults to "
                         "<out_jsonl>.metrics.json.")
    # Generation.
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    # Local-batching knobs.
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Fixed batch size; ignored if --dynamic_bs_max_tokens > 0.")
    ap.add_argument("--dynamic_bs_max_tokens", type=int, default=45000)
    ap.add_argument("--dynamic_bs_max_bs", type=int, default=100)
    # Frame / pixel knobs (per-backend best-effort interpretation).
    ap.add_argument("--video_min_pixels", type=int, default=DEFAULT_VIDEO_MIN_PIXELS)
    ap.add_argument("--video_max_pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    ap.add_argument("--image_min_pixels", type=int, default=DEFAULT_IMAGE_MIN_PIXELS,
                    help="Per-image pixel floor (smart_resize). Default 2x video.")
    ap.add_argument("--image_max_pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS,
                    help="Per-image pixel ceiling (smart_resize). Default 2x video.")
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--min_frames", type=int, default=DEFAULT_MIN_FRAMES)
    # Ablations. ``--no_vision`` drops both image AND video inputs (whichever
    # the row carries).
    ap.add_argument("--no_vision", action="store_true",
                    help="Drop image and video inputs entirely (ts + text only).")
    ap.add_argument("--no_timeseries", action="store_true")
    # API-only knobs.
    ap.add_argument("--api_provider", default="proxy",
                    choices=["proxy", "openai", "google"],
                    help="proxy = openai SDK against an OpenAI-compatible URL.")
    ap.add_argument("--api_base_url", default=None,
                    help="Defaults to the TS_bench proxy when provider=proxy.")
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")
    ap.add_argument("--api_concurrency", type=int, default=8)
    ap.add_argument(
        "--api_checkpoint_every", type=int, default=20,
        help="API backends only: append completed rows to --out_jsonl every "
             "N completions for crash safety. 0 = disabled (final write only).",
    )
    # InternVL-specific knobs are no longer exposed on the CLI: the backend
    # derives per-row ``num_frames`` from --fps / --min_frames / --max_frames
    # (the same Qwen-family knobs above) and forces --batch_size=1.
    # Internal: legacy alias plumbing — keep do_sample_frames=True implicit;
    # exposing a flag would let users hit a hard error on backends that
    # don't support dense frames. No-op pass-through for now.
    args = ap.parse_args()

    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK. Absent → single-process run.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    include_vision = not args.no_vision
    include_timeseries = not args.no_timeseries

    BackendCls = backends.get(args.backend)
    backend = BackendCls(
        args=args,
        include_vision=include_vision,
        include_timeseries=include_timeseries,
    )

    if backend.is_local:
        if world_size > 1:
            # Long timeout: the post-loop all_gather_object is the first NCCL
            # collective (nothing inside the inference loop talks across ranks),
            # so NCCL bootstraps lazily there. With pathological samples (e.g.
            # HoloAssist clips that exceed the 131k context and trigger the
            # per-row OOM-retry path), one rank can spend an extra hour after
            # its peers reach the gather; the default 600 s TCPStore timeout
            # then aborts the bootstrap and the whole job dies. 2 h is enough
            # headroom for any reasonable single-shard skew.
            dist.init_process_group(
                backend="nccl",
                timeout=_dt.timedelta(hours=2),
            )
            torch.cuda.set_device(local_rank)
            device_map = {"": local_rank}
        else:
            device_map = "auto"
        backend.load(device_map=device_map)
    else:
        if world_size > 1:
            raise SystemExit(
                f"--backend {backend.name} is API-only and cannot run under "
                "torchrun; rerun without it (single process, --api_concurrency).",
            )
        backend.load()

    ds = TimeOmniVDataset(args.test_jsonl)

    # Resume support: keep prior rows, skip dataset indices whose id is
    # already covered. Same on every rank for consistent shard slicing.
    existing_results, done_ids = load_existing_predictions(args.out_jsonl)
    if rank == 0 and existing_results:
        print(
            f"[INFER] resume: {len(existing_results)} rows already in "
            f"{args.out_jsonl} ({len(done_ids)} unique ids); "
            f"will only infer the remaining {len(ds) - len(done_ids)} samples",
            flush=True,
        )
    rows = ds.rows
    remaining_indices = [
        i for i in range(len(ds))
        if rows[i].get("id") not in done_ids
    ]

    # Detect the test set's task. Prediction-style datasets (forecasting,
    # regression) carry free-form numeric outputs that the classification
    # parser can't extract — running it would just spam predictions.jsonl
    # with `prediction: null` and fill the metrics sidecar with a huge
    # `ground_truth_distribution` of unique forecast strings. eval.py picks
    # up the regression scoring (parse_forecast → MAE/MSE/MAPE/PCC) from
    # `raw`, so for prediction we route raw text into the `prediction` field
    # as-is and skip the classification-style parser + distributions.
    task_set = {r.get("task") for r in ds.rows if r.get("task")}
    task = next(iter(task_set)) if len(task_set) == 1 else "classification"
    if task == "prediction":
        # Extract just the <forecast>...</forecast> block from raw so the
        # `prediction` field is the clean forecast payload (drops thinking
        # / preamble). When no opening tag is present, parser returns None
        # and eval.py falls back to `raw` for scoring.
        parser = extract_forecast_block
        if rank == 0:
            print(
                f"[INFER] backend={backend.name} task=prediction "
                f"(extracting <forecast>...</forecast> block into the "
                f"`prediction` field; eval.py rescore runs the full "
                f"forecast parser on it).",
                flush=True,
            )
    else:
        # Classification: build the answer parser from the test set's ground-
        # truth labels so the regex auto-adapts (A-D, A/B, 0-39, mixed sets, …).
        label_set = sorted({
            r["answer"] for r in ds.rows
            if isinstance(r.get("answer"), str) and r["answer"]
        })
        parser = build_parser(label_set)
        if rank == 0:
            preview = label_set[:12] + (["…"] if len(label_set) > 12 else [])
            print(
                f"[INFER] backend={backend.name} task={task} parser labels "
                f"(n={len(label_set)}): {preview}",
                flush=True,
            )

    # Branch on backend.is_local. Local: DataLoader (+ optional dynamic-bs)
    # + DDP gather. API: ThreadPoolExecutor; world_size already asserted == 1.
    if backend.is_local:
        local_results = _run_local(
            backend, ds, args, rank, world_size,
            remaining_indices=remaining_indices,
            parser=parser,
        )
    else:
        if remaining_indices:
            api_rows = [ds[i] for i in remaining_indices]
            local_results = run_inference_api(
                backend, api_rows,
                max_new_tokens=args.max_new_tokens,
                parser=parser,
                concurrency=args.api_concurrency,
                show_progress=(rank == 0),
                checkpoint_path=args.out_jsonl,
                checkpoint_every=args.api_checkpoint_every,
                existing_results=existing_results,
            )
        else:
            if rank == 0:
                print(
                    "[INFER] resume: every sample already covered, skipping "
                    "generation; eval.py will rescore the existing predictions.",
                    flush=True,
                )
            local_results = []

    # Gather per-rank lists onto every rank (local + DDP only); rank 0 then
    # prepends the resumed rows and writes the merged file.
    if backend.is_local and world_size > 1:
        # Release KV-cache + last-batch tensors from the PyTorch caching
        # allocator back to CUDA so NCCL's own cudaMalloc inside
        # all_gather_object has free memory; otherwise large models (e.g.
        # Qwen3-VL-30B) OOM here right after inference completes.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist.barrier()
        gathered: list[list[dict]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)
        new_results = [r for shard in gathered for r in shard]
    else:
        new_results = local_results
    all_results = existing_results + new_results

    if rank == 0:
        _write_outputs(args, all_results, existing_results, new_results,
                       backend_name=backend.name,
                       include_vision=include_vision,
                       include_timeseries=include_timeseries,
                       world_size=world_size,
                       task=task)

    if backend.is_local and world_size > 1:
        dist.destroy_process_group()


def _run_local(
    backend: LocalHFBackend, ds: TimeOmniVDataset, args, rank: int, world_size: int,
    *, remaining_indices: list[int], parser: Callable[[str | None], str | None],
) -> list[dict]:
    """Build the local DataLoader (dyn-bs if backend supports it, fixed batch
    otherwise) and drive ``run_inference_local``. Returns the per-rank result
    list (the caller does the all_gather + write).

    The backend's collator is wrapped to also stash the raw row dicts under
    ``batch["_rows"]`` — needed by the per-row OOM retry path in
    ``run_inference_local``. The inner (unwrapped) collator is then handed
    to the runner so retries can re-collate a single row directly.
    """
    inner_collator = backend.make_collator()

    def collator(rows: list[dict]) -> dict:
        # We pass `list(rows)` (not a generator) because some torch sampler
        # paths exhaust the iterable inside the collate_fn before our hook.
        out = inner_collator(list(rows))
        out["_rows"] = list(rows)
        return out

    use_dyn_bs = (
        backend.supports_dynamic_bs and args.dynamic_bs_max_tokens > 0
    )
    if use_dyn_bs:
        lengths = backend.compute_lengths(ds)
        if lengths is None:
            use_dyn_bs = False

    if use_dyn_bs:
        # Resume: feed only the remaining rows + their lengths to the sampler.
        # Lengths cache stays full-dataset (cheap to slice) so resuming a
        # partly-done eval doesn't re-tokenize.
        if len(remaining_indices) < len(ds):
            remaining_lengths = [lengths[i] for i in remaining_indices]
            ds_for_loader = Subset(ds, remaining_indices)
        else:
            remaining_lengths = lengths
            ds_for_loader = ds
        sampler = TokenBudgetBatchSampler(
            lengths=remaining_lengths,
            max_tokens=args.dynamic_bs_max_tokens,
            max_bs=args.dynamic_bs_max_bs,
            rank=rank, world_size=world_size,
            seed=42, truncate_to_world_size=False,
        )
        sampler.set_epoch(0)
        if rank == 0:
            print(
                f"[INFER] dyn-bs enabled: N={len(remaining_lengths)} "
                f"(of {len(lengths)}) max_tokens={args.dynamic_bs_max_tokens} "
                f"max_bs={args.dynamic_bs_max_bs} world_size={world_size} "
                f"batches_this_rank={len(sampler)}",
                flush=True,
            )
        loader = DataLoader(ds_for_loader, batch_sampler=sampler, collate_fn=collator)
    else:
        # Fixed batch + interleaved per-rank shard. Resume: drop indices whose
        # id is already done before building the Subset.
        remaining_set = set(remaining_indices)
        rank_indices = [
            i for i in range(rank, len(ds), world_size) if i in remaining_set
        ]
        ds_shard = Subset(ds, rank_indices)
        loader = DataLoader(
            ds_shard, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator,
        )
        if rank == 0:
            print(
                f"[INFER] fixed batch: n_remaining={len(remaining_set)} "
                f"batch_size={args.batch_size} world_size={world_size} "
                f"batches_this_rank={len(loader)}",
                flush=True,
            )

    if remaining_indices:
        return run_inference_local(
            backend, loader,
            max_new_tokens=args.max_new_tokens,
            parser=parser,
            collate_fn=inner_collator,
            show_progress=(rank == 0),
        )
    if rank == 0:
        print(
            "[INFER] resume: every sample already covered, skipping "
            "generation; eval.py will rescore the existing predictions.",
            flush=True,
        )
    return []


def _write_outputs(
    args, all_results: list[dict], existing_results: list[dict],
    new_results: list[dict], *, backend_name: str,
    include_vision: bool, include_timeseries: bool, world_size: int,
    task: str,
) -> None:
    """Write predictions.jsonl and the metrics sidecar; print the summary.

    Branches on ``task``: classification reports per-class distributions
    and accuracy; prediction reports only generation success / failure
    (regression metrics live in eval.py's metrics.json — running the
    forecast parser here would duplicate that work)."""
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed = 0
    correct = 0
    gt_counter: Counter = Counter()
    pred_counter: Counter = Counter()
    is_classification = task == "classification"
    with args.out_jsonl.open("w") as fout:
        for r in all_results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r.get("failed"):
                failed += 1
                if is_classification:
                    gt_counter[r["ground_truth"]] += 1
                    pred_counter["_FAILED_"] += 1
                continue
            succeeded += 1
            if is_classification:
                gt_counter[r["ground_truth"]] += 1
                pred_counter[r["prediction"] or "_NONE_"] += 1
                if r["prediction"] == r["ground_truth"]:
                    correct += 1
    total = len(all_results)

    success_rate = succeeded / total if total else 0.0

    metrics: dict = {
        "task": task,
        "backend": backend_name,
        "adapter": args.adapter_path or None,
        "test_jsonl": args.test_jsonl,
        "out_jsonl": str(args.out_jsonl),
        "include_vision": include_vision,
        "include_timeseries": include_timeseries,
        "world_size": world_size,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": success_rate,
        "resumed_from_existing": len(existing_results),
        "newly_inferred": len(new_results),
    }
    if is_classification:
        accuracy_over_total = correct / total if total else 0.0
        accuracy_over_succeeded = correct / succeeded if succeeded else 0.0
        metrics.update({
            "correct": correct,
            "accuracy_over_total": accuracy_over_total,
            "accuracy_over_succeeded": accuracy_over_succeeded,
            "ground_truth_distribution": dict(gt_counter),
            "prediction_distribution": dict(pred_counter),
        })

    metrics_path = args.metrics_path or args.out_jsonl.with_suffix(
        args.out_jsonl.suffix + ".metrics.json",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Backend  : {backend_name}")
    print(f"Task     : {task}")
    print(f"Adapter  : {args.adapter_path or '(zero-shot, no adapter)'}")
    print(f"Test set : {args.test_jsonl}  (n={total}, world_size={world_size})")
    if existing_results:
        print(
            f"Resumed: kept {len(existing_results)} prior rows, "
            f"newly inferred {len(new_results)}",
        )
    print(f"Succeeded: {succeeded}/{total} (success_rate={success_rate:.4f})")
    print(f"Failed   : {failed}/{total}")
    if is_classification:
        print(
            f"Accuracy over total    : {correct}/{total} = "
            f"{metrics['accuracy_over_total']:.4f}",
        )
        print(
            f"Accuracy over succeeded: {correct}/{succeeded} = "
            f"{metrics['accuracy_over_succeeded']:.4f}",
        )
        print("Ground truth distribution:", dict(gt_counter))
        print("Prediction distribution :", dict(pred_counter))
    else:
        print("(prediction task — regression metrics in eval.py's metrics.json)")
    print(f"Metrics written to: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
