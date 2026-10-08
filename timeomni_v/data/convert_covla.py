"""Convert CoVLA jsonl (inline timeseries + long video paths) → TimeOmni-v unified format.

In a single pass per row this script:
  1. Parses the inline <timeseries> block into a per-sample CSV.
  2. Slices the source video at [start_frame, end_frame] into a per-sample mp4.
  3. Probes the sliced clip for ``(w, h, source_fps, nb_frames, duration)``
     and writes it as ``video_meta``; writes ``ts_shape=[T, C]`` from the
     TS DataFrame. These are consumed by
     ``timeomni_v.data.length_estimate.compute_exact_lengths`` to skip the
     ffprobe / csv-scan pass entirely (so lr-schedule / max_steps is known
     at startup without a multi-minute probe).
  4. Emits one unified TimeOmni-v jsonl row: original prompt kept intact
     (inline ``<timeseries>...</timeseries>`` body preserved) AND a
     ``timeseries_path`` pointing at the CSV.

Both training modes consume this same jsonl:
  * ``--mode timeomni_v`` strips the inline TS body in the collator and uses
    ``timeseries_path`` (TS-as-tensor).
  * ``--mode baseline`` ignores ``timeseries_path`` and keeps the inline
    TS text (optionally downsampled to a token budget via the collator).

All other CoVLA fields are dropped; the source ``video_id`` is kept under
``id`` for traceability and aligned with the unified schema (image+TS jsonls
also use ``id``). Every row is tagged ``task: "classification"`` since CoVLA
is a multiple-choice braking-reason classification benchmark.

Usage:
    # Fresh conversion (slice videos + write meta):
    python -m timeomni_v.data.convert_covla \\
        --in-jsonl CoVLA-Dataset/all_brake_reasons_final_train.jsonl \\
        --out-jsonl CoVLA-Dataset/all_brake_reasons_final_train.jsonl \\
        --csv-dir CoVLA-Dataset/timeseries \\
        --clip-dir CoVLA-Dataset/video_clips \\
        [--limit 10]   # optional: first N rows only (smoke)

    # Clips + csvs already exist — just backfill video_meta / ts_shape into
    # an existing TimeOmni-v jsonl:
    python -m timeomni_v.data.convert_covla --annotate-only \\
        --jsonl CoVLA-Dataset/all_brake_reasons_final_train.jsonl

This script is the ONLY dataset-specific code in the repo. Model/processor/
dataset/training code is generic and never imports from here.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from timeomni_v.data.probe import probe_ts_shape, probe_video_meta

TS_BLOCK_RE = re.compile(r"<timeseries>(.*?)</timeseries>", re.DOTALL)
DATA_LINE_RE = re.compile(r"^[\d.]+:\s*(.+)$")


def _video_meta_dict(meta) -> dict | None:
    """Shape the 5-tuple from probe_video_meta into the jsonl dict form."""
    if meta is None:
        return None
    w, h, sfps, nf, dur = meta
    return {
        "width": int(w), "height": int(h),
        "source_fps": float(sfps),
        "nb_frames": int(nf),
        "duration": float(dur),
    }


def parse_timeseries_block(text: str) -> pd.DataFrame:
    """Extract numeric rows from a <timeseries>...</timeseries> block into a DataFrame.

    Each data line looks like: "25.5: vEgo=6.517, aEgo=0.197, ..., lead_a=null".
    Returns a DataFrame with one column per variable; "null" or empty becomes NaN.
    """
    match = TS_BLOCK_RE.search(text)
    if not match:
        raise ValueError("no <timeseries> block found")
    body = match.group(1)

    records: list[dict[str, float]] = []
    for line in body.splitlines():
        m = DATA_LINE_RE.match(line.strip())
        if not m:
            continue
        kv_part = m.group(1)
        row: dict[str, float] = {}
        for pair in kv_part.split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            row[k] = float("nan") if v in ("null", "") else float(v)
        if row:
            records.append(row)

    if not records:
        raise ValueError("<timeseries> block had no data rows")

    return pd.DataFrame.from_records(records)


def probe_video_fps(video_path: Path) -> float:
    """Use ffprobe to read the source video's avg_frame_rate as a float."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    num, den = result.stdout.strip().split("/")
    num_f, den_f = float(num), float(den)
    if den_f == 0:
        return 0.0
    return num_f / den_f


def slice_video(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    """Copy a segment with -c copy (stream copy, near-instant). Overwrites dst.

    Caveat: `-ss` before `-i` with `-c copy` seeks to the nearest keyframe, so
    clip start may be offset by up to the GOP length (typically ≤ 1s). For
    video-QA training on multi-second segments this is acceptable.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration_sec:.3f}",
        "-an",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def convert_entry(
    entry: dict,
    csv_dir: Path,
    clip_dir: Path,
    fps_cache: dict[Path, float],
    prompt_only: bool = False,
) -> dict:
    """Convert one CoVLA row into the unified TimeOmni-v row.

    Returns: ``{id, task, video_path, timeseries_path, prompt, answer}`` where
    ``prompt`` retains the inline ``<timeseries>...</timeseries>`` body. The
    ``id`` field carries the source CoVLA ``video_id`` (renamed for parity
    with the image+TS jsonls); ``task`` is ``"classification"`` for CoVLA.

    ``prompt_only=True`` skips TS-CSV writing and video slicing. The derived
    paths must already exist on disk. Use this to refresh prompts after the
    source prompt text changes without re-running the expensive work.
    """
    csv_dir = Path(csv_dir)
    clip_dir = Path(clip_dir)
    if not prompt_only:
        csv_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)

    video_id = entry["video_id"]
    start_frame = int(entry["start_frame"])
    end_frame = int(entry["end_frame"])
    src_video = Path(entry["video_path"])

    # 1. Time-series CSV. In prompt-only mode we read the shape from the
    # existing file; in fresh mode we use the in-memory DataFrame's shape
    # directly (no extra IO).
    csv_name = f"{video_id}_{start_frame}_{end_frame}.csv"
    csv_path = csv_dir / csv_name
    if prompt_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"prompt-only mode: missing csv {csv_path}")
        ts_shape = probe_ts_shape(str(csv_path))
    else:
        df = parse_timeseries_block(entry["prompt"])
        df.to_csv(csv_path, index=False)
        ts_shape = (int(df.shape[0]), int(df.shape[1]))

    # 2. Video clip (ffmpeg-sliced). Reuse if it already exists.
    clip_name = f"{video_id}_{start_frame}_{end_frame}.mp4"
    clip_path = clip_dir / clip_name
    if prompt_only:
        if not (clip_path.exists() and clip_path.stat().st_size > 0):
            raise FileNotFoundError(f"prompt-only mode: missing clip {clip_path}")
    elif not (clip_path.exists() and clip_path.stat().st_size > 0):
        if src_video not in fps_cache:
            fps_cache[src_video] = probe_video_fps(src_video)
        fps = fps_cache[src_video]
        if fps <= 0:
            raise RuntimeError(f"could not probe fps for {src_video}")
        start_sec = start_frame / fps
        duration_sec = max(1e-3, (end_frame - start_frame + 1) / fps)
        slice_video(src_video, clip_path, start_sec, duration_sec)

    # 2b. Probe the sliced clip for the meta that `compute_exact_lengths`
    # needs — same whether we just wrote it or are reusing an existing clip.
    video_meta = _video_meta_dict(probe_video_meta(str(clip_path)))

    # 3. Unified row — prompt retains inline TS body; baseline uses it as-is,
    # timeomni_v strips it in the collator via strip_ts_block(). video_meta /
    # ts_shape are optional hints consumed by the length estimator. The
    # local var `video_id` is the source CoVLA key; the output field is
    # `id` for parity with the image+TS jsonl schema.
    row: dict = {
        "id": video_id,
        "task": "classification",
        "video_path": str(clip_path),
        "timeseries_path": str(csv_path),
        "prompt": entry["prompt"],
        "answer": entry["ground_truth"],
    }
    if video_meta is not None:
        row["video_meta"] = video_meta
    if ts_shape is not None:
        row["ts_shape"] = [int(ts_shape[0]), int(ts_shape[1])]
    return row


def _worker_convert(args: tuple[dict, str, str, bool]) -> tuple[int, dict]:
    """Worker for ProcessPoolExecutor: convert a single entry.

    Returns (line_no, row). fps_cache is per-worker (no sharing).
    """
    entry, csv_dir, clip_dir, prompt_only = args
    line_no = entry["_line_no"]
    _entry = {k: v for k, v in entry.items() if k != "_line_no"}
    row = convert_entry(
        _entry, csv_dir=Path(csv_dir), clip_dir=Path(clip_dir),
        fps_cache={}, prompt_only=prompt_only,
    )
    return line_no, row


def _annotate_only(jsonl_path: Path, num_workers: int, force: bool) -> None:
    """Backfill video_meta / ts_shape into an existing unified TimeOmni-v jsonl.

    Use when clips + csvs already exist on disk and you only need the meta
    fields added. Rewrites `jsonl_path` in place via .tmp + rename. Skips
    rows that already have both fields unless --force is passed."""
    with jsonl_path.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"[ANNOTATE] loaded {len(rows)} rows from {jsonl_path}")

    v_todo = [
        i for i, r in enumerate(rows)
        if force or not isinstance(r.get("video_meta"), dict)
    ]
    t_todo = [
        i for i, r in enumerate(rows)
        if r.get("timeseries_path") and
           (force or not isinstance(r.get("ts_shape"), (list, tuple)))
    ]
    print(f"[ANNOTATE] to probe: {len(v_todo)} videos, {len(t_todo)} TS csvs "
          f"(workers={num_workers})")

    if v_todo:
        done = 0
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = {pool.submit(probe_video_meta, rows[i]["video_path"]): i for i in v_todo}
            for fut in as_completed(futs):
                i = futs[fut]
                vm = _video_meta_dict(fut.result())
                if vm is not None:
                    rows[i]["video_meta"] = vm
                done += 1
                if done % 1000 == 0:
                    print(f"[ANNOTATE]   video {done}/{len(v_todo)}", flush=True)

    if t_todo:
        done = 0
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = {pool.submit(probe_ts_shape, rows[i]["timeseries_path"]): i for i in t_todo}
            for fut in as_completed(futs):
                i = futs[fut]
                shape = fut.result()
                if shape is not None:
                    rows[i]["ts_shape"] = [int(shape[0]), int(shape[1])]
                done += 1
                if done % 2000 == 0:
                    print(f"[ANNOTATE]   ts {done}/{len(t_todo)}", flush=True)

    tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)

    n_vm = sum(1 for r in rows if isinstance(r.get("video_meta"), dict))
    n_ts = sum(1 for r in rows if isinstance(r.get("ts_shape"), (list, tuple)))
    print(f"[ANNOTATE] wrote {jsonl_path}: {n_vm}/{len(rows)} video_meta, "
          f"{n_ts}/{len(rows)} ts_shape")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotate-only", action="store_true",
        help="Skip slicing/parsing. Probe existing clips+csvs and write "
             "video_meta / ts_shape back into --jsonl in place.",
    )
    ap.add_argument("--jsonl", type=Path,
                   help="[annotate-only mode] path to unified timeomni_v jsonl to update in place.")
    ap.add_argument("--force", action="store_true",
                   help="[annotate-only mode] re-probe rows that already have meta.")
    ap.add_argument("--in-jsonl", type=Path)
    ap.add_argument("--out-jsonl", type=Path)
    ap.add_argument("--csv-dir", type=Path)
    ap.add_argument("--clip-dir", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="First N rows only (smoke).")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel ffmpeg / probe workers. 1 = sequential.",
    )
    ap.add_argument(
        "--prompt-only",
        action="store_true",
        help="Refresh prompts only: reuse existing TS CSVs and video clips, "
             "skip ffmpeg slicing and TS parsing. Fails if derived files are missing.",
    )
    args = ap.parse_args()

    if args.annotate_only:
        if args.jsonl is None:
            ap.error("--annotate-only requires --jsonl")
        _annotate_only(args.jsonl, args.num_workers, args.force)
        return

    missing = [n for n in ("in_jsonl", "out_jsonl", "csv_dir", "clip_dir")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"missing required args for conversion: {missing}")

    entries: list[dict] = []
    with args.in_jsonl.open() as fin:
        for line_no, line in enumerate(fin, 1):
            if args.limit is not None and line_no > args.limit:
                break
            entry = json.loads(line)
            entry["_line_no"] = line_no
            entries.append(entry)

    fout = args.out_jsonl.open("w")
    n = 0

    if args.num_workers <= 1:
        fps_cache: dict[Path, float] = {}
        try:
            for entry in entries:
                line_no = entry.pop("_line_no")
                try:
                    row = convert_entry(
                        entry, csv_dir=args.csv_dir, clip_dir=args.clip_dir,
                        fps_cache=fps_cache, prompt_only=args.prompt_only,
                    )
                except (ValueError, KeyError, RuntimeError,
                        subprocess.CalledProcessError) as exc:
                    raise RuntimeError(f"line {line_no}: {exc}") from exc
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        finally:
            fout.close()
    else:
        # Parallel: write rows as they complete (no order guarantee — shuffle-safe).
        work = [
            (e, str(args.csv_dir), str(args.clip_dir), args.prompt_only)
            for e in entries
        ]
        try:
            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                for future in as_completed(
                    [pool.submit(_worker_convert, w) for w in work]
                ):
                    _line_no, row = future.result()
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    n += 1
                    if n % 200 == 0:
                        print(f"  converted {n}/{len(entries)}", flush=True)
        finally:
            fout.close()

    print(f"converted {n} entries → {args.out_jsonl}")


if __name__ == "__main__":
    main()
