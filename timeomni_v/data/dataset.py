"""Generic TimeOmni-v jsonl dataset.

Each line in the jsonl must contain:
    id               — sample identifier (replaces the older `video_id`)
    task             — one of "classification" / "prediction" (eval routing key)
    prompt           — the task prompt (may contain <timeseries></timeseries>)
    answer           — the target answer string
    timeseries_path  — optional, path to a per-sample TS csv (omit for
                       baseline runs that keep TS inline in the prompt)
    video_path       — optional, path to a short pre-sliced mp4 clip
    image_path       — optional, single path or list of paths for image
                       samples. Mutually exclusive with video_path.
    video_meta       — optional, populated by convert_covla
    image_meta       — optional, list of {width,height,channels} per image

Multi-source mode: passing a list of jsonl paths concatenates their rows in
order and exposes ``task_of_index`` (parallel to ``rows``, value = source
index 0..N-1) so a downstream batch sampler can constrain each batch to a
single source. ``source_paths`` and ``task_names`` keep the per-source
metadata for cache-key construction and logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from torch.utils.data import Dataset

# Qwen2.5-Omni's video processor pins temporal_patch_size=2 and requires
# `nframes ∈ [2, total_frames]`. A source clip with <2 frames triggers
# `ValueError: nframes should in interval [2, 1], but got 0` deep inside
# qwen_omni_utils. cuhk_x_har contains a small tail of such 1-frame clips
# (action segments shorter than one IMU sample period); skip them upfront.
MIN_VIDEO_FRAMES = 2


PathLike = Union[str, Path]


class TimeOmniVDataset(Dataset):
    def __init__(self, jsonl_path: PathLike | Iterable[PathLike]):
        if isinstance(jsonl_path, (str, Path)):
            paths = [Path(jsonl_path)]
        else:
            paths = [Path(p) for p in jsonl_path]
            if not paths:
                raise ValueError("TimeOmniVDataset requires at least one jsonl path")

        self.source_paths: list[Path] = paths
        self.task_names: list[str] = [p.name.split(".")[0] for p in paths]
        # Backwards-compatible single-path attribute. For multi-source mode it
        # holds the first path; callers that need every source must use
        # ``self.source_paths`` instead.
        self.path: Path = paths[0]

        kept_rows: list[dict] = []
        task_of_index: list[int] = []
        per_source_kept: list[int] = []
        for src_idx, path in enumerate(paths):
            with path.open() as f:
                raw_rows = [json.loads(l) for l in f if l.strip()]
            n_short = 0
            n_kept_here = 0
            for row in raw_rows:
                vm = row.get("video_meta")
                nb = int(vm["nb_frames"]) if isinstance(vm, dict) and "nb_frames" in vm else None
                if nb is not None and nb < MIN_VIDEO_FRAMES:
                    n_short += 1
                    continue
                kept_rows.append(row)
                task_of_index.append(src_idx)
                n_kept_here += 1
            per_source_kept.append(n_kept_here)
            if n_short:
                print(
                    f"[TimeOmniVDataset] {path.name}: dropped {n_short} clips with "
                    f"nb_frames < {MIN_VIDEO_FRAMES} (Qwen video reader requires "
                    f"≥ temporal_patch_size source frames). kept={n_kept_here}",
                    flush=True,
                )

        self.rows = kept_rows
        self.task_of_index = task_of_index
        self.per_source_kept = per_source_kept

        if len(paths) > 1:
            counts = ", ".join(
                f"{name}={n}" for name, n in zip(self.task_names, per_source_kept)
            )
            print(
                f"[TimeOmniVDataset] multi-source: {len(paths)} jsonls — {counts} "
                f"(total={len(self.rows)})",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "id": row.get("id"),
            "task": row.get("task"),
            "video_path": row.get("video_path"),
            "image_path": row.get("image_path"),
            "timeseries_path": row.get("timeseries_path"),
            # Populated by timeomni_v.data.convert_per_channel_forecasting for
            # forecast-head training rows; ForecastCollator reads it. None for
            # classification / legacy rows.
            "forecast_target_path": row.get("forecast_target_path"),
            "prompt": row["prompt"],
            "answer": row["answer"],
        }
