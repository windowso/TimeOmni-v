"""Lightweight metadata probes for video / TS csv files.

Kept in its own module (no torch / transformers deps) so that
``convert_covla`` can import them without pulling in the training stack.
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional


def probe_video_meta(path: str) -> Optional[tuple[int, int, float, int, float]]:
    """Return (width, height, source_fps, total_frames, duration) or None.

    Handles mp4s where ``nb_frames`` is missing / 0 by falling back to
    ``duration * avg_frame_rate`` (and vice-versa). Any error → None so the
    caller can emit a warning and skip / length-count without meta.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True, timeout=10,
        )
        stream = json.loads(result.stdout)["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        num, den = stream.get("avg_frame_rate", "0/1").split("/")
        num_f, den_f = float(num), float(den)
        source_fps = num_f / den_f if den_f > 0 else 0.0
        nb_frames_raw = stream.get("nb_frames", "")
        nb_frames = int(nb_frames_raw) if nb_frames_raw.isdigit() else 0
        duration_raw = stream.get("duration", "")
        try:
            duration = float(duration_raw) if duration_raw else 0.0
        except ValueError:
            duration = 0.0
        if nb_frames <= 0 and duration > 0 and source_fps > 0:
            nb_frames = int(duration * source_fps)
        if duration <= 0 and nb_frames > 0 and source_fps > 0:
            duration = nb_frames / source_fps
        return (width, height, source_fps, nb_frames, duration)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ValueError, KeyError, IndexError, FileNotFoundError):
        return None


def probe_image_meta(
    path: str | list[str],
) -> Optional[dict] | list[Optional[dict]]:
    """Return ``{width, height, channels}`` for an image, or a list of dicts
    when given a list of paths. Used by future image-dataset converters to
    populate ``image_meta`` in the unified jsonl so the length estimator can
    skip per-image PIL opens at startup.

    Any error returns None for that entry (caller can warn / skip).
    """
    from PIL import Image

    if isinstance(path, list):
        return [probe_image_meta(p) for p in path]
    try:
        with Image.open(path) as im:
            mode = im.mode
            channels = len(im.getbands())
            return {
                "width": int(im.width),
                "height": int(im.height),
                "channels": int(channels),
                "mode": mode,
            }
    except (OSError, ValueError):
        return None


def probe_ts_shape(path: str) -> Optional[tuple[int, int]]:
    """Return (T, C) for a TS csv — rows × columns, header row excluded.

    Mirrors ``load_and_pad_ts_csvs`` shape extraction without float parsing:
    line-count for T, first-line comma-count for C. A column literally
    named ``timestamp`` is excluded from C (load_and_pad_ts_csvs drops it
    too). Empty cells / trailing commas are fine — we only care about
    shape."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header:
                return None
            cols = [c.strip() for c in header.rstrip("\r\n").split(",")]
            n_cols = len(cols) - sum(1 for c in cols if c == "timestamp")
            n_rows = 0
            for _ in f:
                n_rows += 1
        return (n_rows, n_cols)
    except (OSError, ValueError):
        return None
