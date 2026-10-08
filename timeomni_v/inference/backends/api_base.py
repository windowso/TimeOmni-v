"""Shared helpers for API backends: frame extraction, base64 encoding, and a
small retry wrapper.

The frame extractor uses decord (already in the chronos env). We uniform-
sample ``num_segments`` frames the same way Qwen2.5-Omni's video processor
does in spirit — pick segment midpoints — then enforce the per-frame pixel
ceiling via PIL resize so the base64 payload doesn't blow past API request
limits. ``--video_min_pixels`` is intentionally ignored for API backends:
upscaling tiny videos costs paid bandwidth + tokens for no information gain.
"""

from __future__ import annotations

import base64
import io
import math
import random
import time
from typing import Callable

import numpy as np
from PIL import Image

from timeomni_v.data.probe import probe_video_meta


def _uniform_indices(nf: int, num_segments: int) -> list[int]:
    seg_size = nf / max(1, num_segments)
    return [
        int(min(nf - 1, max(0, round(seg_size / 2 + seg_size * i))))
        for i in range(num_segments)
    ]


def _decode_indexed_frames(
    video_path: str, num_segments: int, *, nb_meta: int,
) -> list:
    """Return ``num_segments`` uniformly-sampled RGB numpy frames.

    decord first (fast). On any DECORDError, fall back to PyAV → system
    ffmpeg, which has libdav1d so AV1 clips (e.g. agibot's) decode where
    decord's bundled ffmpeg can't. Also covers other codecs decord rejects
    on this env.
    """
    try:
        import decord
        from decord._ffi.base import DECORDError
    except Exception:
        decord = None
        DECORDError = ()

    if decord is not None:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            nf = len(vr)
            if nf <= 0:
                raise ValueError(f"empty video: {video_path}")
            indices = _uniform_indices(nf, num_segments)
            return [vr[i].asnumpy() for i in indices]
        except DECORDError:
            pass  # fall through to PyAV

    import av  # PyAV; uses system ffmpeg (libdav1d / libaom on this image)

    nf = nb_meta if nb_meta and nb_meta > 0 else 0
    if nf <= 0:
        # ffprobe didn't give us nb_frames — count by streaming decode once.
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            nf = stream.frames or 0
            if nf <= 0:
                nf = sum(1 for _ in container.decode(stream))
    if nf <= 0:
        raise ValueError(f"empty video (PyAV fallback): {video_path}")
    indices = _uniform_indices(nf, num_segments)
    target_set = set(indices)
    collected: dict[int, "object"] = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # AV1 frame-accurate seek is unreliable; sequential decode is fine
        # for short API clips and avoids GOP-key-frame surprises.
        for i, frame in enumerate(container.decode(stream)):
            if i in target_set:
                collected[i] = frame.to_ndarray(format="rgb24")
                if len(collected) == len(target_set):
                    break
    if not collected:
        raise ValueError(f"PyAV fallback decoded 0 frames from {video_path}")
    # Keep order; if any indices were past the actual end of stream, drop
    # them rather than padding with garbage.
    return [collected[i] for i in indices if i in collected]


def extract_frames_b64_jpeg(
    video_path: str, *, fps: float, min_frames: int, max_frames: int,
    max_pixels: int | None = None, jpeg_quality: int = 85,
) -> list[str]:
    """Uniformly sample frames and return them as base64-encoded JPEG strings.

    Frame-count math mirrors what Qwen2.5-Omni / Qwen3-Omni do (target =
    duration × fps, clamped to [min_frames, max_frames]). ``max_pixels`` (when
    set) caps each frame's pixel count by aspect-preserving downscale before
    JPEG encoding — protects per-request size + paid token budget.
    """
    meta = probe_video_meta(video_path)
    if meta is None:
        raise ValueError(f"could not probe video: {video_path}")
    _w, _h, src_fps, nb, dur = meta
    duration = dur if dur > 0 else (nb / src_fps if src_fps else 0.0)
    target = max(1, round(duration * fps))
    num_segments = max(min_frames, min(max_frames, target))

    arrays = _decode_indexed_frames(video_path, num_segments, nb_meta=nb)

    frames_b64: list[str] = []
    for arr in arrays:
        img = Image.fromarray(arr).convert("RGB")
        if max_pixels is not None and (img.width * img.height) > max_pixels:
            scale = math.sqrt(max_pixels / (img.width * img.height))
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return frames_b64


def encode_image_b64_jpeg(
    image_path: str, *, max_pixels: int | None = None, jpeg_quality: int = 85,
) -> str:
    """Read an image from disk, optionally downscale to ``max_pixels`` total
    area, and return it as a base64-encoded JPEG string. Mirrors what
    ``extract_frames_b64_jpeg`` does per-frame for video — kept separate so
    image rows pay no decord / probe cost."""
    img = Image.open(image_path).convert("RGB")
    if max_pixels is not None and (img.width * img.height) > max_pixels:
        scale = math.sqrt(max_pixels / (img.width * img.height))
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def normalize_image_paths(image_path: str | list[str] | None) -> list[str]:
    """Normalize the optional ``image_path`` field (str | list[str] | None)
    to a flat list of paths."""
    if image_path is None:
        return []
    if isinstance(image_path, str):
        return [image_path]
    return list(image_path)


def with_retry(fn: Callable, *args, max_attempts: int = 3,
              base_delay: float = 1.0, retry_exceptions: tuple = (Exception,),
              **kwargs):
    """Call ``fn(*args, **kwargs)`` with exponential backoff. Retries for
    ``retry_exceptions``; lets all other exceptions propagate immediately so
    permanent errors (e.g. ``BadRequestError`` for an oversized payload) fail
    fast and surface as a ``failed=True`` row instead of stalling for 10 s.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except retry_exceptions as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


__all__ = [
    "extract_frames_b64_jpeg",
    "encode_image_b64_jpeg",
    "normalize_image_paths",
    "with_retry",
]
