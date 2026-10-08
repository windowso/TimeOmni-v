"""Exact per-sample post-processor token counts.

Computes the number of tokens each sample will occupy in the model's input
sequence, WITHOUT decoding the video / images or running the processor
end-to-end:

* text tokens: run the tokenizer on the prompt transformed exactly as the
  collator does (optional downsample for baseline) with `<video>...</video>`
  already replaced by the Qwen video marker. Subtract 1 for the single
  `<|video_pad|>` placeholder that the processor expands to N visual tokens.

* visual tokens (video): ffprobe → (width, height, source_fps, total_frames);
  replicate the Qwen2.5-Omni video processor's sampling logic (target fps,
  temporal_patch_size rounding, min/max frame bounds, smart_resize with
  patch*merge factor) to compute `video_grid_thw`; final token count is
  `prod(grid_thw) // merge_size**2`.

* visual tokens (image): per-image ``smart_resize`` to honor the same pixel
  bounds, then ``(h_resized // patch_size) * (w_resized // patch_size) //
  merge_size**2``. Images are 2D (T=1) so the temporal grid contributes 1.

Results are cached to a sidecar JSON so the ffprobe + tokenize pass is paid
once per (dataset, fps, pixel bounds) tuple. Cache is invalidated on any
relevant knob change (see `_cache_key`).
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from timeomni_v.data.collator import DEFAULT_FPS
from timeomni_v.data.probe import probe_image_meta, probe_ts_shape, probe_video_meta

VIDEO_BLOCK_RE = re.compile(r"<video>.*?</video>", re.DOTALL)


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Port of transformers.qwen2_vl smart_resize (identical math)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too extreme: {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _sampled_num_frames(total_frames: int, source_fps: float, target_fps: float,
                      temporal_patch_size: int, min_frames: int, max_frames: int) -> int:
    """Replicate Qwen2VLVideoProcessor.sample_frames arithmetic for the
    `do_sample_frames=True, fps=...` code path. Returns the post-sampling
    frame count (always a multiple of temporal_patch_size)."""
    if total_frames <= 0 or source_fps <= 0:
        return temporal_patch_size
    max_frames = math.floor(min(max_frames, total_frames) / temporal_patch_size) * temporal_patch_size
    num_frames = total_frames / source_fps * target_fps
    num_frames = min(min(max(num_frames, min_frames), max_frames), total_frames)
    num_frames = math.floor(num_frames / temporal_patch_size) * temporal_patch_size
    return max(num_frames, temporal_patch_size)


def _exact_vis_tokens(meta: tuple[int, int, float, int, float] | None,
                    patch_size: int, temporal_patch_size: int, merge_size: int,
                    min_pixels: int, max_pixels: int,
                    do_sample_frames: bool, target_fps: float,
                    min_frames: int, max_frames: int) -> int:
    """Number of `<|video_pad|>` tokens the processor emits for this video.

    Two branches, mirroring the processor exactly:

    * do_sample_frames=True: Qwen2VLVideoProcessor.sample_frames runs —
      num_frames = clip(total/source_fps × target_fps, min_frames, max_frames),
      then floored to a multiple of temporal_patch_size.

    * do_sample_frames=False: the processor keeps every source frame; if the
      total isn't divisible by temporal_patch_size the last frame is repeated
      to pad (we round up).
    """
    if meta is None:
        return 0
    width, height, source_fps, total_frames, _duration = meta
    if total_frames <= 0:
        return 0
    if do_sample_frames:
        num_frames = _sampled_num_frames(
            total_frames, source_fps, target_fps,
            temporal_patch_size, min_frames, max_frames,
        )
    else:
        num_frames = total_frames
        rem = num_frames % temporal_patch_size
        if rem != 0:
            num_frames += temporal_patch_size - rem
    factor = patch_size * merge_size
    resized_h, resized_w = _smart_resize(height, width, factor, min_pixels, max_pixels)
    t_grid = num_frames // temporal_patch_size
    h_grid = resized_h // patch_size
    w_grid = resized_w // patch_size
    return (t_grid * h_grid * w_grid) // (merge_size * merge_size)


def _exact_image_tokens(image_meta: list[dict],
                       patch_size: int, merge_size: int,
                       min_pixels: int, max_pixels: int) -> int:
    """Number of `<|image_pad|>` tokens emitted across all images for one
    sample. Each image goes through ``smart_resize`` against the same pixel
    bounds the video path uses, then the per-image grid is
    ``(h_resized // patch_size) * (w_resized // patch_size)`` post-merge,
    i.e. divided by ``merge_size**2``. Images are 2D so temporal_patch_size
    doesn't apply (T_grid = 1)."""
    if not image_meta:
        return 0
    factor = patch_size * merge_size
    total = 0
    for m in image_meta:
        if not isinstance(m, dict) or "width" not in m or "height" not in m:
            continue
        w, h = int(m["width"]), int(m["height"])
        if w <= 0 or h <= 0:
            continue
        resized_h, resized_w = _smart_resize(h, w, factor, min_pixels, max_pixels)
        h_grid = resized_h // patch_size
        w_grid = resized_w // patch_size
        total += (h_grid * w_grid) // (merge_size * merge_size)
    return total


def _cache_key(video_min_pixels: int, video_max_pixels: int,
              image_min_pixels: int, image_max_pixels: int,
              patch_size: int, temporal_patch_size: int, merge_size: int,
              do_sample_frames: bool, target_fps: float,
              min_frames: int, max_frames: int,
              include_vision: bool,
              include_timeseries: bool, ts_patch_size: int,
              prompt_transform_skip_tasks: "frozenset[str] | set[str] | None",
              extra: dict | None) -> dict:
    return {
        "video_min_pixels": video_min_pixels,
        "video_max_pixels": video_max_pixels,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "patch_size": patch_size,
        "temporal_patch_size": temporal_patch_size,
        "merge_size": merge_size,
        "do_sample_frames": bool(do_sample_frames),
        "target_fps": float(target_fps),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
        "include_vision": bool(include_vision),
        "include_timeseries": bool(include_timeseries),
        "ts_patch_size": int(ts_patch_size),
        "prompt_transform_skip_tasks": (
            sorted(prompt_transform_skip_tasks) if prompt_transform_skip_tasks else []
        ),
        "extra": extra or {},
    }




def compute_exact_lengths(
    dataset,
    processor,
    video_min_pixels: int,
    video_max_pixels: int,
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    do_sample_frames: bool = False,
    target_fps: float = DEFAULT_FPS,
    min_frames: int = 4,
    max_frames: int = 768,
    prompt_transform: Optional[Callable[[str], str]] = None,
    prompt_transform_id: str | None = None,
    prompt_transform_skip_tasks: "frozenset[str] | set[str] | None" = None,
    include_vision: bool = True,
    include_timeseries: bool = False,
    ts_patch_size: int = 16,
    cache_path: Path | str | None = None,
    num_workers: int | None = None,
) -> list[int]:
    """Return exact post-processor token counts for every sample.

    The text count goes through ``processor.apply_chat_template`` so it
    matches the chat-template-wrapped sequence the collator produces (user
    turn + assistant turn + end markers). ``prompt_transform`` runs on the
    raw prompt before it is placed into the conversation's text element.

    ``include_vision`` covers both video and image branches: if the row
    carries ``video_path`` we replicate Qwen2VLVideoProcessor.sample_frames;
    if it carries ``image_path`` we replicate the per-image smart_resize.
    """
    from timeomni_v.data.chat_format import build_conversation

    if image_min_pixels is None:
        image_min_pixels = 2 * video_min_pixels
    if image_max_pixels is None:
        image_max_pixels = 2 * video_max_pixels

    tokenizer = processor.tokenizer
    video_processor = processor.video_processor
    patch_size = video_processor.patch_size
    temporal_patch_size = video_processor.temporal_patch_size
    merge_size = video_processor.merge_size

    key = _cache_key(
        video_min_pixels, video_max_pixels,
        image_min_pixels, image_max_pixels,
        patch_size, temporal_patch_size, merge_size,
        do_sample_frames, target_fps, min_frames, max_frames,
        include_vision, include_timeseries, ts_patch_size,
        prompt_transform_skip_tasks,
        {"n": len(dataset), "prompt_transform_id": prompt_transform_id},
    )
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text())
            if blob.get("key") == key and len(blob.get("lengths", [])) == len(dataset):
                print(f"[LENGTHS] loaded cached lengths from {cache_path}", flush=True)
                return blob["lengths"]
        except (json.JSONDecodeError, OSError):
            pass

    # 1. Build chat-templated text per sample (includes user + assistant
    # turn framing, one <|video_pad|> placeholder for video rows OR one
    # <|image_pad|> per image for image rows). We don't call process_mm_info
    # here — skipping video decode avoids paying ~10 min for a length pass.
    raw_rows = getattr(dataset, "rows", None)

    texts: list[str] = []
    video_paths: list[Optional[str]] = []
    image_paths: list[Optional[list[str]]] = []
    ts_paths: list[str] = []
    cached_video_metas: list[Optional[tuple]] = []
    cached_image_metas: list[Optional[list[dict]]] = []
    cached_ts_shapes: list[Optional[tuple[int, int]]] = []
    for i in range(len(dataset)):
        row = dataset[i]
        conv = build_conversation(
            row,
            answer=row["answer"],
            fps=target_fps,
            max_frames=max_frames,
            min_frames=min_frames,
            video_min_pixels=video_min_pixels,
            video_max_pixels=video_max_pixels,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            include_vision=include_vision,
            include_timeseries=include_timeseries,
            prompt_transform=prompt_transform,
            prompt_transform_skip_tasks=prompt_transform_skip_tasks,
        )
        text = processor.apply_chat_template(
            conv, add_generation_prompt=False, tokenize=False,
        )
        texts.append(text)

        raw = raw_rows[i] if raw_rows is not None else None
        if include_vision and row.get("video_path"):
            video_paths.append(row["video_path"])
            vm = raw.get("video_meta") if isinstance(raw, dict) else None
            if isinstance(vm, dict) and {"width", "height", "source_fps", "nb_frames"}.issubset(vm):
                cached_video_metas.append((
                    int(vm["width"]), int(vm["height"]),
                    float(vm["source_fps"]), int(vm["nb_frames"]),
                    float(vm.get("duration", 0.0)),
                ))
            else:
                cached_video_metas.append(None)
        else:
            video_paths.append(None)
            cached_video_metas.append(None)

        if include_vision and row.get("image_path"):
            paths = row["image_path"] if isinstance(row["image_path"], list) else [row["image_path"]]
            image_paths.append(paths)
            im = raw.get("image_meta") if isinstance(raw, dict) else None
            if isinstance(im, list) and len(im) == len(paths):
                cached_image_metas.append([m if isinstance(m, dict) else None for m in im])
            elif isinstance(im, dict):
                cached_image_metas.append([im])
            else:
                cached_image_metas.append(None)
        else:
            image_paths.append(None)
            cached_image_metas.append(None)

        if include_timeseries:
            ts_paths.append(row["timeseries_path"])
            ts = raw.get("ts_shape") if isinstance(raw, dict) else None
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                cached_ts_shapes.append((int(ts[0]), int(ts[1])))
            else:
                cached_ts_shapes.append(None)

    print(f"[LENGTHS] tokenizing {len(texts)} samples…", flush=True)
    tok_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    text_tokens = [len(ids) for ids in tok_ids]

    # 2. Probe video metadata in parallel — but only for rows where the
    # jsonl didn't already carry a cached `video_meta` block (populated by
    # `timeomni_v.data.annotate_meta`).
    nw = num_workers if num_workers is not None else min(32, (os.cpu_count() or 4))
    metas: list[Optional[tuple]] = list(cached_video_metas)
    if include_vision:
        missing_v = [
            i for i, m in enumerate(metas)
            if m is None and video_paths[i] is not None
        ]
        n_cached_v = sum(1 for i, m in enumerate(metas) if m is not None and video_paths[i] is not None)
        if missing_v:
            print(f"[LENGTHS] probing {len(missing_v)} videos with {nw} workers "
                  f"({n_cached_v} loaded from jsonl cache)…", flush=True)
            with ProcessPoolExecutor(max_workers=nw) as pool:
                fut_to_idx = {pool.submit(probe_video_meta, video_paths[i]): i for i in missing_v}
                done = 0
                for fut in as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    metas[i] = fut.result()
                    done += 1
                    if done % 2000 == 0:
                        print(f"[LENGTHS]   probed {done}/{len(missing_v)}", flush=True)
    else:
        print(f"[LENGTHS] include_vision=False — skipping video/image probes", flush=True)

    # 2a. Probe missing image_meta with PIL. Most image datasets ship
    # image_meta in the jsonl already so this is a fallback.
    image_metas: list[Optional[list[dict]]] = list(cached_image_metas)
    if include_vision:
        missing_i = [
            i for i, m in enumerate(image_metas)
            if m is None and image_paths[i] is not None
        ]
        if missing_i:
            print(f"[LENGTHS] probing {len(missing_i)} image rows with {nw} workers "
                  f"(missing image_meta in jsonl)…", flush=True)
            with ProcessPoolExecutor(max_workers=nw) as pool:
                fut_to_idx = {
                    pool.submit(probe_image_meta, image_paths[i]): i
                    for i in missing_i
                }
                done = 0
                for fut in as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    res = fut.result()
                    if isinstance(res, dict):
                        res = [res]
                    image_metas[i] = [m for m in (res or []) if isinstance(m, dict)]
                    done += 1
                    if done % 2000 == 0:
                        print(f"[LENGTHS]   probed images {done}/{len(missing_i)}", flush=True)

    # 2b. If TS mode, probe each CSV's (T, C) — needed to compute P*C
    # expansion of the single <|ts_placeholder|> the template emits.
    # Same cache-first strategy as videos.
    ts_shapes: list[Optional[tuple[int, int]]] = []
    if include_timeseries:
        ts_shapes = list(cached_ts_shapes)
        missing_t = [i for i, s in enumerate(ts_shapes) if s is None]
        n_cached_t = len(ts_shapes) - len(missing_t)
        if missing_t:
            print(f"[LENGTHS] probing {len(missing_t)} TS csvs with {nw} workers "
                  f"({n_cached_t} loaded from jsonl cache)…", flush=True)
            with ProcessPoolExecutor(max_workers=nw) as pool:
                fut_to_idx = {pool.submit(probe_ts_shape, ts_paths[i]): i for i in missing_t}
                done = 0
                for fut in as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    ts_shapes[i] = fut.result()
                    done += 1
                    if done % 2000 == 0:
                        print(f"[LENGTHS]   probed TS {done}/{len(missing_t)}", flush=True)
        else:
            print(f"[LENGTHS] all {n_cached_t} TS shapes loaded from jsonl cache", flush=True)

    # 3. Compute per-sample exact length.
    lengths: list[int] = []
    missing_v = 0
    missing_ts = 0
    for idx, txt_tok in enumerate(text_tokens):
        seq = txt_tok
        if include_vision:
            if video_paths[idx] is not None:
                meta = metas[idx]
                vis_tok = _exact_vis_tokens(
                    meta,
                    patch_size, temporal_patch_size, merge_size,
                    video_min_pixels, video_max_pixels,
                    do_sample_frames, target_fps, min_frames, max_frames,
                )
                if meta is None:
                    missing_v += 1
                # Subtract the single <|video_pad|> placeholder counted by
                # txt_tok; add the expanded copies. Guard against negative
                # results when txt_tok happens to be 0.
                seq = max(0, seq - 1) + vis_tok
            if image_paths[idx] is not None:
                im_meta = image_metas[idx] or []
                img_tok = _exact_image_tokens(
                    im_meta, patch_size, merge_size,
                    image_min_pixels, image_max_pixels,
                )
                # txt_tok already counted one <|image_pad|> per image
                # (chat template emits one per element); subtract those.
                n_imgs = len(image_paths[idx])
                seq = max(0, seq - n_imgs) + img_tok
        if include_timeseries:
            shape = ts_shapes[idx] if idx < len(ts_shapes) else None
            if shape is None:
                missing_ts += 1
            else:
                T, C = shape
                P = math.ceil(T / ts_patch_size) if T > 0 else 0
                # Subtract the single ts_placeholder that txt_tok already
                # counted, then add P*C expanded copies.
                seq = seq - 1 + P * C
        lengths.append(seq)

    if missing_v:
        print(f"[LENGTHS] WARNING: {missing_v} videos failed to probe; "
              f"those samples are length-counted without visual tokens", flush=True)
    if missing_ts:
        print(f"[LENGTHS] WARNING: {missing_ts}/{len(ts_shapes)} TS csvs failed to probe; "
              f"those samples are length-counted without TS tokens", flush=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"key": key, "lengths": lengths}))
        print(f"[LENGTHS] wrote cache → {cache_path}", flush=True)

    # Print summary stats.
    import statistics
    ls = sorted(lengths)
    print(
        f"[LENGTHS] done: N={len(ls)} "
        f"min={ls[0]} p50={ls[len(ls)//2]} p95={ls[int(len(ls)*0.95)]} max={ls[-1]} "
        f"mean={statistics.mean(ls):.0f}",
        flush=True,
    )
    return lengths
