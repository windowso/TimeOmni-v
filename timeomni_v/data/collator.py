"""Data collators for TimeOmni-v training.

Both collators consume the same unified jsonl schema — rows carry the
inline ``<timeseries>...</timeseries>`` block inside ``prompt`` AND a
``timeseries_path`` to the per-sample CSV. The collators differ only in
how they use those fields:

* ``TimeOmniVDataCollator`` — TS-as-tensor. Strips the inline TS body (via
  ``strip_ts_block``) so the prompt holds only empty ``<timeseries></timeseries>``
  tags, then hands ``timeseries_path`` to ``TimeOmniVProcessor`` via its
  ``timeseries=`` kwarg; the processor loads tensors and expands the
  ``<|ts_placeholder|>`` token into N latent-token embeddings in the model forward.

* ``BaselineCollator`` — TS-as-text. Keeps the inline TS body (optionally
  downsampled via ``downsample_ts_block`` to respect a token budget) and
  ignores ``timeseries_path`` entirely.

One jsonl, two modes. See ``timeomni_v.data.convert_covla`` for the producer.

Videos are passed to the Qwen2.5-Omni processor as file paths (short per-sample
clips from ``timeomni_v.data.slice_videos``). The processor decodes+resizes; we
avoid passing raw decord tensors because the video processor's
``get_image_size`` expects channels-first per-frame and mis-parses them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import torch

from timeomni_v.data.chat_format import apply_template_and_mm, build_conversation
from timeomni_v.processing.processing_timeomni_v import TimeOmniVProcessor
from timeomni_v.utils.warnings import silence_qwen_audio_system_prompt_warning

# DataLoader workers may use spawn (re-importing modules from scratch) — install
# the filter at module import so workers also suppress the per-call warning.
silence_qwen_audio_system_prompt_warning()

# Single source of truth for the video-frame pixel-area bounds used by the
# Qwen2.5-Omni processor. Both training and inference collators / CLIs import
# these defaults so that changing resolution is a one-line edit.
#
# Qwen2.5-Omni uses patch_size=14, spatial merge_size=2, so factor=28 and each
# post-merge token covers a 28x28 pixel block. Per-frame token count =
# resized_pixels / 28**2. Tunable in tokens-per-frame:
#   128 tokens/frame ≈ 100k pixels (low-res, fast iteration)
#   192 tokens/frame ≈ 150k pixels (default, balances quality & length)
DEFAULT_VIDEO_MIN_PIXELS = 64 * 28 * 28
DEFAULT_VIDEO_MAX_PIXELS = 128 * 28 * 28

# Image samples are usually 1-7 stills per row (vs ~20 sampled video frames),
# so we can spend more pixels per image without blowing the per-sample token
# budget. Default = 2× the video bounds.
DEFAULT_IMAGE_MIN_PIXELS = 2 * DEFAULT_VIDEO_MIN_PIXELS
DEFAULT_IMAGE_MAX_PIXELS = 2 * DEFAULT_VIDEO_MAX_PIXELS

# Frame-sampling defaults: enable uniform temporal downsampling via the
# processor. do_sample_frames=True is the master switch (without it the
# processor keeps every source frame); fps + max_frames together cap the
# sampled frame count. For CoVLA 5-12s clips at 20 source fps this yields
# 4-12 frames per clip instead of 90+.
#
# DEFAULT_FPS is the target sampling fps handed to Qwen2.5-Omni's video
# processor (distinct from the source video's own fps). Train/infer CLIs,
# both collators, and length_estimate all import it so changing the
# sampling rate is a one-line edit here.
DEFAULT_DO_SAMPLE_FRAMES = True
DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 80
DEFAULT_MIN_FRAMES = 4

# TS-as-text defaults for BaselineCollator / length_estimate.
# With ~500 Qwen tokens already spent on the non-TS prompt + answer,
# 12 data rows × ~25 tokens/row after 1-decimal rounding ≈ 300 tokens.
DEFAULT_TS_MAX_LINES = 12
DEFAULT_TS_DECIMALS = 1

TS_BLOCK_RE = re.compile(r"<timeseries>(.*?)</timeseries>", re.DOTALL)
# Matches a data line like "25.5: vEgo=6.517, aEgo=0.197, ...". Descriptive
# header lines (format hint, unit docs) don't match and are preserved.
TS_DATA_LINE_RE = re.compile(r"^\s*[\d.]+\s*:\s*\S.*$")
TS_NUM_RE = re.compile(r"-?\d+\.\d+")


def strip_ts_block(prompt: str) -> str:
    """Delete the inline ``<timeseries>...</timeseries>`` block (tags + body).

    Used by ``TimeOmniVDataCollator``: in timeomni_v mode the TS signal is handed
    to the processor as a tensor via a structured ``{"type":"timeseries"}``
    content element, so the inline text copy is redundant and wastes tokens.
    No-op if the prompt has no TS block.
    """
    return TS_BLOCK_RE.sub("", prompt)


def downsample_ts_block(prompt: str, max_lines: int, decimals: int) -> str:
    """Shrink the inline ``<timeseries>`` block in-place.

    * Uniformly downsample data rows to at most ``max_lines`` (keeps first /
      last so onset / offset dynamics survive).
    * Round every decimal number in kept rows to ``decimals`` places.
    * Non-data lines (format hint, null-legend) pass through untouched.

    No-op if the prompt has no ``<timeseries>`` block or the block is already
    under budget.
    """
    m = TS_BLOCK_RE.search(prompt)
    if not m:
        return prompt
    lines = m.group(1).splitlines()
    data_idx = [i for i, l in enumerate(lines) if TS_DATA_LINE_RE.match(l)]

    if max_lines > 0 and len(data_idx) > max_lines:
        import numpy as np
        picks = np.linspace(0, len(data_idx) - 1, max_lines).round().astype(int)
        keep_data = {data_idx[i] for i in dict.fromkeys(picks.tolist())}
    else:
        keep_data = set(data_idx)

    non_data = set(range(len(lines))) - set(data_idx)
    kept = sorted(non_data | keep_data)

    def _round(line: str) -> str:
        return TS_NUM_RE.sub(lambda mm: f"{float(mm.group(0)):.{decimals}f}", line)

    new_body_lines = [
        _round(lines[i]) if i in keep_data else lines[i] for i in kept
    ]
    new_body = "\n".join(new_body_lines)
    return prompt[: m.start(1)] + new_body + prompt[m.end(1) :]


def _mask_labels_to_assistant(input_ids, attn, tokenizer) -> torch.Tensor:
    """Return labels masked so only the assistant reply contributes to loss.

    Keeps the assistant content + ``<|im_end|>`` and masks everything else:
    user turn (incl. video / TS placeholders), the ``<|im_start|>assistant\\n``
    header, padding, AND the trailing ``\\n`` after ``<|im_end|>`` — that
    newline is purely chat-template formatting filler whose target token is
    deterministic, so training on it just dilutes the gradient signal from
    the actual answer tokens.
    """
    labels = input_ids.clone()
    labels[attn == 0] = -100
    header_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    header_t = torch.tensor(header_ids, dtype=input_ids.dtype, device=input_ids.device)
    H = header_t.numel()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for i in range(input_ids.size(0)):
        row_ids = input_ids[i]
        if row_ids.size(0) < H:
            continue
        windows = row_ids.unfold(0, H, 1)
        matches = (windows == header_t).all(dim=1)
        idx = matches.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            labels[i, :] = -100  # No header found → don't train this row.
            continue
        header_end = int(idx[-1].item()) + H
        labels[i, :header_end] = -100
        # Mask trailing template tokens after the assistant's <|im_end|>.
        # Search only within the assistant span (after header_end) so we
        # don't accidentally pick up an im_end belonging to the user turn.
        end_positions = (row_ids[header_end:] == im_end_id).nonzero(as_tuple=True)[0]
        if end_positions.numel() > 0:
            last_end = header_end + int(end_positions[-1].item())
            labels[i, last_end + 1 :] = -100
    return labels


def _log_collator_timing(
    tag: str,
    input_ids,
    attn,
    vpad_id: int,
    *,
    t0: float,
    t_prep: float,
    t_proc: float,
    t_lbl: float,
    ts_pad_id: int | None = None,
    ipad_id: int | None = None,
) -> None:
    n_vis = int((input_ids == vpad_id).sum().item())
    n_total = int(attn.sum().item())
    n_ts = int((input_ids == ts_pad_id).sum().item()) if ts_pad_id is not None else 0
    n_img = int((input_ids == ipad_id).sum().item()) if ipad_id is not None else 0
    n_text = n_total - n_vis - n_img - n_ts
    ts_part = f" ts_tok={n_ts}" if ts_pad_id is not None else ""
    img_part = f" img_tok={n_img}" if ipad_id is not None else ""
    print(
        f"[{tag}] bs={input_ids.size(0)} "
        f"seq={input_ids.size(1)} valid={n_total} "
        f"vis_tok={n_vis}{img_part}{ts_part} text_tok={n_text} | "
        f"prep={(t_prep - t0) * 1000:.1f}ms "
        f"processor={(t_proc - t_prep) * 1000:.1f}ms "
        f"labels={(t_lbl - t_proc) * 1000:.1f}ms "
        f"total={(t_lbl - t0) * 1000:.1f}ms",
        flush=True,
    )


@dataclass
class TimeOmniVDataCollator:
    processor: TimeOmniVProcessor
    fps: float = DEFAULT_FPS
    video_min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS
    video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    image_min_pixels: int = DEFAULT_IMAGE_MIN_PIXELS
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS
    do_sample_frames: bool = DEFAULT_DO_SAMPLE_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES
    min_frames: int = DEFAULT_MIN_FRAMES
    answer_prefix: str = "\nAnswer: "
    log_timing: bool = False
    # Tasks for which the inline <timeseries>...</timeseries> body is kept
    # in the prompt (in addition to the structured TS element fed via the
    # encoder). Default empty: forecasting now goes through the linear
    # forecast head (see timeomni_v.data.forecast_collator and modeling_timeomni_v)
    # which doesn't need redundant inline numerics. Set to a non-empty
    # frozenset only if a task family genuinely benefits from text-form TS.
    keep_inline_ts_tasks: "frozenset[str] | set[str] | None" = frozenset()

    def __call__(self, batch: list[dict]) -> dict:
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError(
                f"TimeOmniVDataCollator requires tokenizer.padding_side='right', "
                f"got {self.processor.tokenizer.padding_side!r}."
            )
        t0 = time.perf_counter() if self.log_timing else 0.0
        full_texts: list[str] = []
        all_videos: list = []
        all_images: list = []
        ts_paths: list[str] = []
        for row in batch:
            ts_path = row.get("timeseries_path")
            if ts_path is None:
                raise ValueError(
                    f"TimeOmniVDataCollator received row without timeseries_path "
                    f"(id={row.get('id')!r}). Use BaselineCollator "
                    f"for TS-as-text inputs, or filter the dataset to TS-bearing rows."
                )
            # Structured video / image / timeseries elements drive modality
            # insertion; the inline <timeseries>...</timeseries> block in the
            # prompt is stripped (redundant with the tensor path). Qwen's chat
            # template turns {"type":"video"} / {"type":"image"} into the
            # corresponding marker, TimeOmniVProcessor turns {"type":"timeseries"}
            # into <|ts_start|><|ts_placeholder|><|ts_end|>, then __call__
            # expands the single TS placeholder to P*C copies once the CSV
            # shape is known.
            conv_full = build_conversation(
                row,
                answer=row["answer"],
                fps=self.fps,
                max_frames=self.max_frames,
                min_frames=self.min_frames,
                video_min_pixels=self.video_min_pixels,
                video_max_pixels=self.video_max_pixels,
                image_min_pixels=self.image_min_pixels,
                image_max_pixels=self.image_max_pixels,
                include_timeseries=True,
                prompt_transform=strip_ts_block,
                prompt_transform_skip_tasks=self.keep_inline_ts_tasks,
            )
            full_text, _, images, videos = apply_template_and_mm(
                self.processor, conv_full, add_generation_prompt=False,
            )
            full_texts.append(full_text)
            all_videos.extend(videos or [])
            all_images.extend(images or [])
            ts_paths.append(ts_path)
        t_prep = time.perf_counter() if self.log_timing else 0.0

        batch_out = self.processor(
            text=full_texts,
            videos=all_videos or None,
            images=all_images or None,
            timeseries=ts_paths,
            videos_kwargs={
                "size": {
                    "shortest_edge": self.video_min_pixels,
                    "longest_edge": self.video_max_pixels,
                },
            },
            images_kwargs={
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
            },
            return_tensors="pt",
            padding=True,
        )
        t_proc = time.perf_counter() if self.log_timing else 0.0
        batch_out["labels"] = _mask_labels_to_assistant(
            batch_out["input_ids"], batch_out["attention_mask"], self.processor.tokenizer,
        )

        if self.log_timing:
            t_lbl = time.perf_counter()
            from timeomni_v.utils.tokens import TS_PLACEHOLDER
            tok = self.processor.tokenizer
            vpad_id = tok.convert_tokens_to_ids(tok.video_token)
            ipad_id = tok.convert_tokens_to_ids(tok.image_token)
            ts_pad_id = tok.convert_tokens_to_ids(TS_PLACEHOLDER)
            _log_collator_timing(
                "TIMEOMNI_V_COLLATOR",
                batch_out["input_ids"], batch_out["attention_mask"], vpad_id,
                t0=t0, t_prep=t_prep, t_proc=t_proc, t_lbl=t_lbl,
                ts_pad_id=ts_pad_id, ipad_id=ipad_id,
            )
        return batch_out


@dataclass
class BaselineCollator:
    """TS-as-text collator using the official Qwen2.5-Omni processor.

    Same chat-template + video pipeline as TimeOmniVDataCollator; differences:
      * no ``timeseries=`` kwarg (TS lives inline in the prompt text);
      * ``downsample_ts_block`` trims the inline block to a token budget;
      * prints per-batch token / timing stats (useful for profiling).
    """

    processor: "object"  # Qwen2_5OmniProcessor — avoids a hard import here
    fps: float = DEFAULT_FPS
    video_min_pixels: int = DEFAULT_VIDEO_MIN_PIXELS
    video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    image_min_pixels: int = DEFAULT_IMAGE_MIN_PIXELS
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS
    do_sample_frames: bool = DEFAULT_DO_SAMPLE_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES
    min_frames: int = DEFAULT_MIN_FRAMES
    ts_max_lines: int = DEFAULT_TS_MAX_LINES
    ts_decimals: int = DEFAULT_TS_DECIMALS
    log_timing: bool = False

    def __call__(self, batch: list[dict]) -> dict:
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError(
                f"BaselineCollator requires tokenizer.padding_side='right', "
                f"got {self.processor.tokenizer.padding_side!r}."
            )
        t0 = time.perf_counter() if self.log_timing else 0.0

        full_texts: list[str] = []
        all_videos: list = []
        all_images: list = []
        ts_transform = lambda p: downsample_ts_block(  # noqa: E731
            p, self.ts_max_lines, self.ts_decimals
        )
        for row in batch:
            conv_full = build_conversation(
                row,
                answer=row["answer"],
                fps=self.fps,
                max_frames=self.max_frames,
                min_frames=self.min_frames,
                video_min_pixels=self.video_min_pixels,
                video_max_pixels=self.video_max_pixels,
                image_min_pixels=self.image_min_pixels,
                image_max_pixels=self.image_max_pixels,
                prompt_transform=ts_transform,
            )
            full_text, _, images, videos = apply_template_and_mm(
                self.processor, conv_full, add_generation_prompt=False,
            )
            full_texts.append(full_text)
            all_videos.extend(videos or [])
            all_images.extend(images or [])
        t_prep = time.perf_counter() if self.log_timing else 0.0

        # Match the processor's resize bounds to the per-message config so
        # it doesn't re-resize the already-sampled tensors with different
        # defaults (which would drift grid_thw).
        batch_out = self.processor(
            text=full_texts,
            videos=all_videos or None,
            images=all_images or None,
            videos_kwargs={
                "size": {
                    "shortest_edge": self.video_min_pixels,
                    "longest_edge": self.video_max_pixels,
                },
            },
            images_kwargs={
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
            },
            return_tensors="pt",
            padding=True,
        )
        t_proc = time.perf_counter() if self.log_timing else 0.0

        input_ids = batch_out["input_ids"]
        attn = batch_out["attention_mask"]
        batch_out["labels"] = _mask_labels_to_assistant(input_ids, attn, self.processor.tokenizer)

        if self.log_timing:
            t_lbl = time.perf_counter()
            tok = self.processor.tokenizer
            vpad_id = tok.convert_tokens_to_ids(tok.video_token)
            ipad_id = tok.convert_tokens_to_ids(tok.image_token)
            _log_collator_timing(
                "BASELINE_COLLATOR",
                input_ids, attn, vpad_id,
                t0=t0, t_prep=t_prep, t_proc=t_proc, t_lbl=t_lbl,
                ipad_id=ipad_id,
            )
        return batch_out
