"""InternVL3.5 backend (HF re-export).

Targets ``InternVL3_5-{4,8}B-HF`` checkpoints — the **HuggingFace
re-exports**, not the original ``OpenGVLab/InternVL3_5-*`` repo. The
re-exports use the official ``InternVLForConditionalGeneration`` class +
``AutoProcessor`` / ``InternVLProcessor`` and *do not* expose the
``trust_remote_code`` ``model.chat()`` / ``model.batch_chat()`` interface.
The pipeline is therefore the same shape as Qwen3-VL::

    inputs = processor.apply_chat_template(
        conversations,                     # list[list[dict]]  (batch)
        tokenize=True, add_generation_prompt=True,
        return_dict=True, padding=True, return_tensors="pt",
        num_frames=N,
    )
    output_ids = model.generate(**inputs, max_new_tokens=...)

The same content-element schema as the rest of the Qwen family works
(``{"type":"video","video":path,...}``), so we reuse ``build_conversation``.

Why ``--batch_size=1`` is enforced:
  ``InternVLVideoProcessor`` stacks every video's ``pixel_values_videos``
  into a single tensor of shape ``(B, F, 3, H, W)``, requiring **F to match
  across rows in the batch**. The HF re-export's per-frame H,W is fixed by
  the checkpoint config (384²), so that's free. F is the only knob that
  varies — and our policy is to derive it per-row from the same fps /
  min_frames / max_frames math the Qwen backends use, which means each row
  has its own num_frames. Mixing those in one batch breaks the stack.
  Forcing bs=1 lets us honor those knobs row-by-row instead of pinning a
  single fixed N for the whole sweep.

Capabilities:
  * Dynamic batching is **off** — the processor doesn't expose the patch /
    temporal-patch / merge-size attributes ``compute_exact_lengths`` keys on.
  * Pixel bounds are not honored as continuous knobs (InternVL uses fixed
    384² tiles); ``supports_pixel_bounds=False``.
  * Dense frames (``do_sample_frames=False``, "all source frames") are not
    supported — the processor would emit unstackable shapes even at bs=1
    when paired with our per-row num_frames; ``supports_dense_frames=False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoProcessor, InternVLForConditionalGeneration

from timeomni_v.data.chat_format import build_conversation
from timeomni_v.data.collator import strip_ts_block
from timeomni_v.data.probe import probe_video_meta
from timeomni_v.inference.backends.base import LocalHFBackend


def _frame_count_for(
    video_path: str, *, fps: float, min_frames: int, max_frames: int,
    fallback: int = 16,
) -> int:
    """Per-row uniform-sample target. Mirrors what the Qwen video processor
    derives from the same knobs: clip(round(duration*fps), min_frames,
    max_frames). Falls back to a small constant if probing fails."""
    meta = probe_video_meta(video_path)
    if meta is None:
        return fallback
    _w, _h, src_fps, nb, dur = meta
    duration = dur if dur > 0 else (nb / src_fps if src_fps else 0.0)
    if duration <= 0:
        return fallback
    target = max(1, int(round(duration * fps)))
    target = max(int(min_frames), min(int(max_frames), target))
    if nb and nb > 0:
        target = min(target, int(nb))
    return max(1, target)


def _sanitize_video_tags(prompt: str) -> str:
    """Replace literal ``<video>`` / ``</video>`` substrings in the prompt
    text with non-conflicting markers.

    InternVL's processor scans for ``self.video_token = "<video>"`` (a plain
    string match — see ``processing_internvl.py::_insert_media_placeholders``)
    and assumes one occurrence per real video. Our CoVLA prompts contain a
    descriptive ``<video>...</video>`` narrative block that is *not* a real
    media placeholder, which inflates the count and triggers an OOB on
    ``video_patch_indices`` (size = batch_size + 1).

    We keep the human-readable content and only swap the angle-bracket tags
    for square brackets. Other backends (Qwen, etc.) use a different special
    token (``<|video_pad|>``) so this rewrite is safe to apply backend-locally
    via ``prompt_transform``."""
    return prompt.replace("<video>", "[video]").replace("</video>", "[/video]")


@dataclass
class _InternVLCollator:
    processor: object
    fps: float
    video_min_pixels: int
    video_max_pixels: int
    max_frames: int
    min_frames: int
    include_vision: bool
    include_timeseries: bool

    def __call__(self, batch: list[dict]) -> dict:
        # Sanity guard — InternVLVideoProcessor stacks per-row pixel tensors,
        # so all rows in a batch must share the same num_frames. Our policy
        # is per-row num_frames computed from fps / min_frames / max_frames,
        # which can vary; we sidestep that by forcing bs=1 in the backend's
        # _validate_args. If something slips through, fail loudly here.
        if self.include_vision and len(batch) > 1:
            raise RuntimeError(
                "InternVL collator received a batch of "
                f"{len(batch)} rows; this backend only supports "
                "--batch_size=1 because num_frames is derived per-row from "
                "fps/min_frames/max_frames and InternVLVideoProcessor "
                "requires a uniform F across the batch.",
            )

        answers = [r["answer"] for r in batch]
        ids = [r.get("id") for r in batch]

        # InternVL has no time-series channel — TS appears only inline as
        # plain text in the prompt; the ablation just toggles whether that
        # block is kept or stripped. On top of the TS toggle we always run
        # ``_sanitize_video_tags`` to neutralize literal ``<video>`` markers
        # in the prompt body that would otherwise collide with InternVL's
        # ``self.video_token`` string-match in ``_insert_media_placeholders``.
        if self.include_timeseries:
            prompt_transform = _sanitize_video_tags
        else:
            def prompt_transform(p: str) -> str:
                return _sanitize_video_tags(strip_ts_block(p))

        conversations: list[list[dict]] = []
        for r in batch:
            conv = build_conversation(
                r,
                answer=None,
                fps=self.fps,
                max_frames=self.max_frames,
                min_frames=self.min_frames,
                video_min_pixels=self.video_min_pixels,
                video_max_pixels=self.video_max_pixels,
                include_vision=self.include_vision,
                include_timeseries=False,
                prompt_transform=prompt_transform,
            )
            conversations.append(conv)

        # Per-row num_frames from probe + fps clamps. With bs=1 there's
        # exactly one value per call. transformers 4.57's
        # ``apply_chat_template`` forwards extra kwargs straight into the
        # processor's ``__call__`` (see processing_utils.py L1620-1660), so
        # ``num_frames``/``padding`` are passed as flat kwargs. Image
        # samples (image_path set, no video_path) are routed through the
        # processor's native image handling; only video rows need num_frames.
        extra_kwargs: dict = {"padding": True}
        if self.include_vision and batch[0].get("video_path"):
            extra_kwargs["num_frames"] = _frame_count_for(
                batch[0]["video_path"],
                fps=self.fps,
                min_frames=self.min_frames,
                max_frames=self.max_frames,
            )

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **extra_kwargs,
        )
        return {"inputs": inputs, "answers": answers, "ids": ids}


class InternVLBackend(LocalHFBackend):
    name = "internvl"
    supports_dynamic_bs = False
    # InternVL-HF uses fixed 384² tiles; the smart_resize-style continuous
    # pixel budget knobs Qwen exposes don't apply.
    supports_pixel_bounds = False
    # We pin per-row num_frames; "all frames" mode would conflict with that.
    supports_dense_frames = False

    def _validate_args(self) -> None:
        super()._validate_args()
        if int(self.args.batch_size) != 1:
            raise SystemExit(
                "--backend internvl_* requires --batch_size=1 "
                "(per-row num_frames derived from fps/min/max breaks the "
                "InternVL pixel_values_videos stack at bs>1).",
            )

    def __init__(self, *, args, include_vision, include_timeseries):
        super().__init__(
            args=args,
            include_vision=include_vision,
            include_timeseries=include_timeseries,
        )
        self.model = None
        self.processor = None

    def load(self, *, device_map=None) -> None:
        a = self.args
        model = InternVLForConditionalGeneration.from_pretrained(
            a.model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(a.model_path)
        # InternVL3.5-HF's video_preprocessor_config.json ships ``size``
        # = {384,384}, which is a bug: the vision tower expects 448 (so the
        # 32×32 patch grid divides evenly under downsample_ratio=0.5 → 16×16
        # = 256 image tokens, matching processor_config's image_seq_length).
        # 384/14 = 27 patches/side, and 27 is odd, so pixel_shuffle's
        # ``view(B, 27, int(27*0.5)=13, …)`` fails with the 27*27 ≠ 27*13*2
        # mismatch. Force-align the video processor to the same 448 tiles
        # the static-image processor uses.
        vc_image_size = model.config.vision_config.image_size
        if isinstance(vc_image_size, (list, tuple)):
            tgt_h, tgt_w = int(vc_image_size[0]), int(vc_image_size[1])
        else:
            tgt_h = tgt_w = int(vc_image_size)
        if hasattr(processor, "video_processor") and processor.video_processor is not None:
            processor.video_processor.size = {"height": tgt_h, "width": tgt_w}
        self.model = model
        self.processor = processor

    def make_collator(self) -> Callable[[list[dict]], dict]:
        a = self.args
        return _InternVLCollator(
            processor=self.processor,
            fps=a.fps,
            video_min_pixels=a.video_min_pixels,
            video_max_pixels=a.video_max_pixels,
            max_frames=a.max_frames,
            min_frames=a.min_frames,
            include_vision=self.include_vision,
            include_timeseries=self.include_timeseries,
        )

    def generate(self, batch_inputs, *, max_new_tokens: int) -> list[str]:
        device = self.model.device
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }
        # Pass pad_token_id explicitly so HF generate doesn't fall back to
        # ``eos_token_id`` and emit ``Setting pad_token_id to eos_token_id`` —
        # the tokenizer already defines ``<|endoftext|>`` (151643) as pad,
        # distinct from ``<|im_end|>`` (151645) as eos. Falling back to eos
        # would also break batched generation (rows that finish early would
        # be padded with the same id used to mark end-of-sequence).
        pad_token_id = self.processor.tokenizer.pad_token_id
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_token_id,
            )
        # Per-sample trim: padding can vary across rows, so use each row's
        # actual prompt length rather than a single shared offset.
        prompt_lens = [len(ids) for ids in inputs["input_ids"]]
        trimmed = [
            out[plen:] for out, plen in zip(output_ids, prompt_lens)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


__all__ = ["InternVLBackend"]
