"""Qwen2.5-Omni baseline backend.

Bare ``Qwen2_5OmniThinkerForConditionalGeneration`` + the canonical
``Qwen2_5OmniProcessor``, optionally wrapped with a LoRA adapter via
``--adapter_path``. Used both for zero-shot eval (no adapter) and for
trained-baseline checkpoints.

This is the original code path that lived inline in ``infer.py`` before the
backend abstraction landed; behaviour is preserved bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)

from timeomni_v.data.chat_format import apply_template_and_mm, build_conversation
from timeomni_v.data.collator import (
    DEFAULT_DO_SAMPLE_FRAMES,
    DEFAULT_FPS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_VIDEO_MAX_PIXELS,
    DEFAULT_VIDEO_MIN_PIXELS,
    strip_ts_block,
)
from timeomni_v.data.length_estimate import compute_exact_lengths
from timeomni_v.inference.backends.base import LocalHFBackend


@dataclass
class _Qwen25OmniCollator:
    """Collator for the Qwen2.5-Omni baseline path. Mirrors the original
    ``InferenceCollator`` baseline branch verbatim — see git history for the
    pre-backend version. Now also routes image samples (image_path) through
    the parent processor's native image handling."""

    processor: object
    fps: float
    video_min_pixels: int
    video_max_pixels: int
    image_min_pixels: int
    image_max_pixels: int
    max_frames: int
    min_frames: int
    do_sample_frames: bool
    include_vision: bool
    include_timeseries: bool

    def __call__(self, batch: list[dict]) -> dict:
        answers = [r["answer"] for r in batch]
        ids = [r.get("id") for r in batch]

        # Baseline never has a structured TS element in the conversation. The
        # only TS handling is whether to keep the inline ``<timeseries>...
        # </timeseries>`` text block in the prompt or strip it.
        prompt_transform = None if self.include_timeseries else strip_ts_block

        prefix_texts: list[str] = []
        all_videos: list = []
        all_images: list = []
        for r in batch:
            conv = build_conversation(
                r,
                answer=None,
                fps=self.fps,
                max_frames=self.max_frames,
                min_frames=self.min_frames,
                video_min_pixels=self.video_min_pixels,
                video_max_pixels=self.video_max_pixels,
                image_min_pixels=self.image_min_pixels,
                image_max_pixels=self.image_max_pixels,
                include_vision=self.include_vision,
                include_timeseries=False,
                prompt_transform=prompt_transform,
            )
            prefix_text, _, images, videos = apply_template_and_mm(
                self.processor, conv, add_generation_prompt=True,
            )
            prefix_texts.append(prefix_text)
            if self.include_vision:
                all_videos.extend(videos or [])
                all_images.extend(images or [])

        videos_kwargs = {
            "size": {
                "shortest_edge": self.video_min_pixels,
                "longest_edge": self.video_max_pixels,
            },
        }
        images_kwargs = {
            "min_pixels": self.image_min_pixels,
            "max_pixels": self.image_max_pixels,
        }
        inputs = self.processor(
            text=prefix_texts,
            videos=all_videos or None,
            images=all_images or None,
            videos_kwargs=videos_kwargs,
            images_kwargs=images_kwargs,
            return_tensors="pt", padding=True,
        )
        return {"inputs": inputs, "answers": answers, "ids": ids}


class Qwen2_5OmniBackend(LocalHFBackend):
    name = "qwen2_5_omni"
    supports_dynamic_bs = True
    supports_pixel_bounds = True
    supports_dense_frames = True

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
        full_cfg = AutoConfig.from_pretrained(a.model_path, trust_remote_code=True)
        thinker_cfg = full_cfg.thinker_config

        tokenizer = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
        processor = Qwen2_5OmniProcessor.from_pretrained(a.model_path, trust_remote_code=True)
        processor.tokenizer = tokenizer

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            a.model_path, config=thinker_cfg, dtype=torch.bfloat16,
            device_map=device_map, attn_implementation="flash_attention_2",
        )
        if a.adapter_path:
            model = PeftModel.from_pretrained(model, a.adapter_path)
        model.eval()
        self.model = model
        self.processor = processor

    def make_collator(self) -> Callable[[list[dict]], dict]:
        a = self.args
        return _Qwen25OmniCollator(
            processor=self.processor,
            fps=a.fps,
            video_min_pixels=a.video_min_pixels,
            video_max_pixels=a.video_max_pixels,
            image_min_pixels=a.image_min_pixels,
            image_max_pixels=a.image_max_pixels,
            max_frames=a.max_frames,
            min_frames=a.min_frames,
            do_sample_frames=DEFAULT_DO_SAMPLE_FRAMES,
            include_vision=self.include_vision,
            include_timeseries=self.include_timeseries,
        )

    def compute_lengths(self, dataset) -> list[int]:
        a = self.args
        # Baseline: TS appears only as inline text (counted as plain tokens by
        # the chat-template tokenizer pass), so include_timeseries=False at
        # the length-estimation level. The prompt-transform id keys the cache.
        if self.include_timeseries:
            prompt_transform = None
            prompt_transform_id = "raw_full_ts"
        else:
            prompt_transform = strip_ts_block
            prompt_transform_id = "strip_ts"
        return compute_exact_lengths(
            dataset=dataset,
            processor=self.processor,
            video_min_pixels=a.video_min_pixels,
            video_max_pixels=a.video_max_pixels,
            image_min_pixels=a.image_min_pixels,
            image_max_pixels=a.image_max_pixels,
            do_sample_frames=DEFAULT_DO_SAMPLE_FRAMES,
            target_fps=a.fps,
            min_frames=a.min_frames,
            max_frames=a.max_frames,
            prompt_transform=prompt_transform,
            prompt_transform_id=prompt_transform_id,
            include_vision=self.include_vision,
            include_timeseries=False,
            ts_patch_size=16,
            cache_path=Path(a.test_jsonl + ".lengths.json"),
        )

    def generate(self, batch_inputs, *, max_new_tokens: int) -> list[str]:
        device = self.model.device
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


# Re-export defaults so callers (CLI / scripts) can import a single module.
__all__ = [
    "Qwen2_5OmniBackend",
    "DEFAULT_DO_SAMPLE_FRAMES",
    "DEFAULT_FPS",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MIN_FRAMES",
    "DEFAULT_VIDEO_MAX_PIXELS",
    "DEFAULT_VIDEO_MIN_PIXELS",
]
