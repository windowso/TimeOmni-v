"""Qwen3-Omni-30B-A3B-Instruct backend.

Same content-element conversation schema as Qwen2.5-Omni (``{"type":"video"
,...}``) and the same ``qwen_omni_utils.process_mm_info`` extractor — so we
reuse ``build_conversation`` and ``apply_template_and_mm`` directly. We
import the **thinker-only** subclass to skip the talker codepath; this
matches the TS_bench reference impl and avoids the structured generate
output (no need for ``thinker_return_dict_in_generate`` / ``.sequences``).

Dynamic batching is supported because the processor's ``video_processor``
exposes the same ``patch_size / temporal_patch_size / merge_size`` knobs
``compute_exact_lengths`` keys on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import (
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from timeomni_v.data.chat_format import apply_template_and_mm, build_conversation
from timeomni_v.data.collator import (
    DEFAULT_DO_SAMPLE_FRAMES,
    strip_ts_block,
)
from timeomni_v.data.length_estimate import compute_exact_lengths
from timeomni_v.inference.backends.base import LocalHFBackend


@dataclass
class _Qwen3OmniCollator:
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

        # No structured TS element — we are using the bare visual-language
        # mode of Qwen3-Omni; TS appears only inline in the prompt.
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
        # Qwen3-Omni's processor accepts ``audio=None`` cleanly; we never
        # extract audio here (use_audio_in_video=False at apply_template_and_mm).
        inputs = self.processor(
            text=prefix_texts,
            videos=all_videos or None,
            images=all_images or None,
            videos_kwargs=videos_kwargs,
            images_kwargs=images_kwargs,
            return_tensors="pt", padding=True,
            use_audio_in_video=False,
        )
        return {"inputs": inputs, "answers": answers, "ids": ids}


class Qwen3OmniBackend(LocalHFBackend):
    name = "qwen3_omni"
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
        # Thinker-only: skips the talker codepath, returns a plain output_ids
        # tensor from generate() — same shape as Qwen2.5-Omni Thinker.
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            a.model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        model.eval()
        processor = Qwen3OmniMoeProcessor.from_pretrained(a.model_path)
        # Push the global pixel bounds into the processor so smart_resize uses
        # them; the per-element bounds in build_conversation override on a
        # per-sample basis, but the processor default is the floor.
        if a.video_min_pixels is not None:
            processor.min_pixels = a.video_min_pixels
        if a.video_max_pixels is not None:
            processor.max_pixels = a.video_max_pixels
        self.model = model
        self.processor = processor

    def make_collator(self) -> Callable[[list[dict]], dict]:
        a = self.args
        return _Qwen3OmniCollator(
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
                use_audio_in_video=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


__all__ = ["Qwen3OmniBackend"]
