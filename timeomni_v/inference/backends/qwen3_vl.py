"""Qwen3-VL backend (covers both Qwen3-VL-8B-Instruct and the
Qwen3-VL-30B-A3B-Instruct MoE variant).

The 8B checkpoint is dense (``Qwen3VLForConditionalGeneration``); the 30B
checkpoint is MoE (``Qwen3VLMoeForConditionalGeneration``, ``model_type:
qwen3_vl_moe``, 128 experts × 8 active per token). The two share the same
processor / chat template / video pipeline, but their ``language_model.mlp``
sub-modules diverge: dense has ``gate_proj/up_proj/down_proj`` directly,
MoE has ``mlp.experts.{i}.{gate,up,down}_proj`` plus ``mlp.gate.weight``
(the router). Hard-coding ``Qwen3VLForConditionalGeneration`` and feeding
it a MoE checkpoint silently random-inits every ``mlp.*`` weight (HF logs
"newly initialized" for the missing dense keys and drops the experts as
"unexpected"), giving garbage outputs. To dispatch to the right class
based on ``config.model_type`` we use ``AutoModelForImageTextToText``,
which carries entries for both ``qwen3_vl`` and ``qwen3_vl_moe`` in
transformers ≥ 4.57's auto registry.

Single-call processor pipeline (no separate ``process_mm_info``)::

    inputs = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True,
        return_dict=True, padding=True, return_tensors="pt",
    )

The conversation content schema matches the rest of the Qwen family —
``{"type":"video","video":path}`` — so we reuse ``build_conversation``.

**Why we mutate ``processor.video_processor`` instance attrs in ``load``
instead of passing them per call:** Qwen3-VL's ``processing_qwen3_vl.py``
does NOT translate flat ``min_pixels`` / ``max_pixels`` into
``size = {"shortest_edge", "longest_edge"}`` (Qwen2.5-VL does, Qwen3-VL
doesn't), and they're not in ``Qwen3VLVideoProcessorInitKwargs.valid_kwargs``
either — so passing them via call-time kwargs gets silently dropped by
``_merge_kwargs``. Setting ``video_processor.size`` / ``.fps`` /
``.min_frames`` / ``.max_frames`` / ``.do_sample_frames`` directly is picked
up by ``BaseVideoProcessor.__call__`` which back-fills missing kwargs from
``getattr(self, k)``.

Batching: bs>1 works fine even with variable per-row frame counts, because
Qwen2VL-family ``pixel_values_videos`` is **flat-concatenated** across the
batch with ``video_grid_thw`` carrying per-row dims (different from
InternVL's stacked layout, which forces bs=1).

Dynamic-bs is still disabled — ``compute_exact_lengths`` keys on the
``video_processor.patch_size / temporal_patch_size / merge_size`` triple
which Qwen3VLVideoProcessor doesn't directly expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from timeomni_v.data.chat_format import build_conversation
from timeomni_v.data.collator import strip_ts_block
from timeomni_v.inference.backends.base import LocalHFBackend


@dataclass
class _Qwen3VLCollator:
    processor: object
    fps: float
    video_min_pixels: int
    video_max_pixels: int
    max_frames: int
    min_frames: int
    include_vision: bool
    include_timeseries: bool

    def __call__(self, batch: list[dict]) -> dict:
        answers = [r["answer"] for r in batch]
        ids = [r.get("id") for r in batch]

        # Qwen3-VL has no time-series channel — TS only ever appears inline as
        # plain text in the prompt; the ablation just toggles whether that
        # block stays or is stripped.
        prompt_transform = None if self.include_timeseries else strip_ts_block

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

        # Sampling/resize knobs are configured on processor.video_processor in
        # Qwen3VLBackend.load (see module docstring). transformers 4.57's
        # ``apply_chat_template`` forwards extra kwargs straight into the
        # processor's ``__call__`` (processing_utils.py L1620-1660), so
        # ``padding`` is passed as a flat kwarg. Image conversation elements
        # are handled natively by Qwen3-VL's processor (no separate kwarg).
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return {"inputs": inputs, "answers": answers, "ids": ids}


class Qwen3VLBackend(LocalHFBackend):
    name = "qwen3_vl"
    # No video_processor.patch_size to key the length-cache on; dynamic-bs off.
    supports_dynamic_bs = False
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
        # AutoModelForImageTextToText reads ``config.model_type`` and dispatches
        # to ``Qwen3VLForConditionalGeneration`` for the 8B dense checkpoint or
        # ``Qwen3VLMoeForConditionalGeneration`` for the 30B MoE checkpoint. See
        # this module's top docstring for why hard-coding the dense class
        # silently corrupts MoE weights.
        model = AutoModelForImageTextToText.from_pretrained(
            a.model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        model.eval()
        # Greedy decoding with do_sample=False — strip sampling-only fields the
        # checkpoint's generation_config ships with, otherwise generate() warns
        # every step that they're being ignored.
        for k in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, k):
                setattr(model.generation_config, k, None)
        processor = AutoProcessor.from_pretrained(a.model_path)
        # Decoder-only generation needs left-padding so the rightmost token of
        # every sequence is the real last prompt token; default for Qwen
        # tokenizers is "right" which makes generate() warn every batch.
        processor.tokenizer.padding_side = "left"
        if self.include_vision:
            vp = processor.video_processor
            vp.fps = float(a.fps)
            vp.min_frames = int(a.min_frames)
            vp.max_frames = int(a.max_frames)
            vp.size = {
                "shortest_edge": int(a.video_min_pixels),
                "longest_edge": int(a.video_max_pixels),
            }
            vp.do_sample_frames = True
            # Same instance-attr trick for the image processor — the Qwen3-VL
            # processor's apply_chat_template doesn't accept per-call image
            # pixel bounds, so we pin them on the image_processor directly.
            ip = getattr(processor, "image_processor", None)
            if ip is not None:
                if hasattr(ip, "min_pixels"):
                    ip.min_pixels = int(a.image_min_pixels)
                if hasattr(ip, "max_pixels"):
                    ip.max_pixels = int(a.image_max_pixels)
        self.model = model
        self.processor = processor

    def make_collator(self) -> Callable[[list[dict]], dict]:
        a = self.args
        return _Qwen3VLCollator(
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
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        # Trim per-sample because left/right padding can vary; the shared
        # processor.batch_decode handles the rest.
        prompt_lens = [len(ids) for ids in inputs["input_ids"]]
        trimmed = [
            out[plen:] for out, plen in zip(output_ids, prompt_lens)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


__all__ = ["Qwen3VLBackend"]
