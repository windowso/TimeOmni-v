"""TimeOmni-v backend — Qwen2.5-Omni + Chronos TS tower + LoRA adapter.

This is the default training/eval path: the unified jsonl carries inline TS
text in ``prompt`` AND a ``timeseries_path`` to the per-sample CSV; the
collator strips the inline text and feeds CSVs to ``TimeOmniVProcessor``, which
expands the TS placeholder to ``P*C`` copies and the model scatters TS
embeddings into ``inputs_embeds`` at those positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from timeomni_v.data.chat_format import apply_template_and_mm, build_conversation
from timeomni_v.data.collator import (
    DEFAULT_DO_SAMPLE_FRAMES,
    strip_ts_block,
)
from timeomni_v.data.length_estimate import compute_exact_lengths
from timeomni_v.inference.backends.base import LocalHFBackend
from timeomni_v.modeling.configuration_timeomni_v import TimeOmniVConfig
from timeomni_v.modeling.modeling_timeomni_v import TimeOmniVForConditionalGeneration
from timeomni_v.processing.processing_timeomni_v import TimeOmniVProcessor
from timeomni_v.utils.tokens import add_ts_tokens


@dataclass
class _TimeOmniVCollator:
    """Collator for the TimeOmni-v path. Always strips the inline TS body; the
    structured ``timeseries`` element is included only when TS is enabled.
    Visual inputs flow through whichever modality the row carries — video
    or one+ images — gated by ``include_vision``."""

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
    # forecast_mode=True switches the collator + backend to head-driven
    # forecasting: no inline TS in the prompt for any task (default behavior
    # in the new flow), and forecast_target_path / forecast_target_lens are
    # surfaced in the batch so the backend can format predictions back into
    # <forecast> text using each row's gt label structure.
    forecast_mode: bool = False
    # Default empty: training flipped this default for the same reason — the
    # forecast head supersedes inline-numeric LM prediction.
    keep_inline_ts_tasks: "frozenset[str] | set[str] | None" = frozenset()

    def __call__(self, batch: list[dict]) -> dict:
        answers = [r["answer"] for r in batch]
        ids = [r.get("id") for r in batch]

        prefix_texts: list[str] = []
        all_videos: list = []
        all_images: list = []
        ts_paths: list[str] = []
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
                include_timeseries=self.include_timeseries,
                prompt_transform=strip_ts_block,
                # `no_timeseries` ablation cuts the encoder TS feed AND the
                # inline TS text — without this gate, prediction rows would
                # still leak the raw numbers through the text channel and
                # the ablation would be a no-op for forecasting tasks.
                prompt_transform_skip_tasks=(
                    self.keep_inline_ts_tasks if self.include_timeseries else None
                ),
            )
            prefix_text, _, images, videos = apply_template_and_mm(
                self.processor, conv, add_generation_prompt=True,
            )
            prefix_texts.append(prefix_text)
            if self.include_vision:
                all_videos.extend(videos or [])
                all_images.extend(images or [])
            if self.include_timeseries:
                ts_paths.append(r["timeseries_path"])

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
        # TimeOmniVProcessor accepts ``timeseries=None`` cleanly when the TS
        # ablation is on (no <|ts_placeholder|> emitted upstream → no scatter).
        inputs = self.processor(
            text=prefix_texts,
            videos=all_videos or None,
            images=all_images or None,
            timeseries=ts_paths if self.include_timeseries else None,
            videos_kwargs=videos_kwargs,
            images_kwargs=images_kwargs,
            return_tensors="pt", padding=True,
        )
        return {"inputs": inputs, "answers": answers, "ids": ids}


class TimeOmniVBackend(LocalHFBackend):
    name = "timeomni_v"
    supports_dynamic_bs = True
    supports_pixel_bounds = True
    supports_dense_frames = True

    def _validate_args(self) -> None:
        super()._validate_args()
        if not self.args.chronos_path:
            raise SystemExit("--chronos_path is required for --backend timeomni_v")

    def __init__(self, *, args, include_vision, include_timeseries):
        super().__init__(
            args=args,
            include_vision=include_vision,
            include_timeseries=include_timeseries,
        )
        self.model = None
        self.processor = None
        # Detected during load() — set when the saved config.json carries
        # forecast_head_config (a forecasting run; PEFT-loaded
        # forecast_head provides the weights).
        self.forecast_mode: bool = False

    def load(self, *, device_map=None) -> None:
        a = self.args
        adapter_dir = Path(a.adapter_path) if a.adapter_path else None

        saved_tokenizer = (
            adapter_dir is not None
            and (adapter_dir / "tokenizer_config.json").exists()
        )
        if saved_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir), trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                a.model_path, trust_remote_code=True,
            )
            add_ts_tokens(tokenizer)

        saved_config_path = (
            adapter_dir / "config.json" if adapter_dir is not None else None
        )
        if saved_config_path is not None and saved_config_path.exists():
            config = TimeOmniVConfig.from_pretrained(
                str(adapter_dir), trust_remote_code=True,
            )
            config.ts_encoder_path = a.chronos_path
        else:
            full_cfg = AutoConfig.from_pretrained(
                a.model_path, trust_remote_code=True,
            )
            thinker_dict = full_cfg.thinker_config.to_dict()
            thinker_dict.pop("model_type", None)
            config = TimeOmniVConfig(
                ts_encoder_path=a.chronos_path,
                ts_encoder_hidden_size=768,
                ts_adapter_hidden_size=2048,
                **thinker_dict,
            )
            config.timeseries_token_id = tokenizer.convert_tokens_to_ids(
                "<|ts_placeholder|>",
            )
            config.timeseries_start_token_id = tokenizer.convert_tokens_to_ids(
                "<|ts_start|>",
            )
            config.timeseries_end_token_id = tokenizer.convert_tokens_to_ids(
                "<|ts_end|>",
            )

        processor = TimeOmniVProcessor.from_pretrained(
            a.model_path, trust_remote_code=True,
        )
        processor.tokenizer = tokenizer
        processor.fusion_mode = a.fusion_mode

        # TimeOmniVForConditionalGeneration.__init__ reads
        # config.forecast_head_config (set by train.py when --pred_len > 0)
        # and attaches a fresh ForecastHead with matching dims. Trained
        # weights flow in via PEFT below.
        model = TimeOmniVForConditionalGeneration.from_pretrained(
            a.model_path, config=config, dtype=torch.bfloat16,
            device_map=device_map, attn_implementation="flash_attention_2",
        )
        if model.ts_tower is not None:
            processor.ts_patch_size = int(model.ts_tower.patch_size)

        if a.adapter_path:
            # PEFT loads adapter_model.safetensors which contains the LoRA
            # deltas + the full weights of every module listed in
            # modules_to_save (ts_tower, ts_adapter, embed_tokens, and —
            # for forecasting runs — forecast_head).
            model = PeftModel.from_pretrained(model, a.adapter_path)
        model.eval()

        # forecast_mode is on iff the model has a head AND the head's
        # parameters are non-trivially populated. The head is constructed
        # in __init__ when forecast_head_config is on the saved config.
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        head = getattr(base, "forecast_head", None)
        if head is not None:
            self.forecast_mode = True
            head_cfg = getattr(config, "forecast_head_config", None) or {}
            print(
                f"[FORECAST_HEAD] loaded via PEFT — "
                f"window={head_cfg.get('window')} pred_len={head_cfg.get('pred_len')}",
                flush=True,
            )

        self.model = model
        self.processor = processor

    def make_collator(self) -> Callable[[list[dict]], dict]:
        a = self.args
        return _TimeOmniVCollator(
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
            forecast_mode=self.forecast_mode,
            keep_inline_ts_tasks=(
                frozenset()
                if self.forecast_mode
                else frozenset({"prediction"})
            ),
        )

    def compute_lengths(self, dataset) -> list[int]:
        a = self.args
        ts_patch_size = int(getattr(self.processor, "ts_patch_size", 16))
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
            prompt_transform=strip_ts_block,
            prompt_transform_id="strip_ts",
            # Gate must mirror _TimeOmniVCollator: keep inline TS for prediction
            # rows only when the encoder TS feed is active. Under
            # `no_timeseries` (include_timeseries=False) we strip everywhere.
            prompt_transform_skip_tasks=(
                frozenset({"prediction"}) if self.include_timeseries else None
            ),
            include_vision=self.include_vision,
            include_timeseries=self.include_timeseries,
            ts_patch_size=ts_patch_size,
            cache_path=Path(a.test_jsonl + ".lengths.json"),
        )

    def generate(self, batch_inputs, *, max_new_tokens: int) -> list[str]:
        if self.forecast_mode:
            return self._forecast(batch_inputs)
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

    def _forecast(self, batch_inputs) -> list[str]:
        """Run the forecast head and format predictions back into
        ``<forecast>(k: v_k)\\n…</forecast>`` text using integer step indices
        as labels. eval.py's ``align_forecast`` finds no label overlap with
        the gt's date labels and falls back to positional alignment, which
        is exactly the right semantics here (the head emits a fixed-order
        step sequence).
        """
        device = self.model.device
        model_inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }
        with torch.no_grad():
            preds = self.model.forecast(**model_inputs)  # (B, pred_len)
        preds = preds.float().cpu().numpy()

        out_texts: list[str] = []
        for i in range(preds.shape[0]):
            lines = ["<forecast>"]
            for k in range(preds.shape[1]):
                lines.append(f"({k}: {preds[i, k]:g})")
            lines.append("</forecast>")
            out_texts.append("\n".join(lines))
        return out_texts


__all__ = ["TimeOmniVBackend"]
