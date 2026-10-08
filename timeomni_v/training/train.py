"""Unified training entrypoint for TimeOmni-v.

Two training modes share one script. Select with ``--mode``:

* ``--mode timeomni_v`` (default) — full TimeOmni-v architecture. TS is a separate
  tensor stream (chronos encoder → TS adapter → scatter into embeddings at
  ``<|ts_placeholder|>``). Trainable: ts_tower (FT), ts_adapter (FT), LLM
  (LoRA), 3 new TS token embedding rows. Frozen: visual, audio_tower, LM
  head, original vocab rows.

* ``--mode baseline`` — plain Qwen2.5-Omni Thinker, TS rendered as text inside
  the prompt (``<timeseries>...``). Trainable: LLM via LoRA only. Frozen:
  visual, audio_tower.

Most scaffolding (LoRA config, frozen-tower wrap, dynamic-BS batching,
length estimation, save) is shared. Mode-specific code is confined to two
``build_*`` builders and the collator choice in ``main``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)

from timeomni_v.data.collator import (
    DEFAULT_FPS,
    DEFAULT_IMAGE_MAX_PIXELS,
    DEFAULT_IMAGE_MIN_PIXELS,
    DEFAULT_TS_DECIMALS,
    DEFAULT_TS_MAX_LINES,
    DEFAULT_VIDEO_MAX_PIXELS,
    DEFAULT_VIDEO_MIN_PIXELS,
    BaselineCollator,
    TimeOmniVDataCollator,
    downsample_ts_block,
)
from timeomni_v.data.forecast_collator import ForecastCollator
from timeomni_v.data.dataset import TimeOmniVDataset
from timeomni_v.data.length_estimate import compute_exact_lengths
from timeomni_v.modeling.configuration_timeomni_v import TimeOmniVConfig
from timeomni_v.modeling.modeling_timeomni_v import TimeOmniVForConditionalGeneration
from timeomni_v.processing.processing_timeomni_v import TimeOmniVProcessor
from timeomni_v.training.dynamic_trainer import DynamicBSTrainer
from timeomni_v.training.step_timer import StepTimerCallback
from timeomni_v.utils.frozen_tower import wrap_frozen_tower_forward
from timeomni_v.utils.tokens import (
    add_ts_tokens,
    freeze_except_new_embedding_rows,
    init_new_token_embeddings_from_mean,
)
from timeomni_v.utils.trainable import dump_trainable_params
from timeomni_v.utils.warnings import (
    silence_qwen_audio_system_prompt_warning,
    silence_rope_scaling_warning,
)

silence_rope_scaling_warning()
silence_qwen_audio_system_prompt_warning()


@dataclass
class RunArgs:
    mode: Literal["timeomni_v", "baseline"] = field(
        default="timeomni_v",
        metadata={"help": "timeomni_v: full TimeOmni-v (TS tensor stream). baseline: Thinker + inline-TS text."},
    )
    log_step_timing: bool = field(
        default=False,
        metadata={"help": "Sample GPU power + SM util via NVML and print a per-logging-step summary (no CUDA synchronize, no fwd/bwd split). Also enables per-batch collator timing."},
    )


@dataclass
class ModelArgs:
    qwen_path: str = field(metadata={"help": "Path to Qwen2.5-Omni checkpoint"})
    chronos_path: str | None = field(
        default=None,
        metadata={"help": "Path to Chronos-2 checkpoint (required for --mode timeomni_v)"},
    )


@dataclass
class DataArgs:
    # Accept one path or many. With a single path the run is unchanged. With
    # multiple paths the rows are concatenated and every batch is constrained
    # to a single source jsonl (see TokenBudgetBatchSampler.task_ids); one
    # epoch still covers every row from every source exactly once.
    train_jsonl: list[str] = field(metadata={"help": "Training jsonl(s); pass multiple to enable per-task batches", "nargs": "+"})
    eval_jsonl: str | None = field(default=None, metadata={"help": "Optional eval jsonl"})
    fps: float = field(default=DEFAULT_FPS, metadata={"help": "Target FPS passed to Qwen video processor"})
    # Per-frame pixel-area bounds; processor rescales each frame so its pixel
    # count falls in [video_min_pixels, video_max_pixels]. With patch=14 and
    # merge=2, tokens_per_frame = pixels / (14*14*4) = pixels / 784.
    video_min_pixels: int = field(default=DEFAULT_VIDEO_MIN_PIXELS, metadata={"help": "Min pixel area per video frame (default ≈128 tokens/frame)"})
    video_max_pixels: int = field(default=DEFAULT_VIDEO_MAX_PIXELS, metadata={"help": "Max pixel area per video frame (default ≈192 tokens/frame)"})
    image_min_pixels: int = field(default=DEFAULT_IMAGE_MIN_PIXELS, metadata={"help": "Min pixel area per image (default = 2x video_min_pixels)"})
    image_max_pixels: int = field(default=DEFAULT_IMAGE_MAX_PIXELS, metadata={"help": "Max pixel area per image (default = 2x video_max_pixels)"})
    # --- timeomni_v mode only ---
    fusion_mode: str = field(
        default="block_adjacent",
        metadata={"help": "[timeomni_v] block_adjacent | time_interleave (see TimeOmniVProcessor)"},
    )
    # --- baseline mode only ---
    ts_max_lines: int = field(default=DEFAULT_TS_MAX_LINES, metadata={"help": "[baseline] Max TS data rows per sample (0 = no downsample)"})
    ts_decimals: int = field(default=DEFAULT_TS_DECIMALS, metadata={"help": "[baseline] Round TS numeric values to N decimals"})
    # --- shared ---
    # Token-budget batching: 0 disables (standard fixed per_device_batch_size),
    # >0 enables TokenBudgetBatchSampler — short samples packed up to the budget,
    # long samples keep bs=1. Good starting point: ~3x median observed seq.
    dynamic_bs_max_tokens: int = field(default=0, metadata={"help": "Token budget per batch; 0 disables dynamic bs"})
    dynamic_bs_max_bs: int = field(default=8, metadata={"help": "Hard cap on samples/batch when packing"})
    # --- forecasting-head mode (timeomni_v only) ---
    # pred_len > 0 enables the linear forecast head: train.py attaches a
    # ForecastHead to the model, swaps in ForecastCollator (which loads
    # forecast_target_path produced by convert_per_channel_forecasting and
    # masks labels off so HF Trainer's CE loss is bypassed), and the model
    # returns MSE in normalized space.
    pred_len: int = field(default=0, metadata={"help": "[timeomni_v] Forecast head output length. 0 disables forecasting."})
    head_window: int = field(default=8, metadata={"help": "[timeomni_v] # of LLM hiddens read from first ts_placeholder."})
    head_dropout: float = field(default=0.0, metadata={"help": "[timeomni_v] Dropout in the forecast head."})


@dataclass
class LoraArgs:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def _assert_flash_attn_2(model) -> None:
    """Fail loud if flash-attention-2 didn't actually attach to the LLM layers.

    ``from_pretrained(attn_implementation='flash_attention_2')`` will silently
    fall back to SDPA if the flash-attn wheel isn't importable at load time —
    the only signal is a short INFO log that's easy to miss. Walk the config
    tree and make sure every submodule that exposes `_attn_implementation`
    actually says ``flash_attention_2``; also verify the `flash_attn` package
    is importable so we're not running eager by accident.
    """
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "flash_attn not importable — model.from_pretrained will fall back to "
            f"SDPA/eager and cost 2-3x on long sequences. Original error: {e}"
        ) from e

    base = model
    for attr in ("base_model", "model"):
        if hasattr(base, attr):
            base = getattr(base, attr)

    cfgs = []
    def _collect(obj, name):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            cfgs.append((name, cfg))
        for sub_name in ("thinker", "talker", "text_config", "vision_config",
                         "audio_config", "thinker_config"):
            sub = getattr(obj, sub_name, None)
            if sub is not None and sub is not obj:
                _collect(sub, f"{name}.{sub_name}")
    _collect(base, "model")
    for cfg in list(cfgs):
        name, c = cfg
        for sub_name in ("text_config", "vision_config", "audio_config",
                         "thinker_config"):
            sub = getattr(c, sub_name, None)
            if sub is not None:
                cfgs.append((f"{name}.{sub_name}", sub))

    bad = []
    seen = False
    for name, cfg in cfgs:
        impl = getattr(cfg, "_attn_implementation", None)
        if impl is None:
            continue
        seen = True
        if impl != "flash_attention_2":
            bad.append(f"{name}: _attn_implementation={impl!r}")
    if not seen:
        print("[FA2] no _attn_implementation field found on any config — "
              "could not verify, assuming FA2 is attached.", flush=True)
        return
    if bad:
        raise RuntimeError(
            "Flash-Attention-2 did not attach to all sublayers:\n  "
            + "\n  ".join(bad)
            + "\nCheck the flash-attn wheel matches your torch/CUDA version."
        )
    print("[FA2] verified: all configs report _attn_implementation="
          "'flash_attention_2'", flush=True)


def _lora_target_regex(lora_args: LoraArgs) -> str:
    """Restrict LoRA to LLM layers only. Audio / vision towers must stay frozen,
    so we match the full path ``model.layers.<i>...<proj>$`` — this excludes
    e.g. ``audio_tower.layers.0.q_proj`` or ``visual.blocks.0.mlp.gate_proj``.
    """
    proj_alt = "|".join(re.escape(n) for n in lora_args.target_modules.split(","))
    return rf"^model\.layers\.\d+\..*\.({proj_alt})$"


def build_timeomni_v_model_and_processor(
    model_args: ModelArgs, lora_args: LoraArgs, data_args: "DataArgs | None" = None
):
    if not model_args.chronos_path:
        raise ValueError("--chronos_path is required when --mode=timeomni_v")

    # Build TimeOmniVConfig from the checkpoint's thinker_config slice.
    full_cfg = AutoConfig.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    thinker_dict = full_cfg.thinker_config.to_dict()
    thinker_dict.pop("model_type", None)
    config = TimeOmniVConfig(
        ts_encoder_path=model_args.chronos_path,
        ts_encoder_hidden_size=768,
        ts_adapter_hidden_size=2048,
        **thinker_dict,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    old_vocab_size = len(tokenizer)
    ids = add_ts_tokens(tokenizer)
    config.timeseries_token_id = ids.placeholder
    config.timeseries_start_token_id = ids.start
    config.timeseries_end_token_id = ids.end

    processor = TimeOmniVProcessor.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    processor.tokenizer = tokenizer
    # fusion_mode is set later in main() from data_args

    model = TimeOmniVForConditionalGeneration.from_pretrained(
        model_args.qwen_path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    _assert_flash_attn_2(model)
    # Qwen2.5-Omni already pads its embedding to 152064 rows while the tokenizer
    # only uses ~151665 — the 399 slot gap covers our 3 new TS specials. So
    # `resize_token_embeddings` is unnecessary (and would actually SHRINK the
    # table, breaking shape compat at load time). We just need to make sure
    # the new IDs fit inside the existing capacity.
    emb = model.get_input_embeddings()
    assert emb.weight.shape[0] > max(ids.placeholder, ids.start, ids.end), (
        f"TS token ids exceed embedding capacity ({emb.weight.shape[0]}); "
        "add resize_token_embeddings back or use a vocab with more padding."
    )
    init_new_token_embeddings_from_mean(
        model, old_vocab_size, [ids.placeholder, ids.start, ids.end]
    )

    # Sync TS patch size from the loaded Chronos checkpoint into the processor
    # so n_patches calculation in the collator matches what ts_tower will emit.
    if model.ts_tower is not None:
        processor.ts_patch_size = int(model.ts_tower.patch_size)

    # Forecasting head — attach BEFORE LoRA wrap so its parameters are
    # registered with PEFT's modules_to_save and end up in the saved
    # checkpoint. d_llm comes from the LLM hidden size in the loaded config.
    if data_args is not None and data_args.pred_len > 0:
        d_llm = config.text_config.hidden_size
        model.attach_forecast_head(
            hidden_size=d_llm,
            window=data_args.head_window,
            pred_len=data_args.pred_len,
            dropout=data_args.head_dropout,
        )
        print(
            f"[FORECAST_HEAD] attached: hidden_size={d_llm} "
            f"window={data_args.head_window} pred_len={data_args.pred_len} "
            f"dropout={data_args.head_dropout}",
            flush=True,
        )

    # modules_to_save lists non-LoRA modules whose **full weights** should be
    # persisted by PeftModel.save_pretrained. Without this, only LoRA adapters
    # are saved and our full-FT ts_tower / ts_adapter / new embedding rows are
    # silently lost — inference would re-init them randomly.
    #
    # ensure_weight_tying must follow the BASE MODEL's tying setting:
    #   - Qwen2.5-Omni Thinker has tie_word_embeddings=False (lm_head and
    #     embed_tokens are independently trained tensors). If we set
    #     ensure_weight_tying=True here, PEFT FORCIBLY re-points lm_head.weight
    #     to the embed_tokens deepcopy's storage, throwing away the original
    #     trained lm_head and using the embedding matrix as output projection.
    #     Logits then explode (we observed loss=481 in eval, ~2000 in train).
    #   - Qwen3-Omni-MoE Thinker has tie_word_embeddings=True. Without
    #     ensure_weight_tying, PEFT's deepcopy of embed_tokens breaks the tie
    #     (lm_head keeps original rows; new TS token rows never reach the head).
    # We pull the actual flag from the loaded config so this stays correct
    # across model swaps.
    base_text_cfg = getattr(model.config, "text_config", model.config)
    tied = bool(getattr(base_text_cfg, "tie_word_embeddings", False))
    print(f"[LORA] tie_word_embeddings={tied} → ensure_weight_tying={tied}", flush=True)
    lora_cfg = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.dropout,
        target_modules=_lora_target_regex(lora_args),
        modules_to_save=(
            ["ts_tower", "ts_adapter", "embed_tokens", "forecast_head"]
            if data_args is not None and data_args.pred_len > 0
            else ["ts_tower", "ts_adapter", "embed_tokens"]
        ),
        ensure_weight_tying=tied,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    # PEFT + reentrant gradient_checkpointing requires the embedding output to
    # carry requires_grad=True, otherwise the checkpointed region's inputs are
    # detached (base model is frozen under LoRA) and loss.backward() aborts.
    model.enable_input_require_grads()

    # modules_to_save wraps ts_tower / ts_adapter / embed_tokens so their full
    # weights are persisted. But those wrappers also mark everything inside
    # them trainable — we want only the 3 new embedding rows trainable, not
    # the original 150k+ vocab. Re-freeze via a gradient hook.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    freeze_except_new_embedding_rows(
        base.get_input_embeddings(),
        new_ids=[ids.placeholder, ids.start, ids.end],
    )

    wrapped = wrap_frozen_tower_forward(base, ["visual", "audio_tower"])
    if wrapped:
        print(f"[FROZEN_TOWER] wrapped forward in no_grad: {wrapped}", flush=True)

    model.print_trainable_parameters()
    return model, processor


def build_baseline_model_and_processor(model_args: ModelArgs, lora_args: LoraArgs):
    """Load Thinker-only Qwen2.5-Omni and apply LoRA on the LLM only."""
    full_cfg = AutoConfig.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    thinker_cfg = full_cfg.thinker_config

    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_args.qwen_path,
        config=thinker_cfg,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    _assert_flash_attn_2(model)

    lora_cfg = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.dropout,
        target_modules=_lora_target_regex(lora_args),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    wrapped = wrap_frozen_tower_forward(base, ["visual", "audio_tower"])
    if wrapped:
        print(f"[FROZEN_TOWER] wrapped forward in no_grad: {wrapped}", flush=True)

    model.print_trainable_parameters()
    return model, processor


def _build_collator(mode: str, processor, data_args: DataArgs, log_timing: bool):
    if mode == "timeomni_v":
        if data_args.pred_len > 0:
            return ForecastCollator(
                processor=processor,
                pred_len=data_args.pred_len,
                fps=data_args.fps,
                video_min_pixels=data_args.video_min_pixels,
                video_max_pixels=data_args.video_max_pixels,
                image_min_pixels=data_args.image_min_pixels,
                image_max_pixels=data_args.image_max_pixels,
                log_timing=log_timing,
            )
        return TimeOmniVDataCollator(
            processor=processor,
            fps=data_args.fps,
            video_min_pixels=data_args.video_min_pixels,
            video_max_pixels=data_args.video_max_pixels,
            image_min_pixels=data_args.image_min_pixels,
            image_max_pixels=data_args.image_max_pixels,
            log_timing=log_timing,
        )
    return BaselineCollator(
        processor=processor,
        fps=data_args.fps,
        video_min_pixels=data_args.video_min_pixels,
        video_max_pixels=data_args.video_max_pixels,
        image_min_pixels=data_args.image_min_pixels,
        image_max_pixels=data_args.image_max_pixels,
        ts_max_lines=data_args.ts_max_lines,
        ts_decimals=data_args.ts_decimals,
        log_timing=log_timing,
    )


def main() -> None:
    parser = HfArgumentParser((RunArgs, ModelArgs, DataArgs, LoraArgs, TrainingArguments))
    run_args, model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    if run_args.mode == "timeomni_v":
        model, processor = build_timeomni_v_model_and_processor(model_args, lora_args, data_args)
        processor.fusion_mode = data_args.fusion_mode
    elif run_args.mode == "baseline":
        if model_args.chronos_path:
            raise ValueError("--chronos_path is ignored in --mode=baseline; remove it.")
        if data_args.pred_len > 0:
            raise ValueError("--pred_len > 0 requires --mode=timeomni_v (forecast head needs ts_tower).")
        model, processor = build_baseline_model_and_processor(model_args, lora_args)
    else:
        raise ValueError(f"unknown --mode {run_args.mode!r}; expected 'timeomni_v' or 'baseline'")

    # Forecast-head runs set `labels=None` in the collator (the head returns
    # MSE directly). HF Trainer's prediction_step gates loss extraction on
    # `all(inputs.get(k) is not None for k in self.label_names)`, defaulting
    # to ["labels"]; with labels=None has_labels=False → eval_loss never gets
    # populated → metric_for_best_model="eval_loss" crashes at first eval.
    # Tell Trainer that forecast_target is the label key for these runs.
    if data_args.pred_len > 0:
        training_args.label_names = ["forecast_target"]

    dump_trainable_params(model, training_args.output_dir)
    train_ds = TimeOmniVDataset(data_args.train_jsonl)
    eval_ds = TimeOmniVDataset(data_args.eval_jsonl) if data_args.eval_jsonl else None
    collator = _build_collator(run_args.mode, processor, data_args, run_args.log_step_timing)

    lengths = None
    eval_lengths = None
    if data_args.dynamic_bs_max_tokens > 0:
        from pathlib import Path
        from timeomni_v.data.collator import strip_ts_block
        base_length_kwargs = dict(
            processor=processor,
            video_min_pixels=data_args.video_min_pixels,
            video_max_pixels=data_args.video_max_pixels,
            image_min_pixels=data_args.image_min_pixels,
            image_max_pixels=data_args.image_max_pixels,
            do_sample_frames=collator.do_sample_frames,
            target_fps=collator.fps,
            min_frames=collator.min_frames,
            max_frames=collator.max_frames,
        )
        if run_args.mode == "baseline":
            base_length_kwargs["prompt_transform"] = lambda p: downsample_ts_block(
                p, data_args.ts_max_lines, data_args.ts_decimals
            )
            base_length_kwargs["prompt_transform_id"] = f"ds{data_args.ts_max_lines}_d{data_args.ts_decimals}"
        else:  # timeomni_v: strip the inline TS body for every row — both
               # classification (was already stripped) and prediction (now
               # going through the forecast head, the model never sees inline
               # numerics). compute_exact_lengths still accounts for the
               # chat-template TS marker that expands to P*C tokens.
            base_length_kwargs["prompt_transform"] = strip_ts_block
            base_length_kwargs["prompt_transform_id"] = "strip_ts"
            base_length_kwargs["prompt_transform_skip_tasks"] = None
            base_length_kwargs["include_timeseries"] = True
            base_length_kwargs["ts_patch_size"] = processor.ts_patch_size
        # Lengths cache lives next to each source jsonl. With multiple sources
        # we run compute_exact_lengths once per file so each source's sidecar
        # stays reusable across single-task and multi-task runs (the cache
        # key already encodes every length-affecting knob; only `n` and the
        # source path change). Concat preserves the row order of train_ds,
        # which loaded sources in the same order.
        per_source_lengths: list[list[int]] = []
        for src_path in train_ds.source_paths:
            src_ds = TimeOmniVDataset(src_path)
            per_source_lengths.append(compute_exact_lengths(
                dataset=src_ds,
                cache_path=Path(str(src_path) + ".lengths.json"),
                **base_length_kwargs,
            ))
        lengths = [L for ls in per_source_lengths for L in ls]
        assert len(lengths) == len(train_ds), (
            f"concat per-source lengths {len(lengths)} != train_ds {len(train_ds)} — "
            "row filtering drift between TimeOmniVDataset(list) and TimeOmniVDataset(single)"
        )
        if eval_ds is not None:
            eval_lengths = compute_exact_lengths(
                dataset=eval_ds,
                cache_path=Path(data_args.eval_jsonl + ".lengths.json"),
                **base_length_kwargs,
            )

    # NVML power + SM util summary, no cuda.synchronize (see step_timer.py).
    # Same flag enables per-batch timing inside the collator.
    callbacks = []
    if run_args.log_step_timing:
        callbacks.append(StepTimerCallback())

    # Only enforce per-task batches when more than one source jsonl was
    # supplied. With a single source `task_of_index` is all zeros and the
    # task-grouped path collapses to the original global packing — but we
    # pass None to keep the legacy code path verbatim and avoid a needless
    # task_ids size check.
    train_task_ids = train_ds.task_of_index if len(train_ds.source_paths) > 1 else None

    trainer = DynamicBSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks or None,
        lengths=lengths,
        eval_lengths=eval_lengths,
        max_tokens=data_args.dynamic_bs_max_tokens,
        max_bs=data_args.dynamic_bs_max_bs,
        task_ids=train_task_ids,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    # timeomni_v mode must persist TimeOmniVConfig (with ts_encoder_path and TS token
    # ids) + tokenizer so infer.py can reconstruct the model. Baseline relies
    # on the official Qwen2.5-Omni config, which the Trainer saves automatically.
    if run_args.mode == "timeomni_v" and trainer.is_world_process_zero():
        base_for_save = model.get_base_model() if hasattr(model, "get_base_model") else model
        # `config.save_pretrained` writes forecast_head_config (set by
        # attach_forecast_head) onto the saved config.json, so a fresh
        # from_pretrained at inference time rebuilds the head with matching
        # dims; PEFT's adapter_model.safetensors already carries the
        # trained weights via modules_to_save=["forecast_head", ...].
        base_for_save.config.save_pretrained(training_args.output_dir)
        processor.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
