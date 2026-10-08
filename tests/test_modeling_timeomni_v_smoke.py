"""End-to-end smoke test for TimeOmni-v.

Loads the full Qwen2.5-Omni Thinker weights + Chronos-2 encoder, runs one
forward pass on a single real CoVLA sample, and verifies the loss is finite.
Also checks that backward flows gradients into the TS encoder + adapter.

These tests require a GPU with enough memory (~16 GB for bf16 7B dense) plus
local copies of the checkpoints and a sample jsonl. They are skipped unless
the following env vars are set:

    TIMEOMNI_V_QWEN_PATH=/path/to/Qwen2.5-Omni-7B
    TIMEOMNI_V_CHRONOS_PATH=/path/to/chronos-2
    TIMEOMNI_V_SMOKE_JSONL=/path/to/sample.jsonl
    pytest tests/test_modeling_timeomni_v_smoke.py -v -m slow
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoTokenizer

from timeomni_v.data.collator import TimeOmniVDataCollator
from timeomni_v.data.dataset import TimeOmniVDataset
from timeomni_v.modeling.configuration_timeomni_v import TimeOmniVConfig
from timeomni_v.modeling.modeling_timeomni_v import TimeOmniVForConditionalGeneration
from timeomni_v.processing.processing_timeomni_v import TimeOmniVProcessor
from timeomni_v.utils.tokens import add_ts_tokens

QWEN_PATH = os.environ.get("TIMEOMNI_V_QWEN_PATH", "")
CHRONOS_PATH = os.environ.get("TIMEOMNI_V_CHRONOS_PATH", "")
JSONL = os.environ.get("TIMEOMNI_V_SMOKE_JSONL", "")

pytestmark = pytest.mark.skipif(
    not (QWEN_PATH and CHRONOS_PATH and JSONL),
    reason="set TIMEOMNI_V_QWEN_PATH / TIMEOMNI_V_CHRONOS_PATH / TIMEOMNI_V_SMOKE_JSONL to run the smoke test",
)


def _build_model_and_processor():
    """Shared setup: load config, tokenizer, processor, model with TS tokens wired."""
    # Build config from the top-level OmniConfig's thinker_config slice.
    # We can't call TimeOmniVConfig.from_pretrained(QWEN_PATH) directly because the
    # checkpoint is a top-level OmniConfig; extract thinker_config and re-instantiate.
    from transformers import AutoConfig

    full_cfg = AutoConfig.from_pretrained(QWEN_PATH, trust_remote_code=True)
    thinker_dict = full_cfg.thinker_config.to_dict()
    # Drop the inner model_type so our TimeOmniVConfig's class-level value wins.
    thinker_dict.pop("model_type", None)
    config = TimeOmniVConfig(
        ts_encoder_path=CHRONOS_PATH,
        ts_encoder_hidden_size=768,
        ts_adapter_hidden_size=2048,
        **thinker_dict,
    )

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    ids = add_ts_tokens(tokenizer)
    config.timeseries_token_id = ids.placeholder
    config.timeseries_start_token_id = ids.start
    config.timeseries_end_token_id = ids.end

    processor = TimeOmniVProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    processor.tokenizer = tokenizer

    model = TimeOmniVForConditionalGeneration.from_pretrained(
        QWEN_PATH,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


@pytest.mark.slow
def test_one_forward_pass():
    assert torch.cuda.is_available(), "smoke test needs a GPU"
    model, processor = _build_model_and_processor()
    model.eval()

    ds = TimeOmniVDataset(JSONL)
    collator = TimeOmniVDataCollator(processor=processor)
    batch = collator([ds[0]])
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
        out = model(**batch)
    assert torch.isfinite(out.loss), f"loss is not finite: {out.loss.item()}"


@pytest.mark.slow
def test_one_backward_pass():
    """Backward only through ts_tower + ts_adapter (LLM frozen).

    The training config freezes the LLM weights and trains them via LoRA;
    computing gradients on the full 7B dense LLM during a bare smoke test
    would OOM. The point of this test is to verify TS path gradients flow,
    not LLM ones.
    """
    assert torch.cuda.is_available(), "smoke test needs a GPU"
    model, processor = _build_model_and_processor()
    model.train()
    model.gradient_checkpointing_enable()

    # Mimic training-time freezing: only TS-side modules get grads.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.ts_tower.parameters():
        p.requires_grad = True
    for p in model.ts_adapter.parameters():
        p.requires_grad = True

    ds = TimeOmniVDataset(JSONL)
    batch = TimeOmniVDataCollator(processor=processor)([ds[0]])
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    out = model(**batch)
    out.loss.backward()

    # Adapter: all params should have finite grads
    for p in model.ts_adapter.parameters():
        assert p.grad is not None, "ts_adapter param has no grad"
        assert torch.isfinite(p.grad).all(), "ts_adapter grad contains non-finite values"

    # TS tower: at least some params must have finite grads
    any_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.ts_tower.parameters()
        if p.requires_grad
    )
    assert any_grad, "no finite grads seen in ts_tower"
