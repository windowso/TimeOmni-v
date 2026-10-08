"""Tests for TimeOmniVProcessor TS-only expansion logic (no parent processor needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from timeomni_v.processing.processing_timeomni_v import (
    TimeOmniVProcessor,
    TS_MARKER,
    _rewrite_ts_elements,
    interleave_video_ts_placeholders,
    load_and_pad_ts_csvs,
)


def test_ts_marker_is_single_placeholder_wrapped():
    """TS_MARKER is what the chat template emits per {"type":"timeseries"}
    element — one placeholder, wrapped in start/end markers. __call__
    expands the placeholder to P*C copies once CSV shape is known."""
    assert TS_MARKER.count("<|ts_placeholder|>") == 1
    assert TS_MARKER.startswith("<|ts_start|>")
    assert TS_MARKER.endswith("<|ts_end|>")


def test_rewrite_ts_elements_maps_structured_to_text_marker():
    conv = [
        {"role": "user", "content": [
            {"type": "video", "video": "/fake.mp4"},
            {"type": "timeseries", "timeseries": "/fake.csv"},
            {"type": "text", "text": "hello"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
    ]
    out = _rewrite_ts_elements(conv)
    user_content = out[0]["content"]
    assert user_content[0] == {"type": "video", "video": "/fake.mp4"}
    assert user_content[1] == {"type": "text", "text": TS_MARKER}
    assert user_content[2] == {"type": "text", "text": "hello"}
    # Assistant turn untouched
    assert out[1] == conv[1]
    # Input not mutated in place
    assert conv[0]["content"][1] == {"type": "timeseries", "timeseries": "/fake.csv"}


def test_rewrite_ts_elements_handles_batched_input():
    conv = [
        [{"role": "user", "content": [{"type": "timeseries", "timeseries": "/a.csv"}]}],
        [{"role": "user", "content": [{"type": "timeseries", "timeseries": "/b.csv"}]}],
    ]
    out = _rewrite_ts_elements(conv)
    assert len(out) == 2
    for c in out:
        assert c[0]["content"][0] == {"type": "text", "text": TS_MARKER}


def test_rewrite_ts_elements_noop_without_ts_elements():
    conv = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _rewrite_ts_elements(conv) == conv


def test_interleave_video_ts_placeholders_orders_by_time():
    video_pos = [0.0, 0.5, 1.0]
    ts_pos = [0.0, 0.7]
    out = interleave_video_ts_placeholders(
        video_positions=video_pos, ts_positions=ts_pos,
        video_tokens_per_step=2, ts_tokens_per_step=3,
    )
    # time sort: v@0.0, ts@0.0, v@0.5, ts@0.7, v@1.0
    assert out == ["V", "V", "T", "T", "T", "V", "V", "T", "T", "T", "V", "V"]


def test_load_and_pad_ts_csvs_basic(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, float("nan"), 30.0]})
    path1 = tmp_path / "s1.csv"
    df.to_csv(path1, index=False)
    df2 = pd.DataFrame({"a": [4.0, 5.0], "b": [40.0, 50.0]})
    path2 = tmp_path / "s2.csv"
    df2.to_csv(path2, index=False)

    ts_values, n_channels, n_timesteps = load_and_pad_ts_csvs([str(path1), str(path2)])
    assert ts_values.shape == (4, 3)
    assert n_channels == [2, 2]
    assert n_timesteps == [3, 2]
    assert torch.isnan(ts_values[1, 1])
    assert torch.isnan(ts_values[2, 2]) and torch.isnan(ts_values[3, 2])


def _mock_processor_for_interleave(
    video_token_id: int,
    ts_token_id: int,
    merge_size: int,
    ts_start_id: int = 101,
    ts_end_id: int = 102,
):
    """Build an TimeOmniVProcessor-like object wired only well enough to exercise
    _postprocess_time_interleave on a synthetic batch.
    """
    proc = TimeOmniVProcessor.__new__(TimeOmniVProcessor)
    proc.tokenizer = MagicMock()
    proc.tokenizer.video_token = "<|video_pad|>"
    # convert_tokens_to_ids is used for video_token, ts_placeholder, ts_start, ts_end.
    id_table = {
        "<|video_pad|>": video_token_id,
        "<|ts_placeholder|>": ts_token_id,
        "<|ts_start|>": ts_start_id,
        "<|ts_end|>": ts_end_id,
    }
    proc.tokenizer.convert_tokens_to_ids = lambda s: id_table[s]
    proc.video_processor = SimpleNamespace(merge_size=merge_size)
    proc.fusion_mode = "time_interleave"
    proc.ts_patch_size = 16
    return proc


def test_time_interleave_postprocess_preserves_length_and_orders_by_time():
    """Synthetic input:
        input_ids = [99, 99,            # prefix text
                     V, V, V, V, V, V,  # video run: T_v=3 time steps × S_v=2 spatial
                     101,               # ts_start marker
                     T, T, T, T,        # ts run: n_patches=2 × n_channels=2
                     102, 55, 55]       # ts_end marker + suffix text

    Video grid_thw=(T=3, H=4, W=2), merge_size=2 → S_v = (4/2)*(2/2) = 2
    Expected: TS patches 0 and 1 map to steps min(2, 0*3//2)=0 and min(2, 1*3//2)=1.
    So interleaved region = [V V  T T  V V  T T  V V] (P0 after step 0, P1 after step 1).
    The new layout wraps the interleaved region with ts_start/ts_end at the
    boundaries (Qwen-style), and strips them + the placeholders from the tail.
    """
    video_id = 1000
    ts_id = 1001
    ts_start = 101
    ts_end = 102
    # Use sentinel ints for V/T to distinguish; reality uses real token ids.
    V = video_id
    T = ts_id

    ids = torch.tensor([
        99, 99,                # prefix
        V, V, V, V, V, V,      # 6 video tokens (T_v=3, S_v=2)
        ts_start,              # ts_start marker
        T, T, T, T,            # 4 ts tokens (P=2, C=2)
        ts_end, 55, 55,        # ts_end + suffix
    ], dtype=torch.long)
    attn = torch.ones_like(ids)
    batch = {
        "input_ids": ids.unsqueeze(0),
        "attention_mask": attn.unsqueeze(0),
        "video_grid_thw": torch.tensor([[3, 4, 2]], dtype=torch.long),
    }
    proc = _mock_processor_for_interleave(
        video_id, ts_id, merge_size=2,
        ts_start_id=ts_start, ts_end_id=ts_end,
    )
    proc._postprocess_time_interleave(
        batch, n_patches_list=[2], n_channels_list=[2],
    )
    out = batch["input_ids"][0].tolist()

    # Length preserved
    assert len(out) == len(ids)

    # Prefix unchanged
    assert out[:2] == [99, 99]

    # ts_start sits immediately before the interleaved region
    assert out[2] == ts_start

    # Interleaved region: V V T T V V T T V V  (10 tokens) = 6 video + 4 ts
    interleaved = out[3:3 + 10]
    assert interleaved == [V, V, T, T, V, V, T, T, V, V]

    # ts_end sits immediately after the interleaved region
    assert out[3 + 10] == ts_end

    # Tail: original [ts_start, T*4, ts_end, 55, 55] had its ts_start /
    # ts_end / placeholders stripped (moved into the new wrapper) → [55, 55]
    assert out[3 + 10 + 1:] == [55, 55]


def test_time_interleave_mixed_image_and_video_batch():
    """Regression: when a batch mixes image+TS samples (no video tokens) with
    video+TS samples, ``video_grid_thw`` only has rows for the video-bearing
    samples — its row count is < batch size. The loop must use a separate
    ``video_idx`` counter that advances only when consuming a video sample;
    otherwise it either IndexError'd at b=N_video or silently shifted the
    grid lookup and emitted "video run length != T*S".
    """
    video_id = 1000
    ts_id = 1001
    ts_start = 101
    ts_end = 102
    V = video_id
    T = ts_id

    # Sample 0: image+TS (no V tokens). Pad with text so length matches sample 1.
    sample0 = [
        99, 99, 99, 99,           # image prefix (no V — image_pad has its own id)
        ts_start, T, T, ts_end,   # TS run (n_patches=2, n_channels=1)
        55, 55, 55, 55, 55, 55, 55, 55,
    ]
    # Sample 1: video+TS. T_v=3, H_v=4, W_v=2, merge_size=2 → S_v=2 → 6 V tokens.
    # n_patches=2, n_channels=2 → 4 T tokens. Same length (16) as sample 0.
    sample1 = [
        99, 99,                   # text prefix
        V, V, V, V, V, V,         # video run
        ts_start, T, T, T, T, ts_end,
        55, 55,                   # suffix
    ]
    assert len(sample0) == len(sample1) == 16

    input_ids = torch.tensor([sample0, sample1], dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        # Only sample 1 has video — only one row in video_grid_thw.
        "video_grid_thw": torch.tensor([[3, 4, 2]], dtype=torch.long),
    }
    proc = _mock_processor_for_interleave(
        video_id, ts_id, merge_size=2,
        ts_start_id=ts_start, ts_end_id=ts_end,
    )
    proc._postprocess_time_interleave(
        batch,
        # n_patches/n_channels are per-sample (both samples have TS).
        n_patches_list=[2, 2],
        n_channels_list=[1, 2],
    )

    # Sample 0 has no video → returned untouched.
    assert batch["input_ids"][0].tolist() == sample0

    # Sample 1 must have been interleaved correctly using video_grid_thw[0]
    # (the only video row), not video_grid_thw[1] (out of bounds).
    out1 = batch["input_ids"][1].tolist()
    assert len(out1) == 16
    assert out1[:2] == [99, 99]
    assert out1[2] == ts_start
    # Interleaved layout (T_v=3, S_v=2, P=2, C=2): V V T T V V T T V V
    assert out1[3:13] == [V, V, T, T, V, V, T, T, V, V]
    assert out1[13] == ts_end
    assert out1[14:] == [55, 55]
