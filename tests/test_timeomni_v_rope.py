"""Tests for TimeOmniVForConditionalGeneration.get_rope_index.

Validates that:
  * V tokens keep 3D vision M-RoPE positions (t·spg·pps, h, w)
  * TS tokens get 2D positions (time, channel, channel)
  * In time_interleave (V and TS share a vision span), TS time is scaled to
    align with the video time axis
  * In block_adjacent (TS alone), TS time is the integer patch index
  * Text-only / generation continuation (no TS placeholders) defers to parent
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from timeomni_v.modeling.modeling_timeomni_v import TimeOmniVForConditionalGeneration


def _mock_timeomni_v_for_rope(
    *,
    video_token_id=1000,
    ts_token_id=1001,
    vision_start_id=900,
    vision_end_id=901,
    ts_start_id=902,
    ts_end_id=903,
    spatial_merge=2,
    pps=1.0,
):
    """Build a bare-minimum TimeOmniVForConditionalGeneration-ish object with just
    the attributes get_rope_index touches. We never call __init__ (which would
    try to load Qwen weights)."""
    obj = TimeOmniVForConditionalGeneration.__new__(TimeOmniVForConditionalGeneration)
    obj.config = SimpleNamespace(
        timeseries_token_id=ts_token_id,
        timeseries_start_token_id=ts_start_id,
        timeseries_end_token_id=ts_end_id,
        vision_start_token_id=vision_start_id,
        video_token_id=video_token_id,
        position_id_per_seconds=pps,
    )
    obj.spatial_merge_size = spatial_merge
    obj._vision_end_token_id = vision_end_id  # bypasses the +1 lazy resolver
    obj._ts_n_channels = None
    obj._ts_n_patches = None
    return obj


def test_block_adjacent_ts_gets_2d_positions():
    """Block-adjacent layout: text prefix + ts_start + 4 TS placeholders + ts_end.

    Expect TS tokens at positions (patch_idx, ch_idx, ch_idx) with patch_idx
    and ch_idx independent dims. n_chan=2, n_patches=2 → patches: 0,1; chans 0,1.
    Sequence positions (time dim): [0, 1, 2, 2, 3, 3, 4]
    """
    P = 50  # text id placeholder (any non-special integer)
    obj = _mock_timeomni_v_for_rope()
    cfg = obj.config
    ids = torch.tensor([
        P, P,                              # 2 prefix text tokens
        cfg.timeseries_start_token_id,     # ts_start
        cfg.timeseries_token_id,           # T patch=0 ch=0
        cfg.timeseries_token_id,           # T patch=0 ch=1
        cfg.timeseries_token_id,           # T patch=1 ch=0
        cfg.timeseries_token_id,           # T patch=1 ch=1
        cfg.timeseries_end_token_id,       # ts_end
        P,                                 # suffix text token
    ], dtype=torch.long).unsqueeze(0)
    attn = torch.ones_like(ids)

    obj._ts_n_channels = torch.tensor([2], dtype=torch.long)

    pos, deltas = obj.get_rope_index(
        input_ids=ids, attention_mask=attn, video_grid_thw=None,
    )

    # Shape (3, 1, T)
    assert pos.shape == (3, 1, ids.shape[1])

    p = pos[:, 0, :].T.tolist()  # (T, 3) — one row per token, columns = (t, h, w)
    # text positions: 0, 1
    assert p[0] == [0, 0, 0]
    assert p[1] == [1, 1, 1]
    # ts_start at base = cur_pos = 2 (all 3 dims same)
    assert p[2] == [2, 2, 2]
    # 4 TS placeholders, content_base = 3
    # patch 0 ch 0 → (3+0, 3+0, 3+0)
    assert p[3] == [3, 3, 3]
    # patch 0 ch 1 → (3+0, 3+1, 3+1)
    assert p[4] == [3, 4, 4]
    # patch 1 ch 0 → (3+1, 3+0, 3+0)
    assert p[5] == [4, 3, 3]
    # patch 1 ch 1 → (3+1, 3+1, 3+1)
    assert p[6] == [4, 4, 4]
    # ts_end: end_base = content_base + max(inner) + 1 = 3 + 1 + 1 = 5
    assert p[7] == [5, 5, 5]
    # suffix text at cur_pos = end_base + 1 = 6
    assert p[8] == [6, 6, 6]


def test_time_interleave_v_keeps_3d_ts_aligned_in_time():
    """time_interleave layout: vision_start + ts_start + interleaved + ts_end + vision_end.

    Synthetic video grid: T_v=2, H=4, W=4, merge=2 → llm_h=2, llm_w=2, S_v=4.
    n_patches=2, n_chan=1. ts_time_per_patch = (T_v · spg · pps) / n_patches
                                            = (2 · 1 · 1) / 2 = 1.0
    So TS patch 0 → t_val=0, patch 1 → t_val=1 — aligned with the two video
    time steps.

    Layout in the span (after vision_start ts_start):
      step 0: V V V V (4 spatial), then T (patch 0)
      step 1: V V V V, then T (patch 1)
    """
    P = 50
    obj = _mock_timeomni_v_for_rope(spatial_merge=2, pps=1.0)
    cfg = obj.config

    # Build the sequence
    seq = [
        P,                               # prefix
        cfg.vision_start_token_id,       # vision_start
        cfg.timeseries_start_token_id,   # ts_start
        # step 0: 4 V then 1 T
        cfg.video_token_id, cfg.video_token_id, cfg.video_token_id, cfg.video_token_id,
        cfg.timeseries_token_id,
        # step 1: 4 V then 1 T
        cfg.video_token_id, cfg.video_token_id, cfg.video_token_id, cfg.video_token_id,
        cfg.timeseries_token_id,
        cfg.timeseries_end_token_id,     # ts_end
        cfg._vision_end_token_id if False else 901,  # vision_end (id 901)
        P,                               # suffix
    ]
    ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    attn = torch.ones_like(ids)

    obj._ts_n_channels = torch.tensor([1], dtype=torch.long)

    grid_thw = torch.tensor([[2, 4, 4]], dtype=torch.long)
    spg = torch.tensor([1.0])
    pos, deltas = obj.get_rope_index(
        input_ids=ids, attention_mask=attn, video_grid_thw=grid_thw,
        second_per_grids=spg,
    )

    p = pos[:, 0, :].T.tolist()  # (T, 3) — one row per token, columns = (t, h, w)
    # prefix text
    assert p[0] == [0, 0, 0]
    # vision_start at base = 1
    assert p[1] == [1, 1, 1]
    # ts_start (right after vision_start) at base + 1 = 2
    assert p[2] == [2, 2, 2]
    # content_base = 3
    # 4 V at step 0: t_val=0; spatial idx 0..3 → (h,w) = (0,0)(0,1)(1,0)(1,1)
    assert p[3] == [3, 3, 3]   # (3+0, 3+0, 3+0)
    assert p[4] == [3, 3, 4]   # (3+0, 3+0, 3+1)
    assert p[5] == [3, 4, 3]   # (3+0, 3+1, 3+0)
    assert p[6] == [3, 4, 4]   # (3+0, 3+1, 3+1)
    # T patch 0: t_val = 0*1=0, ch=0
    assert p[7] == [3, 3, 3]   # (3+0, 3+0, 3+0)
    # 4 V at step 1: t_val=1
    assert p[8] == [4, 3, 3]
    assert p[9] == [4, 3, 4]
    assert p[10] == [4, 4, 3]
    assert p[11] == [4, 4, 4]
    # T patch 1: t_val = 1*1=1, ch=0
    assert p[12] == [4, 3, 3]
    # ts_end: content_base + inner_max + 1 = 3 + 1 + 1 = 5
    assert p[13] == [5, 5, 5]
    # vision_end at end_base + 1 = 6
    assert p[14] == [6, 6, 6]
    # suffix text at cur_pos = end_base + 2 = 7
    assert p[15] == [7, 7, 7]

    # mrope delta = max_pos + 1 - len = 7 + 1 - 16 = -8
    assert deltas.item() == 7 + 1 - ids.shape[1]


def test_no_ts_falls_through_to_parent():
    """If there are no TS tokens in input_ids, must defer to parent's
    get_rope_index. We verify by stubbing super().get_rope_index to return a
    sentinel and confirming our override returns it unchanged."""
    obj = _mock_timeomni_v_for_rope()
    cfg = obj.config

    sentinel = (torch.zeros(3, 1, 4, dtype=torch.long), torch.zeros(1, 1))

    # Monkey-patch the bound super().get_rope_index lookup
    # We can't easily mock super(), so instead set ts_token_id to a value that
    # doesn't appear in input_ids and verify our code reaches the early-return.
    # Use a different sentinel approach: monkeypatch via class.
    called = {"n": 0}
    def fake_super_call(input_ids, *args, **kwargs):
        called["n"] += 1
        return sentinel

    # Replace the parent method on the class temporarily.
    parent = TimeOmniVForConditionalGeneration.__mro__[1]
    original = parent.get_rope_index
    parent.get_rope_index = fake_super_call
    try:
        # Prefix-only input (no TS, no V)
        ids = torch.tensor([[50, 50, 50, 50]], dtype=torch.long)
        attn = torch.ones_like(ids)
        obj._ts_n_channels = None  # explicitly cleared
        out = obj.get_rope_index(input_ids=ids, attention_mask=attn)
    finally:
        parent.get_rope_index = original

    assert called["n"] == 1
    assert out is sentinel
