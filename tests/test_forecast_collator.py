"""CPU-only tests for ForecastCollator helpers.

The full ``__call__`` path needs the Qwen2.5-Omni processor + actual image /
video files on disk; that's exercised by the smoke train. Here we cover the
target-loading + padding contract (the only fundamentally new logic in the
forecast collator).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from timeomni_v.data.forecast_collator import _load_target_csv


def test_load_target_csv_single_column(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"v0": [1.0, 2.5, 3.0]}).to_csv(p, index=False)
    arr = _load_target_csv(str(p))
    assert arr.shape == (3,)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [1.0, 2.5, 3.0])


def test_load_target_csv_drops_timestamp(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"timestamp": ["a", "b"], "v0": [4.0, 5.0]}).to_csv(p, index=False)
    arr = _load_target_csv(str(p))
    assert arr.shape == (2,)
    np.testing.assert_allclose(arr, [4.0, 5.0])


def test_load_target_csv_rejects_multichannel(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="single-channel"):
        _load_target_csv(str(p))


def test_padding_logic():
    # Replicate the (B, pred_len) build the collator does at the end.
    pred_len = 6
    bs = 3
    target_arrs = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([7.0, 8.0], dtype=np.float32),
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32),
    ]
    target = torch.full((bs, pred_len), float("nan"), dtype=torch.float32)
    mask = torch.zeros((bs, pred_len), dtype=torch.bool)
    for i, a in enumerate(target_arrs):
        target[i, : a.shape[0]] = torch.from_numpy(a)
        mask[i, : a.shape[0]] = True
    assert mask[0].tolist() == [True, True, True, False, False, False]
    assert mask[2].tolist() == [True] * 6
    assert torch.isnan(target[0, 3:]).all()
    assert torch.isnan(target[1, 2:]).all()
    assert not torch.isnan(target[2]).any()
