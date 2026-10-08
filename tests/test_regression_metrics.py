"""Tests for timeomni_v.inference.regression_metrics."""

import math

import numpy as np

from timeomni_v.inference.regression_metrics import (
    mae, mape, mse, pcc, per_channel, row_pcc,
)


def test_mae_known_inputs():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.array([[1.5, 2.5], [3.5, 4.5]])
    # |0.5| × 4 / 4 = 0.5
    assert math.isclose(mae(gt, pred), 0.5)


def test_mse_known_inputs():
    gt = np.array([[0.0, 1.0], [2.0, 3.0]])
    pred = np.array([[1.0, 1.0], [2.0, 1.0]])
    # diffs: 1, 0, 0, 2 → squared: 1, 0, 0, 4 → mean = 1.25
    assert math.isclose(mse(gt, pred), 1.25)


def test_mape_skips_zero_ground_truth():
    gt = np.array([1.0, 0.0, 2.0])
    # MAPE on the two non-zero rows: (0.5 + 0) / 2 = 0.25 → ×100 = 25%
    pred = np.array([1.5, 1.0, 2.0])
    val, n_skip = mape(gt, pred)
    assert math.isclose(val, 25.0)
    assert n_skip == 1


def test_mape_returns_nan_when_all_skipped():
    gt = np.array([0.0, 0.0])
    pred = np.array([1.0, 2.0])
    val, n_skip = mape(gt, pred)
    assert math.isnan(val)
    assert n_skip == 2


def test_pcc_perfect_correlation_is_one():
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 4.0, 6.0, 8.0])  # perfect linear → 1.0
    assert math.isclose(pcc(gt, pred), 1.0)


def test_pcc_zero_variance_is_nan():
    gt = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert math.isnan(pcc(gt, pred))


def test_metrics_handle_nan_pairwise():
    gt = np.array([1.0, np.nan, 3.0])
    pred = np.array([1.0, 5.0, 3.0])
    # Only 2 finite pairs survive; both differences = 0.
    assert math.isclose(mae(gt, pred), 0.0)
    assert math.isclose(mse(gt, pred), 0.0)


def test_per_channel_reduces_along_time_axis():
    gt = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    pred = np.array([[1.5, 10.0], [2.5, 20.0], [3.5, 30.0]])
    ch = per_channel(gt, pred)
    assert math.isclose(ch.mae[0], 0.5)
    assert math.isclose(ch.mae[1], 0.0)


def test_row_pcc_single_channel_matches_pcc():
    # row_pcc on a (T, 1) array must collapse to plain pcc on that column.
    gt = np.array([[1.0], [2.0], [3.0], [4.0]])
    pred = np.array([[2.0], [4.0], [6.0], [8.0]])
    assert math.isclose(row_pcc(gt, pred), pcc(gt[:, 0], pred[:, 0]))


def test_row_pcc_averages_finite_channels():
    # ch0: perfect positive (pcc=1). ch1: perfect negative (pcc=-1).
    # ch2: constant column → degenerate, dropped before averaging.
    # Expected row pcc = mean([1, -1]) = 0.
    gt = np.array([[1.0, 1.0, 5.0],
                   [2.0, 2.0, 5.0],
                   [3.0, 3.0, 5.0],
                   [4.0, 4.0, 5.0]])
    pred = np.array([[2.0, 4.0, 0.0],
                     [4.0, 3.0, 1.0],
                     [6.0, 2.0, 2.0],
                     [8.0, 1.0, 3.0]])
    assert math.isclose(row_pcc(gt, pred), 0.0, abs_tol=1e-12)


def test_row_pcc_all_degenerate_is_nan():
    gt = np.array([[1.0, 1.0], [1.0, 1.0]])
    pred = np.array([[2.0, 3.0], [4.0, 5.0]])
    assert math.isnan(row_pcc(gt, pred))
