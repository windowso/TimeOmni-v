"""Regression / forecasting metrics: MAE, MSE, MAPE, PCC.

All metrics operate on a pair of 2D arrays ``gt[T, C]``, ``pred[T, C]``
that came out of ``timeomni_v.inference.forecast_parse.align_forecast`` (so
shapes already match — NaNs in one or the other are treated as missing
and dropped pairwise).

* ``mae``, ``mse`` — straightforward, NaN-aware mean over flat values.
* ``mape`` — returned as a **percentage** (mean of |gt-pred|/|gt| × 100).
  Skips elements where ``|gt| < eps`` to avoid div-by-zero; reports the
  skipped count alongside the metric.
* ``pcc`` — Pearson correlation over a single flat 1-D pair (NaN-aware).
* ``row_pcc`` — canonical per-sample PCC for a forecast: takes (T, C)
  arrays, computes per-channel Pearson, returns the mean over finite
  channels. Single-channel collapses to plain Pearson on that column.
  This is the only PCC aggregator we use; pooling rows or channels
  before correlation produces between-group artefacts (Simpson's
  paradox) and is intentionally not supported.
* Per-channel MAE/MSE/MAPE reduce along the time axis so the caller can
  break down errors by output dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_DEFAULT_EPS = 1e-8


def _valid_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Boolean mask of positions where BOTH arrays are finite."""
    return np.isfinite(gt) & np.isfinite(pred)


def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    m = _valid_mask(gt, pred)
    if not m.any():
        return float("nan")
    return float(np.abs(gt[m] - pred[m]).mean())


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    m = _valid_mask(gt, pred)
    if not m.any():
        return float("nan")
    diff = gt[m] - pred[m]
    return float((diff * diff).mean())


def mape(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    eps: float = _DEFAULT_EPS,
) -> tuple[float, int]:
    """Return ``(mape_percent, n_skipped)``. MAPE is a **percentage**
    (``mean(|gt-pred|/|gt|) × 100``). Skips positions where ``|gt| < eps``
    to avoid div-by-zero — a separate counter so callers can audit how
    much of the population was dropped."""
    m = _valid_mask(gt, pred)
    if not m.any():
        return float("nan"), 0
    g = gt[m]
    p = pred[m]
    nz = np.abs(g) >= eps
    n_skip = int((~nz).sum())
    if not nz.any():
        return float("nan"), n_skip
    return float((np.abs(g[nz] - p[nz]) / np.abs(g[nz])).mean() * 100.0), n_skip


def pcc(gt: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation coefficient over flattened pairs. Returns NaN if
    either side has zero variance after masking."""
    m = _valid_mask(gt, pred)
    if m.sum() < 2:
        return float("nan")
    g = gt[m].astype(np.float64)
    p = pred[m].astype(np.float64)
    g_std = g.std()
    p_std = p.std()
    if g_std == 0 or p_std == 0:
        return float("nan")
    # Manual implementation avoids the scipy import + warning when std=0.
    g0 = g - g.mean()
    p0 = p - p.mean()
    return float((g0 * p0).sum() / (np.sqrt((g0 * g0).sum()) * np.sqrt((p0 * p0).sum())))


def row_pcc(gt: np.ndarray, pred: np.ndarray) -> float:
    """Per-row Pearson correlation: compute Pearson on each channel
    (column) and return the mean over finite per-channel values. Returns
    NaN if every channel is degenerate (constant / all-NaN).

    For single-channel inputs this collapses to ``pcc(gt[:, 0], pred[:, 0])``
    — exactly the natural definition. For multi-channel inputs it is
    robust to inter-channel scale differences (terra ch2 ~1e5 vs ch4
    ~1e-3) because each channel's correlation is computed in its own
    scale before averaging.
    """
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape != pred.shape:
        raise ValueError(
            f"row_pcc needs matching 2-D arrays; got {gt.shape} vs {pred.shape}"
        )
    per_ch: list[float] = []
    for c in range(gt.shape[1]):
        v = pcc(gt[:, c], pred[:, c])
        if np.isfinite(v):
            per_ch.append(v)
    if not per_ch:
        return float("nan")
    return float(np.mean(per_ch))


@dataclass
class ChannelStats:
    mae: list[float]
    mse: list[float]
    mape: list[float]


def per_channel(gt: np.ndarray, pred: np.ndarray) -> ChannelStats:
    """Compute MAE/MSE/MAPE per channel (column). NaNs handled per
    column; an all-NaN column yields NaN for every metric.

    Per-channel PCC is intentionally **not** computed here — see the
    module docstring. Use ``row_pcc`` for per-sample PCC and average
    those values across rows for the dataset-level number.
    """
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape != pred.shape:
        raise ValueError(
            f"per_channel needs matching 2-D arrays; got {gt.shape} vs {pred.shape}"
        )
    n_cols = gt.shape[1]
    out = ChannelStats(mae=[], mse=[], mape=[])
    for c in range(n_cols):
        g = gt[:, c]
        p = pred[:, c]
        out.mae.append(mae(g, p))
        out.mse.append(mse(g, p))
        m, _ = mape(g, p)
        out.mape.append(m)
    return out


__all__ = ["mae", "mse", "mape", "pcc", "row_pcc", "per_channel", "ChannelStats"]
