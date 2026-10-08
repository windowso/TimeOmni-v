"""Parser for forecast-task answers.

Datasets in the prediction family (terra, sp500, pixelrec, …) wrap the
target sequence between ``<forecast>...</forecast>`` tags and emit one line
per timestep::

    <forecast>
    (2023-12-12: 421.6, 0.7, 100789, 69.5, 0.01646, 28.07, 4.77)
    (2023-12-13: 404.0, 0.0, 100847, 67.5, 0.01589, 28.00, 5.28)
    ...
    </forecast>

This module:
  * extracts the body between the tags (closing tag optional — generated
    outputs frequently truncate);
  * parses each ``(label: v1, v2, …)`` line into ``(label_str, ndarray)``;
  * aligns a ground-truth and a prediction sequence by label (date) when
    available, falling back to positional alignment.

Kept torch-free so ``timeomni_v.inference.eval`` can import it without dragging
in the full training stack.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


_FORECAST_TAG_RE = re.compile(r"<forecast>(.*?)(?:</forecast>|$)", re.DOTALL | re.IGNORECASE)
# Match a single `(...)` group; we then split label vs values inside.
_PAREN_RE = re.compile(r"\(([^()\n]+)\)")
# Numeric date prefix the model sometimes leaks into the value portion
# (e.g. `(timestamp: 2023-12-12, 254, 0.2, ...)`). Stripping it before
# running _NUM_RE prevents the date from being shredded into three
# spurious numbers (2023, -12, -12) that shift every channel left.
_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")


ParsedForecast = list[tuple[str, np.ndarray]]


def parse_forecast(text: str | None) -> Optional[ParsedForecast]:
    """Return the list of ``(label, values)`` tuples, or None when nothing
    parseable was found.

    * Strips ``<forecast>...</forecast>`` if present (matches the body up to
      the first closing tag OR the end of input — generated text often loses
      the closing tag).
    * Skips lines that don't match the ``(label: numbers)`` shape.
    * Each parsed value array is float64 (downstream arithmetic).
    """
    if not text:
        return None
    body = text
    m = _FORECAST_TAG_RE.search(text)
    if m is not None:
        body = m.group(1)

    rows: ParsedForecast = []
    for paren in _PAREN_RE.finditer(body):
        inner = paren.group(1).strip()
        # Pick the label / values separator. We prefer the LAST `:` because
        # the canonical schema is `(<label>: <nums>)` but models sometimes
        # prepend metadata, e.g. `(timestamp: 2023-12-12: 254, ...)` —
        # taking the rightmost colon keeps the entire `timestamp: date`
        # blob in the label and the numeric suffix in the value portion.
        # If no colon at all, fall back to the FIRST comma (covers the
        # comma-separated `(date, nums)` schema some models emit).
        if ":" in inner:
            sep_idx = inner.rfind(":")
        elif "," in inner:
            sep_idx = inner.find(",")
        else:
            continue
        label = inner[:sep_idx].strip()
        nums_str = inner[sep_idx + 1:]
        # Defensive: strip any date that leaked into the value portion so
        # _NUM_RE doesn't shred `2023-12-12` into `2023, -12, -12`.
        nums_str = _DATE_RE.sub("", nums_str)
        nums = [float(x) for x in _NUM_RE.findall(nums_str)]
        if not nums:
            continue
        rows.append((label, np.asarray(nums, dtype=np.float64)))
    if not rows:
        return None
    return rows


def align_forecast(
    gt: ParsedForecast,
    pred: ParsedForecast,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align a parsed prediction against a parsed ground-truth.

    Strategy:
      1. If every gt label appears in pred (label-based alignment is
         reliable), build per-label maps and stack values in gt order.
      2. Otherwise positional: zip gt with pred, truncating to the shorter.

    Per-row width is taken from the gt row; if a pred row is shorter we pad
    with NaN (so it falls out of MAE/MSE), if it's longer we trim. Returns
    ``(gt_arr, pred_arr, info)`` where info carries diagnostic flags.
    """
    info: dict = {
        "len_gt": len(gt),
        "len_pred": len(pred),
        "alignment": "positional",
        "length_mismatch": len(gt) != len(pred),
        "missing_labels": 0,
    }
    if not gt or not pred:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, info

    width = max(v.shape[0] for _, v in gt)

    pred_by_label = {label: arr for label, arr in pred}
    have_all_labels = all(label in pred_by_label for label, _ in gt)
    info["missing_labels"] = sum(1 for label, _ in gt if label not in pred_by_label)

    rows_gt: list[np.ndarray] = []
    rows_pred: list[np.ndarray] = []
    if have_all_labels:
        info["alignment"] = "by_label"
        for label, gv in gt:
            pv = pred_by_label[label]
            rows_gt.append(_to_width(gv, width))
            rows_pred.append(_to_width(pv, width))
    else:
        n = min(len(gt), len(pred))
        for i in range(n):
            gv = gt[i][1]
            pv = pred[i][1]
            rows_gt.append(_to_width(gv, width))
            rows_pred.append(_to_width(pv, width))

    return np.stack(rows_gt, axis=0), np.stack(rows_pred, axis=0), info


def _to_width(arr: np.ndarray, width: int) -> np.ndarray:
    """Pad ``arr`` with NaN to ``width`` columns or trim to it."""
    if arr.shape[0] == width:
        return arr.astype(np.float64, copy=False)
    if arr.shape[0] > width:
        return arr[:width].astype(np.float64, copy=False)
    out = np.full((width,), np.nan, dtype=np.float64)
    out[: arr.shape[0]] = arr
    return out


def extract_forecast_block(text: str | None) -> Optional[str]:
    """Return the ``<forecast>...</forecast>`` block (re-wrapped) found in
    ``text``, or None if no opening tag is present. Used by infer.py so the
    `prediction` field carries just the forecast payload instead of the
    full raw generation (which often includes thinking/preamble).
    """
    if not text:
        return None
    m = _FORECAST_TAG_RE.search(text)
    if m is None:
        return None
    return f"<forecast>{m.group(1).rstrip()}\n</forecast>"


__all__ = ["parse_forecast", "align_forecast", "extract_forecast_block", "ParsedForecast"]
