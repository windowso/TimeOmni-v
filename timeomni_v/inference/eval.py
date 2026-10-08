"""Score a predictions.jsonl produced by `timeomni_v.inference.infer`.

Branches on the ``task`` field carried by the source jsonl (now part of the
unified schema):

* ``task == "classification"`` (or missing — legacy default): accuracy,
  per-class precision/recall/F1, macro-F1, weighted-F1, UAR, confusion
  matrix. Unparseable predictions land in a dedicated "None" column.

* ``task == "prediction"``: regression scoring — MAE / MSE / MAPE / PCC
  over parseable rows, plus per-channel breakdowns. The ``answer`` and
  model output are run through the ``<forecast>(label: v1, v2, ...)``
  parser; rows that fail to parse are reported as part of the
  ``parse_rate`` and excluded from metric averages (matches the infer
  ``failed`` convention).

The ``task`` is read from the source test jsonl (``--test_jsonl``) or
falls back to the predictions file (``--predictions``) where each row
also carries ``task`` if it was present at infer time.

Usage:
    python -m timeomni_v.inference.eval predictions.jsonl
    python -m timeomni_v.inference.eval predictions.jsonl --labels A,B,C
    python -m timeomni_v.inference.eval predictions.jsonl --test_jsonl test.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from timeomni_v.inference.forecast_parse import align_forecast, parse_forecast
from timeomni_v.inference.parse import build_parser
from timeomni_v.inference.regression_metrics import (
    mae as _mae,
    mape as _mape,
    mse as _mse,
    per_channel as _per_channel,
    row_pcc as _row_pcc,
)


def load_records(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def write_records(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def reparse_records(records: list[dict], labels: list[str]) -> int:
    """Re-run the auto-tuned answer parser over each row's ``raw`` and update
    ``prediction`` in place. ``labels`` defines the valid answer alphabet —
    the parser only emits tokens from this set, with longest-first
    alternation so ``"10"`` wins over ``"1"`` etc. Returns the number of
    rows whose prediction changed."""
    parser = build_parser(labels)
    n_changed = 0
    for r in records:
        if "raw" not in r:
            continue
        new_pred = parser(r.get("raw"))
        if new_pred != r.get("prediction"):
            r["prediction"] = new_pred
            n_changed += 1
    return n_changed


def load_predictions(path: Path) -> list[tuple[str, str | None]]:
    return [(r["ground_truth"], r["prediction"]) for r in load_records(path)]


# ---------------------------------------------------------------------------
# Classification metrics (existing behavior, untouched).
# ---------------------------------------------------------------------------


def compute_metrics(rows: list[tuple[str, str | None]], labels: list[str]) -> dict:
    pred_cols = [*labels, None]
    cm: dict[str, dict[str | None, int]] = {g: {p: 0 for p in pred_cols} for g in labels}

    total = len(rows)
    correct = 0
    unparseable = 0
    unknown_gt = 0
    for gt, pred in rows:
        if gt not in cm:
            unknown_gt += 1
            continue
        p = pred if pred in labels else None
        cm[gt][p] += 1
        if pred is None:
            unparseable += 1
        if pred == gt:
            correct += 1

    per_class: dict[str, dict[str, float]] = {}
    for c in labels:
        tp = cm[c][c]
        fn = sum(cm[c][p] for p in pred_cols if p != c)
        fp = sum(cm[g][c] for g in labels if g != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    num_classes = len(labels)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / max(1, num_classes)
    uar = sum(m["recall"] for m in per_class.values()) / max(1, num_classes)
    total_support = sum(m["support"] for m in per_class.values())
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
        if total_support else 0.0
    )

    return {
        "task": "classification",
        "total": total,
        "correct": correct,
        "unparseable": unparseable,
        "unknown_gt": unknown_gt,
        "accuracy": correct / max(1, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "uar": uar,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def format_classification_report(m: dict, labels: list[str]) -> str:
    lines: list[str] = []
    lines.append(
        f"n={m['total']}  correct={m['correct']}  "
        f"unparseable={m['unparseable']}  unknown_gt={m['unknown_gt']}"
    )
    lines.append(f"Accuracy    : {m['accuracy']:.4f}")
    lines.append(f"Macro F1    : {m['macro_f1']:.4f}")
    lines.append(f"Weighted F1 : {m['weighted_f1']:.4f}")
    lines.append(f"UAR         : {m['uar']:.4f}  (= macro-averaged recall)")
    lines.append("")
    lines.append(
        f"{'class':>8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"
    )
    for c in labels:
        pc = m["per_class"][c]
        lines.append(
            f"{c:>8} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
            f"{pc['f1']:>10.4f} {pc['support']:>10d}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=gt, cols=pred; trailing col = unparseable):")
    col_headers = [*labels, "None"]
    lines.append(f"{'gt/pred':>10}" + "".join(f"{c:>7}" for c in col_headers))
    cm = m["confusion_matrix"]
    for g in labels:
        lines.append(
            f"{g:>10}" + "".join(f"{cm[g][p]:>7}" for p in [*labels, None])
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression metrics (forecasting / prediction tasks).
# ---------------------------------------------------------------------------


def compute_regression_metrics(records: list[dict]) -> tuple[dict, list[dict]]:
    """Score a prediction-task predictions.jsonl. Returns ``(metrics, debug)``.

    ``metrics`` mirrors the classification metrics shape:
      * ``task = "prediction"``
      * ``total``, ``succeeded``, ``failed``, ``parse_rate``
      * ``mae``, ``mse``, ``mape``, ``pcc`` — **mean over parseable rows**
        of each per-row metric. Per-row MAE/MSE/MAPE are element-wise on
        that row's (T, C) matrix; per-row PCC is ``row_pcc`` (per-channel
        Pearson averaged over channels). Aggregating to row level first
        prevents long / wide rows from dominating the dataset score.
      * ``mape_skipped`` — total elements skipped across rows (|gt| < eps).
      * ``per_channel`` arrays for MAE/MSE/MAPE — for each channel ``c``,
        the mean over rows that had column ``c`` of that row's per-channel
        value.
      * ``n_rows`` — number of parseable rows that contributed.
      * ``length_mismatch_count`` and ``alignment_count`` to surface how
        often the parser had to fall back to positional alignment.

    ``debug`` is the per-row breakdown (one dict per record), useful for
    `--debug-jsonl` to inspect outliers.
    """
    total = len(records)
    row_maes: list[float] = []
    row_mses: list[float] = []
    row_mapes: list[float] = []
    row_pccs: list[float] = []
    mape_skipped_total = 0
    # Per-channel accumulators: channel index -> list of finite per-row values.
    ch_maes: dict[int, list[float]] = {}
    ch_mses: dict[int, list[float]] = {}
    ch_mapes: dict[int, list[float]] = {}
    debug: list[dict] = []
    parsed_ok = 0
    parsed_failed = 0
    length_mismatch = 0
    alignment_counter: Counter = Counter()
    max_width = 0

    for r in records:
        gt_text = r.get("ground_truth")
        pred_text = r.get("raw") if r.get("prediction") is None else r.get("prediction")
        if pred_text is None:
            pred_text = r.get("raw")
        # Use raw text for parsing — the classification parser may have
        # stuffed `prediction` with a single-token guess that won't parse
        # as a forecast.
        if r.get("raw") is not None:
            pred_text = r["raw"]

        gt_parsed = parse_forecast(gt_text)
        pred_parsed = parse_forecast(pred_text)

        row_dbg: dict = {
            "id": r.get("id"),
            "parsed_gt": gt_parsed is not None,
            "parsed_pred": pred_parsed is not None,
            "failed": bool(r.get("failed")),
        }

        if r.get("failed") or gt_parsed is None or pred_parsed is None:
            parsed_failed += 1
            debug.append(row_dbg)
            continue

        gt_arr, pred_arr, info = align_forecast(gt_parsed, pred_parsed)
        if gt_arr.size == 0:
            parsed_failed += 1
            row_dbg["align_info"] = info
            debug.append(row_dbg)
            continue

        parsed_ok += 1
        if info.get("length_mismatch"):
            length_mismatch += 1
        alignment_counter[info.get("alignment", "?")] += 1
        max_width = max(max_width, gt_arr.shape[1])

        # Per-row metrics: element-wise within this row's (T, C) matrix
        # (PCC: per-channel Pearson averaged over channels).
        row_mae = _mae(gt_arr, pred_arr)
        row_mse = _mse(gt_arr, pred_arr)
        row_mape, n_skip = _mape(gt_arr, pred_arr)
        row_pcc_val = _row_pcc(gt_arr, pred_arr)
        ch_stats = _per_channel(gt_arr, pred_arr)

        row_dbg.update({
            "mae": row_mae,
            "mse": row_mse,
            "mape": row_mape,
            "mape_skipped": n_skip,
            "pcc": row_pcc_val,
            "len_gt": info["len_gt"],
            "len_pred": info["len_pred"],
            "alignment": info["alignment"],
            "length_mismatch": info["length_mismatch"],
            "missing_labels": info["missing_labels"],
            "channels": gt_arr.shape[1],
        })
        debug.append(row_dbg)

        if np.isfinite(row_mae):
            row_maes.append(row_mae)
        if np.isfinite(row_mse):
            row_mses.append(row_mse)
        if np.isfinite(row_mape):
            row_mapes.append(row_mape)
        if np.isfinite(row_pcc_val):
            row_pccs.append(row_pcc_val)
        mape_skipped_total += n_skip

        for c in range(gt_arr.shape[1]):
            if np.isfinite(ch_stats.mae[c]):
                ch_maes.setdefault(c, []).append(ch_stats.mae[c])
            if np.isfinite(ch_stats.mse[c]):
                ch_mses.setdefault(c, []).append(ch_stats.mse[c])
            if np.isfinite(ch_stats.mape[c]):
                ch_mapes.setdefault(c, []).append(ch_stats.mape[c])

    def _mean_or_nan(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    if parsed_ok:
        per_channel = {
            "mae": [_mean_or_nan(ch_maes.get(c, [])) for c in range(max_width)],
            "mse": [_mean_or_nan(ch_mses.get(c, [])) for c in range(max_width)],
            "mape": [_mean_or_nan(ch_mapes.get(c, [])) for c in range(max_width)],
        }
        metrics_overall = {
            "mae": _mean_or_nan(row_maes),
            "mse": _mean_or_nan(row_mses),
            "mape": _mean_or_nan(row_mapes),
            "mape_skipped": mape_skipped_total,
            "pcc": _mean_or_nan(row_pccs),
            "per_channel": per_channel,
            "channels": int(max_width),
            "n_rows": parsed_ok,
        }
    else:
        metrics_overall = {
            "mae": float("nan"),
            "mse": float("nan"),
            "mape": float("nan"),
            "mape_skipped": 0,
            "pcc": float("nan"),
            "per_channel": {"mae": [], "mse": [], "mape": []},
            "channels": 0,
            "n_rows": 0,
        }

    metrics = {
        "task": "prediction",
        "total": total,
        "succeeded": parsed_ok,
        "failed": parsed_failed,
        "parse_rate": parsed_ok / max(1, total),
        "length_mismatch_count": length_mismatch,
        "alignment_counts": dict(alignment_counter),
        **metrics_overall,
    }
    return metrics, debug


def format_regression_report(m: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"n={m['total']}  succeeded={m['succeeded']}  failed={m['failed']}  "
        f"parse_rate={m['parse_rate']:.4f}"
    )
    lines.append(
        f"length_mismatch={m['length_mismatch_count']}  "
        f"alignment={m['alignment_counts']}  "
        f"channels={m['channels']}  n_rows={m['n_rows']}"
    )
    lines.append("")
    lines.append(f"MAE  : {m['mae']:.6f}     [mean over rows]")
    lines.append(f"MSE  : {m['mse']:.6f}     [mean over rows]")
    lines.append(
        f"MAPE : {m['mape']:.4f}%   [mean over rows]  "
        f"(skipped {m['mape_skipped']} |gt|<eps cells across rows)"
    )
    lines.append(f"PCC  : {m['pcc']:.6f}     [mean over rows of per-channel Pearson]")
    if m["channels"] > 0:
        lines.append("")
        lines.append("Per-channel:")
        lines.append(f"{'ch':>4} {'MAE':>12} {'MSE':>12} {'MAPE(%)':>12}")
        pc = m["per_channel"]
        for c in range(m["channels"]):
            lines.append(
                f"{c:>4} {pc['mae'][c]:>12.6f} {pc['mse'][c]:>12.6f} "
                f"{pc['mape'][c]:>12.4f}"
            )
    return "\n".join(lines)


def print_classification_report(m: dict, labels: list[str]) -> None:
    print(format_classification_report(m, labels))


def print_regression_report(m: dict) -> None:
    print(format_regression_report(m))


# ---------------------------------------------------------------------------
# Persistence (shared).
# ---------------------------------------------------------------------------


def save_metrics(
    m: dict, labels: list[str] | None, out_path: Path,
    infer_metrics: dict | None = None,
) -> None:
    """Persist metrics as JSON. Confusion-matrix's None pred-key is stringified
    so the file is parseable by any json reader. ``infer_metrics`` carries the
    success/failure counts written by ``infer.py``; merge them in."""
    payload: dict = {**m}
    if labels is not None:
        payload["labels"] = labels
    if "confusion_matrix" in payload:
        payload["confusion_matrix"] = {
            g: {("None" if p is None else p): v for p, v in row.items()}
            for g, row in payload["confusion_matrix"].items()
        }
    if infer_metrics is not None:
        payload["infer"] = infer_metrics
        for k in ("succeeded", "failed", "success_rate"):
            if k in infer_metrics and k not in payload:
                payload[k] = infer_metrics[k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    """Make numpy scalars / arrays JSON-serializable."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def load_infer_metrics(pred_jsonl: Path) -> dict | None:
    """Pick up the success-rate / failure-count metrics written by infer.py.
    Returns None if the file isn't there (e.g. predictions came from an older
    infer run before that path existed)."""
    candidate = pred_jsonl.with_suffix(pred_jsonl.suffix + ".metrics.json")
    if not candidate.exists():
        return None
    try:
        with candidate.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[eval] could not read {candidate}: {e}")
        return None


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------


def _detect_task(records: list[dict], test_jsonl: Path | None) -> str:
    """Look up ``task`` from the source test jsonl, or from the predictions
    rows themselves (some infer paths copy ``task`` through). Defaults to
    ``"classification"`` for backwards compatibility with old predictions."""
    if test_jsonl is not None and test_jsonl.exists():
        with test_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "task" in row:
                    return row["task"]
                break
    for r in records:
        if "task" in r and r["task"]:
            return r["task"]
    return "classification"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_jsonl", type=Path)
    ap.add_argument(
        "--test_jsonl", type=Path, default=None,
        help="Source test jsonl (used to detect task). If absent, falls back to "
             "the `task` field on the predictions rows themselves.",
    )
    ap.add_argument(
        "--task", default=None, choices=["classification", "prediction"],
        help="Force a task type, bypassing auto-detection.",
    )
    ap.add_argument(
        "--labels", default=None,
        help="Comma-separated class labels (classification only).",
    )
    ap.add_argument(
        "--out_metrics", type=Path, default=None,
        help="Where to write metrics.json. Defaults to "
             "<pred_jsonl_dir>/metrics.json.",
    )
    ap.add_argument(
        "--reparse", action="store_true",
        help="Classification only: re-run the answer parser on each row's "
             "`raw` and overwrite predictions.jsonl in place before scoring.",
    )
    ap.add_argument(
        "--debug_jsonl", type=Path, default=None,
        help="Prediction task only: write per-row breakdown (parse status, "
             "per-row metrics) to this path. Defaults to "
             "<pred_jsonl>.regression.jsonl.",
    )
    args = ap.parse_args()

    records = load_records(args.pred_jsonl)
    task = args.task or _detect_task(records, args.test_jsonl)
    print(f"[eval] task={task}")

    if task == "prediction":
        metrics, debug = compute_regression_metrics(records)
        print(f"File: {args.pred_jsonl}")
        print("-" * 72)
        print_regression_report(metrics)
        infer_metrics = load_infer_metrics(args.pred_jsonl)
        out_metrics = args.out_metrics or args.pred_jsonl.parent / "metrics.json"
        save_metrics(metrics, labels=None, out_path=out_metrics,
                     infer_metrics=infer_metrics)
        debug_path = (
            args.debug_jsonl
            or args.pred_jsonl.with_suffix(args.pred_jsonl.suffix + ".regression.jsonl")
        )
        with debug_path.open("w") as f:
            for d in debug:
                f.write(json.dumps(d, ensure_ascii=False, default=_json_default) + "\n")
        print(f"\n[eval] metrics  -> {out_metrics}")
        print(f"[eval] per-row  -> {debug_path}")
        return

    # Classification path (preserves existing behavior).
    if args.labels is None:
        labels = sorted({
            r["ground_truth"] for r in records
            if not r.get("failed") and r.get("ground_truth") is not None
        })
        print(f"[eval] auto-detected labels: {labels}")
    else:
        labels = args.labels.split(",")

    if args.reparse:
        live = [r for r in records if not r.get("failed") and "raw" in r]
        n_changed = reparse_records(live, labels)
        write_records(args.pred_jsonl, records)
        print(
            f"[eval] re-parsed {len(live)} rows; {n_changed} predictions "
            f"changed; wrote {args.pred_jsonl}"
        )
    rows = [
        (r["ground_truth"], r["prediction"])
        for r in records if not r.get("failed")
    ]

    metrics = compute_metrics(rows, labels)
    print(f"File: {args.pred_jsonl}")
    print("-" * 72)
    print_classification_report(metrics, labels)

    infer_metrics = load_infer_metrics(args.pred_jsonl)
    if infer_metrics is not None:
        succeeded = infer_metrics.get("succeeded")
        failed = infer_metrics.get("failed")
        success_rate = infer_metrics.get("success_rate")
        print()
        print(
            f"Infer success rate: {success_rate:.4f}"
            f"  (succeeded={succeeded}, failed={failed})"
            if success_rate is not None
            else f"Infer stats: succeeded={succeeded} failed={failed}"
        )
    else:
        print(
            "\n[eval] no infer-time metrics file found "
            f"({args.pred_jsonl}.metrics.json); "
            "success_rate / failed counts will be omitted."
        )

    out_metrics = args.out_metrics or args.pred_jsonl.parent / "metrics.json"
    save_metrics(metrics, labels, out_metrics, infer_metrics=infer_metrics)
    print(f"\n[eval] metrics written to {out_metrics}")


if __name__ == "__main__":
    main()
