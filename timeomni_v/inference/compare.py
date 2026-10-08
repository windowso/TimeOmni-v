"""Aggregate predictions from multiple runs and compare accuracy.

Usage:
    python -m timeomni_v.inference.compare \
        --predictions \
            interleave:runs/timeomni_v-interleave/predictions.jsonl \
            block_adjacent:runs/timeomni_v-block/predictions.jsonl \
            baseline:runs/baseline/predictions.jsonl \
        --out-csv runs/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def per_option_accuracy(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Return {option: (correct, total)} keyed by the ground_truth label."""
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in rows:
        gt = r["ground_truth"]
        pred = r["prediction"]
        per[gt][1] += 1
        if pred == gt:
            per[gt][0] += 1
    return {k: (v[0], v[1]) for k, v in per.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="list of name:path entries",
    )
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    tables = {}
    for spec in args.predictions:
        name, path = spec.split(":", 1)
        tables[name] = load_predictions(Path(path))

    # Overall + per-option accuracy
    summary_rows = []
    for name, rows in tables.items():
        n = len(rows)
        correct = sum(1 for r in rows if r["prediction"] == r["ground_truth"])
        per_opt = per_option_accuracy(rows)
        pred_dist = Counter(r["prediction"] for r in rows)

        summary_rows.append(
            {
                "name": name,
                "n": n,
                "overall_acc": correct / max(1, n),
                "overall_correct": correct,
                **{f"acc_{k}": v[0] / max(1, v[1]) for k, v in per_opt.items()},
                **{f"total_{k}": v[1] for k, v in per_opt.items()},
                **{f"pred_{k}": pred_dist.get(k, 0) for k in ["A", "B", "C", None]},
            }
        )

    # Print human-readable table
    header = ["name", "n", "overall_acc"]
    for k in ["A", "B", "C"]:
        header += [f"acc_{k}", f"total_{k}"]
    header += ["pred_A", "pred_B", "pred_C", "pred_None"]

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    col_width = max(len(h) for h in header) + 2
    print("  ".join(h.ljust(col_width) for h in header))
    for row in summary_rows:
        vals = [
            row.get("name"),
            row.get("n"),
            row.get("overall_acc"),
        ]
        for k in ["A", "B", "C"]:
            vals.extend([row.get(f"acc_{k}"), row.get(f"total_{k}")])
        vals += [
            row.get("pred_A"),
            row.get("pred_B"),
            row.get("pred_C"),
            row.get(f"pred_None"),
        ]
        print("  ".join(_fmt(v).ljust(col_width) for v in vals))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k) for k in header})
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
