"""Summarize a multi-dataset eval run tree (zero-shot or trained) to a CSV.

Walks any tree shaped like::

    <run_root>/<ablation>/<dataset_stem>/metrics.json

— this matches both ``scripts/eval.sh`` (trained TimeOmni-v runs land under
``runs/<RUN_NAME>/<ablation>/<dataset>/``) and ``scripts/eval_zero_shot.sh``
(zero-shot runs land under ``runs/<RUN_DIR_NAME>/<backend>/<ablation>/<
dataset>/``; point this script at ``runs/<RUN_DIR_NAME>/<backend>/`` for one
backend, or run it once per backend).

``metrics.json`` is the file ``timeomni_v/inference/eval.py`` writes — it
branches on ``task``:

* ``classification`` rows carry ``accuracy``, ``macro_f1``, ``weighted_f1``,
  ``uar`` (plus the merged-in ``success_rate`` from infer.py's sidecar).
* ``prediction`` rows carry ``mae``, ``mse``, ``mape`` (percent), ``pcc``
  (plus ``parse_rate`` and the same ``success_rate``).

By default we pick the column set per-dataset based on the metrics.json's
``task`` field, so a mixed run (some classification + some prediction
datasets in one tree) produces one CSV with the right metric for each.
``--metrics`` overrides this with a uniform list applied to every dataset.

Usage::

    python -m scripts.summarize_eval runs/timeomni_v-mixed2
    python -m scripts.summarize_eval runs/zero_shot_qwen2_5_omni \
        --out /tmp/zs.csv --metrics accuracy,macro_f1,uar

If ``--out`` is omitted, defaults to ``<run_root>/summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Per-task default column set. ``success_rate`` (from infer.py's sidecar)
# is appended to both so every row carries a "did we generate something"
# indicator regardless of task.
DEFAULT_METRICS_CLS = ("accuracy", "macro_f1", "weighted_f1", "success_rate")
DEFAULT_METRICS_PRED = ("mae", "mse", "mape", "pcc", "success_rate")
# Legacy: shared with the old --metrics flag default. Kept for backward
# compatibility when callers explicitly request a uniform list.
DEFAULT_METRICS = DEFAULT_METRICS_CLS


def load_metric(path: Path, key: str):
    """Read ``metrics.json`` and pull out one metric. Returns None if the file
    or key is missing — both happen for partial / failed runs and we want the
    CSV cell to be empty rather than error out."""
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get(key)


def discover(run_root: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """Walk ``run_root`` to find every (ablation, dataset_stem) pair that has
    a metrics.json. Returns sorted unique ablation names + dataset names +
    a ``{dataset_name -> task}`` map (task pulled from the first metrics.json
    found for that dataset; ablations of the same dataset agree on task)."""
    abls: set[str] = set()
    datasets: set[str] = set()
    task_by_ds: dict[str, str] = {}
    for ablation_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in ablation_dir.iterdir() if p.is_dir()):
            mp = dataset_dir / "metrics.json"
            if not mp.is_file():
                continue
            abls.add(ablation_dir.name)
            datasets.add(dataset_dir.name)
            if dataset_dir.name not in task_by_ds:
                try:
                    task = json.loads(mp.read_text()).get("task", "classification")
                except (OSError, json.JSONDecodeError):
                    task = "classification"
                task_by_ds[dataset_dir.name] = task or "classification"
    return sorted(abls), sorted(datasets), task_by_ds


def fmt(v) -> str:
    """Render a metric for CSV. Floats → 4-decimal; missing → empty."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def resolve_metric(payload: dict, key: str, task: str):
    """Pull ``key`` out of a metrics.json payload, with task-aware overrides.

    For ``prediction`` tasks ``success_rate`` is reported as ``infer success
    rate * parse_rate`` — the end-to-end probability that a sample yielded
    a parseable forecast. ``parse_rate`` alone undercounts infer failures,
    and infer ``success_rate`` alone ignores rows whose raw text didn't
    contain a parseable ``<forecast>...</forecast>`` block.
    """
    if key == "success_rate" and task == "prediction":
        infer_sr = payload.get("success_rate")
        parse_rate = payload.get("parse_rate")
        if infer_sr is None or parse_rate is None:
            return None
        return infer_sr * parse_rate
    return payload.get(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path,
                    help="Directory containing <ablation>/<dataset>/metrics.json.")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV path (default: <run_root>/summary.csv).")
    ap.add_argument(
        "--metrics", default=None,
        help="Comma-separated metric keys, in column order, applied "
             "uniformly to every dataset. When omitted (default), the "
             "column set is chosen per-dataset by task: "
             f"classification → {','.join(DEFAULT_METRICS_CLS)}; "
             f"prediction → {','.join(DEFAULT_METRICS_PRED)}.",
    )
    args = ap.parse_args()

    if not args.run_root.is_dir():
        raise SystemExit(f"run_root does not exist: {args.run_root}")

    ablations, datasets, task_by_ds = discover(args.run_root)
    if not ablations or not datasets:
        raise SystemExit(
            f"no metrics.json found under {args.run_root}/<ablation>/<dataset>/",
        )

    if args.metrics is not None:
        uniform = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if not uniform:
            raise SystemExit("--metrics must not be empty")
        metrics_per_ds = {ds: uniform for ds in datasets}
    else:
        metrics_per_ds = {
            ds: list(DEFAULT_METRICS_PRED if task_by_ds.get(ds) == "prediction"
                     else DEFAULT_METRICS_CLS)
            for ds in datasets
        }

    # Column layout: first column = ablation, then one column per
    # (dataset, metric) pair grouped by dataset so all metrics for a dataset
    # are adjacent.
    header = ["ablation"]
    for ds in datasets:
        for m in metrics_per_ds[ds]:
            header.append(f"{ds}/{m}")

    out_path = args.out or (args.run_root / "summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header)
        for ab in ablations:
            row = [ab]
            for ds in datasets:
                metrics_path = args.run_root / ab / ds / "metrics.json"
                payload = None
                if metrics_path.is_file():
                    try:
                        payload = json.loads(metrics_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        payload = None
                task = task_by_ds.get(ds, "classification")
                for m in metrics_per_ds[ds]:
                    row.append(fmt(resolve_metric(payload, m, task) if payload else None))
            w.writerow(row)

    print(f"[summarize] wrote {out_path}")
    print(f"            ablations={ablations}")
    print(f"            datasets ={datasets}")
    print(f"            tasks    ={task_by_ds}")
    print(f"            columns  =per-dataset by task" if args.metrics is None
          else f"            columns  ={metrics_per_ds[datasets[0]]} (uniform)")


if __name__ == "__main__":
    main()
