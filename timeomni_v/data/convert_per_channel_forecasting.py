"""Explode forecasting jsonl rows into single-channel rows + per-channel target CSVs.

For each ``task=="prediction"`` row we:
  1. Parse the ``<forecast>...</forecast>`` body in ``answer`` into a
     ``(T_target, C_target)`` matrix via :mod:`timeomni_v.inference.forecast_parse`.
  2. Load the multi-channel history CSV at ``timeseries_path`` (drops the
     ``timestamp`` column the same way ``load_and_pad_ts_csvs`` does at
     train time) → ``(T_hist, C_hist)``. Asserts ``C_hist == C_target``.
  3. For each channel ``c`` writes:
        - ``<csv-out-dir>/<id>__ch{c}.csv`` (history, single column)
        - ``<csv-out-dir>/<id>__ch{c}.target.csv`` (target, single column)
  4. Emits one new jsonl row per channel:
        - ``id = f"{orig_id}__ch{c}"``
        - ``timeseries_path`` -> per-channel history CSV
        - new field ``forecast_target_path`` -> per-channel target CSV
        - ``ts_shape = [T_hist, 1]``
        - ``answer = ""`` (the forecast head replaces text generation; the
          target lives in the CSV instead).

Rows with ``task != "prediction"`` pass through unchanged.

Usage::

    python -m timeomni_v.data.convert_per_channel_forecasting \\
        --in  data/image_ts/jsonl/terra.jsonl \\
              data/image_ts/jsonl/terra_test.jsonl \\
        --csv-out-dir data/image_ts/csv/terra_perch \\
        --suffix percha
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timeomni_v.inference.forecast_parse import parse_forecast


def _load_history(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load a multi-channel history CSV; drop ``timestamp`` to mirror runtime."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    return df, list(df.columns)


def _stack_target(parsed: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Stack parsed ``[(label, values), ...]`` into ``(T_target, C_target)``."""
    width = parsed[0][1].shape[0]
    rows: list[np.ndarray] = []
    for _, vals in parsed:
        if vals.shape[0] != width:
            raise ValueError(
                f"target rows have mixed widths (first={width}, got={vals.shape[0]})"
            )
        rows.append(vals)
    return np.stack(rows, axis=0)  # (T_target, C_target)


def _format_forecast_answer(
    parsed: list[tuple[str, np.ndarray]], channel: int
) -> str:
    """Rebuild a ``<forecast>...</forecast>`` body restricted to one channel.

    Preserves the original labels (dates / step indices) so eval.py's
    ``parse_forecast`` + label-based alignment keep working when scoring.
    """
    lines = ["<forecast>"]
    for label, vals in parsed:
        v = float(vals[channel])
        # Match the precision of the original CoVLA / sp500 / pixelrec data
        # (~6 significant digits is plenty; eval re-parses to float64).
        lines.append(f"({label}: {v:g})")
    lines.append("</forecast>")
    return "\n".join(lines)


def explode_row(
    row: dict, csv_out_dir: Path
) -> list[dict]:
    """Return the list of new jsonl rows produced from ``row``.

    Non-prediction rows are returned as ``[row]`` unchanged.
    """
    if row.get("task") != "prediction":
        return [row]

    orig_id = row["id"]
    ans = row.get("answer", "")
    parsed = parse_forecast(ans)
    if parsed is None:
        raise ValueError(f"row id={orig_id!r}: <forecast> body did not parse")
    target_arr = _stack_target(parsed)  # (T_target, C_target)

    ts_path = row["timeseries_path"]
    hist_df, hist_cols = _load_history(ts_path)
    hist_arr = hist_df.to_numpy(dtype=np.float32)  # (T_hist, C_hist)

    if hist_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(
            f"row id={orig_id!r}: history channels {hist_arr.shape[1]} != "
            f"target channels {target_arr.shape[1]}"
        )

    csv_out_dir.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []
    for c in range(hist_arr.shape[1]):
        col_name = hist_cols[c]
        # Single-column DataFrames keep the original column name so debugging
        # prints stay readable; the loader just drops the header at runtime.
        hist_path = csv_out_dir / f"{orig_id}__ch{c}.csv"
        target_path = csv_out_dir / f"{orig_id}__ch{c}.target.csv"
        pd.DataFrame({col_name: hist_arr[:, c]}).to_csv(hist_path, index=False)
        pd.DataFrame({col_name: target_arr[:, c]}).to_csv(target_path, index=False)

        new_row = dict(row)  # shallow copy preserves video/image/prompt fields
        new_row["id"] = f"{orig_id}__ch{c}"
        new_row["timeseries_path"] = str(hist_path)
        new_row["forecast_target_path"] = str(target_path)
        new_row["ts_shape"] = [int(hist_arr.shape[0]), 1]
        # Keep a single-channel <forecast> body in `answer` so eval.py can
        # parse the ground truth via the existing forecast parser. Training
        # ignores this — the head reads the per-channel target CSV.
        new_row["answer"] = _format_forecast_answer(parsed, c)
        new_row["channel_index"] = c
        new_row["channel_name"] = col_name
        new_rows.append(new_row)
    return new_rows


def convert_file(in_path: Path, out_path: Path, csv_out_dir: Path) -> tuple[int, int]:
    n_in = 0
    n_out = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            for new_row in explode_row(row, csv_out_dir):
                fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                n_out += 1
    return n_in, n_out


def _derive_out_path(in_path: Path, suffix: str) -> Path:
    """Splice ``suffix`` in before the optional dot-tag and ``.jsonl``.

    ``foo.jsonl`` → ``foo.percha.jsonl``; ``foo.<tag>.jsonl`` → ``foo.percha.<tag>.jsonl``.
    """
    stem, dot, tag = in_path.stem.partition(".")
    return in_path.with_name(f"{stem}.{suffix}{dot}{tag}{in_path.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", nargs="+", required=True, type=Path,
                    help="Input forecasting jsonl(s) with task=prediction rows.")
    ap.add_argument("--csv-out-dir", required=True, type=Path,
                    help="Where per-channel history+target CSVs are written.")
    ap.add_argument("--suffix", default="percha",
                    help="Suffix to splice into output jsonl name "
                         "(<stem>.<suffix>[.<tag>].jsonl). Default: percha.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Optional override for output jsonl directory; "
                         "defaults to each input's directory.")
    args = ap.parse_args()

    for in_path in args.in_paths:
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / _derive_out_path(in_path, args.suffix).name
        else:
            out_path = _derive_out_path(in_path, args.suffix)
        n_in, n_out = convert_file(in_path, out_path, args.csv_out_dir)
        print(f"[convert_perch] {in_path.name}: {n_in} in -> {n_out} out  "
              f"(jsonl={out_path}, csvs={args.csv_out_dir})", flush=True)


if __name__ == "__main__":
    main()
