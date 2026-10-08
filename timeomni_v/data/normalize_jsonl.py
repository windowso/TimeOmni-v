"""One-shot normalizer for the unified TimeOmni-v jsonl schema.

Aligns the older video-only schema (``video_id``, no ``task``) with the
image+TS schema (``id``, ``task``). Per row:

* If ``video_id`` is present and ``id`` is not, set ``id = video_id`` and drop
  ``video_id``.
* If ``task`` is missing, set ``task = "classification"``. Image jsonls
  already carry the right ``task`` and are left alone.

Idempotent — re-running on an already-normalized file is a no-op. Unknown
fields pass through untouched.

Usage::

    python -m timeomni_v.data.normalize_jsonl --in <jsonl> [--in <jsonl> ...] --inplace
    python -m timeomni_v.data.normalize_jsonl --in <jsonl> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TASK = "classification"


def normalize_row(row: dict) -> dict:
    """Apply the schema rename + task default to a single row in place."""
    if "video_id" in row:
        row["id"] = row.pop("video_id")
    if "task" not in row:
        row["task"] = DEFAULT_TASK
    return row


def normalize_file(in_path: Path, out_path: Path) -> tuple[int, int, int]:
    """Read ``in_path``, normalize each row, write to ``out_path``.

    Returns ``(n_rows, n_id_renamed, n_task_added)`` for reporting.
    """
    n_rows = 0
    n_id = 0
    n_task = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with in_path.open() as fin, tmp.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            had_video_id = "video_id" in row and "id" not in row
            had_no_task = "task" not in row
            normalize_row(row)
            if had_video_id:
                n_id += 1
            if had_no_task:
                n_task += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
    tmp.replace(out_path)
    return n_rows, n_id, n_task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in", dest="inputs", action="append", required=True, type=Path,
        help="Input jsonl path. Repeat for multiple files.",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inplace", action="store_true",
                   help="Rewrite each input file in place.")
    g.add_argument("--out-dir", type=Path,
                   help="Write normalized files into this directory, "
                        "preserving each input's basename.")
    args = ap.parse_args()

    for in_path in args.inputs:
        if not in_path.exists():
            raise SystemExit(f"missing: {in_path}")
        out_path = in_path if args.inplace else args.out_dir / in_path.name
        n_rows, n_id, n_task = normalize_file(in_path, out_path)
        print(
            f"[normalize] {in_path} -> {out_path}: "
            f"rows={n_rows} renamed_id={n_id} added_task={n_task}",
            flush=True,
        )


if __name__ == "__main__":
    main()
