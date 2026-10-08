"""Helpers for inspecting trainable parameters of a PEFT/Transformers model."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path


def _is_rank_zero() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    return rank == 0


def _summarize_dtypes(params: list[tuple[str, "torch.nn.Parameter"]]) -> list[str]:
    """Return human-readable lines like `  bfloat16 : 1234 tensors  (e.g. base_model...)`."""
    by_dtype: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for n, p in params:
        by_dtype[str(p.dtype)].append((n, tuple(p.shape)))
    lines: list[str] = []
    for dt, entries in sorted(by_dtype.items(), key=lambda kv: -len(kv[1])):
        sample_name = entries[0][0]
        lines.append(f"  {dt:<16} : {len(entries):>6} tensors   e.g. {sample_name}")
    return lines


def dump_trainable_params(model, output_dir: str | os.PathLike, filename: str = "trainable_params.txt") -> None:
    """Print a dtype breakdown (trainable vs frozen) to stdout and write a
    trainable-parameter listing to <output_dir>/<filename>.

    Only rank 0 writes the file to avoid clobbering under DDP.
    """
    trainable: list[tuple[str, "torch.nn.Parameter"]] = []
    frozen: list[tuple[str, "torch.nn.Parameter"]] = []
    for n, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append((n, p))

    header = (
        f"[TRAINABLE] {len(trainable)} trainable tensors | "
        f"{len(frozen)} frozen tensors"
    )
    print(header, flush=True)
    print("[TRAINABLE] trainable dtype breakdown:", flush=True)
    trainable_lines = _summarize_dtypes(trainable)
    for line in trainable_lines:
        print("[TRAINABLE] " + line, flush=True)
    print("[TRAINABLE] frozen dtype breakdown:", flush=True)
    frozen_lines = _summarize_dtypes(frozen)
    for line in frozen_lines:
        print("[TRAINABLE] " + line, flush=True)

    if not _is_rank_zero():
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with path.open("w") as f:
        f.write(header + "\n")
        f.write("# trainable dtype breakdown\n")
        for line in trainable_lines:
            f.write(line + "\n")
        f.write("# frozen dtype breakdown\n")
        for line in frozen_lines:
            f.write(line + "\n")
        f.write("\n# trainable tensors\n")
        for n, p in trainable:
            f.write(f"{n}\tshape={tuple(p.shape)}\tdtype={p.dtype}\n")
    print(f"[TRAINABLE] wrote summary + {len(trainable)} entries to {path}", flush=True)
