"""Wrap frozen sub-towers (vision / audio) in torch.no_grad.

When a tower is fully frozen (all params have requires_grad=False), its
forward still builds an autograd graph that connects to the downstream LLM —
not because its own weights need grads, but because the LLM masked-scatters
the tower's output into an embedding tensor that *does* require grad. That
graph costs memory and, under non-reentrant gradient checkpointing, forces
the activation save/restore machinery to track tower intermediates too.

Wrapping the tower's forward in no_grad:
  * skips autograd graph construction inside the tower;
  * returns a detached tensor that enters masked_scatter cleanly;
  * is a no-op when any tower param is trainable (so it won't silently
    break a future config where the vision encoder is unfrozen).
"""

from __future__ import annotations

from typing import Iterable

import torch


def _all_frozen(module: torch.nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())


def wrap_frozen_tower_forward(
    root: torch.nn.Module,
    attr_paths: Iterable[str],
) -> list[str]:
    """For each attribute path on `root` that resolves to a Module with all
    params frozen, replace its `forward` with a no_grad wrapper. Returns the
    list of wrapped attribute paths (useful for logging).
    """
    wrapped: list[str] = []
    for path in attr_paths:
        obj: torch.nn.Module | None = root
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None or not isinstance(obj, torch.nn.Module):
            continue
        if not _all_frozen(obj):
            continue
        orig_forward = obj.forward

        def _no_grad_forward(*args, _orig=orig_forward, **kwargs):
            with torch.no_grad():
                return _orig(*args, **kwargs)

        obj.forward = _no_grad_forward  # type: ignore[assignment]
        wrapped.append(path)
    return wrapped
