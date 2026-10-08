"""2-layer MLP mapping Chronos-2 encoder hidden dim → LLM hidden dim.

The output passes through a final RMSNorm whose scale is initialized so that
TS-token embeddings start with the **same magnitude as the LLM's text token
embeddings**. Without this, default Linear init produces TS embeddings whose
L2 norm is ~15× larger than text embeddings (Chronos features have norm ~25,
adapter shrinks to ~11, vs text-embed norm ~0.77 in Qwen2.5-Omni). That
magnitude mismatch dominates the first attention layer at step 0, blows up
hidden states, and produces nonsensical CE losses (we observed loss≈1577 vs
the bound ln(vocab)≈12 — the bound is for uniform logits; here one logit
explodes far above the target). The RMSNorm caps output magnitude regardless
of input scale; its scalar weight remains trainable so the model can grow
TS contribution as training progresses.

`init_output_std` default 0.013 matches Qwen2.5-Omni's per-token embed RMS.
For other LLMs, derive from `model.get_input_embeddings().weight.std(dim=-1).mean()`.

**Critical**: HF's PreTrainedModel._init_weights matches our nn.RMSNorm via the
substring check `"RMSNorm" in module.__class__.__name__` and forces weight=1.0,
which clobbers our small init. So __init__ stashes the desired value on the
module and TimeOmniVForConditionalGeneration.from_pretrained calls
``ts_adapter.reset_out_norm_init()`` AFTER ``super().from_pretrained`` to
restore it. Same pattern as the chronos ts_tower reload.
"""

from __future__ import annotations

import torch
from torch import nn


class TsAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 2048,
        init_output_std: float = 0.013,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_norm = nn.RMSNorm(out_dim)
        self._init_output_std = float(init_output_std)
        self.reset_out_norm_init()

    def reset_out_norm_init(self) -> None:
        """(Re-)apply the small init to ``out_norm.weight``.

        Idempotent. Call after HF ``from_pretrained`` to undo HF's
        _init_weights forcing the RMSNorm scale to 1.0.
        """
        nn.init.constant_(self.out_norm.weight, self._init_output_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.fc2(self.act(self.fc1(x))))
