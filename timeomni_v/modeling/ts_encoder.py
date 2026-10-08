"""Thin wrapper around Chronos-2's encoder for feature extraction.

Input: float tensor of shape (N_variates, T_steps) with NaN allowed.
Output: hidden states over **data patches only** (special REG token and
output-patch tokens are dropped), shape (N_variates, n_data_patches, 768).

Chronos-2 encoder output layout:
    [data_patch_0, ..., data_patch_{P-1}, REG_token, output_patch_0, ...]

Chronos-2 does its own instance normalization and NaN masking internally.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from chronos import Chronos2Pipeline

if TYPE_CHECKING:
    from pathlib import Path


class TsEncoder(nn.Module):
    """Wraps Chronos2Model.encode() to return just the data-patch hidden states."""

    def __init__(self, inner: nn.Module, patch_size: int = 16):
        super().__init__()
        self.inner = inner  # Chronos2Model
        self.patch_size = patch_size
        self.hidden_size = getattr(inner.config, "d_model", 768)

    @classmethod
    def from_pretrained(cls, path: str | Path, device_map: str | None = None) -> "TsEncoder":
        pipe = Chronos2Pipeline.from_pretrained(str(path), device_map=device_map)
        inner = pipe.model
        patch_size = getattr(inner.chronos_config, "input_patch_size", 16)
        return cls(inner, patch_size=patch_size)

    def num_data_patches(self, n_timesteps: int) -> int:
        """Number of non-special patches produced for a context of length n_timesteps."""
        return math.ceil(n_timesteps / self.patch_size)

    def forward(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """context: (N, T), group_ids: (N,) long. Returns (N, P_data, D).

        Chronos-2 encoder output layout:
            [data_patch_0, ..., data_patch_{P-1}, REG_token, output_patch_0, ...]

        We slice [:, :n_data_patches, :] to extract only the data patches.
        """
        if context.ndim != 2:
            raise ValueError(f"expected 2D context (N, T); got shape {tuple(context.shape)}")
        n_variates, n_timesteps = context.shape
        n_data_patches = self.num_data_patches(n_timesteps)

        if group_ids is None:
            group_ids = torch.zeros(n_variates, dtype=torch.long, device=context.device)

        enc_out, *_ = self.inner.encode(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,  # minimal; we discard these anyway
        )
        last_hidden = enc_out.last_hidden_state  # (N, P_all, D)
        # Chronos-2 layout: [data_patches..., REG_token, output_patches...]
        # Data patches occupy indices 0 through n_data_patches-1.
        return last_hidden[:, :n_data_patches, :]
