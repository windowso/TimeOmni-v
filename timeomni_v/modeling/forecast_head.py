"""TimeLLM-style forecasting head for TimeOmni-v.

Reads a fixed-length window of last-layer LLM hidden states (one window per
sample, starting at the first ``<|ts_placeholder|>`` position), flattens
them, and projects to ``pred_len`` numbers in normalized space. The caller
(``TimeOmniVForConditionalGeneration``) is responsible for de-normalizing
predictions at inference time using the per-sample input statistics.
"""

from __future__ import annotations

import torch
from torch import nn


class ForecastHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        window: int,
        pred_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.window = int(window)
        self.pred_len = int(pred_len)
        in_dim = self.hidden_size * self.window
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, self.pred_len)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        # hiddens: (B, window, hidden_size) -> (B, pred_len)
        if hiddens.dim() != 3:
            raise ValueError(
                f"ForecastHead expects (B, window, hidden), got {tuple(hiddens.shape)}"
            )
        b, w, h = hiddens.shape
        if w != self.window or h != self.hidden_size:
            raise ValueError(
                f"shape mismatch: expected (*, {self.window}, {self.hidden_size}), "
                f"got (*, {w}, {h})"
            )
        x = hiddens.reshape(b, w * h)
        x = self.linear(self.drop(self.norm(x)))
        return x


__all__ = ["ForecastHead"]
