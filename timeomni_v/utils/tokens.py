"""Add and track TimeOmni-v special tokens on a Qwen2.5-Omni tokenizer/model pair."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

TS_PLACEHOLDER = "<|ts_placeholder|>"
TS_START = "<|ts_start|>"
TS_END = "<|ts_end|>"
ALL_TS_SPECIALS = [TS_PLACEHOLDER, TS_START, TS_END]


@dataclass
class TsTokenIds:
    placeholder: int
    start: int
    end: int


def add_ts_tokens(tokenizer: PreTrainedTokenizerBase) -> TsTokenIds:
    """Add the three TS specials to the tokenizer.

    Returns the three token IDs (in the order: placeholder, start, end).
    Does NOT resize model embeddings — callers build the model separately and
    then call `init_new_token_embeddings_from_mean` after resize.
    """
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ALL_TS_SPECIALS}
    )
    return TsTokenIds(
        placeholder=tokenizer.convert_tokens_to_ids(TS_PLACEHOLDER),
        start=tokenizer.convert_tokens_to_ids(TS_START),
        end=tokenizer.convert_tokens_to_ids(TS_END),
    )


def init_new_token_embeddings_from_mean(
    model: nn.Module, old_vocab_size: int, new_ids: list[int]
) -> None:
    """Copy the mean of the original vocab embedding into each new row.

    Better starting point than HF's random init — the new rows land in the
    same distribution as the pretrained vocab, so training starts from a
    plausible point and converges faster.
    """
    emb = model.get_input_embeddings()
    with torch.no_grad():
        mean_row = emb.weight[:old_vocab_size].mean(dim=0)
        for tid in new_ids:
            emb.weight[tid].copy_(mean_row)


def freeze_except_new_embedding_rows(
    embedding: nn.Embedding, new_ids: list[int]
) -> None:
    """Make only the rows at new_ids trainable; freeze the rest via a gradient hook.

    We keep requires_grad=True on the whole embedding so the optimizer sees it,
    but zero-out grads for the frozen rows inside a backward hook.
    """
    embedding.weight.requires_grad_(True)
    new_ids_t = torch.tensor(sorted(set(new_ids)), dtype=torch.long)

    def _zero_old_rows(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
        mask[new_ids_t.to(grad.device)] = True
        out = grad.clone()
        out[~mask] = 0
        return out

    embedding.weight.register_hook(_zero_old_rows)
