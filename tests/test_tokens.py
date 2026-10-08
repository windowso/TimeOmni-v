import torch
from torch import nn

from timeomni_v.utils.tokens import freeze_except_new_embedding_rows


def test_freeze_except_new_embedding_rows_zeros_old_rows():
    emb = nn.Embedding(10, 4)
    freeze_except_new_embedding_rows(emb, new_ids=[7, 9])
    loss = emb.weight.sum()
    loss.backward()
    # rows 0..6 and 8 should have zero grad; rows 7 and 9 should have grad of 1s
    for i in range(10):
        if i in (7, 9):
            assert emb.weight.grad[i].abs().sum() > 0, f"row {i} should have nonzero grad"
        else:
            assert emb.weight.grad[i].abs().sum() == 0, f"row {i} should be zeroed"
