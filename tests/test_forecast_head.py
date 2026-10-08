import pytest
import torch

from timeomni_v.modeling.forecast_head import ForecastHead


def test_forward_shape():
    head = ForecastHead(hidden_size=4096, window=8, pred_len=20)
    x = torch.randn(3, 8, 4096)
    y = head(x)
    assert y.shape == (3, 20)


def test_rejects_bad_shape():
    head = ForecastHead(hidden_size=64, window=4, pred_len=5)
    with pytest.raises(ValueError):
        head(torch.randn(2, 3, 64))  # window mismatch
    with pytest.raises(ValueError):
        head(torch.randn(2, 4, 32))  # hidden mismatch
    with pytest.raises(ValueError):
        head(torch.randn(8, 64))  # 2-D


def test_dropout_in_train_mode_changes_output():
    torch.manual_seed(0)
    head = ForecastHead(hidden_size=16, window=4, pred_len=3, dropout=0.5)
    x = torch.randn(2, 4, 16)
    head.train()
    y1 = head(x)
    y2 = head(x)
    # With p=0.5 dropout the two passes will diverge with overwhelming probability.
    assert not torch.allclose(y1, y2)
    head.eval()
    y3 = head(x)
    y4 = head(x)
    assert torch.allclose(y3, y4)
