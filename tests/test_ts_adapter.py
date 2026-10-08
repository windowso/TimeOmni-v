import torch

from timeomni_v.modeling.ts_adapter import TsAdapter


def test_ts_adapter_shape():
    adapter = TsAdapter(in_dim=768, hidden_dim=2048, out_dim=2048)
    x = torch.randn(2, 35, 768)  # (batch, tokens, ts hidden)
    y = adapter(x)
    assert y.shape == (2, 35, 2048)


def test_ts_adapter_grads_flow():
    adapter = TsAdapter(768, 2048, 2048)
    x = torch.randn(1, 10, 768, requires_grad=True)
    y = adapter(x).sum()
    y.backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in adapter.parameters())
