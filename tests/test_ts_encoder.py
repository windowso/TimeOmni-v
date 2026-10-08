import os

import numpy as np
import pytest
import torch

from timeomni_v.modeling.ts_encoder import TsEncoder

CHRONOS_PATH = os.environ.get("TIMEOMNI_V_CHRONOS_PATH", "")

pytestmark = pytest.mark.skipif(
    not CHRONOS_PATH,
    reason="set TIMEOMNI_V_CHRONOS_PATH to a local chronos-2 checkpoint to run TS encoder tests",
)


@pytest.fixture(scope="module")
def encoder():
    return TsEncoder.from_pretrained(CHRONOS_PATH).eval()


def test_encode_raw_floats_returns_patch_embeddings(encoder):
    ts = torch.tensor(np.random.randn(5, 93).astype(np.float32))  # 5 channels, 93 steps
    out = encoder(context=ts)  # shape: (5, n_data_patches, 768)
    assert out.ndim == 3
    assert out.shape[0] == 5
    assert out.shape[2] == encoder.hidden_size  # 768
    assert out.shape[1] >= 1
    assert out.shape[1] == encoder.num_data_patches(n_timesteps=93)


def test_encode_handles_nan_without_crash(encoder):
    arr = np.random.randn(5, 93).astype(np.float32)
    arr[3, :] = np.nan            # one entire channel all-NaN
    arr[0, 10:20] = np.nan        # partial NaN in another channel
    ts = torch.tensor(arr)
    out = encoder(context=ts)
    assert torch.isfinite(out).all(), "encoder must produce finite outputs even with NaN input"
