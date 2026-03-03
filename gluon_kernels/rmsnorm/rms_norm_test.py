import torch
import pytest
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from gluon_kernels.rmsnorm.rms_norm import rms_norm as gluon_rms_norm

DEVICE = torch.device("cuda")


def ground_truth_rms_norm(x, eps=1e-10, gamma=None):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    normed = x / rms
    if gamma is not None:
        normed = normed * gamma
    return normed

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),
        (4, 256),
        (32, 512),
        (64, 1024),
        (128, 4096),
    ],
)
def test_gluon_vs_ground_truth(shape, dtype):
    """Verify that the Gluon kernel matches the PyTorch reference."""
    torch.manual_seed(42)
    n_rows, n_cols = shape
    x = torch.randn(n_rows, n_cols, device=DEVICE, dtype=dtype)
    gamma = torch.ones(n_cols, device=DEVICE, dtype=dtype)

    gluon_out = gluon_rms_norm(x, eps=1e-6, gamma=gamma)
    ref_out = ground_truth_rms_norm(x, eps=1e-6, gamma=gamma)

    atol = 1e-2 if dtype == torch.float16 else 1e-5
    rtol = 1e-2 if dtype == torch.float16 else 1e-5

    assert torch.allclose(gluon_out, ref_out, atol=atol, rtol=rtol), (
        f"Gluon vs PyTorch ref mismatch for shape={shape}, dtype={dtype}\n"
        f"Max abs diff: {(gluon_out - ref_out).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
