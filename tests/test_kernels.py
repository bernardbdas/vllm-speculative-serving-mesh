import torch

from src.kernels.fused_rmsnorm import fused_rmsnorm
from src.kernels.fused_rope import fused_rotary_emb
from src.kernels.fused_swiglu import fused_swiglu


def test_fused_rmsnorm():
    x = torch.randn(2, 4, 64)
    residual = torch.randn(2, 4, 64)
    weight = torch.ones(64)

    normed, updated_res = fused_rmsnorm(x, residual, weight)

    assert normed.shape == x.shape
    assert updated_res.shape == residual.shape
    assert not torch.isnan(normed).any()


def test_fused_swiglu():
    gate = torch.randn(4, 128)
    up = torch.randn(4, 128)

    out = fused_swiglu(gate, up)

    assert out.shape == gate.shape
    assert not torch.isnan(out).any()


def test_fused_rope():
    q = torch.randn(2, 8, 4, 32)
    k = torch.randn(2, 8, 4, 32)
    cos = torch.ones(2, 8, 1, 32)
    sin = torch.zeros(2, 8, 1, 32)

    q_rot, k_rot = fused_rotary_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # With cos=1 and sin=0, rotary embedding is identity
    assert torch.allclose(q_rot, q, atol=1e-5)
