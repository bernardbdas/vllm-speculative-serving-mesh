"""Fused RMSNorm + Residual Add Kernel with Triton and PyTorch Fast Paths."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        stride_x,
        stride_res,
        stride_out,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start_x = row_idx * stride_x
        row_start_res = row_idx * stride_res
        row_start_out = row_idx * stride_out

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input and residual
        x = tl.load(x_ptr + row_start_x + col_offsets, mask=mask, other=0.0)
        res = tl.load(residual_ptr + row_start_res + col_offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)

        # Fused add
        x_res = x + res
        # Save residual back in-place if needed
        tl.store(residual_ptr + row_start_res + col_offsets, x_res, mask=mask)

        # RMSNorm computation
        variance = tl.sum(x_res * x_res, axis=0) / n_cols
        rsqrt = 1.0 / tl.sqrt(variance + eps)
        out = x_res * rsqrt * w

        tl.store(out_ptr + row_start_out + col_offsets, out, mask=mask)


def fused_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies fused (x + residual) and RMSNorm.
    Returns:
        Tuple of (normalized_output, updated_residual)
    """
    if HAS_TRITON and x.is_cuda and residual.is_cuda:
        n_rows = x.numel() // x.shape[-1]
        n_cols = x.shape[-1]
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        out = torch.empty_like(x)
        _fused_rmsnorm_kernel[(n_rows,)](
            x,
            residual,
            weight,
            out,
            x.stride(0) if x.dim() > 1 else n_cols,
            residual.stride(0) if residual.dim() > 1 else n_cols,
            out.stride(0) if out.dim() > 1 else n_cols,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out, residual
    else:
        # Native PyTorch vectorized path
        updated_res = x + residual
        var = updated_res.pow(2).mean(-1, keepdim=True)
        normed = updated_res * torch.rsqrt(var + eps) * weight
        return normed, updated_res
