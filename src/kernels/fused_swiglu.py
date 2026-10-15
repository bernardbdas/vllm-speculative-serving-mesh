"""Fused SwiGLU Kernel for High-Throughput MLP Evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _swiglu_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

        # Swish(gate) = gate * sigmoid(gate)
        # sigmoid(g) = 1 / (1 + exp(-g))
        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
        swish = gate * sigmoid_gate

        res = swish * up
        tl.store(out_ptr + offsets, res, mask=mask)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Computes element-wise SwiGLU: SiLU(gate) * up."""
    if HAS_TRITON and gate.is_cuda and up.is_cuda:
        n_elements = gate.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        out = torch.empty_like(gate)

        _swiglu_kernel[grid](
            gate,
            up,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    else:
        return F.silu(gate) * up
