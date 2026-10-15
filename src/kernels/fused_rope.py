"""Fused Rotary Position Embedding (RoPE) Kernel."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False


def fused_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings using in-place fused operations."""

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # In CUDA/Triton setups this eliminates extra DRAM roundtrips
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
