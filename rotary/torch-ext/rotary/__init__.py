from typing import Optional, Tuple
import torch

from ._ops import ops


def apply_rotary(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool,
) -> None:
    ops.apply_rotary(x1, x2, cos, sin, out1, out2, conj)


def apply_rotary_transformers(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary kernel implementation wrapper
    Adapts rotary kernel implementation to match transformers apply_rotary_pos_emb signature
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Get half dimension for rotation
    half_dim = q.shape[-1] // 2
    if cos.shape[-1] != half_dim:
        # Trim cos/sin to match half_dim
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]

    # Write into fresh output buffers, reading directly from q/k. This avoids the
    # extra full read+write of cloning q/k before an in-place rotation (the kernel
    # supports out != in), roughly halving the memory traffic of this wrapper.
    q_rotated = torch.empty_like(q)
    k_rotated = torch.empty_like(k)

    apply_rotary(
        q[..., :half_dim],
        q[..., half_dim:],
        cos,
        sin,
        q_rotated[..., :half_dim],
        q_rotated[..., half_dim:],
        False,
    )
    apply_rotary(
        k[..., :half_dim],
        k[..., half_dim:],
        cos,
        sin,
        k_rotated[..., :half_dim],
        k_rotated[..., half_dim:],
        False,
    )
    return q_rotated, k_rotated


# Add torch compile support for functions
apply_rotary_transformers.can_torch_compile = True


__all__ = ["apply_rotary", "apply_rotary_transformers"]
