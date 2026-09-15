# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""3D rotary positional embedding for ``(T, H, W)`` token grids.

Adapted (PyTorch-only, no TransformerEngine) from
``imaginaire.networks.video_backbone.VideoRopePosition3DEmb``: the head dim
is split into three chunks ``(dim_t, dim_h, dim_w)``, each axis contributes
its own rotary frequency band, and the resulting angles are duplicated
across the two rotation halves (GPT-NeoX style) so that
:func:`apply_rotary_emb` can rotate Q/K with a single tensor.

For a query at position ``p_q = (t_q, h_q, w_q)`` and a key at
``p_k = (t_k, h_k, w_k)``, the dot product after rotation depends only on
``p_q - p_k``. If the query and key share the same value along some axis
(e.g. a factored *spatial* attention block where both are on the same
frame), the rotation along that axis cancels out -- so this single 3D
RoPE specializes correctly to 2D RoPE for spatial-only blocks and 1D RoPE
for temporal-only blocks.
"""

from __future__ import annotations

import torch
from torch import nn


class VideoRopePosition3DEmb(nn.Module):
    """3D RoPE producing angles for a ``(T, H, W)`` token grid.

    Args:
        head_dim: Per-head channel count. Must satisfy
            ``head_dim - 2 * (head_dim // 6 * 2) > 0``.
        len_t, len_h, len_w: Maximum supported grid sizes (frequencies are
            cached for these). ``forward`` accepts any smaller ``(T, H, W)``.
        base_theta: RoPE base (``10000.0`` matches LLaMA/Cosmos).
        t_extrapolation_ratio, h_extrapolation_ratio, w_extrapolation_ratio:
            NTK-aware extrapolation factors, applied as
            ``theta *= ratio ** (dim_axis / (dim_axis - 2))``.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        len_t: int,
        len_h: int,
        len_w: int,
        base_theta: float = 10000.0,
        t_extrapolation_ratio: float = 1.0,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        dim = head_dim
        dim_h = (dim // 6) * 2
        dim_w = dim_h
        dim_t = dim - dim_h - dim_w
        assert dim_t >= 2, f"head_dim={dim} too small for 3D RoPE (dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w})"
        self.head_dim = dim
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.max_t = len_t
        self.max_h = len_h
        self.max_w = len_w

        # NTK-aware extrapolation exponent ``dim_axis / (dim_axis - 2)`` is
        # only well-defined for ``dim_axis > 2``. Smaller axes are degenerate
        # (only one frequency bucket) and would also raise ZeroDivisionError
        # below regardless of the ratio. Fall back to base_theta in that
        # case -- when the ratio is 1.0 (the default) the expression is a
        # no-op anyway, so we lose nothing.
        def _theta(ratio: float, dim: int) -> float:
            if ratio == 1.0 or dim <= 2:
                return base_theta
            return base_theta * (ratio ** (dim / (dim - 2)))

        h_theta = _theta(h_extrapolation_ratio, dim_h)
        w_theta = _theta(w_extrapolation_ratio, dim_w)
        t_theta = _theta(t_extrapolation_ratio, dim_t)

        idx_h = torch.arange(0, dim_h, 2, dtype=torch.float32) / dim_h
        idx_w = torch.arange(0, dim_w, 2, dtype=torch.float32) / dim_w
        idx_t = torch.arange(0, dim_t, 2, dtype=torch.float32) / dim_t
        self.register_buffer("freqs_h", 1.0 / (h_theta**idx_h), persistent=False)
        self.register_buffer("freqs_w", 1.0 / (w_theta**idx_w), persistent=False)
        self.register_buffer("freqs_t", 1.0 / (t_theta**idx_t), persistent=False)
        self.register_buffer("seq_h", torch.arange(len_h, dtype=torch.float32), persistent=False)
        self.register_buffer("seq_w", torch.arange(len_w, dtype=torch.float32), persistent=False)
        self.register_buffer("seq_t", torch.arange(len_t, dtype=torch.float32), persistent=False)

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:  # returns [T*H*W,D]
        """Returns rope angles of shape ``(T*H*W, head_dim)``.

        Each half of the last dim duplicates the per-axis angles
        ``[t_half | h_half | w_half | t_half | h_half | w_half]`` so that
        :func:`apply_rotary_emb` (GPT-NeoX style) can rotate Q/K via
        ``[x_lo, x_hi] @ rot``.
        """
        assert T <= self.max_t and H <= self.max_h and W <= self.max_w, (
            f"Requested ({T},{H},{W}) exceeds cached ({self.max_t},{self.max_h},{self.max_w})"
        )
        ang_t = torch.outer(self.seq_t[:T], self.freqs_t)  # [T,Dt/2]
        ang_h = torch.outer(self.seq_h[:H], self.freqs_h)  # [H,Dh/2]
        ang_w = torch.outer(self.seq_w[:W], self.freqs_w)  # [W,Dw/2]
        ang_t_b = ang_t.view(T, 1, 1, -1).expand(T, H, W, -1)  # [T,H,W,Dt/2]
        ang_h_b = ang_h.view(1, H, 1, -1).expand(T, H, W, -1)  # [T,H,W,Dh/2]
        ang_w_b = ang_w.view(1, 1, W, -1).expand(T, H, W, -1)  # [T,H,W,Dw/2]
        half = torch.cat([ang_t_b, ang_h_b, ang_w_b], dim=-1)  # [T,H,W,D/2]
        full = torch.cat([half, half], dim=-1)  # [T,H,W,D]
        return full.reshape(T * H * W, -1)  # [T*H*W,D]


def apply_rotary_emb(
    x: torch.Tensor, rope_emb: torch.Tensor
) -> torch.Tensor:  # x: [...,S,N,D], rope_emb: [S,D], returns [...,S,N,D]
    """Rotate the last dim of ``x`` via GPT-NeoX rotary embeddings.

    Args:
        x: ``(..., S, num_heads, head_dim)`` (any leading shape).
        rope_emb: ``(S, head_dim)`` angles tensor produced by
            :class:`VideoRopePosition3DEmb`.

    Notes:
        Standard GPT-NeoX rotation pairs channel ``i`` with channel
        ``i + head_dim/2`` (not adjacent channels). With ``rope_emb``'s
        ``[half | half]`` duplication, ``cos(rope_emb[:half]) ==
        cos(rope_emb[half:])`` so both halves of a pair rotate by the same
        angle.
    """
    rope_emb = rope_emb.to(x.dtype).unsqueeze(-2)  # [S,1,D]
    cos = rope_emb.cos()  # [S,1,D]
    sin = rope_emb.sin()  # [S,1,D]
    half = x.shape[-1] // 2
    x_lo, x_hi = x[..., :half], x[..., half:]  # [...,S,N,D/2], [...,S,N,D/2]
    cos_lo, cos_hi = cos[..., :half], cos[..., half:]  # [S,1,D/2], [S,1,D/2]
    sin_lo, sin_hi = sin[..., :half], sin[..., half:]  # [S,1,D/2], [S,1,D/2]
    out_lo = x_lo * cos_lo - x_hi * sin_lo  # [...,S,N,D/2]
    out_hi = x_hi * cos_hi + x_lo * sin_hi  # [...,S,N,D/2]
    return torch.cat([out_lo, out_hi], dim=-1)  # [...,S,N,D]
