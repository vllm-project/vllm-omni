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

from __future__ import annotations

# =============================================================================
# Transformer VAE with hierarchical hourglass architecture,
# neighborhood attention (FlexAttention) for spatial processing, and
# causal temporal attention for video.
#
# Architecture:
#   Encoder: Tokenizer → [spatial blocks + PatchMerging] (level 0, no temporal)
#            → [spatial→temporal blocks + PatchMerging (+ TemporalMerging)] × (N-1)
#            → global spatial+temporal → z
#   Decoder: z → global temporal+spatial
#            → [TemporalExpanding + PatchExpanding + temporal→spatial blocks] × (N-1)
#            → [PatchExpanding + spatial blocks] (level 0, no temporal) → Detokenizer
#
# No skip connections between encoder and decoder.
#
# Based on: https://github.com/crowsonkb/k-diffusion/blob/master/
#           k_diffusion/models/image_transformer_v2.py
# =============================================================================
import functools
import math
from typing import Literal

import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.modules.utils import _pair

from vllm_omni.diffusion.models.cosmos3.lidar_encoder import ops
from vllm_omni.diffusion.models.cosmos3.lidar_encoder.neighborhood_attention import neighborhood_attention_2d
from vllm_omni.diffusion.models.cosmos3.lidar_encoder.rope3d import (
    VideoRopePosition3DEmb,
    apply_rotary_emb,
)

__all__ = ["Encoder", "Decoder"]


# =============================================================================
# Normalization
# =============================================================================


class RMSNorm(torch.nn.Module):
    def __init__(self, in_dim: int, scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(in_dim)) if scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * self.scale).to(x)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}"


# =============================================================================
# Rotary positional encoding (spatial)
# =============================================================================


class AxialRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_harmonics: list[int]):
        super().__init__()
        freqs_h = self.setup_freqs(num_heads * dim // 4, max_harmonics[0])
        freqs_w = self.setup_freqs(num_heads * dim // 4, max_harmonics[1])
        self.register_buffer("freqs_h", freqs_h.view(dim // 4, num_heads).T)
        self.register_buffer("freqs_w", freqs_w.view(dim // 4, num_heads).T)

    def setup_freqs(self, dim: int, max_harmonics: int):
        return torch.linspace(math.log(1), math.log(max_harmonics), dim).exp().round()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = einops.rearrange(coords, "b c h w -> b h w c")
        radian_h = coords[..., None, [0]] * self.freqs_h
        radian_w = coords[..., None, [1]] * self.freqs_w
        return torch.cat((radian_h, radian_w), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * theta.cos() - x2 * theta.sin()
        y2 = x1 * theta.sin() + x2 * theta.cos()
        return torch.cat((y1, y2), dim=-1)

    def extra_repr(self) -> str:
        return f"freqs_h={tuple(self.freqs_h.shape)}, freqs_w={tuple(self.freqs_w.shape)}"


# =============================================================================
# Spatial self-attention blocks
# =============================================================================


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_max_harmonics: list[int] = (1, 1),
        bias=False,
        eps=1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.eps = eps

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([self.num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.rope = AxialRoPE(self.head_dim, num_heads, rope_max_harmonics)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)

    def scale_qk(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scale = self.scale.clamp(max=math.log(100)).exp().sqrt()
        q = (F.normalize(q, p=2, dim=-1, eps=self.eps) * scale).to(q.dtype)
        k = (F.normalize(k, p=2, dim=-1, eps=self.eps) * scale).to(k.dtype)
        return q, k

    def apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        theta = self.rope(coords)
        d = theta.shape[-1] * 2
        assert (q.shape[-1] >= d) and (k.shape[-1] >= d)
        q[..., :d] = self.rope.rotate(q[..., :d], theta).to(q.dtype)
        k[..., :d] = self.rope.rotate(k[..., :d], theta).to(k.dtype)
        return q, k

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim)
        q, k = self.scale_qk(q, k)
        q, k = self.apply_rope_qk(q, k, coords)
        q = einops.rearrange(q, "B H W N D -> B N (H W) D")
        k = einops.rearrange(k, "B H W N D -> B N (H W) D")
        v = einops.rearrange(v, "B H W N D -> B N (H W) D")
        h = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        h = einops.rearrange(h, "B N (H W) D -> B H W (N D)", H=H, W=W)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


class CircularNeighborhoodSelfAttentionBlock(GlobalSelfAttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: list[int],
        dilation: list[int] = 1,
        dropout: float = 0.0,
        rope_max_harmonics: list[int] = (1, 1),
        # When True (default), the W (last spatial) axis is treated as
        # periodic: it is circularly padded before the neighborhood attention
        # so the seam wraps (range-image azimuth, column 0 ~ column W-1). When
        # False, no wrap -- the neighborhood shifts inward at the W edges
        # (the same boundary handling it already uses for H). Set False for a
        # Cartesian grid (e.g. BEV x-axis) where the two W edges are unrelated.
        circular: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            rope_max_harmonics=rope_max_harmonics,
        )
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.circular = circular

    def before_attn(self, q, k, v):
        if not self.circular:
            return q, k, v
        padding = self.kernel_size[1] // 2
        if padding == 0:
            return q, k, v
        q = F.pad(q, (0, 0, 0, 0, padding, padding), mode="circular")
        k = F.pad(k, (0, 0, 0, 0, padding, padding), mode="circular")
        v = F.pad(v, (0, 0, 0, 0, padding, padding), mode="circular")
        return q, k, v

    def after_attn(self, x):
        if not self.circular:
            return x
        padding = self.kernel_size[1] // 2
        if padding == 0:
            return x
        x = x[:, :, padding:-padding]
        return x

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim)
        q, k = self.scale_qk(q, k)
        q, k = self.apply_rope_qk(q, k, coords)
        q, k, v = self.before_attn(q, k, v)
        h = neighborhood_attention_2d(
            query=q,
            key=k,
            value=v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            scale=1.0,
        )
        h = einops.rearrange(h, "B H W N D -> B H W (N D)")
        h = self.after_attn(h)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"kernel_size={self.kernel_size}, dilation={self.dilation}"
        )


# =============================================================================
# Causal temporal self-attention
# =============================================================================


class CausalTemporalAttention(nn.Module):
    """Causal self-attention along the temporal dimension.

    Input/output shape: (N, T, C) where N = B*H*W.
    Each position attends only to itself and earlier positions.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)
        self.dropout = nn.Dropout(dropout)
        self._SDPA_MAX_BATCH = 65535

    def _project_qkv(self, x: torch.Tensor):
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        return einops.rearrange(
            qkv,
            "N T (three heads D) -> three N heads T D",
            three=3,
            D=self.head_dim,
        )

    def _sdpa(self, q, k, v, *, attn_mask=None, is_causal=False):
        N = q.shape[0]
        if N <= self._SDPA_MAX_BATCH:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        chunks = []
        for i in range(0, N, self._SDPA_MAX_BATCH):
            j = min(i + self._SDPA_MAX_BATCH, N)
            chunks.append(
                F.scaled_dot_product_attention(
                    q[i:j],
                    k[i:j],
                    v[i:j],
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )
            )
        return torch.cat(chunks, dim=0)

    def _finish(self, x, h):
        h = einops.rearrange(h, "N heads T D -> N T (heads D)")
        h = self.dropout(h)
        h = self.out_proj(h)
        return x + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        return self._finish(x, self._sdpa(q, k, v, is_causal=True))

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend new temporal tokens to cached history and this causal block."""
        q, k_new, v_new = self._project_qkv(x)
        past = 0
        if kv_cache is not None:
            k_past, v_past = kv_cache
            if (
                k_past.shape[:2] != k_new.shape[:2]
                or k_past.shape[3:] != k_new.shape[3:]
                or v_past.shape != k_past.shape
            ):
                raise ValueError(
                    "temporal KV cache shape mismatch: "
                    f"cached k/v={tuple(k_past.shape)}/{tuple(v_past.shape)}, "
                    f"new k={tuple(k_new.shape)}"
                )
            past = k_past.shape[2]
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
        else:
            k, v = k_new, v_new

        T_new = q.shape[2]
        q_index = past + torch.arange(T_new, device=q.device)[:, None]
        k_index = torch.arange(past + T_new, device=q.device)[None, :]
        causal_offset_mask = k_index <= q_index
        out = self._finish(
            x,
            self._sdpa(
                q,
                k,
                v,
                attn_mask=causal_offset_mask,
                is_causal=False,
            ),
        )
        return out, (k.detach(), v.detach())

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


# =============================================================================
# Feed-forward
# =============================================================================


class GEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features * 2, bias=bias)

    def forward(self, x):
        h = super().forward(x)
        h, gate = h.chunk(2, dim=-1)
        return h * F.gelu(gate)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, mid_dim, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(ops.zero_out)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gegelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


# =============================================================================
# Spatial block (spatial attention + FFN)
# =============================================================================


class SpatialBlock(nn.Module):
    """Spatial attention + FFN.  Operates on (BT, H, W, C) per-frame."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        attn_type: Literal["local", "global"] = "global",
        kernel_size: list[int] | None = None,
        dilation: list[int] = 1,
        rope_max_harmonics: list[int] = (1, 1),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        # Forwarded to the local (neighborhood) attention: whether the W axis
        # wraps (periodic azimuth) or clamps (Cartesian). No effect for global.
        circular: bool = True,
    ):
        super().__init__()

        if attn_type == "global":
            self.residual_attn = GlobalSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                dropout=dropout,
                rope_max_harmonics=rope_max_harmonics,
            )
        elif attn_type == "local":
            self.residual_attn = CircularNeighborhoodSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                rope_max_harmonics=rope_max_harmonics,
                circular=circular,
            )

        self.residual_ffn = FeedForwardNetwork(
            dim=in_dim,
            mid_dim=int(in_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.residual_attn(x, coords)
        x = self.residual_ffn(x)
        return x


class TemporalBlock(nn.Module):
    """Causal temporal attention + FFN.  Operates on (BHW, T, C)."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal_attn = CausalTemporalAttention(in_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_attn(x)
        x = self.ffn(x)
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.temporal_attn.forward_stream(x, kv_cache)
        x = self.ffn(x)
        return x, kv_cache


class CausalTemporalConv(nn.Module):
    """Causal temporal convolution along T (WAN-style).

    The conv analog of :class:`CausalTemporalAttention` -- same
    (N, T, C) interface (N = B*H*W), same zero-init residual so the
    block is a residual add at init. Implemented as a depth-preserving
    ``Conv1d`` over the T axis with left-only causal padding (kernel-1),
    so each output only depends on the current and past frames -- the
    ``kernel_size x 1 x 1`` case of WanVAE's ``CausalConv3d`` (spatial
    mixing is left to the surrounding neighborhood-attention blocks, so
    swapping only the temporal mixer stays a clean ablation).
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False).apply(ops.zero_out)
        self.dropout = nn.Dropout(dropout)
        self.causal_pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        h = self.norm(x)
        h = einops.rearrange(h, "N T C -> N C T")
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv(h)
        h = einops.rearrange(h, "N C T -> N T C")
        h = self.dropout(h)
        h = self.out_proj(h)
        return x + h


class TemporalConvBlock(nn.Module):
    """Causal temporal conv + FFN.  Operates on (BHW, T, C).

    Drop-in replacement for :class:`TemporalBlock` (identical constructor
    signature and (N, T, C) forward) that swaps the temporal *attention*
    for a WAN-style causal temporal *convolution*. ``num_heads`` is
    accepted only for signature parity (unused).
    """

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.temporal_conv = CausalTemporalConv(in_dim, kernel_size, dropout)
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.ffn(x)
        return x


# =============================================================================
# Joint 3D self-attention block
# =============================================================================


class Joint3DSelfAttention(nn.Module):
    """Joint self-attention over flattened (T*H*W) tokens.

    Mirrors :class:`GlobalSelfAttentionBlock` but operates jointly over time
    and space instead of per-frame.

    Two positional-encoding modes are supported via ``use_3d_rope``:

    * **Default (``use_3d_rope=False``)** -- spatial position via the
      existing :class:`AxialRoPE` on ``(h, w)`` (broadcast across time, since
      polar coords don't vary with time), and time position via a *learnable*
      1D embedding added to the pre-attention features. This is what
      original training runs used; kept as the default for backward compat.
      Caveat: the learnable temporal slots past the largest ``T`` seen in
      training are uninitialized noise, so this mode does *not* extrapolate
      cleanly to longer-than-trained sequence lengths at inference.
    * **``use_3d_rope=True``** -- replace both the spatial :class:`AxialRoPE`
      and the learnable temporal PE with a single 3D RoPE
      (:class:`VideoRopePosition3DEmb`) over ``(t, h, w)``. RoPE is a
      rotation (length-preserving), and is fully parameter-free, so the
      angles extrapolate to longer sequences with no untrained parameters.
      The spatial axes of the 3D RoPE play the same role :class:`AxialRoPE`
      did, but without coupling to polar coords -- the bottleneck operates
      on a regular (H_z, W_z) grid so a Cartesian RoPE is well-defined and
      consistent with how the rest of the literature does video VAEs.

    Causality is *orthogonal* to the PE choice: ``causal_time=True`` masks
    the joint attention so each ``(t, h, w)`` token only attends to
    ``(t', h', w')`` with ``t' <= t``. The causal mask is materialized as
    a ``(T*H*W, T*H*W)`` bool tensor so memory scales with ``(T*H*W)^2``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_t: int,
        rope_max_harmonics: list[int] = (1, 1),
        dropout: float = 0.0,
        causal_time: bool = False,
        bias: bool = False,
        eps: float = 1e-6,
        # If True, encode both spatial and temporal position with a single
        # 3D RoPE (length-preserving rotation, no learnable PE buffer) and
        # skip the original 2D AxialRoPE + learnable `temporal_pe` path.
        # ``len_h``/``len_w`` set the spatial cap of the 3D RoPE cache.
        use_3d_rope: bool = False,
        len_h: int = 1,
        len_w: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.causal_time = causal_time
        self.use_3d_rope = use_3d_rope
        self.max_t = max_t
        self.max_h = len_h if use_3d_rope else 0
        self.max_w = len_w if use_3d_rope else 0

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        if use_3d_rope:
            self.rope = None
            self.temporal_pe = None
            self.rope3d = VideoRopePosition3DEmb(
                head_dim=self.head_dim,
                len_t=max_t,
                len_h=len_h,
                len_w=len_w,
            )
        else:
            self.rope = AxialRoPE(self.head_dim, num_heads, rope_max_harmonics)
            self.temporal_pe = nn.Parameter(torch.zeros(max_t, dim))
            nn.init.trunc_normal_(self.temporal_pe, std=0.02)
            self.rope3d = None
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)

    def _scale_qk(self, q: torch.Tensor, k: torch.Tensor):
        scale = self.scale.clamp(max=math.log(100)).exp().sqrt()
        q = (F.normalize(q, p=2, dim=-1, eps=self.eps) * scale).to(q.dtype)
        k = (F.normalize(k, p=2, dim=-1, eps=self.eps) * scale).to(k.dtype)
        return q, k

    def _apply_spatial_rope(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        theta = self.rope(coords)  # (B_c, H, W, N, D//2)
        d = theta.shape[-1] * 2
        assert (q.shape[-1] >= d) and (k.shape[-1] >= d)
        theta = theta.unsqueeze(1)  # broadcast over T → (B_c, 1, H, W, N, D//2)
        q[..., :d] = self.rope.rotate(q[..., :d], theta).to(q.dtype)
        k[..., :d] = self.rope.rotate(k[..., :d], theta).to(k.dtype)
        return q, k

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        if T > self.max_t:
            raise ValueError(
                f"Joint3DSelfAttention received T={T} but was constructed with "
                f"max_t={self.max_t}. Increase bottleneck_3d_max_t."
            )
        if self.use_3d_rope and (H > self.max_h or W > self.max_w):
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w}). "
                f"This usually means the bottleneck spatial shape changed -- "
                f"rebuild the decoder with larger len_h/len_w."
            )
        h = self.norm(x)

        if self.use_3d_rope:
            # 3D RoPE path. We flatten (T, H, W) to a single sequence axis
            # so `apply_rotary_emb` can broadcast its (S, head_dim) rope
            # tensor over batch + num_heads in one shot.
            qkv = self.qkv_proj(h)
            qkv = einops.rearrange(
                qkv,
                "B T H W (three N D) -> three B (T H W) N D",
                three=3,
                D=self.head_dim,
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = self._scale_qk(q, k)  # rotation preserves L2 norm
            rope_emb = self.rope3d(T, H, W)  # (T*H*W, head_dim)
            q = apply_rotary_emb(q, rope_emb)
            k = apply_rotary_emb(k, rope_emb)
            q = einops.rearrange(q, "B S N D -> B N S D")
            k = einops.rearrange(k, "B S N D -> B N S D")
            v = einops.rearrange(v, "B S N D -> B N S D")
        else:
            # Original: learnable temporal PE + 2D AxialRoPE over (h, w).
            h = h + self.temporal_pe[:T].view(1, T, 1, 1, C)
            qkv = self.qkv_proj(h)
            q, k, v = einops.rearrange(
                qkv,
                "B T H W (three N D) -> three B T H W N D",
                three=3,
                D=self.head_dim,
            )
            q, k = self._scale_qk(q, k)
            q, k = self._apply_spatial_rope(q, k, coords)
            q = einops.rearrange(q, "B T H W N D -> B N (T H W) D")
            k = einops.rearrange(k, "B T H W N D -> B N (T H W) D")
            v = einops.rearrange(v, "B T H W N D -> B N (T H W) D")

        if self.causal_time:
            S = H * W
            t_idx = torch.arange(T, device=q.device).repeat_interleave(S)
            # mask[i, j] = True iff query t_i can attend to key t_j (t_j <= t_i)
            mask = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)  # (T*S, T*S)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)

        out = einops.rearrange(out, "B N (T H W) D -> B T H W (N D)", T=T, H=H, W=W)
        out = self.dropout(out)
        out = self.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend a new temporal chunk to a sliding causal KV cache.

        WAN-style long-video path: only ``x`` (the new frames) is projected to
        Q; K/V are concatenated with a cached prefix and trimmed to
        ``max_t`` frames. RoPE / temporal PE are applied over the *window*
        indices ``0 .. T_win-1``, so a pure prefix stream with
        ``T_total <= max_t`` matches :meth:`forward` exactly.

        Cache tensors are **unrotated** ``(B, N, T, H*W, D)`` so the window
        can be re-indexed when older frames are dropped.

        Requires ``causal_time=True``.
        """
        if not self.causal_time:
            raise NotImplementedError("Joint3DSelfAttention.forward_stream requires causal_time=True")
        B, T_new, H, W, C = x.shape
        if T_new > self.max_t:
            raise ValueError(
                f"streaming chunk T={T_new} exceeds bottleneck_3d_max_t={self.max_t}; use a smaller chunk_frames"
            )
        del coords  # 3D RoPE path does not use polar coords
        if self.use_3d_rope and (H > self.max_h or W > self.max_w):
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w})."
            )

        S = H * W
        h = self.norm(x)
        if not self.use_3d_rope:
            raise NotImplementedError(
                "Joint3DSelfAttention.forward_stream currently requires use_3d_rope=True "
                "(learnable temporal PE streaming is not supported)"
            )
        qkv = self.qkv_proj(h)
        qkv = einops.rearrange(
            qkv,
            "B T H W (three N D) -> three B (T H W) N D",
            three=3,
            D=self.head_dim,
        )
        q_flat, k_flat, v_flat = qkv[0], qkv[1], qkv[2]
        q_flat, k_flat = self._scale_qk(q_flat, k_flat)
        # Cache / stream in (B, N, T, S, D) so frames can be trimmed.
        q_new = einops.rearrange(q_flat, "B (T S) N D -> B N T S D", T=T_new, S=S)
        k_new = einops.rearrange(k_flat, "B (T S) N D -> B N T S D", T=T_new, S=S)
        v_new = einops.rearrange(v_flat, "B (T S) N D -> B N T S D", T=T_new, S=S)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k_all = torch.cat([k_past, k_new], dim=2)
            v_all = torch.cat([v_past, v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new

        T_total = k_all.shape[2]
        if T_total > self.max_t:
            keep = self.max_t
            k_all = k_all[:, :, -keep:].contiguous()
            v_all = v_all[:, :, -keep:].contiguous()
            T_total = keep

        # Window-local temporal positions 0 .. T_total-1 (matches one-shot when
        # the stream never exceeds max_t).
        rope_full = self.rope3d(T_total, H, W)  # (T_total*S, D)
        rope_full = einops.rearrange(rope_full, "(T S) D -> T S D", T=T_total, S=S)
        rope_q = rope_full[-T_new:].reshape(T_new * S, -1)
        rope_k = rope_full.reshape(T_total * S, -1)
        q = einops.rearrange(q_new, "B N T S D -> B (T S) N D")
        k = einops.rearrange(k_all, "B N T S D -> B (T S) N D")
        v = einops.rearrange(v_all, "B N T S D -> B (T S) N D")
        q = apply_rotary_emb(q, rope_q)
        k = apply_rotary_emb(k, rope_k)
        q = einops.rearrange(q, "B S N D -> B N S D")
        k = einops.rearrange(k, "B S N D -> B N S D")
        v = einops.rearrange(v, "B S N D -> B N S D")

        # With one newly appended frame, every cached key is at or before the
        # query frame. Avoiding an explicit all-true mask lets SDPA use its
        # memory-efficient backend; at LiDAR resolution the dense mask alone can
        # otherwise consume several GiB. Multi-frame chunks still need temporal
        # causality between their newly appended frames.
        mask = None
        if T_new > 1:
            t_q = torch.arange(T_total - T_new, T_total, device=q.device).repeat_interleave(S)
            t_k = torch.arange(T_total, device=q.device).repeat_interleave(S)
            mask = t_k.unsqueeze(0) <= t_q.unsqueeze(1)  # (T_new*S, T_total*S)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = einops.rearrange(out, "B N (T H W) D -> B T H W (N D)", T=T_new, H=H, W=W)
        out = self.dropout(out)
        out = self.out_proj(out)
        return x + out, (k_all.detach(), v_all.detach())

    def extra_repr(self) -> str:
        pe = (
            f"rope3d(len_t={self.max_t}, len_h={self.max_h}, len_w={self.max_w})"
            if self.use_3d_rope
            else f"max_t={self.max_t}, axial_rope_2d"
        )
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}, {pe}, causal_time={self.causal_time}"


class Bottleneck3DBlock(nn.Module):
    """Joint 3D self-attention + FFN. Operates on (B, T, H, W, C).

    Set ``use_3d_rope=True`` (with ``len_h`` / ``len_w`` set to the
    bottleneck's spatial shape) to swap the default learnable temporal PE
    + 2D AxialRoPE for a single 3D RoPE over ``(t, h, w)`` -- see
    :class:`Joint3DSelfAttention` for the tradeoffs. Backward-compatible:
    leaving ``use_3d_rope`` at its default keeps the original behavior.
    """

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        max_t: int,
        rope_max_harmonics: list[int] = (1, 1),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        causal_time: bool = False,
        use_3d_rope: bool = False,
        len_h: int = 1,
        len_w: int = 1,
    ):
        super().__init__()
        self.attn = Joint3DSelfAttention(
            dim=in_dim,
            num_heads=num_heads,
            max_t=max_t,
            rope_max_harmonics=rope_max_harmonics,
            dropout=dropout,
            causal_time=causal_time,
            use_3d_rope=use_3d_rope,
            len_h=len_h,
            len_w=len_w,
        )
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, coords)
        x = self.ffn(x)
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.attn.forward_stream(x, coords, kv_cache)
        x = self.ffn(x)
        return x, kv_cache


# =============================================================================
# Patch / temporal resampling
# =============================================================================


class PatchMerging(nn.Sequential):
    """2x2 spatial downsample: (B, 2H, 2W, C) -> (B, H, W, 2C)"""

    def __init__(self, dim: int):
        super().__init__(
            Rearrange("B (H P1) (W P2) C -> B H W (P1 P2 C)", P1=2, P2=2),
            nn.Linear(4 * dim, 2 * dim, bias=False),
        )


class PatchExpanding(nn.Module):
    """2x2 spatial upsample: (B, H, W, C) -> (B, 2H, 2W, C//2)"""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, "B H W (P1 P2 C) -> B (H P1) (W P2) C", P1=2, P2=2)
        return x


class TemporalMerging(nn.Module):
    """Causal 2x temporal downsample (mimics WanVAE downsample3d).

    Uses Conv1d(kernel=3, stride=2) with left-only causal padding.
    Each output only depends on the current and past frames.
    Input:  (B, T, H, W, C)
    Output: (B, ceil(T/2), H, W, C)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=0, bias=False)
        self.causal_pad = 2  # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "B T H W C -> (B H W) C T")
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)
        x = einops.rearrange(x, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        return x


class TemporalMergingFirstFrameSpecial(nn.Module):
    """First-frame-special causal 2x temporal downsample (WAN-style).

    Mirrors :class:`WanVAE2P1`'s singleton-first-frame behavior at the
    transformer-VAE merge stage: the first input slot becomes ``z[0]``
    via a per-frame ``Linear`` (no temporal mixing), and the remaining
    ``T_in - 1`` frames are merged in *disjoint* 2-frame chunks via a
    stride-2 ``Conv1d`` with ``kernel=2`` and **no** causal padding --
    each pair stays inside its own chunk:

        ``in[0:1]``  ->  z[0]                   per-frame Linear
        ``in[2i-1:2i+1]``  ->  z[i>=1]          stride-2 kernel-2 conv

    Pair this with :class:`TemporalExpandingWanStyle` in the decoder so
    ``T_decoded = 2*T_z - 1`` exactly matches ``T_in = 1 + 2*(T_z - 1)``.
    No front-crop needed, no boundary information loss.

    Compared to plain :class:`TemporalMerging` (kernel=3, stride=2,
    causal-pad-2):
    * z[0] genuinely depends on in[0] alone (not "in[0] + 2 pad"
      activations); the Linear is the cleanest analog of WAN's CausalConv
      with a fully zero-cached history.
    * z[i>=1] sees both frames of its 2-frame chunk (vs. only the first
      frame + 2 history frames in plain merging) -- so each latent has
      *full* info on the chunk it represents, eliminating the
      "z[i] never sees in[2i+1]" information loss.

    Requires ``T_in`` to be ODD (i.e. ``T_in - 1`` divisible by 2).

    Input:  (B, T_in, H, W, C),  T_in = 2*T_z - 1
    Output: (B, T_z,  H, W, C)
    """

    def __init__(self, dim: int):
        super().__init__()
        # First-frame projection: like WAN's CausalConv3d on a length-1
        # input with zero-padded kernel history -- a linear function of
        # the single frame's features.
        self.first_frame_proj = nn.Linear(dim, dim, bias=False)
        # Disjoint pair merger: kernel=2, stride=2, no padding. Each
        # output index i corresponds to ``rest[2i, 2i+1]`` exclusively
        # (no kernel overlap across chunks, no causal-pad zero leakage).
        self.conv = nn.Conv1d(dim, dim, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, H, W, C = x.shape
        if T_in == 1:
            return self.first_frame_proj(x)
        first = self.first_frame_proj(x[:, :1])  # (B, 1, H, W, C)
        rest = x[:, 1:]
        if rest.shape[1] % 2 != 0:
            raise ValueError(
                f"TemporalMergingFirstFrameSpecial requires (T_in - 1) "
                f"divisible by 2 at this level, got T_in={T_in}. With "
                f"``temporal_first_frame_special=True`` and N levels of "
                f"temporal merge, the input T must satisfy "
                f"T = 1 + (2 ** N) * (T_z - 1) -- e.g. T=9 for N=1, T_z=5."
            )
        rest = einops.rearrange(rest, "B T H W C -> (B H W) C T")
        rest = self.conv(rest)  # (N, C, (T_in - 1) // 2)
        rest = einops.rearrange(rest, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        return torch.cat([first, rest], dim=1)


class TemporalExpanding(nn.Module):
    """Causal 2x temporal upsample (mimics WanVAE upsample3d).

    Uses a causal Conv1d that doubles channels, then interleaves the two
    channel halves along time to achieve 2x upsampling.  No future leakage
    because the underlying conv is causal (left-pad only, stride=1).
    Input:  (B, T//2, H, W, C)
    Output: (B, T, H, W, C)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.causal_pad = 2  # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, H, W, C = x.shape
        x = einops.rearrange(x, "B T H W C -> (B H W) C T")
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)  # (N, 2C, T_in)
        x = einops.rearrange(
            x,
            "N (two C) T -> N C (T two)",
            two=2,
        )  # interleave: [t0_a, t0_b, t1_a, t1_b, ...] → 2*T_in frames
        x = einops.rearrange(x, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        return x


class TemporalExpandingWanStyle(nn.Module):
    """Causal 2x temporal upsample with WAN-style first-frame handling.

    Mirrors :class:`WanVAE2P1.decode`: the first input frame produces a
    single output frame (just the "a" half of the causal conv's doubled
    output -- i.e. no temporal duplication), while every subsequent input
    frame produces 2 output frames. Composed twice (matching the WAN
    encoder's 4x temporal compression with its special first-frame chunk),
    a latent of length ``T_z`` decodes to ``4*T_z - 3`` output frames -- the
    same length as the original input when ``T_in = 1 + 4*(T_z - 1)``, so
    no leading-frame crop is needed.

    Input:  (B, T_in, H, W, C)
    Output: (B, 2*T_in - 1, H, W, C)  (or 1 frame if T_in == 1)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.causal_pad = 2  # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, H, W, C = x.shape
        h = einops.rearrange(x, "B T H W C -> (B H W) C T")
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv(h)  # (N, 2C, T_in)
        h = einops.rearrange(
            h,
            "N (two C) T -> N C (T two)",
            two=2,
        )  # interleave: [t0_a, t0_b, t1_a, t1_b, ...] → 2*T_in frames along T
        h = einops.rearrange(h, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        # Keep t0_a as the single "first frame" output; drop t0_b. Keep both
        # halves for every other input frame.
        if T_in == 1:
            return h[:, :1]
        return torch.cat([h[:, :1], h[:, 2:]], dim=1)


class Tokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: list[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=False,
            ),
            Rearrange("B C H W -> B H W C"),
        )


class Detokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: list[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            RMSNorm(in_channels),
            nn.Linear(
                in_channels,
                out_channels * patch_size[0] * patch_size[1],
                bias=False,
            ),
            Rearrange(
                "B H W (P1 P2 C) -> B C (H P1) (W P2)",
                P1=patch_size[0],
                P2=patch_size[1],
            ),
        )


# =============================================================================
# Positional embedding
# =============================================================================


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, out_dim: int, resolution: list[int]):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, *resolution, out_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, coords=None):
        return self.embedding

    def extra_repr(self):
        return f"resolution={tuple(self.embedding.shape[1:3])}, out_dim={self.embedding.shape[3]}"


# =============================================================================
# Encoder
# =============================================================================


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: list[int],
        in_channels: int,
        z_dim: int,
        base_channels: int = 128,
        patch_size: list[int] = (1, 4),
        window_size: list[int] = (3, 9),
        depths: list[int] = (3, 3, 3, 3),
        num_heads: list[int] = (2, 4, 8, 16),
        dilation: list[int] = (1, 1, 1, 1),
        temporal_downsample: list[bool] = (True, False, False),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: str = "learnable_embedding",
        # If True, every level with ``temporal_downsample[i]=True`` uses
        # :class:`TemporalMergingFirstFrameSpecial` instead of the plain
        # :class:`TemporalMerging`. Yields clean WAN-style chunk
        # boundaries (z[0]=enc(in[0]), z[i>=1]=enc(disjoint 2-frame
        # chunk)) at the cost of requiring T_in - 1 to be divisible
        # by ``2 ** sum(temporal_downsample)``.
        temporal_first_frame_special: bool = False,
        # ---- Bottleneck options (mirror Decoder) ----
        # If True, replace the encoder bottleneck's factorized (global spatial
        # + temporal) attention with a single joint 3D self-attention over
        # (T*H*W) tokens. Down-levels are unchanged. Default False keeps the
        # original factorized bottleneck (bit-identical to existing configs).
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        # If False, the local (neighborhood) attention does NOT wrap the W
        # axis (clamps at the edges instead). Default True preserves the
        # range-image azimuth-periodic behavior. Set False for a Cartesian
        # BEV grid where the left/right W edges are unrelated.
        circular_padding: bool = True,
        # Temporal-mixing mechanism for the down/mid temporal blocks:
        # "attention" (:class:`CausalTemporalAttention`, default) or "conv"
        # (WAN-style :class:`CausalTemporalConv`, kernel ``temporal_conv_kernel``).
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
    ):
        super().__init__()
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.depths = depths
        self.temporal_downsample = temporal_downsample
        self.bottleneck_3d = bottleneck_3d
        self._temporal_cls = (
            functools.partial(TemporalConvBlock, kernel_size=temporal_conv_kernel)
            if temporal_mixer == "conv"
            else TemporalBlock
        )

        token_size = torch.tensor(self.resolution) // torch.tensor(self.patch_size)
        max_harmonics = (token_size / 2).int()

        self.tokenizer = Tokenizer(in_channels, base_channels, patch_size)

        if positional_embedding != "learnable_embedding":
            raise ValueError(
                "Only positional_embedding='learnable_embedding' is supported "
                f"(got {positional_embedding!r}). Alternate absolute PE modes were removed."
            )
        # mapping_depth is retained for config compatibility with the shipped checkpoint.
        _ = mapping_depth
        self.spatial_pe = LearnablePositionalEmbedding(
            out_dim=base_channels,
            resolution=token_size.tolist(),
        )

        # Down levels: spatial blocks + temporal blocks + merging
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.down_levels = nn.ModuleDict()
        for i, num_blocks in enumerate(depths[:-1]):
            dim_i = base_channels << i
            spatial_blocks = nn.ModuleList()
            for j in range(num_blocks):
                spatial_blocks.append(
                    SpatialBlock(
                        in_dim=dim_i,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.down_levels[f"spatial_{i}"] = spatial_blocks
            if i > 0:
                temporal_blocks = nn.ModuleList()
                for j in range(num_blocks):
                    temporal_blocks.append(
                        self._temporal_cls(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                    )
                self.down_levels[f"temporal_{i}"] = temporal_blocks
            self.down_levels[f"merge_{i}"] = PatchMerging(dim_i)
            if temporal_downsample[i]:
                merge_cls = TemporalMergingFirstFrameSpecial if temporal_first_frame_special else TemporalMerging
                self.down_levels[f"temporal_merge_{i}"] = merge_cls(dim_i * 2)

        # Bottleneck: either factorized (global spatial + temporal), or a
        # single joint 3D self-attention over (T*H*W) tokens.
        i = len(depths) - 1
        bottleneck_dim = base_channels << i
        bottleneck_size = (token_size >> i).tolist()
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        causal_time=bottleneck_3d_causal_time,
                        use_3d_rope=bottleneck_3d_rope,
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_spatial = nn.ModuleList()
            self.mid_temporal = nn.ModuleList()
            for j in range(depths[-1]):
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                    )
                )
                self.mid_temporal.append(
                    self._temporal_cls(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )

        # Project to z_dim * 2 (mean + logvar)
        self.head = nn.Sequential(
            RMSNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, z_dim * 2, bias=False),
            Rearrange("B H W C -> B C H W"),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video tensor
            coords: (1, 2, H, W) polar coordinates
        Returns:
            (B, z_dim*2, T', H_z, W_z) latent parameters
        """
        B, C, T, H, W = x.shape
        c = F.avg_pool2d(coords, kernel_size=self.patch_size, stride=self.patch_size)

        # Tokenize per-frame
        x = einops.rearrange(x, "B C T H W -> (B T) C H W")
        h = self.tokenizer(x) + self.spatial_pe(c)  # (BT, H', W', C0)
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T)

        T_cur = T
        for i in range(len(self.depths) - 1):
            _, _, Hc, Wc, _ = h.shape
            has_temporal = f"temporal_{i}" in self.down_levels
            t_blocks = (
                self.down_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.down_levels[f"spatial_{i}"])
            )
            for s_block, t_block in zip(self.down_levels[f"spatial_{i}"], t_blocks):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                if t_block is not None:
                    h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                    h = t_block(h)
                    h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)
                else:
                    h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

            # Spatial downsample per-frame
            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.down_levels[f"merge_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)
            c = F.avg_pool2d(c, kernel_size=2, stride=2)

            # Temporal downsample
            if self.temporal_downsample[i]:
                h = self.down_levels[f"temporal_merge_{i}"](h)
                T_cur = h.shape[1]

        # Bottleneck: joint 3D attention, or factorized spatial-then-temporal.
        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            for block in self.mid_3d:
                h = block(h, c)
        else:
            for s_block, t_block in zip(self.mid_spatial, self.mid_temporal):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                h = t_block(h)
                h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)

        # Head per-frame
        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.head(h)  # (BT, z_dim*2, H_z, W_z)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        return h

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Encode a new temporal chunk with causal KV caches (WAN-style).

        Spatial ops run only on ``x``; temporal / joint-3D layers attend to the
        cached prefix. Requires no temporal downsampling (1x temporal latent).
        """
        if any(self.temporal_downsample):
            raise NotImplementedError("encoder streaming requires temporal_downsample=False everywhere")
        B, C, T, H, W = x.shape
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        c = F.avg_pool2d(coords, kernel_size=self.patch_size, stride=self.patch_size)

        x = einops.rearrange(x, "B C T H W -> (B T) C H W")
        h = self.tokenizer(x) + self.spatial_pe(c)
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T)

        T_cur = T
        for i in range(len(self.depths) - 1):
            _, _, Hc, Wc, _ = h.shape
            has_temporal = f"temporal_{i}" in self.down_levels
            t_blocks = (
                self.down_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.down_levels[f"spatial_{i}"])
            )
            for block_idx, (s_block, t_block) in enumerate(zip(self.down_levels[f"spatial_{i}"], t_blocks)):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                if t_block is not None:
                    if not isinstance(t_block, TemporalBlock):
                        raise NotImplementedError(
                            "encoder streaming currently supports temporal attention, not temporal convolution"
                        )
                    h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                    cache_key = f"down_levels.temporal_{i}.{block_idx}"
                    h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                    h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)
                else:
                    h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.down_levels[f"merge_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)
            c = F.avg_pool2d(c, kernel_size=2, stride=2)

        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            for block_idx, block in enumerate(self.mid_3d):
                cache_key = f"mid_3d.{block_idx}"
                h, new_cache[cache_key] = block.forward_stream(h, c, old_cache.pop(cache_key, None))
        else:
            for block_idx, (s_block, t_block) in enumerate(zip(self.mid_spatial, self.mid_temporal)):
                if not isinstance(t_block, TemporalBlock):
                    raise NotImplementedError(
                        "encoder streaming currently supports temporal attention, not temporal convolution"
                    )
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                cache_key = f"mid_temporal.{block_idx}"
                h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)

        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.head(h)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        missing = set(old_cache) - set(new_cache)
        if missing:
            raise ValueError(f"unused temporal KV cache keys: {sorted(missing)}")
        return h, new_cache


# Decoder ported from imaginaire4 9ca7bd6adfe; shared spatial attention uses FlexAttention.
class Decoder(nn.Module):
    def __init__(
        self,
        resolution: list[int],
        out_channels: int,
        z_dim: int,
        base_channels: int = 128,
        patch_size: list[int] = (1, 4),
        window_size: list[int] = (3, 9),
        depths: list[int] = (3, 3, 3, 3),
        num_heads: list[int] = (2, 4, 8, 16),
        dilation: list[int] = (1, 1, 1, 1),
        temporal_downsample: list[bool] = (True, False, False),
        temporal_upsample: list[bool] | None = None,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: str = "learnable_embedding",
        stem_patchify: bool = False,
        # ---- Bottleneck options ----
        # If True, replace the bottleneck's factorized (temporal + global
        # spatial) attention with a single joint 3D self-attention over
        # (T*H*W) tokens. The up-levels are unchanged (still skip temporal
        # attn at the highest resolution as before).
        bottleneck_3d: bool = False,
        # Max T accepted by the bottleneck-3D's learnable temporal PE. Only
        # used when bottleneck_3d=True. Set this >= the largest T_z (encoder
        # latent frames) you'll ever feed the decoder.
        bottleneck_3d_max_t: int = 32,
        # If True, the joint 3D attention is causal in time (token (t, h, w)
        # only attends to (t', h', w') with t' <= t). The causal mask is
        # materialized as a (T*H*W, T*H*W) bool tensor, so memory scales as
        # O((T*H*W)^2). Bidirectional is the default for non-streaming use.
        bottleneck_3d_causal_time: bool = False,
        # If True (and bottleneck_3d=True), encode position in the joint 3D
        # attention with a parameter-free 3D RoPE (`VideoRopePosition3DEmb`)
        # over (t, h, w) instead of the default (learnable temporal PE + 2D
        # spatial AxialRoPE). Recommended for length-extrapolation: the
        # learnable temporal PE has uninitialized slots past the largest T
        # seen in training (= noise at inference); 3D RoPE has none. The
        # spatial axes of the 3D RoPE replace AxialRoPE -- the bottleneck
        # operates on a regular (H_z, W_z) grid so Cartesian RoPE is fine
        # and decouples positional encoding from polar `coords`.
        # Orthogonal to `bottleneck_3d_causal_time` -- both can be combined.
        bottleneck_3d_rope: bool = False,
        # If True, replace each TemporalExpanding with TemporalExpandingWanStyle,
        # which mirrors the WAN decoder's first-frame-special behavior:
        # T_in -> 2*T_in - 1 per level. Composed twice this maps T_z latents
        # to 4*T_z - 3 frames, so when paired with the WAN encoder there's no
        # over-production and no leading-frame crop needed downstream.
        temporal_expand_wan_style: bool = False,
        # ---- Detokenizer (output) options ----
        # Patch size for the final unpatchify (pixel-shuffle) layer. Defaults
        # to `patch_size` (symmetric with the stem, the original behavior).
        # Setting this larger than `patch_size` lets the decoder run the top
        # (highest-resolution) up-levels at a coarser feature grid and rely
        # on the final per-pixel Linear + rearrange to upsample to the output
        # resolution -- substantially cheaper since the top level dominates
        # the activation memory. The user is responsible for making sure
        # `bottleneck_size = resolution // out_patch_size >> n_down` matches
        # the encoder's latent spatial shape (i.e. reduce `depths` by
        # `log2(out_patch_size / patch_size)` to compensate).
        out_patch_size: list[int] | None = None,
        # If False, the local (neighborhood) attention does NOT wrap the W
        # axis (clamps at the edges). Default True keeps the range-image
        # azimuth-periodic behavior. Set False for a Cartesian BEV grid.
        circular_padding: bool = True,
        # Temporal-mixing mechanism for the mid/up temporal blocks:
        # "attention" (:class:`CausalTemporalAttention`, default) or "conv"
        # (WAN-style :class:`CausalTemporalConv`, kernel ``temporal_conv_kernel``).
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
    ):
        super().__init__()
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.out_patch_size = _pair(out_patch_size) if out_patch_size is not None else self.patch_size
        self.depths = depths
        self.temporal_upsample = temporal_upsample if temporal_upsample is not None else temporal_downsample
        self.stem_patchify = stem_patchify
        self.bottleneck_3d = bottleneck_3d
        self._temporal_cls = (
            functools.partial(TemporalConvBlock, kernel_size=temporal_conv_kernel)
            if temporal_mixer == "conv"
            else TemporalBlock
        )
        self.temporal_expand_wan_style = temporal_expand_wan_style
        temporal_expand_cls = TemporalExpandingWanStyle if temporal_expand_wan_style else TemporalExpanding

        # `token_size` is the feature grid at the top up-level, i.e. *before*
        # the final detokenizer's unpatchify -- this is what RoPE harmonics
        # and the coordinate pyramid index. With `out_patch_size > patch_size`
        # the top up-level sits at a coarser grid than the output resolution.
        token_size = torch.tensor(self.resolution) // torch.tensor(self.out_patch_size)
        max_harmonics = (token_size / 2).int()
        n_down = len(depths) - 1
        bottleneck_dim = base_channels << n_down
        bottleneck_size = (token_size >> n_down).tolist()

        # Project from z_dim to bottleneck channels.
        # By default the stem is a per-pixel channel projection (used in the
        # full TransformerVAE, where the Encoder's Tokenizer has already
        # patchified the input by `patch_size`). When `stem_patchify=True`,
        # we instead apply a Conv2d with stride=patch_size, so the decoder
        # can be paired with an external encoder (e.g. the WAN encoder) whose
        # latent spatial shape is patch_size larger than the bottleneck.
        if stem_patchify and tuple(self.patch_size) != (1, 1):
            self.stem = nn.Sequential(
                nn.Conv2d(
                    z_dim,
                    bottleneck_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            self.stem = nn.Sequential(
                Rearrange("B C H W -> B H W C"),
                nn.Linear(z_dim, bottleneck_dim, bias=False),
            )

        # Positional embedding at bottleneck resolution
        if positional_embedding != "learnable_embedding":
            raise ValueError(
                "Only positional_embedding='learnable_embedding' is supported "
                f"(got {positional_embedding!r}). Alternate absolute PE modes were removed."
            )
        # mapping_depth is retained for config compatibility with the shipped checkpoint.
        _ = mapping_depth
        self.spatial_pe = LearnablePositionalEmbedding(
            out_dim=bottleneck_dim,
            resolution=bottleneck_size,
        )

        # Bottleneck: either factorized (temporal + global spatial), or a
        # single joint 3D self-attention over (T*H*W) tokens.
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        rope_max_harmonics=(max_harmonics >> n_down).clamp(min=1),
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        causal_time=bottleneck_3d_causal_time,
                        use_3d_rope=bottleneck_3d_rope,
                        # 3D RoPE caches per-axis frequencies up to (len_t,
                        # len_h, len_w). Spatial caps come from the
                        # bottleneck's own grid; the temporal cap is shared
                        # with the learnable-PE path via `bottleneck_3d_max_t`.
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_temporal = nn.ModuleList()
            self.mid_spatial = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_temporal.append(
                    self._temporal_cls(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> n_down).clamp(min=1),
                    )
                )

        # Up levels: temporal expand + spatial expand + temporal→spatial blocks
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.up_levels = nn.ModuleDict()
        for i in reversed(range(n_down)):
            dim_i = base_channels << i
            dim_above = base_channels << (i + 1)

            if self.temporal_upsample[i]:
                self.up_levels[f"temporal_expand_{i}"] = temporal_expand_cls(dim_above)

            self.up_levels[f"expand_{i}"] = PatchExpanding(dim_above)

            spatial_blocks = nn.ModuleList()
            for j in range(depths[i]):
                spatial_blocks.append(
                    SpatialBlock(
                        in_dim=dim_i,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.up_levels[f"spatial_{i}"] = spatial_blocks
            # Temporal blocks are needed at every level that does a temporal
            # expand (to mix the newly-interpolated frames into a coherent
            # sequence), plus -- as a cheap-but-useful compute heuristic --
            # at every level deeper than the top one. The previous condition
            # was just `i > 0`, which silently skipped temporal attention at
            # the top level even when temporal_upsample[0] was True, leaving
            # the just-expanded frames un-mixed.
            if i > 0 or self.temporal_upsample[i]:
                temporal_blocks = nn.ModuleList()
                for j in range(depths[i]):
                    temporal_blocks.append(
                        self._temporal_cls(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                    )
                self.up_levels[f"temporal_{i}"] = temporal_blocks

        self.detokenizer = Detokenizer(base_channels, out_channels, self.out_patch_size)

    def forward(
        self,
        z: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_temporal_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            z: (B, z_dim, T', H_z, W_z) latent
            coords: (1, 2, H, W) polar coordinates
        Returns:
            (B, C, T, H, W) reconstructed video
        """
        B = z.shape[0]
        T_cur = z.shape[2]
        n_down = len(self.depths) - 1
        streaming = return_temporal_kv_cache
        if temporal_kv_cache is not None and not streaming:
            raise ValueError("temporal_kv_cache requires return_temporal_kv_cache=True")
        if streaming:
            if any(self.temporal_upsample):
                raise NotImplementedError(
                    "streaming decode currently requires temporal_upsample=False at every decoder level"
                )
            if self.bottleneck_3d:
                if not all(
                    getattr(block.attn, "causal_time", False) and getattr(block.attn, "use_3d_rope", False)
                    for block in self.mid_3d
                ):
                    raise NotImplementedError(
                        "streaming decode with bottleneck_3d requires causal_time=True and use_3d_rope=True"
                    )
            elif not all(isinstance(block, TemporalBlock) for block in self.mid_temporal):
                raise NotImplementedError(
                    "streaming decode currently supports temporal attention, not temporal convolution"
                )
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Build coordinate pyramid. The base sits at the *top up-level* token
        # grid (= resolution // out_patch_size), which is the highest-res grid
        # the attention blocks actually see -- the final detokenizer's
        # pixel-shuffle never gets coords.
        c_base = F.avg_pool2d(coords, kernel_size=self.out_patch_size, stride=self.out_patch_size)
        c_levels = [c_base]
        for i in range(n_down):
            c_levels.append(F.avg_pool2d(c_levels[-1], kernel_size=2, stride=2))

        # Stem per-frame + spatial PE
        z = einops.rearrange(z, "B C T H W -> (B T) C H W")
        h = self.stem(z) + self.spatial_pe(c_levels[n_down])  # (BT, H_z, W_z, C)
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Bottleneck
        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            for block_idx, block in enumerate(self.mid_3d):
                if streaming:
                    cache_key = f"mid_3d.{block_idx}"
                    h, new_cache[cache_key] = block.forward_stream(h, c_levels[n_down], old_cache.pop(cache_key, None))
                else:
                    h = block(h, c_levels[n_down])
        else:
            # Factorized: temporal then global-spatial (decoder order).
            for block_idx, (t_block, s_block) in enumerate(zip(self.mid_temporal, self.mid_spatial)):
                h = einops.rearrange(h, "B T H W C -> (B H W) T C")
                if streaming:
                    cache_key = f"mid_temporal.{block_idx}"
                    h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                else:
                    h = t_block(h)
                h = einops.rearrange(h, "(B H W) T C -> (B T) H W C", B=B, H=Hc, W=Wc)
                h = s_block(h, c_levels[n_down])
                h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Up levels
        for i in reversed(range(n_down)):
            # Temporal upsample
            if self.temporal_upsample[i]:
                h = self.up_levels[f"temporal_expand_{i}"](h)
                T_cur = h.shape[1]

            # Spatial upsample per-frame
            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.up_levels[f"expand_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

            _, _, Hc, Wc, _ = h.shape

            has_temporal = f"temporal_{i}" in self.up_levels
            t_blocks = self.up_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.up_levels[f"spatial_{i}"])
            for block_idx, (t_block, s_block) in enumerate(zip(t_blocks, self.up_levels[f"spatial_{i}"])):
                if t_block is not None:
                    h = einops.rearrange(h, "B T H W C -> (B H W) T C")
                    if streaming:
                        if not isinstance(t_block, TemporalBlock):
                            raise NotImplementedError(
                                "streaming decode currently supports temporal attention, not temporal convolution"
                            )
                        cache_key = f"up_levels.temporal_{i}.{block_idx}"
                        h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                    else:
                        h = t_block(h)
                    h = einops.rearrange(h, "(B H W) T C -> (B T) H W C", B=B, H=Hc, W=Wc)
                else:
                    h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c_levels[i])
                h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Detokenize per-frame
        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.detokenizer(h)  # (BT, C_out, H, W)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        if streaming:
            missing = set(old_cache) - set(new_cache)
            if missing:
                raise ValueError(f"unused temporal KV cache keys: {sorted(missing)}")
            return h, new_cache
        return h
