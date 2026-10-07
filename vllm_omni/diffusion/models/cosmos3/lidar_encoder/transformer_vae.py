# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

# Inference-only V1.2 LiDAR VAE: spatial hourglass with causal temporal
# attention and no temporal resampling. Encoder blocks run spatial→temporal;
# decoder blocks run temporal→spatial. The bottleneck is factorized or joint 3D.
# Based on https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py.
import math
from typing import Literal

import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.modules.utils import _pair

from vllm_omni.diffusion.models.cosmos3.lidar_encoder.neighborhood_attention import neighborhood_attention_2d
from vllm_omni.diffusion.models.cosmos3.lidar_encoder.rope3d import (
    VideoRopePosition3DEmb,
    apply_rotary_emb,
)

__all__ = ["Encoder", "Decoder"]


def _zero_out(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.data.zero_()


def _scale_qk(q: torch.Tensor, k: torch.Tensor, log_scale: torch.Tensor, eps: float):
    scale = log_scale.clamp(max=math.log(100)).exp().sqrt()
    q = (F.normalize(q, p=2, dim=-1, eps=eps) * scale).to(q.dtype)
    k = (F.normalize(k, p=2, dim=-1, eps=eps) * scale).to(k.dtype)
    return q, k


def _validate_inference_options(
    *,
    temporal_resample: list[bool],
    num_levels: int,
    positional_embedding: str,
    temporal_mixer: str,
    dropout: float,
    bottleneck_3d: bool,
    bottleneck_3d_causal_time: bool,
    bottleneck_3d_rope: bool,
) -> None:
    if len(temporal_resample) != num_levels or any(temporal_resample):
        raise ValueError("LiDAR inference requires temporal resampling disabled at every level.")
    if positional_embedding != "learnable_embedding":
        raise ValueError("LiDAR inference requires positional_embedding='learnable_embedding'.")
    if temporal_mixer != "attention":
        raise ValueError("LiDAR inference requires temporal_mixer='attention'.")
    if bottleneck_3d and not (bottleneck_3d_causal_time and bottleneck_3d_rope):
        raise ValueError("LiDAR 3D bottlenecks require causal_time=True and use_3d_rope=True.")
    if not 0 <= dropout <= 1:
        raise ValueError("dropout probability must be between 0 and 1.")


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
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(_zero_out)

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
        q, k = _scale_qk(q, k, self.scale, self.eps)
        q, k = self.apply_rope_qk(q, k, coords)
        q = einops.rearrange(q, "B H W N D -> B N (H W) D")
        k = einops.rearrange(k, "B H W N D -> B N (H W) D")
        v = einops.rearrange(v, "B H W N D -> B N (H W) D")
        h = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        h = einops.rearrange(h, "B N (H W) D -> B H W (N D)", H=H, W=W)
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
        q, k = _scale_qk(q, k, self.scale, self.eps)
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
        h = self.out_proj(h)
        return h

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

    def __init__(self, dim: int, num_heads: int, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(_zero_out)
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
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(_zero_out)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gegelu(x)
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
        # Forwarded to the local (neighborhood) attention: whether the W axis
        # wraps (periodic azimuth) or clamps (Cartesian). No effect for global.
        circular: bool = True,
    ):
        super().__init__()

        if attn_type == "global":
            self.residual_attn = GlobalSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                rope_max_harmonics=rope_max_harmonics,
            )
        elif attn_type == "local":
            self.residual_attn = CircularNeighborhoodSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                rope_max_harmonics=rope_max_harmonics,
                circular=circular,
            )

        self.residual_ffn = FeedForwardNetwork(
            dim=in_dim,
            mid_dim=int(in_dim * mlp_ratio),
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
    ):
        super().__init__()
        self.temporal_attn = CausalTemporalAttention(in_dim, num_heads)
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio))

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


class Joint3DSelfAttention(nn.Module):
    """Causal joint attention over (T, H, W), with 3D rotary positions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_t: int,
        len_h: int,
        len_w: int,
        bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.max_t = max_t
        self.max_h = len_h
        self.max_w = len_w

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.rope3d = VideoRopePosition3DEmb(
            head_dim=self.head_dim,
            len_t=max_t,
            len_h=len_h,
            len_w=len_w,
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(_zero_out)

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        if T > self.max_t:
            raise ValueError(
                f"Joint3DSelfAttention received T={T} but was constructed with "
                f"max_t={self.max_t}. Increase bottleneck_3d_max_t."
            )
        if H > self.max_h or W > self.max_w:
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w})."
            )
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        qkv = einops.rearrange(
            qkv,
            "B T H W (three N D) -> three B (T H W) N D",
            three=3,
            D=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _scale_qk(q, k, self.scale, self.eps)
        rope_emb = self.rope3d(T, H, W)
        q = apply_rotary_emb(q, rope_emb)
        k = apply_rotary_emb(k, rope_emb)
        q = einops.rearrange(q, "B S N D -> B N S D")
        k = einops.rearrange(k, "B S N D -> B N S D")
        v = einops.rearrange(v, "B S N D -> B N S D")

        S = H * W
        t_idx = torch.arange(T, device=q.device).repeat_interleave(S)
        mask = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = einops.rearrange(out, "B N (T H W) D -> B T H W (N D)", T=T, H=H, W=W)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend new frames to a sliding cache of unrotated (B, N, T, H*W, D)
        keys and values. Positions are reindexed when the window exceeds max_t.
        """
        B, T_new, H, W, C = x.shape
        if T_new > self.max_t:
            raise ValueError(
                f"streaming chunk T={T_new} exceeds bottleneck_3d_max_t={self.max_t}; use a smaller chunk_frames"
            )
        del coords  # 3D RoPE path does not use polar coords
        if H > self.max_h or W > self.max_w:
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w})."
            )

        S = H * W
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        qkv = einops.rearrange(
            qkv,
            "B T H W (three N D) -> three B (T H W) N D",
            three=3,
            D=self.head_dim,
        )
        q_flat, k_flat, v_flat = qkv[0], qkv[1], qkv[2]
        q_flat, k_flat = _scale_qk(q_flat, k_flat, self.scale, self.eps)
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
        out = self.out_proj(out)
        return x + out, (k_all.detach(), v_all.detach())

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"rope3d(len_t={self.max_t}, len_h={self.max_h}, len_w={self.max_w})"
        )


class Bottleneck3DBlock(nn.Module):
    """Causal joint 3D attention + FFN on (B, T, H, W, C)."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        max_t: int,
        mlp_ratio: float = 3.0,
        len_h: int = 1,
        len_w: int = 1,
    ):
        super().__init__()
        self.attn = Joint3DSelfAttention(
            dim=in_dim,
            num_heads=num_heads,
            max_t=max_t,
            len_h=len_h,
            len_w=len_w,
        )
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio))

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
# Spatial resampling
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

    def forward(self):
        return self.embedding

    def extra_repr(self):
        return f"resolution={tuple(self.embedding.shape[1:3])}, out_dim={self.embedding.shape[3]}"


# =============================================================================
# Encoder
# =============================================================================


class Encoder(nn.Module):
    """Inference-only encoder; exported configs must disable temporal resampling."""

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
        temporal_first_frame_special: bool = False,
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        circular_padding: bool = True,
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
    ):
        super().__init__()
        _validate_inference_options(
            temporal_resample=temporal_downsample,
            num_levels=len(depths) - 1,
            positional_embedding=positional_embedding,
            temporal_mixer=temporal_mixer,
            dropout=dropout,
            bottleneck_3d=bottleneck_3d,
            bottleneck_3d_causal_time=bottleneck_3d_causal_time,
            bottleneck_3d_rope=bottleneck_3d_rope,
        )
        # mapping_depth, dropout, temporal_conv_kernel and temporal-resampling
        # variant flags remain accepted for exported-config compatibility.
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.depths = depths
        self.bottleneck_3d = bottleneck_3d

        token_size = torch.tensor(self.resolution) // torch.tensor(self.patch_size)
        max_harmonics = (token_size / 2).int()

        self.tokenizer = Tokenizer(in_channels, base_channels, patch_size)

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
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.down_levels[f"spatial_{i}"] = spatial_blocks
            if i > 0:
                temporal_blocks = nn.ModuleList()
                for j in range(num_blocks):
                    temporal_blocks.append(
                        TemporalBlock(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                        )
                    )
                self.down_levels[f"temporal_{i}"] = temporal_blocks
            self.down_levels[f"merge_{i}"] = PatchMerging(dim_i)

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
                        mlp_ratio=mlp_ratio,
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
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                    )
                )
                self.mid_temporal.append(
                    TemporalBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                    )
                )

        # Project to z_dim * 2 (mean + logvar)
        self.head = nn.Sequential(
            RMSNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, z_dim * 2, bias=False),
            Rearrange("B H W C -> B C H W"),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Encode (B, C, T, H, W) pixels into (B, 2*z_dim, T, H_z, W_z)."""
        return self._forward(x, coords, streaming=False)[0]

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Encode new frames, attending to the cached temporal prefix."""
        return self._forward(x, coords, temporal_kv_cache, streaming=True)

    def _forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        streaming: bool,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        B, C, T, H, W = x.shape
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        c = F.avg_pool2d(coords, kernel_size=self.patch_size, stride=self.patch_size)

        x = einops.rearrange(x, "B C T H W -> (B T) C H W")
        h = self.tokenizer(x) + self.spatial_pe()
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
                    h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                    if streaming:
                        cache_key = f"down_levels.temporal_{i}.{block_idx}"
                        h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                    else:
                        h = t_block(h)
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
                if streaming:
                    cache_key = f"mid_3d.{block_idx}"
                    h, new_cache[cache_key] = block.forward_stream(h, c, old_cache.pop(cache_key, None))
                else:
                    h = block(h, c)
        else:
            for block_idx, (s_block, t_block) in enumerate(zip(self.mid_spatial, self.mid_temporal)):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                if streaming:
                    cache_key = f"mid_temporal.{block_idx}"
                    h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                else:
                    h = t_block(h)
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
    """Inference-only decoder with spatial upsampling and causal temporal mixing."""

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
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        temporal_expand_wan_style: bool = False,
        out_patch_size: list[int] | None = None,
        circular_padding: bool = True,
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
    ):
        super().__init__()
        _validate_inference_options(
            temporal_resample=temporal_upsample if temporal_upsample is not None else temporal_downsample,
            num_levels=len(depths) - 1,
            positional_embedding=positional_embedding,
            temporal_mixer=temporal_mixer,
            dropout=dropout,
            bottleneck_3d=bottleneck_3d,
            bottleneck_3d_causal_time=bottleneck_3d_causal_time,
            bottleneck_3d_rope=bottleneck_3d_rope,
        )
        if stem_patchify:
            raise ValueError("LiDAR inference does not support the external-encoder patchifying stem.")
        # mapping_depth, dropout, temporal_conv_kernel and temporal-resampling
        # variant flags remain accepted for exported-config compatibility.
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.out_patch_size = _pair(out_patch_size) if out_patch_size is not None else self.patch_size
        self.depths = depths
        self.bottleneck_3d = bottleneck_3d

        # Attention runs on the output-patch grid before final unpatchifying.
        token_size = torch.tensor(self.resolution) // torch.tensor(self.out_patch_size)
        max_harmonics = (token_size / 2).int()
        n_down = len(depths) - 1
        bottleneck_dim = base_channels << n_down
        bottleneck_size = (token_size >> n_down).tolist()

        self.stem = nn.Sequential(
            Rearrange("B C H W -> B H W C"),
            nn.Linear(z_dim, bottleneck_dim, bias=False),
        )

        # Positional embedding at bottleneck resolution
        self.spatial_pe = LearnablePositionalEmbedding(
            out_dim=bottleneck_dim,
            resolution=bottleneck_size,
        )

        # Bottleneck: temporal→spatial, or joint causal 3D attention.
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        mlp_ratio=mlp_ratio,
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_temporal = nn.ModuleList()
            self.mid_spatial = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_temporal.append(
                    TemporalBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                    )
                )
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        rope_max_harmonics=(max_harmonics >> n_down).clamp(min=1),
                    )
                )

        # Up levels: spatial expand + temporal→spatial blocks
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.up_levels = nn.ModuleDict()
        for i in reversed(range(n_down)):
            dim_i = base_channels << i
            dim_above = base_channels << (i + 1)

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
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.up_levels[f"spatial_{i}"] = spatial_blocks
            if i > 0:
                temporal_blocks = nn.ModuleList()
                for j in range(depths[i]):
                    temporal_blocks.append(
                        TemporalBlock(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
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
        h = self.stem(z) + self.spatial_pe()  # (BT, H_z, W_z, C)
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
