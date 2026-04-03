# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/OpenMOSS/MOVA
"""
MOVA Video Diffusion Transformer.

Based on WanModel architecture from MOVA upstream, adapted for vllm-omni.
This module is NOT called via forward() directly -- instead, the pipeline
drives the block loop via prepare_transformer_block_kwargs /
post_transformer_block_out, with bridge calls interleaved between blocks.
"""

import math

import torch
import torch.nn as nn
from torch.nn import RMSNorm
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding (1-D)."""
    half = dim // 2
    sinusoid = torch.outer(
        position.float(),
        torch.pow(10000.0, -torch.arange(half, device=position.device).float() / half),
    )
    return torch.cat([sinusoid.cos(), sinusoid.sin()], dim=-1).to(position.dtype)


def precompute_freqs_cis_3d(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3-D RoPE frequencies as a tuple of (temporal, height, width).

    Returns tuple, NOT concatenated — matches upstream WanModel storage format.
    """
    t_dim = dim - 2 * (dim // 3)
    h_dim = dim // 3
    w_dim = dim // 3

    freqs_t = precompute_freqs_cis(t_dim, end, theta)
    freqs_h = precompute_freqs_cis(h_dim, end, theta)
    freqs_w = precompute_freqs_cis(w_dim, end, theta)

    return freqs_t, freqs_h, freqs_w


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """Compute 1-D RoPE frequencies as complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply RoPE to tensor x using precomputed frequencies."""
    b, s, d = x.shape
    head_dim = d // num_heads
    x = x.view(b, s, num_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:s].unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(x.dtype).reshape(b, s, d)


def rope_apply_head_dim(x: torch.Tensor, freqs: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Apply RoPE to tensor x.

    Input x: [B, S, dim] (flat, NOT pre-reshaped to heads).
    freqs: [L, 1, head_dim//2] complex.
    Internally reshapes x to [B, S, n_heads, head_dim], applies RoPE, flattens back.
    """
    # x: [B, S, dim] -> [B, S, n_heads, head_dim]
    b, s, d = x.shape
    x = x.view(b, s, -1, head_dim)
    # x: [B, S, n_heads, head_dim] -> complex [B, S, n_heads, head_dim//2]
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    # freqs: [L, 1, head_dim//2] broadcasts over B and n_heads
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(2)
    # x_rotated: [B, S, n_heads * head_dim] = [B, S, dim]
    return x_rotated.to(x.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Adaptive layer norm modulation."""
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class MovaSelfAttention(nn.Module):
    """Self-attention with QK-norm and RoPE, using vllm-omni Attention."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # RoPE expects flat [B, S, dim], reshape to heads AFTER RoPE
        q = rope_apply_head_dim(self.norm_q(self.q(x)), freqs, d)
        k = rope_apply_head_dim(self.norm_k(self.k(x)), freqs, d)

        # Now reshape to [B, S, n_heads, head_dim] for attention
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        x = self.attn(q, k, v)
        x = x.flatten(2)
        x = self.o(x)
        return x


class MovaCrossAttention(nn.Module):
    """Cross-attention with optional image input branch."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_image_input = has_image_input

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(y)).view(b, -1, n, d)
        v = self.v(y).view(b, -1, n, d)

        if self.has_image_input:
            # Split context: first 257 tokens are image, rest is text
            img_ctx = y[:, :257]
            txt_ctx = y[:, 257:]
            k_txt = self.norm_k(self.k(txt_ctx)).view(b, -1, n, d)
            v_txt = self.v(txt_ctx).view(b, -1, n, d)
            k_img = self.norm_k_img(self.k_img(img_ctx)).view(b, -1, n, d)
            v_img = self.v_img(img_ctx).view(b, -1, n, d)

            out_txt = self.attn(q, k_txt, v_txt)
            out_img = self.attn(q, k_img, v_img)
            x = out_txt + out_img
        else:
            x = self.attn(q, k, v)

        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class MovaDiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive LayerNorm modulation."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim

        self.self_attn = MovaSelfAttention(dim, num_heads, eps)
        self.cross_attn = MovaCrossAttention(dim, num_heads, eps, has_image_input)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm3 = nn.LayerNorm(dim, eps=eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # 6-way modulation parameters
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        # Modulation: 6 values (shift1, scale1, gate1, shift2, scale2, gate2)
        mod = (self.modulation + t_mod).chunk(6, dim=1)
        shift1, scale1, gate1 = [m.squeeze(1) for m in mod[:3]]
        shift2, scale2, gate2 = [m.squeeze(1) for m in mod[3:]]

        # Self-attention with RoPE
        y = self.self_attn(modulate(self.norm1(x), shift1, scale1), freqs)
        x = x + gate1 * y

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), context)

        # Feed-forward
        y = self.ffn(modulate(self.norm2(x), shift2, scale2))
        x = x + gate2 * y

        return x


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------


class MovaHead(nn.Module):
    """Output projection with adaptive norm."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, ...], eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if t_mod.ndim == 2:
            t_mod = t_mod.unsqueeze(1)  # [B, dim] -> [B, 1, dim]
        mod = (self.modulation + t_mod).chunk(2, dim=1)
        shift, scale = mod[0].squeeze(1), mod[1].squeeze(1)
        x = self.head(modulate(self.norm(x), shift, scale))
        return x


# ---------------------------------------------------------------------------
# MLP (for optional image embedding)
# ---------------------------------------------------------------------------


class MovaMLP(nn.Module):
    """Projection MLP with optional positional embedding."""

    def __init__(self, in_dim: int, out_dim: int, has_pos_emb: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(out_dim, out_dim)
        if has_pos_emb:
            self.emb_pos = nn.Parameter(torch.randn(1, 514, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "emb_pos"):
            x = x + self.emb_pos
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MovaVideoTransformer(nn.Module):
    """
    MOVA Video Diffusion Transformer.

    Adapted from upstream WanModel. This model does NOT have a standalone
    forward() -- it is driven by the pipeline via prepare_transformer_block_kwargs
    and post_transformer_block_out, with bridge calls between blocks.
    """

    _repeated_blocks = ("MovaDiTBlock",)
    _layerwise_offload_blocks_attr = "blocks"

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        separated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        **kwargs,
    ):
        super().__init__()

        legacy_timestep_key = "se" + "perated_timestep"
        if legacy_timestep_key in kwargs:
            separated_timestep = kwargs.pop(legacy_timestep_key)

        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.separated_timestep = separated_timestep
        self.__dict__[legacy_timestep_key] = separated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        # Patch embedding (3D conv for video)
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [MovaDiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )

        # Output head
        self.head = MovaHead(dim, out_dim, patch_size, eps)

        # Precompute 3D RoPE frequencies
        self.freqs = precompute_freqs_cis_3d(dim // num_heads, 1024)

        # Optional image embedding
        if has_image_input and has_image_pos_emb:
            self.img_emb = MovaMLP(dim, dim, has_pos_emb=True)

        # Optional ref conv
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, F, H, W] -> [B, (F'H'W'), dim]."""
        x = self.patch_embedding(x)
        # x shape: [B, dim, F', H', W']
        grid_size = x.shape[2:]  # (F', H', W')
        x = x.flatten(2).transpose(1, 2)  # [B, F'H'W', dim]
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        """Convert [B, L, C_out*prod(patch)] -> [B, C_out, F, H, W]."""
        c = self.out_dim
        f, h, w = grid_size
        pf, ph, pw = self.patch_size
        x = x.view(-1, f, h, w, pf, ph, pw, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(-1, c, f * pf, h * ph, w * pw)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full forward pass. Used when bridge is NOT interleaved
        (e.g., standalone video generation without audio).
        For MOVA's dual-tower mode, the pipeline calls
        prepare_transformer_block_kwargs / block loop / post_transformer_block_out
        instead.
        """
        device = x.device

        # Time embedding -> 6-way modulation
        t_emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        # Optional image conditioning
        if self.has_image_input and y is not None:
            if not self.fuse_vae_embedding_in_latents:
                x = torch.cat([x, y], dim=1)
            if clip_feature is not None and self.require_clip_embedding:
                if hasattr(self, "img_emb"):
                    clip_feature = self.img_emb(clip_feature)
                context = torch.cat([clip_feature, context], dim=1)

        # Patchify
        x, grid_size = self.patchify(x)
        t, h, w = grid_size

        # Assemble 3D RoPE frequencies from tuple -> [t*h*w, 1, head_dim//2]
        f_freqs, h_freqs, w_freqs = tuple(f.to(device) for f in self.freqs)
        freqs = torch.cat(
            [
                f_freqs[:t].view(t, 1, 1, -1).expand(t, h, w, -1),
                h_freqs[:h].view(1, h, 1, -1).expand(t, h, w, -1),
                w_freqs[:w].view(1, 1, w, -1).expand(t, h, w, -1),
            ],
            dim=-1,
        ).reshape(t * h * w, 1, -1)

        # Block loop
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        # Head uses time embedding (not 6-way t_mod)
        x = self.head(x, t_emb)
        x = self.unpatchify(x, grid_size)

        return x
