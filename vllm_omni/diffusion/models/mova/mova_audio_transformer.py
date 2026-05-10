# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/OpenMOSS/MOVA
"""
MOVA Audio Diffusion Transformer.

Based on WanAudioModel architecture from MOVA upstream.
Shares DiTBlock with the video transformer but uses 1D patch embedding
and 1D RoPE for audio sequences.
"""

from typing import Literal

import torch
import torch.nn as nn
from vllm.logger import init_logger

from .mova_video_transformer import MovaDiTBlock, MovaHead, precompute_freqs_cis, sinusoidal_embedding_1d

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Audio-specific RoPE
# ---------------------------------------------------------------------------


def precompute_freqs_cis_1d_oobleck(
    dim: int,
    end: int = 16384,
    theta: float = 10000.0,
    base_tps: float = 4.0,
    target_tps: float = 44100.0 / 2048.0,
) -> torch.Tensor:
    """
    Audio RoPE for 'oobleck' VAE type.
    Applies time-scale factor and sets H/W dims to ones (no spatial encoding).
    """
    s = base_tps / target_tps
    t_dim = dim - 2 * (dim // 3)
    h_dim = dim // 3
    w_dim = dim // 3

    # Scale temporal frequencies
    t = torch.arange(end, dtype=torch.float32) * s
    freqs_scaled = 1.0 / (theta ** (torch.arange(0, t_dim, 2, dtype=torch.float32) / t_dim))
    freqs_t = torch.polar(
        torch.ones(end, len(freqs_scaled)),
        torch.outer(t, freqs_scaled),
    )

    # H/W dimensions: ones (no positional encoding)
    freqs_h = torch.ones(end, h_dim // 2, dtype=torch.cfloat)
    freqs_w = torch.ones(end, w_dim // 2, dtype=torch.cfloat)

    return freqs_t, freqs_h, freqs_w


def precompute_freqs_cis_1d_dac(
    dim: int,
    end: int = 16384,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Audio RoPE for 'dac' VAE type.
    Returns tuple of 3 tensors (matches upstream storage format).
    """
    freqs = precompute_freqs_cis(dim, end, theta)
    return tuple(freqs.chunk(3, dim=1))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MovaAudioTransformer(nn.Module):
    """
    MOVA Audio Diffusion Transformer.

    Same block architecture as the video transformer (MovaDiTBlock),
    but with 1D patch embedding and 1D RoPE for audio sequences.
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
        patch_size: tuple[int, ...],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        separated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
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
        self.vae_type = vae_type
        self.separated_timestep = separated_timestep
        self.__dict__[legacy_timestep_key] = separated_timestep

        # Patch embedding: Conv1d to project from in_dim to dim
        # Checkpoint has: patch_embedding.weight [dim, in_dim, 1]
        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=1)

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Transformer blocks (same DiTBlock as video)
        self.blocks = nn.ModuleList(
            [MovaDiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )

        # Output head
        self.head = MovaHead(dim, out_dim, patch_size, eps)

        # Precompute 1D RoPE frequencies
        head_dim = dim // num_heads
        if vae_type == "oobleck":
            self.freqs = precompute_freqs_cis_1d_oobleck(head_dim, 16384)
        else:
            self.freqs = precompute_freqs_cis_1d_dac(head_dim, 16384)

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int]]:
        """Project audio [B, in_dim, L] -> [B, L, dim] via Conv1d patch embedding."""
        x = self.patch_embedding(x)  # [B, in_dim, L] -> [B, dim, L]
        grid_size = (x.shape[2],)
        x = x.transpose(1, 2)  # [B, dim, L] -> [B, L, dim]
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int]) -> torch.Tensor:
        """Convert [B, L, C*patch] -> [B, C, L*patch]."""
        c = self.out_dim
        p = self.patch_size[0]
        length = grid_size[0]
        x = x[:, :length]
        x = x.view(-1, length, p, c)
        x = x.permute(0, 3, 1, 2).reshape(-1, c, length * p)
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
        Full forward pass for standalone audio generation.
        For dual-tower mode, the pipeline drives the block loop directly.
        """
        device = x.device

        # Time embedding -> 6-way modulation
        t_emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        # Patchify (1D)
        x, grid_size = self.patchify(x)
        f = grid_size[0]

        # Assemble 1D RoPE from tuple -> [f, 1, head_dim//2]
        af_parts = tuple(freq.to(device) for freq in self.freqs)
        freqs = torch.cat([af_parts[0][:f], af_parts[1][:f], af_parts[2][:f]], dim=-1).reshape(f, 1, -1)

        # Block loop
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        # Head uses time embedding (not 6-way t_mod)
        x = self.head(x, t_emb)
        x = self.unpatchify(x, grid_size)

        return x
