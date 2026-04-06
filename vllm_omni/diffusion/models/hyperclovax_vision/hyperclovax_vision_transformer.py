# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NAVER Cloud Corp. vision-decoder-api

"""
HyperCLOVAX Vision Transformer for vision token to image generation.

This module implements the VisionTransformer diffusion model that converts
vision token embeddings to latent representations for image generation.
"""

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .layers import (
    EmbedAND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


class HyperCLOVAXVisionTransformer2DModel(nn.Module):
    """
    Vision Transformer for vision token to image generation.

    This transformer processes vision token embeddings concatenated with
    noisy latents to predict noise for the diffusion process.

    Architecture:
        - Input projection: (in_channels + context_in_dim) -> hidden_size
        - Time embedding: 256 -> hidden_size
        - Vector embedding: context_in_dim -> hidden_size
        - Position embedding: EmbedAND with 3D axes
        - Single stream blocks: 35 parallel attention+MLP blocks
        - Output layer: hidden_size -> out_channels

    Args:
        od_config: OmniDiffusionConfig containing model configuration
        in_channels: Number of latent channels (default: 16)
        vec_in_dim: Vision pooler output dimension (default: 1536)
        context_in_dim: Vision hidden state dimension (default: 1536)
        hidden_size: Transformer hidden dimension (default: 1920)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        num_heads: Number of attention heads (default: 24)
        depth_single_blocks: Number of single stream blocks (default: 35)
        axes_dim: Position embedding axes dimensions (default: [8, 36, 36])
        theta: RoPE theta parameter (default: 10000)
        use_patchify: Whether to use 2x2 patchification (default: False)
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        in_channels: int = 16,
        vec_in_dim: int = 1536,
        context_in_dim: int = 1536,
        hidden_size: int = 1920,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth_single_blocks: int = 35,
        axes_dim: tuple[int, int, int] = (8, 36, 36),
        theta: int = 10_000,
        use_patchify: bool = False,
    ):
        super().__init__()

        self.od_config = od_config
        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_patchify = use_patchify
        self.depth_single_blocks = depth_single_blocks

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        axes_dim_list = list(axes_dim)
        if sum(axes_dim_list) != pe_dim:
            raise ValueError(f"Got {axes_dim_list} but expected positional dim {pe_dim}")

        # Position embedding
        self.pe_embedder = EmbedAND(dim=pe_dim, theta=theta, axes_dim=axes_dim_list)

        # Input projections
        self.img_in = nn.Linear(in_channels + context_in_dim, hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)

        # Single stream blocks
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
        )

        # Output layer
        self.final_layer = LastLayer(hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            img: Input tensor (B, L, in_channels + context_in_dim)
                 Concatenation of noisy latents and vision spatial features
            img_ids: Position IDs tensor (B, L, 3)
            timesteps: Sigma/timestep tensor (B,) in [0, 1]
            y: Vision pooler output tensor (B, vec_in_dim)

        Returns:
            Output tensor (B, L, out_channels) - predicted noise
        """
        if img.ndim != 3:
            raise ValueError("Input img tensor must have 3 dimensions.")

        # Project input
        img = self.img_in(img)

        # Time and vector embedding
        vec = self.time_in(
            timestep_embedding(timesteps, 256).to(dtype=self.time_in.in_layer.weight.dtype, device=img.device)
        )
        vec = vec + self.vector_in(y)

        # Position embedding
        pe = self.pe_embedder(img_ids)

        # Single stream blocks
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)

        # Final projection
        img = self.final_layer(img, vec)

        return img
