# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
SkyReels-V3 Transformer Model Implementation.

This module implements the transformer architecture for SkyReels-V3,
a multimodal video generation model supporting:
- Image-to-Video (R2V)
- Video-to-Video (V2V)
- Audio-to-Video (A2V)
"""

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)


def apply_rotary_emb_skyreels(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensors.

    Args:
        hidden_states: Input tensor of shape [B, S, H, D]
        freqs_cos: Cosine frequencies
        freqs_sin: Sine frequencies

    Returns:
        Tensor with rotary embeddings applied
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class SkyReelsRotaryPosEmbed(nn.Module):
    """
    Rotary position embeddings for 3D video data (temporal + spatial dimensions).
    Adapted for SkyReels-V3 architecture.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # Split dimensions for temporal, height, width
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = self._get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype)
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    @staticmethod
    def _get_1d_rotary_pos_embed(
        dim: int,
        max_seq_len: int,
        theta: float,
        freqs_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D rotary position embeddings."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim))
        t = torch.arange(max_seq_len, dtype=freqs_dtype)
        freqs = torch.outer(t, freqs)
        # Repeat interleave for real representation
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1)
        return freqs_cos, freqs_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings.

        Args:
            hidden_states: Input tensor [B, S, H, D]
            t: Temporal dimension
            h: Height dimension
            w: Width dimension

        Returns:
            Tensor with rotary embeddings applied
        """
        # Get position indices
        seq_len = t * h * w
        freqs_cos = self.freqs_cos[:seq_len]  # type: ignore
        freqs_sin = self.freqs_sin[:seq_len]  # type: ignore

        return apply_rotary_emb_skyreels(hidden_states, freqs_cos, freqs_sin)


class SkyReelsSelfAttention(nn.Module):
    """
    Optimized self-attention module for SkyReels-V3 using vLLM layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # Fused QKV projection using vLLM's optimized layer
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=True,
            disable_tp=True,
        )

        # QK normalization using vLLM's RMSNorm
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=True),
                nn.Dropout(dropout),
            ]
        )

        # Unified attention layer
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Fused QKV projection
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape for multi-head attention
        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        # Apply rotary embeddings
        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = apply_rotary_emb_skyreels(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb_skyreels(key, freqs_cos, freqs_sin)

        # Compute attention using unified attention layer
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class SkyReelsCrossAttention(nn.Module):
    """
    Optimized cross-attention module for SkyReels-V3 using vLLM layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        text_dim: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
        cross_attn_norm: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.cross_attn_norm = cross_attn_norm

        # Query projection
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)

        # Key and Value projections for encoder
        self.to_kv = ReplicatedLinear(text_dim, self.inner_dim * 2, bias=True)

        # QK normalization
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

        # Optional encoder normalization
        if cross_attn_norm:
            self.norm_encoder = RMSNorm(text_dim, eps=eps)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=True),
                nn.Dropout(dropout),
            ]
        )

        # Unified attention layer
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize encoder if needed
        if self.cross_attn_norm:
            encoder_hidden_states = self.norm_encoder(encoder_hidden_states)

        # Project query
        query, _ = self.to_q(hidden_states)
        query = self.norm_q(query)

        # Project key and value
        kv, _ = self.to_kv(encoder_hidden_states)
        key, value = kv.chunk(2, dim=-1)
        key = self.norm_k(key)

        # Reshape for multi-head attention
        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        # Compute attention
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class SkyReelsTransformerBlock(nn.Module):
    """
    Transformer block for SkyReels-V3.
    Includes self-attention, cross-attention, and feed-forward layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        text_dim: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Self-attention
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn1 = SkyReelsSelfAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            eps=eps,
            dropout=dropout,
        )

        # Cross-attention
        self.norm2 = RMSNorm(dim, eps=eps)
        self.attn2 = SkyReelsCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            text_dim=text_dim,
            eps=eps,
            dropout=dropout,
            cross_attn_norm=cross_attn_norm,
        )

        # Feed-forward
        self.norm3 = RMSNorm(dim, eps=eps)
        ffn_dim = ffn_dim or dim * 4
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            mult=ffn_dim // dim,
            dropout=dropout,
            activation_fn="gelu-approximate",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            hidden_states: Input tensor [B, S, D]
            encoder_hidden_states: Encoder outputs for cross-attention [B, T, D_text]
            rotary_emb: Rotary position embeddings (cos, sin)

        Returns:
            Output tensor [B, S, D]
        """
        # Self-attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + attn_output

        # Cross-attention
        if encoder_hidden_states is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            hidden_states = hidden_states + attn_output

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        return hidden_states


class SkyReelsTransformer3DModel(nn.Module):
    """
    SkyReels-V3 3D Transformer Model for video generation.

    Supports multiple modalities:
    - Text-to-Video
    - Image-to-Video (R2V)
    - Video-to-Video (V2V)
    - Audio-to-Video (A2V)
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 16,
        out_channels: int | None = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        text_dim: int = 4096,
        ffn_dim: int | None = None,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        max_seq_len: int = 16384,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.added_kv_proj_dim = added_kv_proj_dim

        # Input projection
        self.proj_in = Conv3dLayer(
            in_channels=in_channels,
            out_channels=self.inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        # Timestep embedding
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=self.inner_dim,
        )

        # Text projection
        self.text_proj = PixArtAlphaTextProjection(
            in_features=text_dim,
            hidden_size=self.inner_dim,
        )

        # Optional image conditioning projection
        if image_dim is not None:
            self.image_proj = ReplicatedLinear(
                image_dim,
                self.inner_dim,
                bias=True,
            )

        # Rotary position embeddings
        self.pos_embed = SkyReelsRotaryPosEmbed(
            attention_head_dim=attention_head_dim,
            patch_size=patch_size,
            max_seq_len=max_seq_len,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SkyReelsTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    text_dim=self.inner_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.norm_out = RMSNorm(self.inner_dim, eps=eps)
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            self.out_channels * patch_size[0] * patch_size[1] * patch_size[2],
            bias=True,
        )

        # Sequence parallel plan (for distributed training)
        self._sp_plan = {
            "input": SequenceParallelInput(split_dim=1, expected_dims=3),
            "output": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        image_hidden_states: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass through the SkyReels-V3 transformer.

        Args:
            hidden_states: Latent video tensor [B, C, T, H, W]
            timestep: Diffusion timestep [B]
            encoder_hidden_states: Text embeddings [B, seq_len, text_dim]
            encoder_attention_mask: Attention mask for text
            image_hidden_states: Optional image conditioning [B, image_dim]
            return_dict: Whether to return a dict or tuple

        Returns:
            Transformer output
        """
        batch_size, channels, num_frames, height, width = hidden_states.shape

        # Project input
        hidden_states = self.proj_in(hidden_states)  # [B, inner_dim, T', H', W']
        
        # Reshape to sequence
        t_out, h_out, w_out = (
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        )
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, T'*H'*W', inner_dim]

        # Timestep embedding
        timestep_emb = self.time_proj(timestep)
        timestep_emb = self.time_embedding(timestep_emb)  # [B, inner_dim]
        timestep_emb = timestep_emb.unsqueeze(1)  # [B, 1, inner_dim]

        # Text projection
        encoder_hidden_states = self.text_proj(encoder_hidden_states)  # [B, seq_len, inner_dim]

        # Add timestep to encoder hidden states
        encoder_hidden_states = torch.cat([timestep_emb, encoder_hidden_states], dim=1)

        # Optional image conditioning
        if image_hidden_states is not None and self.image_dim is not None:
            image_emb = self.image_proj(image_hidden_states).unsqueeze(1)  # [B, 1, inner_dim]
            encoder_hidden_states = torch.cat([encoder_hidden_states, image_emb], dim=1)

        # Get rotary position embeddings
        seq_len = t_out * h_out * w_out
        freqs_cos = self.pos_embed.freqs_cos[:seq_len]  # type: ignore
        freqs_sin = self.pos_embed.freqs_sin[:seq_len]  # type: ignore
        rotary_emb = (freqs_cos, freqs_sin)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb=rotary_emb,
            )

        # Output projection
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Reshape back to video format
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            self.out_channels,
            t_out,
            h_out,
            w_out,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
        )
        hidden_states = hidden_states.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(
            batch_size, self.out_channels, num_frames, height, width
        )

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load model weights."""
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            default_weight_loader(param, loaded_weight)

