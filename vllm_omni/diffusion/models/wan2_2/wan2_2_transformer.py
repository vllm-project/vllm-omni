# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def apply_rotary_emb_wan(
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


class WanRotaryPosEmbed(nn.Module):
    """
    Rotary position embeddings for 3D video data (temporal + spatial dimensions).
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
        return freqs_cos.float(), freqs_sin.float()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class WanImageEmbedding(nn.Module):
    """Image embedding module for I2V tasks."""

    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len: int | None = None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    """Combined time, text, and image condition embeddings."""

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):
    """
    Optimized self-attention module using vLLM layers.
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
            query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)

        # Compute attention using unified attention layer
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class WanCrossAttention(nn.Module):
    """
    Optimized cross-attention module using vLLM layers.
    Handles both text cross-attention and optional image cross-attention (I2V).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.kv_inner_dim = head_dim * num_heads  # For cross-attention, K/V come from encoder

        # Query projection
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)

        # Separate K and V projections for cross-attention
        self.to_k = ReplicatedLinear(dim, self.kv_inner_dim, bias=True)
        self.to_v = ReplicatedLinear(dim, self.kv_inner_dim, bias=True)

        # QK normalization
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.kv_inner_dim, eps=eps)

        # Optional added KV projections for I2V (image embeddings)
        self.added_kv_proj_dim = added_kv_proj_dim
        if added_kv_proj_dim is not None:
            self.add_k_proj = ReplicatedLinear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = ReplicatedLinear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = RMSNorm(self.inner_dim, eps=eps)
        else:
            self.add_k_proj = None
            self.add_v_proj = None
            self.norm_added_k = None

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
        # Handle I2V case where encoder_hidden_states contains both image and text
        encoder_hidden_states_img = None
        if self.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Query projection
        query, _ = self.to_q(hidden_states)
        query = self.norm_q(query)

        # KV projection from encoder
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)
        key = self.norm_k(key)

        # Reshape for multi-head attention
        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        # I2V: Additional attention with image embeddings
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, _ = self.add_k_proj(encoder_hidden_states_img)
            value_img, _ = self.add_v_proj(encoder_hidden_states_img)
            key_img = self.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (self.num_heads, -1))
            value_img = value_img.unflatten(2, (self.num_heads, -1))

            hidden_states_img = self.attn(query, key_img, value_img)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Main cross-attention using unified attention layer
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Add image attention output if present
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class WanTransformerBlock(nn.Module):
    """
    Transformer block for Wan model with self-attention, cross-attention, and FFN.
    Uses scale-shift modulation from timestep embeddings.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
    ):
        super().__init__()

        head_dim = dim // num_heads

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
        )

        # 2. Cross-attention
        self.attn2 = WanCrossAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # Scale-shift table for modulation
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(nn.Module):
    """
    Optimized Wan Transformer model for video generation using vLLM layers.

    This is an optimized version of the diffusers WanTransformer3DModel that uses
    vLLM's efficient QKVParallelLinear and RMSNorm implementations.

    Sequence Parallelism:
        This model supports non-intrusive SP via _sp_plan. The plan specifies:
        - RoPE (cos/sin) splitting via rope module's split_output
        - hidden_states splitting at first transformer block input
        - Output gathering at proj_out layer

        The video sequence (flattened patches) is parallelized across GPUs.

        Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in diffusers.

    Args:
        patch_size: 3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
        num_attention_heads: Number of attention heads
        attention_head_dim: Dimension of each attention head
        in_channels: Number of input channels
        out_channels: Number of output channels
        text_dim: Input dimension for text embeddings
        freq_dim: Dimension for sinusoidal time embeddings
        ffn_dim: Intermediate dimension in feed-forward network
        num_layers: Number of transformer blocks
        cross_attn_norm: Enable cross-attention normalization
        eps: Epsilon value for normalization layers
        image_dim: Optional image embedding dimension for I2V
        added_kv_proj_dim: Optional added KV projection dimension for I2V
        rope_max_seq_len: Maximum sequence length for rotary embeddings
        pos_embed_seq_len: Optional position embedding sequence length
    """

    _repeated_blocks = ["WanTransformerBlock"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    # Sequence Parallelism for Wan (following diffusers' _cp_plan pattern)
    #
    # The _sp_plan specifies sharding/gathering at module boundaries:
    # - rope: Split both RoPE outputs (freqs_cos, freqs_sin) via split_output=True
    # - blocks.0: Split hidden_states input at the first transformer block
    # - proj_out: Gather outputs after the final projection layer
    #
    # Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
    _sp_plan = {
        # Shard RoPE embeddings after rope module computes them
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # freqs_cos [1, seq, 1, dim]
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # freqs_sin [1, seq, 1, dim]
        },
        # Shard hidden_states at first transformer block input
        # (after patch_embedding + flatten + transpose)
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),  # [B, seq, dim]
        },
        # Gather at proj_out (final linear projection before unpatchify)
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        # Store config for compatibility
        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "text_dim": text_dim,
                "freq_dim": freq_dim,
                "ffn_dim": ffn_dim,
                "num_layers": num_layers,
                "cross_attn_norm": cross_attn_norm,
                "eps": eps,
                "image_dim": image_dim,
                "added_kv_proj_dim": added_kv_proj_dim,
                "rope_max_seq_len": rope_max_seq_len,
                "pos_embed_seq_len": pos_embed_seq_len,
            },
        )()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = Conv3dLayer(
            in_channels=in_channels,
            out_channels=inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, eps, added_kv_proj_dim, cross_attn_norm)
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Compute RoPE embeddings (sharded by _sp_plan via split_output=True)
        rotary_emb = self.rope(hidden_states)

        # Patch embedding and flatten to sequence
        # (hidden_states is sharded at blocks.0 input by _sp_plan)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Handle timestep shape
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from a pretrained model, handling the mapping from
        separate Q/K/V projections to fused QKV projections for self-attention.

        Diffusers weight names:
        - blocks.N.attn1.to_q/to_k/to_v -> fused to blocks.N.attn1.to_qkv (self-attention)
        - blocks.N.attn2.to_q/to_k/to_v -> kept separate (cross-attention)
        - blocks.N.attn1.norm_q/norm_k -> QK normalization for self-attention

        Returns:
            Set of parameter names that were successfully loaded.
        """
        # Stacked params mapping for self-attention QKV fusion
        # Format: (param_name, shard_name, shard_id)
        # Note: Only fuse attn1 (self-attention), NOT attn2 (cross-attention)
        stacked_params_mapping = [
            # self-attention QKV fusion (attn1 only)
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
