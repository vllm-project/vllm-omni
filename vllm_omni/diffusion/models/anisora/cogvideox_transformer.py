# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CogVideoX Transformer using vLLM-Omni parallelized layers.

Replaces the stock diffusers CogVideoXTransformer3DModel with an optimized
version that uses QKVParallelLinear, RowParallelLinear, ColumnParallelLinear,
and vLLM-Omni's Attention backend for tensor/sequence parallelism support.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNorm
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config container — exposes the same attributes the pipeline reads via
# ``self.transformer.config.<attr>``
# ---------------------------------------------------------------------------
class _CogVideoXTransformerConfig:
    """Lightweight config object that mirrors the diffusers ConfigMixin API."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Column-parallel GELU (same helper as in flux_transformer.py)
# ---------------------------------------------------------------------------
class ColumnParallelApproxGELU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


# ---------------------------------------------------------------------------
# Feed-forward (ColumnParallel GELU → RowParallel)
# ---------------------------------------------------------------------------
class CogVideoXFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        # nn.ModuleList indices kept identical to diffusers for weight name
        # compatibility: net.0 = GELU proj, net.1 = (identity/dropout), net.2 = out proj.
        self.net = nn.ModuleList(
            [
                ColumnParallelApproxGELU(
                    dim,
                    inner_dim,
                    approximate="tanh",
                    bias=bias,
                    quant_config=quant_config,
                ),
                nn.Identity(),
                RowParallelLinear(
                    inner_dim,
                    dim,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


# ---------------------------------------------------------------------------
# Adaptive LayerNorm Zero — keeps diffusers naming for weight loading
# ---------------------------------------------------------------------------
class CogVideoXLayerNormZero(nn.Module):
    """Adaptive LN that produces 6 modulation vectors (shift/scale/gate × 2 streams)."""

    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ):
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


# ---------------------------------------------------------------------------
# CogVideoX Attention (fused QKV with RoPE on image tokens only)
# ---------------------------------------------------------------------------
class CogVideoXAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        bias: bool = True,
        out_bias: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        parallel_config: DiffusionParallelConfig | None = None,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.parallel_config = parallel_config
        inner_dim = num_attention_heads * attention_head_dim

        # Fused Q/K/V projection
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=attention_head_dim,
            total_num_heads=num_attention_heads,
            bias=bias,
            quant_config=quant_config,
        )

        # Q/K normalization (LayerNorm, matching diffusers qk_norm="layer_norm")
        if qk_norm:
            self.norm_q = nn.LayerNorm(attention_head_dim, eps=eps, elementwise_affine=True)
            self.norm_k = nn.LayerNorm(attention_head_dim, eps=eps, elementwise_affine=True)
        else:
            self.norm_q = None
            self.norm_k = None

        # Rotary embeddings (interleaved style, same as diffusers CogVideoX)
        self.rope = RotaryEmbedding(is_neox_style=False)

        # vLLM-Omni attention backend
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

        # Output projection
        self.to_out = RowParallelLinear(
            inner_dim,
            query_dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)

        # Concatenate text + image (CogVideoX does joint self-attention)
        combined = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Fused QKV projection
        qkv, _ = self.to_qkv(combined)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to [B, S, num_heads, head_dim]
        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        # Q/K normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE only to image tokens (not text)
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            # vllm's RotaryEmbedding expects half-dim cos/sin (it doubles internally)
            # diffusers' get_3d_rotary_pos_embed returns full-dim with repeat_interleave(2):
            # (c1, c1, c2, c2, ...) — take every other element to get (c1, c2, ...)
            cos = cos[..., ::2]
            sin = sin[..., ::2]
            # Split text and image portions
            q_text, q_image = query[:, :text_seq_length], query[:, text_seq_length:]
            k_text, k_image = key[:, :text_seq_length], key[:, text_seq_length:]
            # Apply RoPE to image tokens only
            q_image = self.rope(q_image, cos, sin)
            k_image = self.rope(k_image, cos, sin)
            # Recombine
            query = torch.cat([q_text, q_image], dim=1)
            key = torch.cat([k_text, k_image], dim=1)

        # Attention — SP-aware: text tokens are replicated (not sharded) across SP ranks,
        # so pass them as joint (front) context while image tokens are the sharded sequence.
        sp_size = self.parallel_config.sequence_parallel_size if self.parallel_config else None
        use_sp_joint = (
            sp_size is not None
            and sp_size > 1
            and is_forward_context_available()
            and not get_forward_context().split_text_embed_in_sp
        )
        if use_sp_joint:
            # SP-aware attention: image shard attends to [text, image_full] via Ulysses all-to-all.
            # Text tokens are replicated (not sharded) and passed as joint context.
            # Ulysses gathers the full image sequence before attention, then scatters back,
            # so each rank computes full-sequence attention for its head/sequence shard.
            # Output: [B, text + image_shard, H, D] — text portion gathered via AllGather on heads
            # (identical on all ranks), image portion via reverse AllToAll (shard for this rank).
            hidden_states = self.attn(
                query[:, text_seq_length:],
                key[:, text_seq_length:],
                value[:, text_seq_length:],
                AttentionMetadata(
                    joint_query=query[:, :text_seq_length],
                    joint_key=key[:, :text_seq_length],
                    joint_value=value[:, :text_seq_length],
                    joint_strategy="front",
                ),
            )
        else:
            hidden_states = self.attn(query, key, value)

        # Flatten heads and project output
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out(hidden_states)

        # Split back into text and image streams
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# CogVideoX Transformer Block
# ---------------------------------------------------------------------------
class CogVideoXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        ff_inner_dim: int | None = None,
        bias: bool = True,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        quant_config: QuantizationConfig | None = None,
        parallel_config: DiffusionParallelConfig | None = None,
    ):
        super().__init__()

        # 1. Attention path
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn1 = CogVideoXAttention(
            query_dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            quant_config=quant_config,
            parallel_config=parallel_config,
        )

        # 2. FFN path
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = CogVideoXFeedForward(
            dim=dim,
            inner_dim=ff_inner_dim,
            bias=bias,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)

        # Attention with AdaLN modulation
        norm_hidden, norm_enc_hidden, gate_msa, enc_gate_msa = self.norm1(hidden_states, encoder_hidden_states, temb)
        attn_hidden, attn_enc_hidden = self.attn1(norm_hidden, norm_enc_hidden, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate_msa * attn_hidden
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_enc_hidden

        # FFN with AdaLN modulation
        norm_hidden, norm_enc_hidden, gate_ff, enc_gate_ff = self.norm2(hidden_states, encoder_hidden_states, temb)
        # Concatenate for joint FFN (text first, matching diffusers)
        combined = torch.cat([norm_enc_hidden, norm_hidden], dim=1)
        ff_output = self.ff(combined)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# CogVideoX Transformer 3D Model (main model)
# ---------------------------------------------------------------------------
class CogVideoXTransformer3DModel(nn.Module):
    """
    Optimized CogVideoX Transformer using vLLM-Omni parallelized layers.

    Uses QKVParallelLinear for fused Q/K/V projections, ColumnParallelLinear
    and RowParallelLinear for FFN, and vLLM-Omni's Attention backend.

    Sequence Parallelism:
        Supports non-intrusive SP via _sp_plan (Ulysses or hybrid Ulysses+Ring).
        The plan splits image tokens across SP ranks while keeping text tokens
        (encoder_hidden_states) replicated. The attention uses joint_strategy="front"
        to combine replicated text context with sharded image tokens correctly.

        - image_rotary_emb: split at model entry (root ""), dim=0
        - hidden_states (image tokens): split at transformer_blocks.0, dim=1
        - proj_out output: gathered back to full sequence for unpatchify
    """

    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    # Sequence Parallelism for CogVideoX (following diffusers' _cp_plan pattern).
    #
    # CogVideoX uses joint text+image self-attention in every block. For SP, only
    # image tokens are sharded; text tokens stay replicated on all ranks and are
    # passed as AttentionMetadata joint context (joint_strategy="front").
    #
    # - "": split image_rotary_emb (tuple of [img_seq, head_dim] tensors) before patch_embed
    # - "transformer_blocks.0": split hidden_states (image tokens [B, img_seq, D]) at first block
    # - "proj_out": gather image tokens after the final linear projection (before unpatchify)
    _sp_plan = {
        # Split both RoPE tensors (cos, sin) along the sequence dim at model entry.
        # image_rotary_emb is (cos, sin), each shaped [img_seq, head_dim].
        "": {
            "image_rotary_emb": [
                SequenceParallelInput(split_dim=0, expected_dims=2, auto_pad=True),
                SequenceParallelInput(split_dim=0, expected_dims=2, auto_pad=True),
            ],
        },
        # Split image hidden_states along sequence dim at the first block's input.
        # After patch_embed + manual split, hidden_states is [B, img_seq, D].
        "transformer_blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        # Gather image tokens after proj_out; unpatchify then operates on the full sequence.
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int | None = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: int | None = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: int | None = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # Store parallel config for SP-aware attention
        self.parallel_config = od_config.parallel_config

        # Store config for pipeline access (self.transformer.config.*)
        self.config = _CogVideoXTransformerConfig(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
        )

        # 1. Patch embedding (kept from diffusers — runs once, not a bottleneck)
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Timestep embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(ofs_embed_dim, ofs_embed_dim, timestep_activation_fn)

        # 3. Transformer blocks (optimized with vLLM-Omni layers)
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    quant_config=quant_config,
                    parallel_config=self.parallel_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks (kept from diffusers — runs once)
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            output_dim = patch_size * patch_size * out_channels
        else:
            output_dim = patch_size * patch_size * patch_size_t * out_channels
        self.proj_out = nn.Linear(inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | int | float,
        timestep_cond: torch.Tensor | None = None,
        ofs: torch.LongTensor | int | float | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Timestep embedding
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_final(hidden_states)

        # 4. Output
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map diffusers separate Q/K/V weights to our fused QKV
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            # Handle stacked Q/K/V parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    break
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Diffusers uses attn1.to_out.0 (ModuleList); we use attn1.to_out (direct)
                if ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")

                if lookup_name not in params_dict:
                    logger.debug("Skipping weight %s (not in model)", original_name)
                    continue

                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
