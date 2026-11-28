# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Bria Fibo-2.0 Dev Transformer, ported from the HuggingFace diffusers
# implementation of BriaFibo2Transformer2DModel (diffusers PR #14991).
# Architecture: Z-Image-style single-stream DiT with
#   - a 2-layer noise refiner over the (patchified) noisy image tokens,
#   - a perceiver resampler that squeezes the Qwen3-VL caption features
#     (6 bundles x 3 layers of hidden states, concat to cap_feat_dim) into
#     gist tokens,
#   - 44 main blocks; the blocks at `injection_layer_ids` additionally
#     cross-attend to their own caption bundle,
#   - gist tokens are placed on RoPE plane 1, the image on plane
#     `max_num_gist_tokens + 1`, context (edit) images on later planes.

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_qk_norm_rope,
    fused_qk_norm_rope_min_tokens,
    pack_qk_norm_rope_table,
)
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk
from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)

ADALN_EMBED_DIM = 256


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _restore_linear_output_shape(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    if input_.dim() > 2 and output.dim() == 2:
        token_count = math.prod(input_.shape[:-1])
        if output.shape[0] == token_count:
            return output.reshape(*input_.shape[:-1], output.shape[-1])
    return output


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Per-token selection between the noisy/clean modulation branches."""
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


def compute_gist_count(
    text_lengths: list[int], min_gist: int, max_gist: int, gist_step: int, min_text_len: int, max_text_len: int
) -> list[int]:
    """Number of gist tokens for each prompt, from its text length in tokens.

    Lengths between `min_text_len` and `max_text_len` map linearly onto
    `[min_gist, max_gist]`, rounded up to a multiple of `gist_step`.
    """
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        quantized = int(math.ceil(raw / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, quantized)))
    return counts


def _packed_qk_norm_rope_table(
    cos: torch.Tensor, sin: torch.Tensor, dtype: torch.dtype, batch_size: int | None = None
) -> torch.Tensor | None:
    """Repeat cos[0]/sin[0] for all batches, matching RotaryEmbedding.
    Fibo's positions are batch-independent, so cos/sin stay batch-1 and the
    real batch must be passed explicitly. ``None`` below the token threshold
    or outside the CUDA-matching contract (BF16, D=128) keeps the eager
    norm+rope chain."""
    num_batch = batch_size if batch_size is not None else cos.shape[0]
    if num_batch * cos.shape[1] < fused_qk_norm_rope_min_tokens(0):
        return None
    return pack_qk_norm_rope_table(cos[0], sin[0], num_batch, dtype=dtype, cuda_rms_norm=True)


def _kv_attn_metadata(attn_mask: torch.Tensor | None) -> AttentionMetadata | None:
    """(B, kv_len) bool validity mask -> dense 4D SDPA mask, diffusers layout."""
    if attn_mask is None:
        return None
    return AttentionMetadata(attn_mask=attn_mask[:, None, None, :])


class TimestepEmbedder(nn.Module):
    def __init__(
        self, out_size, mid_size=None, frequency_embedding_size=256, quant_config: "QuantizationConfig | None" = None
    ):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        # Time embedding MLP is kept full precision (quant_config=None) —
        # small layers that feed adaLN; precision-sensitive (see #2728).
        self.mlp = nn.Sequential(
            ReplicatedLinear(
                frequency_embedding_size,
                mid_size,
                bias=True,
                quant_config=None,
                return_bias=False,
            ),
            nn.SiLU(),
            ReplicatedLinear(
                mid_size,
                out_size,
                bias=True,
                quant_config=None,
                return_bias=False,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].bias.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return _restore_linear_output_shape(t_emb, t_freq)


class Fibo2SelfAttention(nn.Module):
    """Self-attention over one sequence, with QK RMSNorm and (optionally) RoPE.

    Mirrors ZImageAttention: `to_qkv` is fused (same input dim for q/k/v); the
    output projection is row-parallel so TP all-reduces after attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-5,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_heads

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_qkv"),
        )
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, "to_out.0"),
                )
            ]
        )
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        qk_norm_rope_table: torch.Tensor | None = None,
    ):
        qkv, _ = self.to_qkv(hidden_states)
        qkv = _restore_linear_output_shape(qkv, hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        query, key, value = qkv.split([q_size, q_size, q_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        # The fused op reproduces the CUDA RMSNorm reduction of norm_q/norm_k
        # plus the interleaved rotation; eager mode is bit-identical to the
        # explicit chain below (same contract as Z-Image).
        if (
            qk_norm_rope_table is not None
            and cos is not None
            and sin is not None
            and self.norm_q.weight.dtype == self.norm_k.weight.dtype == query.dtype
            and _fused_cuda_supported(query, key, self.head_dim, qk_norm_rope_table.shape[-1], interleaved=True)
        ):
            batch_size, seq_len, num_heads, head_dim = query.shape
            num_kv_heads = key.shape[2]
            query, key = fused_qk_norm_rope(
                query.reshape(batch_size * seq_len, num_heads, head_dim),
                key.reshape(batch_size * seq_len, num_kv_heads, head_dim),
                self.norm_q.weight,
                self.norm_k.weight,
                qk_norm_rope_table,
                self.norm_q.variance_epsilon,
                interleaved=True,
                cuda_rms_norm=True,
            )
            query = query.view(batch_size, seq_len, num_heads, head_dim)
            key = key.view(batch_size, seq_len, num_kv_heads, head_dim)
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)
            if cos is not None and sin is not None:
                query, key = apply_rope_to_qk(self.rope, query, key, (cos, sin))

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        hidden_states = self.attn(query, key, value, _kv_attn_metadata(attn_mask))

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        to_out_input = hidden_states
        hidden_states = self.to_out[0](hidden_states)
        return _restore_linear_output_shape(hidden_states, to_out_input)


class Fibo2CrossAttention(nn.Module):
    """Cross-attention: queries from the token stream, keys/values from the
    caption features (`cap_feat_dim`-wide, so q and kv projections cannot be
    fused). No RoPE — matches the diffusers processor, which only applies
    rotary embeddings when freqs are given.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int,
        eps: float = 1e-5,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = query_dim // num_heads
        self.total_num_heads = num_heads
        from vllm.distributed import get_tensor_model_parallel_world_size

        self.num_heads = num_heads // get_tensor_model_parallel_world_size()

        self.to_q = ColumnParallelLinear(
            query_dim,
            query_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_q"),
        )
        self.to_k = ColumnParallelLinear(
            kv_dim,
            query_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_k"),
        )
        self.to_v = ColumnParallelLinear(
            kv_dim,
            query_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_v"),
        )
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    query_dim,
                    query_dim,
                    bias=False,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, "to_out.0"),
                )
            ]
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        # These projections pass return_bias=False, so they return plain tensors.
        query = self.to_q(hidden_states)
        query = _restore_linear_output_shape(query, hidden_states)
        key = self.to_k(encoder_hidden_states)
        key = _restore_linear_output_shape(key, encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        value = _restore_linear_output_shape(value, encoder_hidden_states)

        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        hidden_states = self.attn(query, key, value, _kv_attn_metadata(attn_mask))

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        to_out_input = hidden_states
        hidden_states = self.to_out[0](hidden_states)
        return _restore_linear_output_shape(hidden_states, to_out_input)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim] * 2,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "w13"),
        )
        self.act = SiluAndMul()
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "w2"),
        )

    def forward(self, x):
        hidden_states = self.w13(x)
        hidden_states = _restore_linear_output_shape(hidden_states, x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.w2(hidden_states)
        return _restore_linear_output_shape(hidden_states, x)


def _ffn_hidden_dim(dim: int) -> int:
    # Same rounding as diffusers: (dim * 8 // 3) // 64 * 64 (3584 -> 9536).
    return (dim * 8 // 3) // 64 * 64


class Fibo2TransformerBlock(nn.Module):
    """Z-Image's block. With `cross_attention_dim` it becomes an injection
    block that also reads the text encoder; with `noise_mask` inputs it does
    per-token (noisy/clean) modulation for the edit path.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        norm_eps: float,
        cross_attention_dim: int | None = None,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = Fibo2SelfAttention(
            dim=dim,
            num_heads=n_heads,
            eps=norm_eps,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attention"),
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=_ffn_hidden_dim(dim),
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "feed_forward"),
        )

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        # Modulation linear is kept at full precision (quant_config=None)
        # — it produces scale/gate values that are precision-sensitive
        # (see #2728, mirrors OmniGen2 fix).
        self.adaLN_modulation = nn.Sequential(
            ReplicatedLinear(
                min(dim, ADALN_EMBED_DIM),
                4 * dim,
                bias=True,
                quant_config=None,
                return_bias=False,
            ),
        )

        if cross_attention_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Fibo2CrossAttention(
                query_dim=dim,
                kv_dim=cross_attention_dim,
                num_heads=n_heads,
                eps=norm_eps,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "cross_attention"),
            )
            self.cross_attn_norm2 = RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        qk_norm_rope_table: torch.Tensor | None = None,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation: different modulation for noisy/clean tokens
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)

            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

            gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
            gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

            scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
            scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

            scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
            scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
            gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
            gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
        else:
            # Global modulation: same modulation for all tokens
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # Attention block
        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            cos=cos,
            sin=sin,
            attn_mask=attn_mask,
            qk_norm_rope_table=qk_norm_rope_table,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        # Gated cross-attention to the text encoder (injection blocks only)
        if self.cross_attention is not None:
            cross_out = self.cross_attention(
                self.cross_attn_norm1(x),
                encoder_hidden_states=encoder_hidden_states,
                attn_mask=encoder_attention_mask,
            )
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(cross_out)

        # FFN block
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))

        return x


class Fibo2PerceiverLayer(nn.Module):
    """Gist tokens read the text encoder (cross-attention), then each other
    (self-attention), then an FFN. No RoPE anywhere, matching diffusers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cross_attention_dim: int,
        norm_eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Fibo2CrossAttention(
            query_dim=dim,
            kv_dim=cross_attention_dim,
            num_heads=n_heads,
            eps=norm_eps,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "cross_attention"),
        )
        self.self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.self_attention = Fibo2SelfAttention(
            dim=dim,
            num_heads=n_heads,
            eps=norm_eps,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "self_attention"),
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=_ffn_hidden_dim(dim),
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "ffn"),
        )

    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        gist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = latents + self.cross_attention(
            self.cross_attn_norm(latents),
            encoder_hidden_states=encoder_hidden_states,
            attn_mask=encoder_attention_mask,
        )
        latents = latents + self.self_attention(self.self_attn_norm(latents), attn_mask=gist_mask)
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class Fibo2PerceiverResampler(nn.Module):
    """Squeezes the text encoder's hidden states into gist tokens."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cross_attention_dim: int,
        num_queries: int,
        num_layers: int,
        norm_eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                Fibo2PerceiverLayer(
                    dim,
                    n_heads,
                    cross_attention_dim,
                    norm_eps,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, f"layers.{i}"),
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        num_gist_tokens: int,
        encoder_attention_mask: torch.Tensor | None = None,
        gist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states.shape[0]
        latents = self.latent_queries[:num_gist_tokens].unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, gist_mask)
        return self.norm_out(latents)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, quant_config: "QuantizationConfig | None" = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Final output projection and its modulation are precision-sensitive
        # (produce the output latent); keep at full precision (see #2728).
        self.linear = ReplicatedLinear(
            hidden_size,
            out_channels,
            bias=True,
            quant_config=None,
            return_bias=False,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                min(hidden_size, ADALN_EMBED_DIM),
                hidden_size,
                bias=True,
                quant_config=None,
                return_bias=False,
            ),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None) -> torch.Tensor:
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            # Original global modulation
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


class RopeEmbedder:
    """cos/sin twin of diffusers' complex-exponential RopeEmbedder. The complex
    multiplication view_as_complex(x) * freqs_cis is numerically equivalent to
    the interleaved rotate-half form applied here (verified against the
    diffusers baseline)."""

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (2048, 512, 512),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.cos_cached = None
        self.sin_cached = None

    @staticmethod
    def precompute_freqs(dim: tuple[int, ...], end: tuple[int, ...], theta: float = 256.0):
        with torch.device("cpu"):
            cos_list = []
            sin_list = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                cos_list.append(torch.cos(freqs))
                sin_list.append(torch.sin(freqs))
            return cos_list, sin_list

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.cos_cached is None:
            self.cos_cached, self.sin_cached = self.precompute_freqs(self.axes_dims, self.axes_lens, theta=self.theta)
            self.cos_cached = [c.to(device) for c in self.cos_cached]
            self.sin_cached = [s.to(device) for s in self.sin_cached]
        else:
            # Ensure cached tensors are on the same device as ids
            if self.cos_cached[0].device != device:
                self.cos_cached = [c.to(device) for c in self.cos_cached]
                self.sin_cached = [s.to(device) for s in self.sin_cached]

        cos_result = []
        sin_result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            cos_result.append(self.cos_cached[i][index])
            sin_result.append(self.sin_cached[i][index])

        return torch.cat(cos_result, dim=-1), torch.cat(sin_result, dim=-1)


class Fibo2Transformer2DModel(CachedTransformer):
    """Bria Fibo-2.0 Dev Transformer for image generation/editing.

    Sequence layout (text-to-image, no batch padding): gist tokens first (RoPE
    plane 1), then image tokens (plane max_num_gist_tokens + 1). With batch
    padding of the gist, the image comes first and a bool attention mask marks
    the real gist tokens. Context (edit) images join after the image, each on
    its own plane, and use per-token noisy/clean modulation.
    """

    # Fused projections and their checkpoint shard names (mirrors
    # stacked_params_mapping in load_weights).
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "w13": ["w1", "w3"],
    }

    _repeated_blocks = ["Fibo2TransformerBlock"]

    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        dim: int = 3584,
        n_layers: int = 44,
        n_refiner_layers: int = 2,
        n_heads: int = 28,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 7680,
        injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38),
        perceiver_num_layers: int = 2,
        min_num_gist_tokens: int = 256,
        max_num_gist_tokens: int = 512,
        gist_step: int = 64,
        gist_min_text_len: int = 256,
        gist_max_text_len: int = 2048,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (2048, 512, 512),
        quant_config: "QuantizationConfig | None" = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # NOTE: `DiffusersPipelineLoader.load_model()` initializes this module
        # under `set_default_torch_dtype(od_config.dtype)`, so using the current
        # default dtype keeps `self.dtype` consistent with the actually loaded
        # weights. This dtype is used by the pipeline to cast inputs.
        self.dtype = torch.get_default_dtype()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.norm_eps = norm_eps
        self.cap_feat_dim = cap_feat_dim
        self.injection_layer_ids = tuple(injection_layer_ids)
        self.min_num_gist_tokens = min_num_gist_tokens
        self.max_num_gist_tokens = max_num_gist_tokens
        self.gist_step = gist_step
        self.gist_min_text_len = gist_min_text_len
        self.gist_max_text_len = gist_max_text_len
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        ffn_hidden_dim = _ffn_hidden_dim(dim)
        logger.info_once(
            "Fibo-2.0 transformer init: dim=%d n_heads=%d n_layers=%d ffn_hidden_dim=%d "
            "injection_layer_ids=%s cap_feat_dim=%d gist=[%d,%d] axes_dims=%s axes_lens=%s",
            dim,
            n_heads,
            n_layers,
            ffn_hidden_dim,
            self.injection_layer_ids,
            cap_feat_dim,
            min_num_gist_tokens,
            max_num_gist_tokens,
            tuple(axes_dims),
            tuple(axes_lens),
        )

        self.x_embedder = ReplicatedLinear(
            patch_size * patch_size * in_channels,
            dim,
            bias=True,
            quant_config=None,
            return_bias=False,
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.noise_refiner = nn.ModuleList(
            [
                Fibo2TransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    quant_config=quant_config,
                    prefix=f"noise_refiner.{layer_id}",
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.perceiver = Fibo2PerceiverResampler(
            dim,
            n_heads,
            cap_feat_dim,
            max_num_gist_tokens,
            perceiver_num_layers,
            norm_eps,
            quant_config=quant_config,
            prefix="perceiver",
        )
        self.layers = nn.ModuleList(
            [
                Fibo2TransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    cross_attention_dim=cap_feat_dim if i in self.injection_layer_ids else None,
                    quant_config=quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size * patch_size * in_channels)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # 2x2 latent patches become tokens, features ordered (channel, dy, dx)
        batch_size, channels, height, width = latents.shape
        patch_size = self.patch_size
        latents = latents.view(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
        return latents.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, (height // patch_size) * (width // patch_size), -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        context_latents: list[torch.Tensor] | None = None,
    ):
        """Predicts velocity.

        Args:
            hidden_states: (batch, in_channels, height, width) noisy latents.
            timestep: (batch,) noise level in [0, 1].
            encoder_hidden_states: (batch, 1 + len(injection_layer_ids),
                text_len, cap_feat_dim) caption bundles; bundle 0 feeds the
                perceiver, the others the injection blocks in
                `injection_layer_ids` order.
            encoder_attention_mask: optional (batch, text_len) bool mask of
                real text tokens; None when nothing is padded.
            context_latents: optional list of (batch, in_channels, h, w)
                context (edit) images, each with its own size.
        """
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.patch_size
        height_tokens, width_tokens = height // patch_size, width // patch_size
        num_image_tokens = height_tokens * width_tokens
        device = hidden_states.device

        # 1. Timestep embeddings: image tokens get the real timestep, clean
        # tokens (the gist and the context) get t = 0
        adaln_noisy = self.t_embedder(timestep * self.t_scale)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep))

        # 2. Patchify the image and the context images, which share the patch
        # embedding. The image sits on RoPE plane `max_num_gist_tokens + 1`
        # whatever the gist count, and context image k on the k-th plane after
        # it with its own rows and columns
        image_plane = self.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self.create_coordinate_grid((1, height_tokens, width_tokens), (image_plane, 0, 0), device)]
        for k, latents in enumerate(context_latents or []):
            tokens.append(self._patchify(latents))
            context_size = (1, latents.shape[2] // patch_size, latents.shape[3] // patch_size)
            ids.append(self.create_coordinate_grid(context_size, (image_plane + k + 1, 0, 0), device))
        hidden_states = self.x_embedder(torch.cat(tokens, dim=1))
        hidden_states = _restore_linear_output_shape(hidden_states, tokens[0])
        image_cos, image_sin = self.rope_embedder(torch.cat([grid.flatten(0, 2) for grid in ids]))
        image_cos = image_cos.unsqueeze(0)
        image_sin = image_sin.unsqueeze(0)

        # 3. Noise refiner. With context images, the noise mask marks the
        # image tokens as the only noisy ones
        refiner_noise_mask = None
        if context_latents:
            refiner_noise_mask = torch.zeros(batch_size, hidden_states.shape[1], dtype=torch.long, device=device)
            refiner_noise_mask[:, :num_image_tokens] = 1
        refiner_qk_norm_rope_table = (
            _packed_qk_norm_rope_table(image_cos, image_sin, hidden_states.dtype, batch_size)
            if image_cos is not None
            else None
        )
        for block in self.noise_refiner:
            hidden_states = block(
                hidden_states,
                image_cos,
                image_sin,
                qk_norm_rope_table=refiner_qk_norm_rope_table,
                adaln_input=adaln_noisy,
                noise_mask=refiner_noise_mask,
                adaln_noisy=adaln_noisy,
                adaln_clean=adaln_clean,
            )

        # 4. Gist count per prompt from its text length, then the Perceiver
        # turns the first bundle into gist tokens.
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * batch_size
        gist_counts = compute_gist_count(
            text_lengths,
            min_gist=self.min_num_gist_tokens,
            max_gist=self.max_num_gist_tokens,
            gist_step=self.gist_step,
            min_text_len=self.gist_min_text_len,
            max_text_len=self.gist_max_text_len,
        )
        num_gist_tokens = max(gist_counts)
        num_context_tokens = hidden_states.shape[1] - num_image_tokens
        gist_mask = None
        if min(gist_counts) < num_gist_tokens:
            assert num_context_tokens == 0, "batch-padded gist with context images is not supported yet"
            gist_mask = torch.arange(num_gist_tokens, device=device) < torch.tensor(gist_counts, device=device)[:, None]
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist_tokens, encoder_attention_mask, gist_mask)
        gist_ids = self.create_coordinate_grid((num_gist_tokens, 1, 1), (1, 0, 0), device).flatten(0, 2)

        # 5. One stream: gist tokens first, then the image tokens, then the
        # context tokens, the order Fibo 2.0 was trained with. When a batch
        # pads gist tokens, the gist goes last instead, so the padding ends
        # every row as varlen attention backends expect. Positions come from
        # RoPE, so the order changes nothing but float rounding. The noise
        # mask marks which tokens are noisy
        gist_cos, gist_sin = self.rope_embedder(gist_ids)
        gist_cos = gist_cos.unsqueeze(0)
        gist_sin = gist_sin.unsqueeze(0)
        if gist_mask is None:
            image_start = num_gist_tokens
            hidden_states = torch.cat([gist, hidden_states], dim=1)
            cos = torch.cat([gist_cos, image_cos], dim=1)
            sin = torch.cat([gist_sin, image_sin], dim=1)
            attention_mask = None
        else:
            image_start = 0
            hidden_states = torch.cat([hidden_states, gist], dim=1)
            cos = torch.cat([image_cos, gist_cos], dim=1)
            sin = torch.cat([image_sin, gist_sin], dim=1)
            # Padded batch: image+context tokens are all real, gist rows carry
            # the per-prompt validity mask
            image_mask = torch.ones(batch_size, num_image_tokens + num_context_tokens, dtype=torch.bool, device=device)
            attention_mask = torch.cat([image_mask, gist_mask], dim=1)
        unified_qk_norm_rope_table = _packed_qk_norm_rope_table(cos, sin, hidden_states.dtype, batch_size)
        noise_mask = torch.zeros(batch_size, hidden_states.shape[1], dtype=torch.long, device=device)
        noise_mask[:, image_start : image_start + num_image_tokens] = 1

        # 6. The main blocks. Injection blocks also read their own bundle, in
        # `injection_layer_ids` order. Like diffusers, the main blocks always
        # use per-token modulation (the noise mask marks the noisy tokens);
        # only the refiner falls back to global modulation without context.
        bundle_index = {layer_id: index + 1 for index, layer_id in enumerate(self.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            text = encoder_hidden_states[:, bundle_index[layer_id]] if layer_id in bundle_index else None
            hidden_states = block(
                hidden_states,
                cos,
                sin,
                attn_mask=attention_mask,
                qk_norm_rope_table=unified_qk_norm_rope_table,
                noise_mask=noise_mask,
                adaln_noisy=adaln_noisy,
                adaln_clean=adaln_clean,
                encoder_hidden_states=text,
                encoder_attention_mask=encoder_attention_mask,
            )

        # 7. Final layer on the image tokens only, then unpatchify to latents
        image_tokens = hidden_states[:, image_start : image_start + num_image_tokens]
        hidden_states = self.final_layer(image_tokens, c=adaln_noisy)
        hidden_states = _restore_linear_output_shape(hidden_states, image_tokens)
        batch_frames = hidden_states.view(batch_size, height_tokens, width_tokens, channels, patch_size, patch_size)
        output = batch_frames.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
        return output, {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Note the leading dot in each pattern: ".attention." must not match
        # ".cross_attention." / ".self_attention." paths.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn (refiner + main blocks)
            (".attention.to_qkv.", ".attention.to_q.", "q"),
            (".attention.to_qkv.", ".attention.to_k.", "k"),
            (".attention.to_qkv.", ".attention.to_v.", "v"),
            # perceiver self-attn (diffusers names it self_attention)
            (".self_attention.to_qkv.", ".self_attention.to_q.", "q"),
            (".self_attention.to_qkv.", ".self_attention.to_k.", "k"),
            (".self_attention.to_qkv.", ".self_attention.to_v.", "v"),
            # ffn (blocks: feed_forward, perceiver: ffn)
            (".feed_forward.w13", ".feed_forward.w1", 0),
            (".feed_forward.w13", ".feed_forward.w3", 1),
            (".ffn.w13", ".ffn.w1", 0),
            (".ffn.w13", ".ffn.w3", 1),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        loaded_params = set[str]()
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
                if name not in params_dict:
                    logger.warning("Unexpected weight name in checkpoint: %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
