# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Native vLLM-Omni port of the Boogu-Image transformer.
#
# Ported from the upstream `boogu` package
# (boogu/models/transformers/transformer_boogu.py and friends) with the
# following changes:
#   - Diffusers mixins (ModelMixin/ConfigMixin/PeftAdapterMixin) removed;
#     configuration is read from `od_config.tf_model_config`.
#   - diffusers Attention + custom processors replaced by the vLLM-Omni
#     `Attention` layer (backend/SP/KV-quant handled by the framework).
#   - Q/K/V and FFN projections use vLLM parallel linears for TP support.
#   - Training-time features (prompt tuning, gradient checkpointing) and
#     inference caches (TeaCache/TaylorSeer) are not ported.

import itertools
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, get_1d_rotary_pos_embed
from einops import rearrange, repeat
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings in complex form (upstream `use_real=False` path).

    Args:
        x: Query or key tensor of shape [B, S, H, D].
        freqs_cis: Complex frequency tensor of shape [B, S, D // 2].
    """
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class LuminaRMSNormZero(nn.Module):
    """AdaRMS modulation: projects `temb` into scale/gate terms."""

    def __init__(self, embedding_dim: int, norm_eps: float) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(min(embedding_dim, 1024), 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    """Final AdaLN + optional output projection (upstream `norm_out`)."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias) if out_dim is not None else None

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x) * (1 + scale)[:, None, :]
        if self.linear_2 is not None:
            x = self.linear_2(x)
        return x


class LuminaFeedForward(nn.Module):
    """SwiGLU feed-forward with tensor-parallel projections."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ) -> None:
        super().__init__()
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = ColumnParallelLinear(dim, inner_dim, bias=False)  # gate
        self.linear_3 = ColumnParallelLinear(dim, inner_dim, bias=False)  # input
        self.linear_2 = RowParallelLinear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, _ = self.linear_1(x)
        h2, _ = self.linear_3(x)
        out, _ = self.linear_2(swiglu(h1, h2))
        return out


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        instruction_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(instruction_feat_dim, eps=norm_eps),
            nn.Linear(instruction_feat_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(instruction_hidden_states)
        return time_embed, caption_embed


class BooguImageDoubleStreamRotaryPosEmbed(nn.Module):
    """3-axis RoPE producing per-segment frequency tensors.

    Returns separate frequency tensors for instruction tokens, reference-image
    tokens, noise-image tokens, the full joint sequence, and the combined
    (ref + noise) image sequence used by the double-stream blocks.
    """

    def __init__(
        self,
        theta: int,
        axes_dim: tuple[int, int, int],
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(
        axes_dim: tuple[int, int, int], axes_lens: tuple[int, int, int], theta: int
    ) -> list[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for d, e in zip(axes_dim, axes_lens):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis: list[torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis: list[torch.Tensor],
        attention_mask: torch.Tensor,
        l_effective_ref_img_len: list[list[int]],
        l_effective_img_len: list[int],
        ref_img_sizes: list[list[tuple[int, int]] | None],
        img_sizes: list[tuple[int, int]],
        device: torch.device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max(sum(ref_img_len) for ref_img_len in l_effective_ref_img_len)
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            position_ids[i, :cap_seq_len] = repeat(
                torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3"
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len

                    row_ids = repeat(
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device),
                        "h -> h w",
                        w=ref_W_tokens,
                    ).flatten()
                    col_ids = repeat(
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device),
                        "w -> h w",
                        h=ref_H_tokens,
                    ).flatten()
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = repeat(torch.arange(H_tokens, dtype=torch.int32, device=device), "h -> h w", w=W_tokens).flatten()
            col_ids = repeat(torch.arange(W_tokens, dtype=torch.int32, device=device), "w -> h w", h=H_tokens).flatten()

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        cap_freqs_cis = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size, max_ref_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs_cis = torch.zeros(batch_size, max_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype)

        combined_img_seq_lengths = [
            sum(ref_img_len) + img_len for ref_img_len, img_len in zip(l_effective_ref_img_len, l_effective_img_len)
        ]
        max_combined_img_len = max(combined_img_seq_lengths)

        combined_img_freqs_cis = torch.zeros(
            batch_size, max_combined_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]

            combined_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            combined_img_freqs_cis[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = freqs_cis[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
            combined_img_freqs_cis,
            combined_img_seq_lengths,
        )


def _concat_instruction_image_features(
    img_tensors: list[torch.Tensor],
    instruct_tensors: list[torch.Tensor],
    encoder_seq_lengths: list[int],
    seq_lengths: list[int],
) -> list[torch.Tensor]:
    """Pack per-stream tensors into joint sequences (instruction first, then image)."""
    assert len(img_tensors) == len(instruct_tensors)

    batch_size = img_tensors[0].shape[0]
    max_seq_len = max(seq_lengths)

    concatenated_list = []
    for img_tensor, instruct_tensor in zip(img_tensors, instruct_tensors):
        feature_dim = img_tensor.shape[-1]
        concatenated = img_tensor.new_zeros(batch_size, max_seq_len, feature_dim)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            concatenated[i, :encoder_seq_len] = instruct_tensor[i, :encoder_seq_len]
            concatenated[i, encoder_seq_len:seq_len] = img_tensor[i, : seq_len - encoder_seq_len]
        concatenated_list.append(concatenated)

    return concatenated_list


def _split_instruction_image_features(
    hidden_states: torch.Tensor,
    encoder_seq_lengths: list[int],
    seq_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a joint sequence back into (instruction, image) streams."""
    batch_size = hidden_states.shape[0]
    feature_dim = hidden_states.shape[-1]

    max_instruct_len = max(encoder_seq_lengths)
    max_img_len = max(seq_len - encoder_seq_len for seq_len, encoder_seq_len in zip(seq_lengths, encoder_seq_lengths))

    instruct_hidden_states = hidden_states.new_zeros(batch_size, max_instruct_len, feature_dim)
    img_hidden_states = hidden_states.new_zeros(batch_size, max_img_len, feature_dim)

    for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        img_len = seq_len - encoder_seq_len
        instruct_hidden_states[i, :encoder_seq_len] = hidden_states[i, :encoder_seq_len]
        img_hidden_states[i, :img_len] = hidden_states[i, encoder_seq_len:seq_len]

    return instruct_hidden_states, img_hidden_states


class BooguImageSelfAttention(nn.Module):
    """GQA self-attention with QK RMSNorm and complex RoPE.

    Replaces the upstream diffusers `Attention` + Boogu attention processors;
    the vLLM-Omni `Attention` layer picks the kernel backend and handles
    SP/KV-cache concerns.
    """

    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        kv_dim = self.head_dim * num_kv_heads

        self.to_q = ColumnParallelLinear(dim, dim, bias=False)
        self.to_k = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.to_v = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-5)
        self.to_out = RowParallelLinear(dim, dim, bias=False)

        self.num_local_heads = self.to_q.output_size_per_partition // self.head_dim
        self.num_local_kv_heads = self.to_k.output_size_per_partition // self.head_dim

        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.num_local_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.num_local_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_local_kv_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_local_kv_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        query, key = query.to(dtype), key.to(dtype)

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        attn_output = self.attn(query, key, value, attn_metadata)
        attn_output = attn_output.flatten(2, 3).type_as(hidden_states)

        attn_output, _ = self.to_out(attn_output)
        return attn_output


class BooguImageJointAttention(nn.Module):
    """Joint instruction+image attention for the double-stream blocks.

    The upstream implementation keeps the per-stream Q/K/V and output
    projections on the attention *processor*
    (`BooguImageDoubleStreamSelfAttnProcessor*`); here they are promoted onto
    this module. The final joint output projection corresponds to the upstream
    `img_instruct_attn.to_out[0]`.
    """

    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        kv_dim = self.head_dim * num_kv_heads

        self.img_to_q = ColumnParallelLinear(dim, dim, bias=False)
        self.img_to_k = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.img_to_v = ColumnParallelLinear(dim, kv_dim, bias=False)

        self.instruct_to_q = ColumnParallelLinear(dim, dim, bias=False)
        self.instruct_to_k = ColumnParallelLinear(dim, kv_dim, bias=False)
        self.instruct_to_v = ColumnParallelLinear(dim, kv_dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-5)

        # Per-stream output projections (attention output is head-sharded
        # under TP, hence row-parallel).
        self.instruct_out = RowParallelLinear(dim, dim, bias=False)
        self.img_out = RowParallelLinear(dim, dim, bias=False)
        # Final joint projection applied to the merged full-dim sequence.
        self.to_out = ReplicatedLinear(dim, dim, bias=False)

        self.num_local_heads = self.img_to_q.output_size_per_partition // self.head_dim
        self.num_local_kv_heads = self.img_to_k.output_size_per_partition // self.head_dim

        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.num_local_kv_heads,
        )

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        joint_attention_mask: torch.Tensor | None,
        rotary_emb: torch.Tensor | None,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> torch.Tensor:
        dtype = img_hidden_states.dtype
        batch_size = img_hidden_states.shape[0]

        img_query, _ = self.img_to_q(img_hidden_states)
        img_key, _ = self.img_to_k(img_hidden_states)
        img_value, _ = self.img_to_v(img_hidden_states)

        instruct_query, _ = self.instruct_to_q(instruct_hidden_states)
        instruct_key, _ = self.instruct_to_k(instruct_hidden_states)
        instruct_value, _ = self.instruct_to_v(instruct_hidden_states)

        query, key, value = _concat_instruction_image_features(
            [img_query, img_key, img_value],
            [instruct_query, instruct_key, instruct_value],
            encoder_seq_lengths,
            seq_lengths,
        )

        query = query.view(batch_size, -1, self.num_local_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_local_kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_local_kv_heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        query, key = query.to(dtype), key.to(dtype)

        attn_metadata = AttentionMetadata(attn_mask=joint_attention_mask) if joint_attention_mask is not None else None
        attn_output = self.attn(query, key, value, attn_metadata)
        attn_output = attn_output.flatten(2, 3).to(dtype)

        instruct_attn_out, img_attn_out = _split_instruction_image_features(
            attn_output, encoder_seq_lengths, seq_lengths
        )
        instruct_projected, _ = self.instruct_out(instruct_attn_out)
        img_projected, _ = self.img_out(img_attn_out)

        merged = _concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )[0]
        merged, _ = self.to_out(merged)
        return merged


class BooguImageTransformerBlock(nn.Module):
    """Single-stream block: GQA attention + SwiGLU FFN with optional AdaRMS modulation."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        self.attn = BooguImageSelfAttention(dim, num_attention_heads, num_kv_heads)
        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(norm_hidden_states, attention_mask, image_rotary_emb)
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(norm_hidden_states, attention_mask, image_rotary_emb)
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class BooguImageNoiseRefinerTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageRefImgRefinerTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageContextRefinerTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageSingleStreamTransformerBlock(BooguImageTransformerBlock):
    pass


class BooguImageDoubleStreamTransformerBlock(nn.Module):
    """Double-stream block: instruction and image tokens are processed in
    parallel streams coupled by a joint attention, plus an extra image
    self-attention."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.modulation = modulation
        self.hidden_size = dim

        self.img_instruct_attn = BooguImageJointAttention(dim, num_attention_heads, num_kv_heads)
        self.img_self_attn = BooguImageSelfAttention(dim, num_attention_heads, num_kv_heads)

        self.img_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Image modulation terms: cross-attn, MLP, self-attn.
            self.img_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
            self.img_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
            self.img_norm3 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
        else:
            self.img_norm1 = RMSNorm(dim, eps=norm_eps)
            self.img_norm2 = RMSNorm(dim, eps=norm_eps)
            self.img_norm3 = RMSNorm(dim, eps=norm_eps)

        self.img_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Instruction modulation terms: cross-attn, MLP.
            self.instruct_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
            self.instruct_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps)
        else:
            self.instruct_norm1 = RMSNorm(dim, eps=norm_eps)
            self.instruct_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        img_hidden_states: torch.Tensor,  # [B, L_img, D] (ref_img + noise_img)
        instruct_hidden_states: torch.Tensor,  # [B, L_instruct, D]
        img_attention_mask: torch.Tensor | None,  # [B, L_img]
        joint_attention_mask: torch.Tensor | None,  # [B, L_total]
        image_rotary_emb: torch.Tensor | None,  # [B, L_img, head_dim // 2]
        rotary_emb: torch.Tensor | None,  # [B, L_total, head_dim // 2]
        temb: torch.Tensor | None = None,
        encoder_seq_lengths: list[int] | None = None,
        seq_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modulation and temb is None:
            raise ValueError("temb must be provided when modulation is enabled")

        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]
        L_img = img_hidden_states.shape[1]

        if self.modulation:
            img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
            img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
            img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

            (
                instruct_norm1_out,
                instruct_gate_msa,
                instruct_scale_mlp,
                instruct_gate_mlp,
            ) = self.instruct_norm1(instruct_hidden_states, temb)
            instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

            joint_attn_out = self.img_instruct_attn(
                img_norm1_out,
                instruct_norm1_out,
                joint_attention_mask,
                rotary_emb,
                encoder_seq_lengths,
                seq_lengths,
            )

            # Split the joint output back into instruction/image segments.
            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            img_self_attn_out = self.img_self_attn(img_norm3_out, img_attention_mask, image_rotary_emb)

            img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(
                img_self_attn_out
            )

            img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
            img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + instruct_gate_msa.unsqueeze(
                1
            ).tanh() * self.instruct_attn_norm(instruct_attn_out)

            instruct_mlp_input = (
                1 + instruct_scale_mlp.unsqueeze(1)
            ) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
            instruct_hidden_states = instruct_hidden_states + instruct_gate_mlp.unsqueeze(
                1
            ).tanh() * self.instruct_ffn_norm2(instruct_mlp_out)

        else:
            img_norm1_out = self.img_norm1(img_hidden_states)
            img_norm3_out = self.img_norm3(img_hidden_states)
            instruct_norm1_out = self.instruct_norm1(instruct_hidden_states)

            joint_attn_out = self.img_instruct_attn(
                img_norm1_out,
                instruct_norm1_out,
                joint_attention_mask,
                rotary_emb,
                encoder_seq_lengths,
                seq_lengths,
            )

            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            img_self_attn_out = self.img_self_attn(img_norm3_out, img_attention_mask, image_rotary_emb)

            img_hidden_states = img_hidden_states + self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + self.img_self_attn_norm(img_self_attn_out)
            img_norm2_out = self.img_norm2(img_hidden_states)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_norm2_out))
            img_hidden_states = img_hidden_states + self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + self.instruct_attn_norm(instruct_attn_out)
            instruct_norm2_out = self.instruct_norm2(instruct_hidden_states)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_norm2_out))
            instruct_hidden_states = instruct_hidden_states + self.instruct_ffn_norm2(instruct_mlp_out)

        return img_hidden_states, instruct_hidden_states


def _cal_preprocessed_instruction_feat_dim(instruction_feature_configs: dict) -> int:
    num_instruction_feature_layers = max(instruction_feature_configs.get("num_instruction_feature_layers", 1), 1)
    instruction_feat_dim = instruction_feature_configs.get("instruction_feat_dim", 4096)
    reduce_type = instruction_feature_configs.get("reduce_type", "concat")
    if "cat" in reduce_type.lower():
        return num_instruction_feature_layers * instruction_feat_dim
    elif "mean" in reduce_type.lower():
        return instruction_feat_dim
    else:
        raise ValueError(f"Invalid reduce_type: {reduce_type}")


class BooguImageTransformer2DModel(nn.Module):
    """Boogu-Image transformer with mixed stream topology.

    Early layers use double-stream (dual-stream) processing, then switch to
    single-stream joint processing. Context/noise/reference-image refiner
    blocks run before the main stack.
    """

    # Noise refiner, reference image refiner, and double stream layers are
    # kept out of regional torch.compile for numerical stability (matches the
    # upstream `_repeated_blocks`).
    _repeated_blocks = [
        "BooguImageTransformerBlock",
        "BooguImageContextRefinerTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
    ]
    _layerwise_offload_blocks_attrs = ["single_stream_layers", "double_stream_layers"]

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        cfg = od_config.tf_model_config

        patch_size = cfg.get("patch_size", 2)
        in_channels = cfg.get("in_channels", 16)
        out_channels = cfg.get("out_channels", None)
        hidden_size = cfg.get("hidden_size", 2304)
        num_layers = cfg.get("num_layers", 26)
        num_double_stream_layers = cfg.get("num_double_stream_layers", 2)
        num_refiner_layers = cfg.get("num_refiner_layers", 2)
        num_attention_heads = cfg.get("num_attention_heads", 24)
        num_kv_heads = cfg.get("num_kv_heads", 8)
        multiple_of = cfg.get("multiple_of", 256)
        ffn_dim_multiplier = cfg.get("ffn_dim_multiplier", None)
        norm_eps = cfg.get("norm_eps", 1e-5)
        axes_dim_rope = tuple(cfg.get("axes_dim_rope", (40, 40, 40)))
        axes_lens = tuple(cfg.get("axes_lens", (2048, 1664, 1664)))
        instruction_feature_configs = cfg.get(
            "instruction_feature_configs",
            {"instruction_feat_dim": 1024, "reduce_type": "mean", "num_instruction_feature_layers": 1},
        )
        prompt_tuning_configs = cfg.get("prompt_tuning_configs", {"use_prompt_tuning": False})
        timestep_scale = cfg.get("timestep_scale", 1.0)

        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )
        if num_double_stream_layers > num_layers:
            raise ValueError(
                f"num_double_stream_layers ({num_double_stream_layers}) cannot be greater than "
                f"num_layers ({num_layers})"
            )
        if prompt_tuning_configs.get("use_prompt_tuning", False):
            raise NotImplementedError(
                "Prompt tuning is a training-time feature and is not supported by the "
                "native vLLM-Omni BooguImageTransformer2DModel."
            )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_double_stream_layers = num_double_stream_layers
        self.num_single_stream_layers = num_layers - num_double_stream_layers
        self.axes_dim_rope = axes_dim_rope
        self.axes_lens = axes_lens
        self.instruction_feature_configs = instruction_feature_configs
        self.preprocessed_instruction_feat_dim = _cal_preprocessed_instruction_feat_dim(instruction_feature_configs)

        self.rope_embedder = BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )
        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            instruction_feat_dim=self.preprocessed_instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        self.noise_refiner = nn.ModuleList(
            [
                BooguImageNoiseRefinerTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                BooguImageRefImgRefinerTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                BooguImageContextRefinerTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # Mixed architecture: dual-stream first, then single-stream.
        self.double_stream_layers = nn.ModuleList(
            [
                BooguImageDoubleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_double_stream_layers)
            ]
        )

        self.single_stream_layers = nn.ModuleList(
            [
                BooguImageSingleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(self.num_single_stream_layers)
            ]
        )

        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Distinguish multiple reference images (max 5).
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))

    def preprocess_instruction_hidden_states(self, raw_instruction_hidden_states):
        """Reduce the raw MLLM hidden states to the transformer feature dim.

        Mirrors upstream ``preprocess_instruction_hidden_states``: a single
        tensor passes through unchanged; a list of per-layer states is combined
        by ``concat`` or ``mean`` according to ``instruction_feature_configs``.
        """
        cfg = self.instruction_feature_configs
        num_instruction_feature_layers = max(cfg.get("num_instruction_feature_layers", 1), 1)
        reduce_type = cfg.get("reduce_type", "concat")

        if isinstance(raw_instruction_hidden_states, torch.Tensor):
            instruction_hidden_states = raw_instruction_hidden_states
        elif isinstance(raw_instruction_hidden_states, (list, tuple)):
            assert len(raw_instruction_hidden_states) == num_instruction_feature_layers
            if "cat" in reduce_type.lower():
                instruction_hidden_states = torch.cat(raw_instruction_hidden_states, dim=-1)
            elif "mean" in reduce_type.lower():
                instruction_hidden_states = torch.mean(torch.stack(raw_instruction_hidden_states), dim=0)
            else:
                raise ValueError(f"Invalid reduce_type: {reduce_type}")
        else:
            raise ValueError(
                "Invalid type of raw_instruction_hidden_states, expected torch.Tensor or list, "
                f"but got {type(raw_instruction_hidden_states)}"
            )

        assert self.preprocessed_instruction_feat_dim == instruction_hidden_states.shape[-1]
        return instruction_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        """Flatten patch tokens and pad to batched sequences.

        Ported from upstream; for text-to-image ``ref_image_hidden_states`` is
        ``None`` and the reference-image branch collapses to zero-length.
        """
        batch_size = len(hidden_states)
        p = self.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_img_sizes = [
                [(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None
                for imgs in ref_image_hidden_states
            ]
            l_effective_ref_img_len = [
                [(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes]
                if _ref_img_sizes is not None
                else [0]
                for _ref_img_sizes in ref_img_sizes
            ]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max(sum(ref_img_len) for ref_img_len in l_effective_ref_img_len)
        max_img_len = max(l_effective_img_len)

        # Reference-image patch embeddings.
        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    ref_img = rearrange(ref_img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
                    imgs.append(ref_img)
                flat_ref_img_hidden_states.append(torch.cat(imgs, dim=0))
            else:
                flat_ref_img_hidden_states.append(None)

        # Noise-image patch embeddings.
        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()
            img = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
            flat_hidden_states.append(img)

        padded_ref_img_hidden_states = torch.zeros(
            batch_size,
            max_ref_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, : sum(l_effective_ref_img_len[i])] = flat_ref_img_hidden_states[i]
                padded_ref_img_mask[i, : sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(
            batch_size,
            max_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            padded_hidden_states[i, : l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )

    def img_patch_embed_and_refine(
        self,
        hidden_states,
        ref_image_hidden_states,
        padded_img_mask,
        padded_ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    ):
        """Embed image patches and run the refiner blocks.

        The reference-image refiner is skipped when there are no reference-image
        tokens (text-to-image), which is numerically identical to upstream (the
        combined sequence only reads ``[:sum(ref_img_len)]`` = empty) while
        avoiding a degenerate zero-length attention.
        """
        batch_size = len(hidden_states)
        max_combined_img_len = max(
            img_len + sum(ref_img_len) for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)
        )

        hidden_states = self.x_embedder(hidden_states)
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift : shift + ref_img_len, :] = (
                    ref_image_hidden_states[i, shift : shift + ref_img_len, :] + self.image_index_embedding[j]
                )
                shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
        num_ref_images = len(flat_l_effective_ref_img_len)
        max_ref_img_len = max(flat_l_effective_ref_img_len)

        if max_ref_img_len > 0:
            batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, dtype=torch.bool)
            batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
                num_ref_images, max_ref_img_len, self.hidden_size
            )
            batch_ref_img_rotary_emb = hidden_states.new_zeros(
                num_ref_images, max_ref_img_len, ref_img_rotary_emb.shape[-1], dtype=ref_img_rotary_emb.dtype
            )
            batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

            # Flatten reference images into a temporary batch.
            idx = 0
            for i in range(batch_size):
                shift = 0
                for ref_img_len in l_effective_ref_img_len[i]:
                    batch_ref_img_mask[idx, :ref_img_len] = True
                    batch_ref_image_hidden_states[idx, :ref_img_len] = ref_image_hidden_states[
                        i, shift : shift + ref_img_len
                    ]
                    batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[i, shift : shift + ref_img_len]
                    batch_temb[idx] = temb[i]
                    shift += ref_img_len
                    idx += 1

            for layer in self.ref_image_refiner:
                batch_ref_image_hidden_states = layer(
                    batch_ref_image_hidden_states, batch_ref_img_mask, batch_ref_img_rotary_emb, batch_temb
                )

            # Restore reference-image sequence layout.
            idx = 0
            for i in range(batch_size):
                shift = 0
                for ref_img_len in l_effective_ref_img_len[i]:
                    ref_image_hidden_states[i, shift : shift + ref_img_len] = batch_ref_image_hidden_states[
                        idx, :ref_img_len
                    ]
                    shift += ref_img_len
                    idx += 1

        combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len, self.hidden_size)
        for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
            combined_img_hidden_states[i, : sum(ref_img_len)] = ref_image_hidden_states[i, : sum(ref_img_len)]
            combined_img_hidden_states[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Denoise one step: refiner -> double-stream -> fuse -> single-stream -> unpatchify.

        Ported from upstream ``BooguImageTransformer2DModel.forward`` with the
        TeaCache/TaylorSeer/PEFT/gradient-checkpointing branches removed. Returns
        the velocity prediction as a ``[B, C, H_lat, W_lat]`` tensor.
        """
        instruction_hidden_states = self.preprocess_instruction_hidden_states(instruction_hidden_states)

        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)
        if is_hidden_states_tensor:
            assert hidden_states.ndim == 4
            hidden_states = [_hidden_states for _hidden_states in hidden_states]

        device = hidden_states[0].device

        # Timestep and instruction embedding.
        temb, instruction_hidden_states = self.time_caption_embed(
            timestep, instruction_hidden_states, hidden_states[0].dtype
        )

        # Flatten and pad token sequences.
        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        # Build rotary embeddings and sequence lengths.
        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
            combined_img_rotary_emb,
            combined_img_seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            instruction_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # Context refinement.
        for layer in self.context_refiner:
            instruction_hidden_states = layer(instruction_hidden_states, instruction_attention_mask, context_rotary_emb)

        # Image patch embedding and refinement.
        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        instruct_hidden_states = instruction_hidden_states
        img_hidden_states = combined_img_hidden_states

        # Joint mask for [instruct + image].
        max_seq_len = max(seq_lengths)
        joint_attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            joint_attention_mask[i, :seq_len] = True

        # Dual-stream (double-stream) stage.
        if self.num_double_stream_layers > 0:
            max_img_len = max(combined_img_seq_lengths)
            img_attention_mask = hidden_states.new_zeros(batch_size, max_img_len, dtype=torch.bool)
            for i, img_seq_len in enumerate(combined_img_seq_lengths):
                img_attention_mask[i, :img_seq_len] = True

            for layer in self.double_stream_layers:
                img_hidden_states, instruct_hidden_states = layer(
                    img_hidden_states,
                    instruct_hidden_states,
                    img_attention_mask,
                    joint_attention_mask,
                    combined_img_rotary_emb,
                    rotary_emb,
                    temb,
                    encoder_seq_lengths,
                    seq_lengths,
                )

        # Fuse streams to joint sequence.
        joint_hidden_states = hidden_states.new_zeros(batch_size, max(seq_lengths), self.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            joint_hidden_states[i, :encoder_seq_len] = instruct_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = img_hidden_states[i, : seq_len - encoder_seq_len]

        # Single-stream stage.
        hidden_states = joint_hidden_states
        for layer in self.single_stream_layers:
            hidden_states = layer(hidden_states, joint_attention_mask, rotary_emb, temb)

        # Output projection.
        hidden_states = self.norm_out(hidden_states, temb)

        # Reshape back to image format.
        p = self.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            img_tokens = hidden_states[i][seq_len - img_len : seq_len]
            img_output = rearrange(
                img_tokens,
                "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                h=height // p,
                w=width // p,
                p1=p,
                p2=p,
            )
            output.append(img_output)

        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load diffusers-named checkpoint weights into the native module.

        Two name promotions relative to upstream (see step 8/10 findings):

        - ``*.img_instruct_attn.processor.{img,instruct}_{to_q,to_k,to_v}`` /
          ``{instruct,img}_out`` -> drop ``.processor`` (upstream keeps the
          joint-attention projections on the attention processor; the native
          module hosts them directly).
        - ``*.to_out.0.weight`` -> ``*.to_out.weight`` (diffusers wraps the
          output projection in a ``ModuleList``; the native module uses a plain
          linear).
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            if ".img_instruct_attn.processor." in name:
                name = name.replace(".img_instruct_attn.processor.", ".img_instruct_attn.")
            if ".to_out.0." in name:
                name = name.replace(".to_out.0.", ".to_out.")

            if name not in params_dict:
                logger.warning("Skipping unexpected checkpoint weight %s", original_name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = sorted(params_dict.keys() - loaded_params)
        if unloaded_params:
            logger.warning("Model parameters not loaded from checkpoint: %s", unloaded_params)

        return loaded_params
