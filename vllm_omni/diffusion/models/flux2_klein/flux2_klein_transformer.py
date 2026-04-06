# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

logger = init_logger(__name__)
if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class Flux2SwiGLU(nn.Module):
    """SwiGLU activation used by Flux2."""

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.linear_in = MergedColumnParallelLinear(
            dim,
            [inner_dim, inner_dim],
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
        )
        self.act_fn = Flux2SwiGLU()
        self.linear_out = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        return self.linear_out(x)


class Flux2Attention(nn.Module):
    def __init__(
        self,
        parallel_config: DiffusionParallelConfig,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.parallel_config = parallel_config
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.dropout = dropout
        self.added_kv_proj_dim = added_kv_proj_dim

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                ),
                nn.Dropout(dropout),
            ]
        )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
                quant_config=quant_config,
            )
            self.add_query_num_heads = self.add_kv_proj.num_heads
            self.add_kv_num_heads = self.add_kv_proj.num_kv_heads
            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.query_num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.kv_num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.query_num_heads, -1))
        key = key.unflatten(-1, (self.kv_num_heads, -1))
        value = value.unflatten(-1, (self.kv_num_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (self.add_query_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_num_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_num_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            sp_size = self.parallel_config.sequence_parallel_size
            forward_ctx = get_forward_context()
            use_sp_joint_attention = sp_size is not None and sp_size > 1 and not forward_ctx.split_text_embed_in_sp

            if use_sp_joint_attention and image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                cos = cos.to(query.dtype)
                sin = sin.to(query.dtype)
                txt_len = encoder_query.shape[1]
                txt_cos, img_cos = cos[:txt_len], cos[txt_len:]
                txt_sin, img_sin = sin[:txt_len], sin[txt_len:]
                query = self.rope(query, img_cos, img_sin)
                key = self.rope(key, img_cos, img_sin)
                encoder_query = self.rope(encoder_query, txt_cos, txt_sin)
                encoder_key = self.rope(encoder_key, txt_cos, txt_sin)

                attn_metadata = AttentionMetadata(
                    joint_query=encoder_query,
                    joint_key=encoder_key,
                    joint_value=encoder_value,
                    joint_strategy="front",
                )
                hidden_states_mask: torch.Tensor | None = kwargs.get("hidden_states_mask", None)
                encoder_hidden_states_mask: torch.Tensor | None = kwargs.get("encoder_hidden_states_mask", None)
                if hidden_states_mask is not None:
                    attn_metadata.attn_mask = hidden_states_mask
                if encoder_hidden_states_mask is not None:
                    attn_metadata.joint_attn_mask = encoder_hidden_states_mask

                hidden_states = self.attn(query, key, value, attn_metadata)
                hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

                txt_len = encoder_hidden_states.shape[1]
                encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                    [txt_len, hidden_states.shape[1] - txt_len],
                    dim=1,
                )
                # Contiguous for FP8 quantization in RowParallelLinear
                encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            else:
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

                if image_rotary_emb is not None:
                    cos, sin = image_rotary_emb
                    cos = cos.to(query.dtype)
                    sin = sin.to(query.dtype)
                    query = self.rope(query, cos, sin)
                    key = self.rope(key, cos, sin)

                attn_metadata = None
                if attention_mask is not None:
                    if attention_mask.dim() == 3:
                        attention_mask = attention_mask.unsqueeze(1)
                    attn_metadata = AttentionMetadata(attn_mask=attention_mask)

                hidden_states = self.attn(query, key, value, attn_metadata)
                hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

                context_len = encoder_hidden_states.shape[1]
                encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                    [context_len, hidden_states.shape[1] - context_len],
                    dim=1,
                )
                # Contiguous for FP8 quantization in RowParallelLinear
                encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
        else:
            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                cos = cos.to(query.dtype)
                sin = sin.to(query.dtype)
                query = self.rope(query, cos, sin)
                key = self.rope(key, cos, sin)

            attn_metadata = None
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                attn_metadata = AttentionMetadata(attn_mask=attention_mask)

            hidden_states = self.attn(query, key, value, attn_metadata)
            hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        hidden_states = self.to_out[0](hidden_states.contiguous())
        hidden_states = self.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class Flux2ParallelSelfAttention(nn.Module):
    """
    Parallel attention block that fuses QKV projections with MLP input projections.
    """

    def __init__(
        self,
        parallel_config: DiffusionParallelConfig,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.parallel_config = parallel_config
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.dropout = dropout

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        self.to_qkv_mlp_proj = ColumnParallelLinear(
            self.query_dim,
            self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
        )
        self.mlp_act_fn = Flux2SwiGLU()

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_out = ColumnParallelLinear(
            self.inner_dim + self.mlp_hidden_dim,
            self.out_dim,
            bias=out_bias,
            gather_output=True,
            quant_config=quant_config,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states, _ = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states,
            [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        sp_size = self.parallel_config.sequence_parallel_size
        forward_ctx = get_forward_context()
        text_seq_len = kwargs.get("text_seq_len", None)
        use_sp_single_stream = (
            sp_size is not None and sp_size > 1 and not forward_ctx.split_text_embed_in_sp and text_seq_len is not None
        )

        if use_sp_single_stream and image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            txt_cos, img_cos = cos[:text_seq_len], cos[text_seq_len:]
            txt_sin, img_sin = sin[:text_seq_len], sin[text_seq_len:]

            img_query = query[:, text_seq_len:]
            img_key = key[:, text_seq_len:]
            img_value = value[:, text_seq_len:]
            text_query = query[:, :text_seq_len]
            text_key = key[:, :text_seq_len]
            text_value = value[:, :text_seq_len]

            img_query = self.rope(img_query, img_cos, img_sin)
            img_key = self.rope(img_key, img_cos, img_sin)
            text_query = self.rope(text_query, txt_cos, txt_sin)
            text_key = self.rope(text_key, txt_cos, txt_sin)

            attn_metadata = AttentionMetadata(
                joint_query=text_query,
                joint_key=text_key,
                joint_value=text_value,
                joint_strategy="front",
            )
            hidden_states_mask: torch.Tensor | None = kwargs.get("hidden_states_mask", None)
            encoder_hidden_states_mask: torch.Tensor | None = kwargs.get("encoder_hidden_states_mask", None)
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            if encoder_hidden_states_mask is not None:
                attn_metadata.joint_attn_mask = encoder_hidden_states_mask

            attn_output = self.attn(img_query, img_key, img_value, attn_metadata)
        else:
            query, key = apply_rope_to_qk(self.rope, query, key, image_rotary_emb)

            attn_metadata = None
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                attn_metadata = AttentionMetadata(attn_mask=attention_mask)

            attn_output = self.attn(query, key, value, attn_metadata)

        attn_output = attn_output.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states, _ = self.to_out(hidden_states)
        return hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        parallel_config: DiffusionParallelConfig,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2ParallelSelfAttention(
            parallel_config=parallel_config,
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Flux2SingleTransformerBlock with SP support.

        In SP mode: image hidden_states is chunked (B, img_len/SP, D),
        text encoder_hidden_states is full (B, txt_len, D).
        The block concatenates them for joint attention.
        """
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        return hidden_states


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        parallel_config: DiffusionParallelConfig,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            parallel_config=parallel_config,
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias, quant_config=quant_config)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        temb_mod_params_txt: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(len(self.axes_dim)):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Flux2RopePrepare(nn.Module):
    """Prepares RoPE embeddings for sequence parallel.

    This module encapsulates the RoPE computation for Flux.2-klein.
    For dual-stream attention, text components (outputs 0, 1) are replicated
    across SP ranks, while image components (outputs 2, 3) are sharded.

    NOTE: The hidden_states projection is handled separately in forward()
    so that _sp_plan can shard it at the root level.
    """

    def __init__(self, pos_embed: Flux2PosEmbed):
        super().__init__()
        self.pos_embed = pos_embed

    def forward(
        self,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute RoPE embeddings for text and image sequences.

        Args:
            img_ids: Image position IDs (img_seq_len, n_axes)
            txt_ids: Text position IDs (txt_seq_len, n_axes)

        Returns:
            Tuple of cosine / sine components for text & image
            in the order: (txt_cos, txt_sin, img_cos, img_sin)

        NOTE: careful about output orders if this is refactored in the
        future; we need to match the _sp_plan indices, since text
        components (0 & 1) need to be replicated across SP ranks,
        while image components (2 & 3) must be sharded.
        """
        img_freqs_cos, img_freqs_sin = self.pos_embed(img_ids)
        txt_freqs_cos, txt_freqs_sin = self.pos_embed(txt_ids)
        return txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )

        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels,
                time_embed_dim=embedding_dim,
                sample_proj_bias=bias,
            )
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor | None) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))

        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))
            return timesteps_emb + guidance_emb
        return timesteps_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2Transformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux 2.

    Supports Sequence Parallelism (Ulysses and Ring) when configured via OmniDiffusionConfig.
    """

    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return ("transformer_blocks" in name or "single_transformer_blocks" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
    _sp_plan = {
        "": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "rope_prepare": {
            2: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
            3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
    """SP plan: shard hidden_states at root level, shard img_freqs at rope_prepare, gather output at proj_out."""

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        od_config: OmniDiffusionConfig = None,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.config = SimpleNamespace(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            guidance_embeds=guidance_embeds,
        )

        if od_config is not None:
            self.parallel_config = od_config.parallel_config
        else:
            from vllm_omni.diffusion.data import DiffusionParallelConfig

            self.parallel_config = DiffusionParallelConfig()

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        self.rope_prepare = Flux2RopePrepare(self.pos_embed)

        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    parallel_config=self.parallel_config,
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    quant_config=quant_config,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    parallel_config=self.parallel_config,
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    quant_config=quant_config,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=eps,
            bias=False,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        joint_attention_kwargs = joint_attention_kwargs or {}

        num_txt_tokens = encoder_hidden_states.shape[1]

        sp_size = self.parallel_config.sequence_parallel_size
        if sp_size is not None and sp_size > 1:
            get_forward_context().split_text_embed_in_sp = False

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin = self.rope_prepare(img_ids, txt_ids)

        concat_rotary_emb = (
            torch.cat([txt_freqs_cos, img_freqs_cos], dim=0),
            torch.cat([txt_freqs_sin, img_freqs_sin], dim=0),
        )

        # Create separate masks for image and text portions for Ulysses SP joint attention
        hidden_states_mask = None
        encoder_hidden_states_mask = None
        ctx = get_forward_context()
        if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
            batch_size = hidden_states.shape[0]
            img_padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size

            hidden_states_mask = torch.ones(
                batch_size,
                img_padded_seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            hidden_states_mask[:, ctx.sp_original_seq_len :] = False
            if hidden_states_mask.all():
                hidden_states_mask = None

        if hidden_states_mask is not None:
            joint_attention_kwargs["hidden_states_mask"] = hidden_states_mask
        if encoder_hidden_states_mask is not None:
            joint_attention_kwargs["encoder_hidden_states_mask"] = encoder_hidden_states_mask

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                text_seq_len=num_txt_tokens,
            )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv.", ".to_q.", "q"),
            (".to_qkv.", ".to_k.", "k"),
            (".to_qkv.", ".to_v.", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        # Expose packed shard mappings for LoRA handling of fused projections.
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                name = original_name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is None:
                    break
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                mapped = True
                break
            if mapped:
                continue

            name = original_name
            if name not in params_dict and ".to_out.0." in name:
                name = name.replace(".to_out.0.", ".to_out.")
            # Some GGUF checkpoints include quantized tensors for modules that
            # are intentionally left unquantized in this model.
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
