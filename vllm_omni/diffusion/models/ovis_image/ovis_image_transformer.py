# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Alibaba Ovis-Image Team and The HuggingFace. All rights reserved.
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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    CombinedTimestepLabelEmbeddings,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZeroSingle
from diffusers.utils import is_torch_npu_available
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


def _safe_quant_config(quant_config: "QuantizationConfig | None") -> "QuantizationConfig | None":
    """Return quant_config only if it is safe to propagate here, else None.

    Dual-stream transformer_blocks, norm modulation layers, and norm_out are
    kept at full precision for FP8 (see #2728). Offline quantization (e.g.
    INC/AutoRound W4A16) needs the config propagated so packed weights load
    correctly.
    """
    if quant_config is None:
        return None
    from vllm.model_executor.layers.quantization.inc import INCConfig

    if isinstance(quant_config, INCConfig):
        return quant_config
    return None


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int | None = None,
        norm_type="layer_norm",
        bias=True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            embedding_dim, 6 * embedding_dim, bias=bias, quant_config=quant_config, prefix=f"{prefix}.linear"
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type})")

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ):
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        emb, _ = self.linear(self.silu(emb))

        chunks = emb.chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        inner_dim: int,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in, inner_dim * 2, bias=bias, gather_output=False, quant_config=quant_config, prefix=f"{prefix}.proj"
        )
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        # Weight layout: [hidden_local, gate_local] interleaved within each TP shard
        hidden_states, _ = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        out = hidden_states * self.activation(gate)
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "swiglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias, quant_config=quant_config, prefix=f"{prefix}.net.0")
        else:
            raise ValueError(f"Unsupported activation function type: {activation_fn}")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            RowParallelLinear(
                inner_dim,
                dim_out,
                bias=bias,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.net.2",
            )
        )

        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for layer in self.net:
            if isinstance(layer, RowParallelLinear):
                # Contiguous for FP8 quantization in RowParallelLinear
                hidden_states, _ = layer(hidden_states.contiguous())
            else:
                output = layer(hidden_states)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
        return hidden_states


class OvisImageAttention(nn.Module):
    def __init__(
        self,
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
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            )
            self.to_out.append(nn.Dropout(dropout))

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.add_kv_proj",
            )

            self.to_add_out = RowParallelLinear(
                input_size=self.inner_dim,
                output_size=query_dim,
                bias=out_bias,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.to_add_out",
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Ensure contiguous for FP8 quantized linear layers
        hidden_states = hidden_states.contiguous()
        qkv, _ = self.to_qkv(hidden_states)

        local_num_heads = self.to_qkv.num_heads
        head_dim = self.head_dim
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (local_num_heads, head_dim))
        key = key.unflatten(-1, (local_num_heads, head_dim))
        value = value.unflatten(-1, (local_num_heads, head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

            encoder_query = encoder_query.unflatten(-1, (local_num_heads, head_dim))
            encoder_key = encoder_key.unflatten(-1, (local_num_heads, head_dim))
            encoder_value = encoder_value.unflatten(-1, (local_num_heads, head_dim))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb  # [S, D/2]
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        hidden_states = self.attn(
            query,
            key,
            value,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :enc_len],
                hidden_states[:, enc_len:],
            )
            # Contiguous for FP8 quantization in RowParallelLinear
            hidden_states, _ = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class OvisImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = ColumnParallelLinear(
            dim,
            self.mlp_hidden_dim * 2,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_mlp",
        )
        self.act_mlp = nn.SiLU()
        self.proj_out = RowParallelLinear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_out",
        )

        self.attn = OvisImageAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]

        combined_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = combined_states

        norm_hidden_states, gate = self.norm(combined_states, emb=temb)

        proj_mlp_output, _ = self.proj_mlp(norm_hidden_states)
        mlp_hidden, mlp_gate = proj_mlp_output.chunk(2, dim=-1)
        mlp_hidden_states = self.act_mlp(mlp_gate) * mlp_hidden

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        combined_features = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        # Contiguous for FP8 quantization in RowParallelLinear
        proj_output, _ = self.proj_out(combined_features.contiguous())

        proj_output = gate.unsqueeze(1) * proj_output

        hidden_states = residual + proj_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class OvisImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, quant_config=_safe_quant_config(quant_config), prefix=f"{prefix}.norm1")
        self.norm1_context = AdaLayerNormZero(
            dim, quant_config=_safe_quant_config(quant_config), prefix=f"{prefix}.norm1_context"
        )

        self.attn = OvisImageAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, dim_out=dim, activation_fn="swiglu", quant_config=quant_config, prefix=f"{prefix}.ff"
        )

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="swiglu", quant_config=quant_config, prefix=f"{prefix}.ff_context"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class OvisImagePosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class OvisImageTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Ovis-Image.

    Reference: https://github.com/AIDC-AI/Ovis-Image

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `6`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `27`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `2048`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _repeated_blocks = ["OvisImageTransformerBlock", "OvisImageSingleTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 6,
        num_single_layers: int = 27,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 2048,
        axes_dims_rope: tuple[int] = (16, 56, 56),
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        model_config = od_config.tf_model_config
        num_layers = model_config.num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = OvisImagePosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        self.context_embedder_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                OvisImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=_safe_quant_config(quant_config),
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                OvisImageSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                    prefix=f"single_transformer_blocks.{i}",
                )
                for i in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`OvisImageTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            img_ids: (`torch.Tensor`):
                The position ids for image tokens.
            txt_ids (`torch.Tensor`):
                The position ids for text tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        timesteps_proj = self.time_proj(timestep)
        temb = self.timestep_embedder(timesteps_proj.to(device=hidden_states.device, dtype=hidden_states.dtype))

        encoder_hidden_states = self.context_embedder_norm(encoder_hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # self attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        # Expose packed shard mappings for LoRA handling of fused projections.
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for name, loaded_weight in weights:
            # 1. Single Block proj_out (RowParallelLinear) weight restructuring
            if "single_transformer_blocks" in name and "proj_out.weight" in name:
                if tp_size > 1:
                    w_attn, w_mlp = loaded_weight.split([self.inner_dim, self.inner_dim * 4], dim=1)
                    w_attn_local = w_attn.chunk(tp_size, dim=1)[tp_rank]
                    w_mlp_local = w_mlp.chunk(tp_size, dim=1)[tp_rank]
                    local_weight = torch.cat([w_attn_local, w_mlp_local], dim=1)

                    if name in params_dict:
                        params_dict[name].data.copy_(local_weight)
                        loaded_params.add(name)
                    continue

            # 2. All SwiGLU linear layers weight/bias restructuring
            if ("net.0.proj.weight" in name) or ("proj_mlp.weight" in name):
                if tp_size > 1:
                    w_hidden, w_gate = loaded_weight.chunk(2, dim=0)
                    w_h_local = w_hidden.chunk(tp_size, dim=0)[tp_rank]
                    w_g_local = w_gate.chunk(tp_size, dim=0)[tp_rank]
                    local_weight = torch.cat([w_h_local, w_g_local], dim=0)

                    if name in params_dict:
                        params_dict[name].data.copy_(local_weight)
                        loaded_params.add(name)
                    continue

            if ("net.0.proj.bias" in name) or ("proj_mlp.bias" in name):
                if tp_size > 1:
                    b_hidden, b_gate = loaded_weight.chunk(2, dim=0)
                    b_h_local = b_hidden.chunk(tp_size, dim=0)[tp_rank]
                    b_g_local = b_gate.chunk(tp_size, dim=0)[tp_rank]
                    local_bias = torch.cat([b_h_local, b_g_local], dim=0)

                    if name in params_dict:
                        params_dict[name].data.copy_(local_bias)
                        loaded_params.add(name)
                    continue

            # 3. Stacked QKV weight loading
            found_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    target_param_name = name.replace(weight_name, param_name)
                    if target_param_name in params_dict:
                        param = params_dict[target_param_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(target_param_name)
                        found_stacked = True
                        break
            if found_stacked:
                continue

            # 4. Standard parameter loading
            if name not in params_dict:
                logger.warning("Unexpected parameter in checkpoint: %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
