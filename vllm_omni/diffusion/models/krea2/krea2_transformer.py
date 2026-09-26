# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

# Devices that do not support float64; RoPE frequencies fall back to float32 there.
_FP64_UNSUPPORTED_DEVICES = frozenset({"mps", "npu", "neuron"})


def _rope_freqs_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type in _FP64_UNSUPPORTED_DEVICES else torch.float64


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _linear(
    in_features: int,
    out_features: int,
    bias: bool,
    quant_config: "QuantizationConfig | None",
    prefix: str,
) -> ReplicatedLinear:
    # ReplicatedLinear (not nn.Linear) so the diffusion LoRA manager can wrap these projections.
    return ReplicatedLinear(
        in_features,
        out_features,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
        return_bias=False,
    )


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: the effective multiplier is ``1 + weight``, matching the Krea 2 checkpoint
    format. Activations are upcast so the normalization runs in float32."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(hidden_states.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps)
        return hidden_states.to(dtype)


class Krea2Attention(nn.Module):
    """Self-attention with grouped-query projections, q/k RMSNorm, rotary embeddings and a sigmoid output gate.

    Q/K/V layout is ``[B, seq, heads, head_dim]``; ``attention_mask`` is a 2D boolean key-padding mask
    ``(batch, key_seq_len)`` that the backend broadcasts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
        role: str = "self",
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        q_dim = self.head_dim * self.num_heads
        kv_dim = self.head_dim * self.num_kv_heads
        self.to_q = _linear(hidden_size, q_dim, False, quant_config, _join_prefix(prefix, "to_q"))
        self.to_k = _linear(hidden_size, kv_dim, False, quant_config, _join_prefix(prefix, "to_k"))
        self.to_v = _linear(hidden_size, kv_dim, False, quant_config, _join_prefix(prefix, "to_v"))
        self.to_gate = _linear(hidden_size, hidden_size, False, quant_config, _join_prefix(prefix, "to_gate"))
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList(
            [_linear(hidden_size, hidden_size, False, quant_config, _join_prefix(prefix, "to_out.0")), nn.Dropout(0.0)]
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.num_kv_heads,
            role=role,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query = query.to(value.dtype)
        key = key.to(value.dtype)

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self, dim: int, hidden_dim: int, quant_config: "QuantizationConfig | None" = None, prefix: str = ""
    ) -> None:
        super().__init__()
        self.gate = _linear(dim, hidden_dim, False, quant_config, _join_prefix(prefix, "gate"))
        self.up = _linear(dim, hidden_dim, False, quant_config, _join_prefix(prefix, "up"))
        self.down = _linear(hidden_dim, dim, False, quant_config, _join_prefix(prefix, "down"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm transformer block (no rotary embeddings, no time modulation) used by the text fusion stage."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(
            dim,
            num_heads,
            num_kv_heads,
            eps=eps,
            role="self",
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn"),
        )
        self.ff = Krea2SwiGLU(dim, intermediate_size, quant_config=quant_config, prefix=_join_prefix(prefix, "ff"))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    """Fuses the stack of tapped text-encoder hidden states into a single sequence of text features.

    Two ``layerwise_blocks`` attend across the ``num_text_layers`` axis independently for every token, a linear
    ``projector`` collapses that axis, and two ``refiner_blocks`` attend across the token sequence.
    """

    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(
                    dim,
                    num_heads,
                    num_kv_heads,
                    intermediate_size,
                    eps,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, f"layerwise_blocks.{i}"),
                )
                for i in range(num_layerwise_blocks)
            ]
        )
        self.projector = _linear(num_text_layers, 1, False, quant_config, _join_prefix(prefix, "projector"))
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(
                    dim,
                    num_heads,
                    num_kv_heads,
                    intermediate_size,
                    eps,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, f"refiner_blocks.{i}"),
                )
                for i in range(num_refiner_blocks)
            ]
        )

    def forward(self, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            eps=norm_eps,
            role="self",
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn"),
        )
        self.ff = Krea2SwiGLU(
            hidden_size, intermediate_size, quant_config=quant_config, prefix=_join_prefix(prefix, "ff")
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # temb: (B, 1, 6 * hidden_size), shared across all blocks; each block only learns an additive table.
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding (cos-first, input scaled by 1000) followed by a two-layer MLP.

    Keeps the sequence dimension at size 1 so the per-block modulations broadcast over tokens.
    """

    def __init__(
        self, embed_dim: int, hidden_size: int, quant_config: "QuantizationConfig | None" = None, prefix: str = ""
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = _linear(embed_dim, hidden_size, True, quant_config, _join_prefix(prefix, "linear_1"))
        self.linear_2 = _linear(hidden_size, hidden_size, True, quant_config, _join_prefix(prefix, "linear_2"))

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2TextProjection(nn.Module):
    """Projects the fused text features into the transformer width."""

    def __init__(
        self,
        text_dim: int,
        hidden_size: int,
        eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = _linear(text_dim, hidden_size, True, quant_config, _join_prefix(prefix, "linear_1"))
        self.linear_2 = _linear(hidden_size, hidden_size, True, quant_config, _join_prefix(prefix, "linear_2"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    """Final adaptive RMSNorm and output projection."""

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = _linear(hidden_size, out_channels, True, quant_config, _join_prefix(prefix, "linear"))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2RotaryPosEmbed(nn.Module):
    """Multi-axis (t, h, w) rotary position embedding, following the Flux/Krea 2 convention."""

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = _rope_freqs_dtype(ids.device)
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Krea2Transformer2DModel(nn.Module):
    r"""The single-stream MMDiT flow-matching backbone used by the Krea 2 pipeline (vLLM-Omni port).

    Text conditioning enters as a stack of hidden states tapped from several layers of a multimodal text encoder. A
    small text-fusion transformer collapses the layer axis and refines the token sequence; the result is concatenated
    with the patchified image latents into a single ``[text, image]`` sequence processed by the transformer blocks.
    The timestep conditions every block through one shared modulation vector plus per-block learned tables.
    """

    _repeated_blocks = ["Krea2TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "transformer_blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
        od_config: "OmniDiffusionConfig | None" = None,
        quant_config: "QuantizationConfig | None" = None,
    ) -> None:
        super().__init__()
        # The pipeline reads ``self.dtype`` to cast inputs before the transformer forward.
        self.dtype = od_config.dtype if od_config is not None else torch.get_default_dtype()
        self.od_config = od_config

        hidden_size = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim={attention_head_dim}"
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.img_in = _linear(in_channels, hidden_size, True, quant_config, "img_in")
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size, quant_config, "time_embed")
        self.time_mod_proj = _linear(hidden_size, 6 * hidden_size, True, quant_config, "time_mod_proj")
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
            quant_config=quant_config,
            prefix="text_fusion",
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, norm_eps, quant_config, "txt_in")
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    norm_eps=norm_eps,
                    quant_config=quant_config,
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.final_layer = Krea2FinalLayer(hidden_size, in_channels, norm_eps, quant_config, "final_layer")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Predict the flow-matching velocity for the image tokens.

        Args:
            hidden_states: Packed (patchified) noisy image latents ``(batch, image_seq_len, in_channels)``.
            encoder_hidden_states: Tapped text-encoder hidden states
                ``(batch, text_seq_len, num_text_layers, text_hidden_dim)``.
            timestep: Flow-matching time in ``[0, 1]`` of shape ``(batch,)``.
            position_ids: ``(t, h, w)`` rotary coordinates ``(text_seq_len + image_seq_len, 3)``.
            encoder_attention_mask: Boolean mask marking valid text tokens ``(batch, text_seq_len)`` or ``None``.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}.")

        batch_size, image_seq_len, _ = hidden_states.shape

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        # 2D key-padding masks: padded text tokens are excluded as keys; their lanes are dropped at the output slice.
        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask.bool()
            image_mask = text_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([text_attention_mask, image_mask], dim=1)

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        return self.final_layer(hidden_states, temb)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Module/parameter names mirror the diffusers ``Krea2Transformer2DModel`` exactly, so weights map 1:1.
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
