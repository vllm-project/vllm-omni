# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from cache_dit import ForwardPattern
from torch import nn
from transformers import Qwen3VLConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3MLP

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def _prefix(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


class HiDreamO1RotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = rope_parameters.get("mrope_section", [24, 20, 20])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        inv_freq = self.inv_freq[None, None, :, None].float().to(hidden_states.device)
        inv_freq = inv_freq.expand(3, position_ids.shape[1], -1, 1)
        positions = position_ids[:, :, None, :].float()
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ positions).transpose(2, 3)
            freqs_t = freqs[0]
            for dim, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dim] * 3
                freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


class HiDreamO1Attention(nn.Module):
    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={config.num_attention_heads} must be divisible by tensor_parallel_size={tp_size}"
            )
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads={config.num_key_value_heads} must be divisible by "
                    f"tensor_parallel_size={tp_size}"
                )
        elif tp_size % config.num_key_value_heads != 0:
            raise ValueError(
                f"tensor_parallel_size={tp_size} must be divisible by num_key_value_heads={config.num_key_value_heads}"
            )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=_prefix(prefix, "qkv_proj"),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=_prefix(prefix, "o_proj"),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            num_heads=self.qkv_proj.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.scaling,
            num_kv_heads=self.qkv_proj.num_kv_heads,
            prefix=_prefix(prefix, "attn"),
            role="self",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        full_attn_spans: list[list[tuple[int, int]]] | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.qkv_proj.num_heads * self.head_dim
        kv_size = self.qkv_proj.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.view(batch_size, seq_len, self.qkv_proj.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.qkv_proj.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.qkv_proj.num_kv_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary_pos_emb(query, key, *position_embeddings)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = self.attn(
            query,
            key,
            value,
            AttentionMetadata(
                attn_mask=attention_mask,
                full_attn_spans=full_attn_spans,
            ),
        )
        attn_output = attn_output.reshape(batch_size, seq_len, q_size)
        attn_output, _ = self.o_proj(attn_output)
        return attn_output


class HiDreamO1DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = HiDreamO1Attention(
            config,
            quant_config=quant_config,
            prefix=_prefix(prefix, "self_attn"),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        full_attn_spans: list[list[tuple[int, int]]] | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            full_attn_spans,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class HiDreamO1TextModel(nn.Module):
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"layers": ForwardPattern.Pattern_3},
        has_separate_cfg=True,
    )

    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                HiDreamO1DecoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=_prefix(prefix, f"layers.{layer_idx}"),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HiDreamO1RotaryEmbedding(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        full_attn_spans: list[list[tuple[int, int]]] | None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                full_attn_spans=full_attn_spans,
            )
        return self.norm(hidden_states)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=_prefix(prefix, "mlp.0"),
                return_bias=False,
            ),
            nn.SiLU(),
            RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=_prefix(prefix, "mlp.2"),
                return_bias=False,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timestep.device) / half
        )
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = self.timestep_embedding(timestep * 1000, self.frequency_embedding_size)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        patch_size: int,
        in_channels: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        bottleneck_dim = hidden_size // 4
        self.proj1 = ColumnParallelLinear(
            patch_size * patch_size * in_channels,
            bottleneck_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=_prefix(prefix, "proj1"),
            return_bias=False,
        )
        self.proj2 = RowParallelLinear(
            bottleneck_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=_prefix(prefix, "proj2"),
            return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(hidden_states))


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        patch_size: int,
        out_channels: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear = ColumnParallelLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=_prefix(prefix, "linear"),
            return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


@dataclass
class HiDreamO1ImageOutput:
    x_pred: torch.Tensor


class HiDreamO1ImageModel(nn.Module):
    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = HiDreamO1TextModel(
            config.text_config,
            quant_config=quant_config,
            prefix=_prefix(prefix, "language_model"),
        )
        self.patch_size = 32
        self.in_channels = 3
        hidden_size = config.text_config.hidden_size
        self.t_embedder1 = TimestepEmbedder(
            hidden_size,
            quant_config=quant_config,
            prefix=_prefix(prefix, "t_embedder1"),
        )
        self.x_embedder = BottleneckPatchEmbed(
            hidden_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            quant_config=quant_config,
            prefix=_prefix(prefix, "x_embedder"),
        )
        self.final_layer2 = FinalLayer(
            hidden_size,
            patch_size=self.patch_size,
            out_channels=self.in_channels,
            quant_config=quant_config,
            prefix=_prefix(prefix, "final_layer2"),
        )
        self.tms_token_id = 151673

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None,
        full_attn_spans: list[list[tuple[int, int]]] | None,
    ) -> HiDreamO1ImageOutput:
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        timestep_embeds = self.t_embedder1(timestep.to(inputs_embeds.device))
        timestep_mask = input_ids == self.tms_token_id
        inputs_embeds = torch.where(
            timestep_mask.unsqueeze(-1),
            timestep_embeds.unsqueeze(1).expand_as(inputs_embeds),
            inputs_embeds,
        )

        image_embeds = self.x_embedder(vinputs.to(inputs_embeds.device)).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, image_embeds], dim=1)
        batch_size, total_seq_len, _ = inputs_embeds.shape

        if attention_mask is not None:
            if attention_mask.shape[0] == 1 and batch_size > 1:
                attention_mask = attention_mask.expand(batch_size, -1, -1, -1)
            expected_mask_shape = (batch_size, 1, total_seq_len, total_seq_len)
            if attention_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"attention_mask must have shape {expected_mask_shape}, got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        if full_attn_spans is None or len(full_attn_spans) != batch_size:
            raise ValueError(
                f"full_attn_spans must contain one entry per batch item; got {full_attn_spans!r} "
                f"for batch_size={batch_size}"
            )

        hidden_states = self.language_model(
            inputs_embeds,
            position_ids,
            attention_mask,
            full_attn_spans,
        )
        return HiDreamO1ImageOutput(x_pred=self.final_layer2(hidden_states))


class HiDreamO1ImageTransformer(nn.Module):
    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = HiDreamO1ImageModel(
            config,
            quant_config=quant_config,
            prefix=_prefix(prefix, "model"),
        )

    @property
    def supports_piecewise_attention(self) -> bool:
        return all(
            layer.self_attn.attn.attn_backend.supports_piecewise_spans for layer in self.model.language_model.layers
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None,
        full_attn_spans: list[list[tuple[int, int]]] | None = None,
        **_: object,
    ) -> HiDreamO1ImageOutput:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=vinputs,
            timestep=timestep,
            attention_mask=attention_mask,
            full_attn_spans=full_attn_spans,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.startswith("model.visual.") or name == "lm_head.weight":
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params:
                    continue
                param = params[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


__all__ = ["HiDreamO1ImageTransformer"]
