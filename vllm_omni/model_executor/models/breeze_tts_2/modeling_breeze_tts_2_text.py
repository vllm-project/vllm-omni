# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Breeze T5Gemma2 text encoder.

The Breeze checkpoint contains a T5Gemma2 encoder, not the older T5Gemma
implementation used by some diffusion models. This module keeps the encoder
local to Breeze so checkpoint names and attention semantics stay explicit,
while using vLLM tensor-parallel layers for the expensive projections.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _RMSNorm(nn.Module):
    """T5Gemma2 RMSNorm, whose checkpoint stores a zero-centered weight."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).to(hidden_states.dtype)


class _RotaryEmbedding(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        for layer_type in dict.fromkeys(config.layer_types):
            params = dict(config.rope_parameters.get(layer_type, {}))
            base = float(params.get("rope_theta", 10_000.0))
            factor = float(params.get("factor", 1.0))
            dim = int(config.head_dim)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            if params.get("rope_type") == "linear":
                inv_freq = inv_freq / factor
            elif params.get("rope_type", "default") != "default":
                raise ValueError(f"Unsupported T5Gemma2 rope type: {params.get('rope_type')!r}")
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        expanded = inv_freq[None, :, None].to(device=hidden_states.device)
        positions = position_ids[:, None, :].float()
        freqs = (expanded.float() @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, heads, repeats, seq_len, head_dim)
        .reshape(batch, heads * repeats, seq_len, head_dim)
    )


def _attention_mask(
    padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    layer_type: str,
    sliding_window: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if layer_type == "full_attention" and padding_mask is None:
        return None
    valid = (
        torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        if padding_mask is None
        else padding_mask.to(device=device, dtype=torch.bool)
    )
    allowed = valid[:, None, None, :].expand(batch_size, 1, seq_len, seq_len)
    if layer_type == "sliding_attention":
        if sliding_window is None:
            raise ValueError("T5Gemma2 sliding_attention requires sliding_window")
        q_idx = torch.arange(seq_len, device=device)[:, None]
        kv_idx = torch.arange(seq_len, device=device)[None, :]
        left = (int(sliding_window) + 1) // 2
        right = int(sliding_window) // 2 + 1
        distance = q_idx - kv_idx
        local = ((distance >= 0) & (distance < left)) | ((distance < 0) & (-distance < right))
        allowed = allowed & local[None, None, :, :]
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask


class _SelfAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size != 0:
            raise ValueError("T5Gemma2 attention heads must be divisible by tensor parallel size")
        self.layer_type = config.layer_type
        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads) // tp_size
        self.num_kv_heads = max(1, int(config.num_key_value_heads) // tp_size)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = float(config.query_pre_attn_scalar) ** -0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
        )
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_norm(query.view(shape).transpose(1, 2))
        key = self.k_norm(key.view(shape).transpose(1, 2))
        value = value.view(shape).transpose(1, 2)
        query, key = _apply_rotary(query, key, *position_embeddings)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        # SDPA scales by head_dim**-0.5; adjust Q to T5Gemma2's scalar.
        query = query * (self.scaling / (self.head_dim**-0.5))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        output = output.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
        output, _ = self.o_proj(output)
        return output


class _MLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.act_fn = get_act_fn(config.hidden_activation)
        self.dropout = nn.Dropout(float(config.dropout_rate))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(self.dropout(self.act_fn(gate) * up))
        return output


class _EncoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        layer_config = type("LayerConfig", (), dict(vars(config)))()
        layer_config.layer_type = config.layer_types[layer_idx]
        self.attention_type = layer_config.layer_type
        self.pre_self_attn_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = _SelfAttention(layer_config)
        self.post_self_attn_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _MLP(config)
        self.post_feedforward_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.dropout = nn.Dropout(float(config.dropout_rate))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + self.dropout(self.post_self_attn_layernorm(hidden_states))
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(self.post_feedforward_layernorm(hidden_states))
        return hidden_states


class BreezeTTS2TextEncoder(nn.Module):
    """Frozen T5Gemma2 encoder returning ``(B, S, hidden_size)`` states."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.embed_scale = float(config.hidden_size**0.5)
        self.eoi_token_index = int(getattr(config, "eoi_token_index", 256000))
        # T5Gemma2 stores this learned replacement vector under the input
        # embedding module (``embed_tokens.eoi_embedding``), so keep the same
        # parameter path for checkpoint loading and tied-weight inspection.
        self.embed_tokens.eoi_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.layers = nn.ModuleList([_EncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)
        self.dropout = nn.Dropout(float(config.dropout_rate))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Breeze text encoder expects 2D input_ids, got {tuple(input_ids.shape)}")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Breeze text encoder attention_mask must match input_ids: "
                f"{tuple(attention_mask.shape)} != {tuple(input_ids.shape)}"
            )
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        eoi_mask = input_ids == self.eoi_token_index
        if bool(eoi_mask.any()):
            hidden_states = torch.where(
                eoi_mask.unsqueeze(-1),
                self.embed_tokens.eoi_embedding.to(dtype=hidden_states.dtype),
                hidden_states,
            )
        hidden_states = self.dropout(hidden_states)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, positions, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        masks = {
            layer_type: _attention_mask(
                attention_mask,
                batch_size=input_ids.shape[0],
                seq_len=input_ids.shape[1],
                layer_type=layer_type,
                sliding_window=self.config.sliding_window,
                dtype=hidden_states.dtype,
                device=input_ids.device,
            )
            for layer_type in set(self.config.layer_types)
        }
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings[layer.attention_type],
                masks[layer.attention_type],
            )
        return self.dropout(self.norm(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            target = name
            shard_id: str | int | None = None
            for packed_name, source_name, source_id in stacked_params_mapping:
                marker = f".{source_name}."
                if marker in name:
                    target = name.replace(marker, f".{packed_name}.")
                    shard_id = source_id
                    break
            parameter = params.get(target)
            if parameter is None:
                continue
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(parameter, tensor)
            else:
                loader(parameter, tensor, shard_id)
            loaded.add(target)
        return loaded


__all__ = ["BreezeTTS2TextEncoder"]
