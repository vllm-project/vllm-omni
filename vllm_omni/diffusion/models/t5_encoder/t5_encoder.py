# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class T5SelfAttention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0, f"num_heads ({self.n_heads}) must be divisible by tp_size ({tp_size})"
        self.n_heads_per_partition = self.n_heads // tp_size

        # Fused Q/K/V projection, sharded across heads.
        # HF has separate q, k, v — we fuse into qkv_proj and use
        # stacked_params_mapping in load_weights() to load each shard.
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,  # T5 uses MHA
            bias=False,
        )

        # Output projection: all-reduce back to full d_model
        # Named ``o`` to match HF's ``SelfAttention.o``
        self.o = RowParallelLinear(
            self.inner_dim,
            self.d_model,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
        )

        if has_relative_attention_bias:
            # Store full embedding; slice heads per rank in forward
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Compute relative position bias, returning only the local head shard."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # values: (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # Slice to local heads for this TP rank
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        # (1, local_heads, query_length, key_length)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Fused QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape: (batch, seq, local_heads, d_kv) -> (batch, local_heads, seq, d_kv)
        query_states = query_states.view(batch_size, seq_length, self.n_heads_per_partition, self.d_kv).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.n_heads_per_partition, self.d_kv).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.n_heads_per_partition, self.d_kv).transpose(1, 2)

        # Attention scores: (batch, local_heads, seq, seq)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(seq_length, seq_length, device=scores.device)
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)

        # (batch, local_heads, seq, d_kv) -> (batch, seq, local_heads * d_kv)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)

        attn_output = self.o(attn_output)

        return attn_output, position_bias


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model,
            [config.d_ff, config.d_ff],
            bias=False,
            gather_output=False,
        )
        self.wo = RowParallelLinear(
            config.d_ff,
            config.d_model,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
        )
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MergedColumnParallelLinear outputs concatenated [gate, up]
        gate_up, _ = self.wi(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__()
        self.wi = ColumnParallelLinear(
            config.d_model,
            config.d_ff,
            bias=False,
            gather_output=False,
            return_bias=False,
        )
        self.wo = RowParallelLinear(
            config.d_ff,
            config.d_model,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
        )
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False, prefix: str = ""):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias, prefix=f"{prefix}.SelfAttention")
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(normed, mask=mask, position_bias=position_bias)
        hidden_states = hidden_states + attn_output

        # Clamp for fp16 stability (matching HF T5)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config, prefix=f"{prefix}.DenseReluDense")
        else:
            self.DenseReluDense = T5DenseActDense(config, prefix=f"{prefix}.DenseReluDense")
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False, prefix: str = ""):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                T5LayerSelfAttention(config, has_relative_attention_bias, prefix=f"{prefix}.layer.0"),
                T5LayerFF(config, prefix=f"{prefix}.layer.1"),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](hidden_states, mask=mask, position_bias=position_bias)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, shared: nn.Embedding, prefix: str = ""):
        super().__init__()
        self.embed_tokens = shared
        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=(i == 0), prefix=f"{prefix}.block.{i}")
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Build attention mask: (batch, seq) -> (batch, 1, 1, seq)
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            extended_mask = (1.0 - extended_mask) * torch.finfo(hidden_states.dtype).min
        else:
            extended_mask = None

        position_bias = None
        for block in self.block:
            hidden_states, position_bias = block(
                hidden_states,
                mask=extended_mask,
                position_bias=position_bias,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5EncoderModel(nn.Module):
    """T5 encoder model applying upstream vLLM layers"""

    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__()
        self.config = config
        self.shared = VocabParallelEmbedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared, prefix=f"{prefix}.encoder")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.shared(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.encoder(input_ids, attention_mask=attention_mask)
        return (hidden_states,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q", "q"),
            ("qkv_proj", "k", "k"),
            ("qkv_proj", "v", "v"),
            ("wi", "wi_0", 0),
            ("wi", "wi_1", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                lookup_name = name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
