# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class T5GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Normal RMSNorm but T5Gemma requires (1 + weight)
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


class T5GemmaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.act_fn = get_act_fn(hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x, _ = self.down_proj(x)
        return x


class T5GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        cache_config: VllmConfig | None = None,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters={"base": rope_theta},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)

        # Scale Q appropriately. T5Gemma uses query_pre_attn_scalar=256 => 256**-0.5 = 1/16
        # The standard scaling is head_dim**-0.5. For T5Gemma, head_dim=256.
        # So we don't need to manually scale if F.scaled_dot_product_attention scales by head_dim.
        # But we must reshape.
        batch_size, seq_len, _ = hidden_states.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA repeat KV
        if self.num_kv_heads != self.num_heads:
            num_repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.q_size)

        output, _ = self.o_proj(attn_output)
        return output


class T5GemmaEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.self_attn = T5GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.mlp = T5GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_activation,
        )
        self.pre_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class T5GemmaEncoderModelTP(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList([T5GemmaEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Scaling inputs
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        # Simple position ids for RoPE
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        # Build attention mask: (batch, seq) -> (batch, 1, 1, seq)
        # Assuming typical bidirectional causal mask handling in HF: T5Gemma uses non-causal encoder.
        if attention_mask is not None:
            # HuggingFace expects boolean mask for scaled_dot_product_attention
            # or additive mask (0 and -inf). Let's use boolean matching FA patterns.
            # SDPA expects attention_mask to be boolean (True = keep, False = masking)
            bool_mask = attention_mask.to(torch.bool)
            extended_mask = bool_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        else:
            extended_mask = None

        for idx, layer in enumerate(self.layers):
            # T5Gemma has layer_types switching between "sliding_attention" and "full_attention"
            # However, for text encoder inference, the sequences are typically < max sequence length
            # and local sliding window only affects very long contexts. For simplicity we use full.
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                attention_mask=extended_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # HF checkpoint keys may carry a "model." prefix (e.g.
            # "model.encoder.layers.0...").  Strip it so the rest of the
            # logic only needs to handle the "encoder.*" namespace.
            if name.startswith("model."):
                name = name[len("model.") :]

            if not name.startswith("encoder."):
                continue

            # Strip "encoder." prefix as this model only wraps the encoder
            name = name[len("encoder.") :]

            # Map self_attn to self_attn and correct normalization names
            # HF: layers.0.pre_self_attn_layernorm.weight -> Ours: layers.0.pre_self_attn_layernorm.weight

            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                lookup_name = name.replace(f".{weight_name}.", f".{param_name}.")
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

            loaded_params.add("encoder." + name)
            loaded_params.add("encoder." + lookup_name)

        return loaded_params
