# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
)
from transformers.utils import ModelOutput
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.bagel import BagelConfig

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata as DiffusionAttentionMetadata
from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W) or (3, H, W)
    x: (N, L, patch_size**2 *3) or (L, patch_size**2 *3)
    """
    is_batch = imgs.ndim == 4
    if not is_batch:
        imgs = imgs.unsqueeze(0)

    # n: batch, c: channel, h: grid_h, p: patch_h, w: grid_w, q: patch_w
    x = imgs.reshape(imgs.shape[0], 3, imgs.shape[2] // p, p, imgs.shape[3] // p, p)
    # Permute to (n, grid_h, grid_w, c, patch_h, patch_w) to match Conv2d (c, h, w) flattening
    x = torch.einsum("nchpwq->nhwcpq", x)
    x = x.reshape(imgs.shape[0], -1, 3 * p**2)

    if not is_batch:
        x = x.squeeze(0)
    return x


class MLPconnector(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation="gelu_pytorch_tanh",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_dim, output_dim, bias=True, gather_output=False, quant_config=quant_config, prefix=f"{prefix}.fc1"
        )
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.ReLU()
        self.fc2 = RowParallelLinear(
            output_dim, output_dim, bias=True, input_is_parallel=True, quant_config=quant_config, prefix=f"{prefix}.fc2"
        )

    def forward(self, x):
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        return self.fc2(x_parallel)[0]


class BagelRotaryEmbedding(nn.Module):
    """Standalone rotary embedding that generates cos/sin from position ids.

    Replaces HuggingFace's Qwen2RotaryEmbedding while preserving full
    ``rope_scaling`` support.  When ``config.rope_scaling`` is set (e.g.
    linear, dynamic-NTK, YaRN, …), we delegate the ``inv_freq`` /
    ``attention_scaling`` computation to HF's ``ROPE_INIT_FUNCTIONS`` so
    that the frequency basis and scaling factor are identical to the
    original checkpoint.  This module has no learnable parameters.
    """

    def __init__(self, config):
        super().__init__()

        if config.rope_scaling is not None:
            # Delegate to HF's rope-scaling helpers for non-default types.
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(config, device=None)
        else:
            dim = config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate cos/sin embeddings for given position ids.

        Args:
            x: Input tensor (only used for dtype inference).
            position_ids: Position indices, shape (batch_size, seq_len).

        Returns:
            cos, sin: Rotary embeddings, each of shape (batch_size, seq_len, dim).
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BagelMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            input_is_parallel=True,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x, _ = self.down_proj(x)
        return x


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)


class Qwen2MoTConfig(Qwen2Config):
    """Configuration for Qwen2MoT (Mixture of Tokens) model.

    This is fundamentally different from Qwen2, hence the distinct name.
    """

    model_type = "qwen2_mot"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="eager",
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: NaiveCache | None = None


class PackedAttentionMoT(nn.Module):
    """Packed attention with Mixture-of-Tokens routing for understanding/generation.

    Uses vLLM's QKVParallelLinear and RowParallelLinear for tensor parallelism
    support, following the same pattern as vLLM's Qwen2Attention.

    The q/k/v projections are stacked into a single QKVParallelLinear:
      - qkv_proj      : stacks q_proj + k_proj + v_proj  (understanding + gen text)
      - qkv_proj_moe_gen : stacks q_proj_moe_gen + k_proj_moe_gen + v_proj_moe_gen (gen vae)
    """

    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.parallel_config = parallel_config

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Understanding mode projections (stacked q/k/v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Generation mode MoE projections (stacked q/k/v)
        self.qkv_proj_moe_gen = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_moe_gen",
        )
        self.o_proj_moe_gen = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_moe_gen",
        )

        # QK normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_op = RotaryEmbedding(is_neox_style=True)

        # SP (Ulysses / Ring) attention for generation mode denoising
        sp_size = parallel_config.sequence_parallel_size if parallel_config is not None else 1
        if sp_size is not None and sp_size > 1:
            self.sp_attn = DiffusionAttention(
                num_heads=self.total_num_heads,
                head_size=self.head_dim,
                softmax_scale=1.0 / (self.head_dim**0.5),
                causal=False,
                num_kv_heads=self.total_num_kv_heads,
            )
        else:
            self.sp_attn = None

    def _is_sp_active(self) -> bool:
        """Check if SP is active for this attention layer."""
        if self.sp_attn is None:
            return False
        if not is_forward_context_available():
            return False
        return get_forward_context().sp_active

    def _forward_sp_gen(
        self,
        packed_query_sequence: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        past_key_values: NaiveCache | None,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_indexes: torch.Tensor,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        """SP-aware attention for gen mode denoising.

        Converts packed format to batched (1, S, H, D) and uses the diffusion
        Attention layer (Ulysses / Ring) with joint mechanism:
          - Main Q/K/V: VAE tokens (split across SP ranks)
          - Joint Q: text marker Q (replicated)
          - Joint K/V: KV cache K/V + text marker K/V (replicated)
        """
        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

        packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
        packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

        # Project text tokens through base qkv
        text_qkv, _ = self.qkv_proj(packed_text_query_sequence)
        text_q, text_k, text_v = text_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Project vae tokens through moe_gen qkv
        vae_qkv, _ = self.qkv_proj_moe_gen(packed_vae_query_sequence)
        vae_q, vae_k, vae_v = vae_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape to (tokens, heads, head_dim)
        text_q = text_q.view(-1, self.num_heads, self.head_dim)
        text_k = text_k.view(-1, self.num_kv_heads, self.head_dim)
        text_v = text_v.view(-1, self.num_kv_heads, self.head_dim)
        vae_q = vae_q.view(-1, self.num_heads, self.head_dim)
        vae_k = vae_k.view(-1, self.num_kv_heads, self.head_dim)
        vae_v = vae_v.view(-1, self.num_kv_heads, self.head_dim)

        # Apply QK norms
        text_q = self.q_norm(text_q.to(torch.float32))
        text_k = self.k_norm(text_k.to(torch.float32))
        vae_q = self.q_norm_moe_gen(vae_q.to(torch.float32))
        vae_k = self.k_norm_moe_gen(vae_k.to(torch.float32))

        # Apply RoPE - need to build per-token cos/sin for text and vae separately
        # packed_query_position_embeddings are ordered as the packed sequence
        cos_full, sin_full = [x[..., : self.head_dim // 2] for x in packed_query_position_embeddings]

        # Extract cos/sin for text and vae positions
        text_cos = cos_full[packed_text_indexes]
        text_sin = sin_full[packed_text_indexes]
        vae_cos = cos_full[packed_vae_token_indexes]
        vae_sin = sin_full[packed_vae_token_indexes]

        text_q = self.rotary_op(text_q.to(text_cos.dtype).unsqueeze(0), text_cos, text_sin).squeeze(0)
        text_k = self.rotary_op(text_k.to(text_cos.dtype).unsqueeze(0), text_cos, text_sin).squeeze(0)
        vae_q = self.rotary_op(vae_q.to(vae_cos.dtype).unsqueeze(0), vae_cos, vae_sin).squeeze(0)
        vae_k = self.rotary_op(vae_k.to(vae_cos.dtype).unsqueeze(0), vae_cos, vae_sin).squeeze(0)

        text_q = text_q.to(torch.bfloat16)
        text_k = text_k.to(torch.bfloat16)
        text_v = text_v.to(torch.bfloat16)
        vae_q = vae_q.to(torch.bfloat16)
        vae_k = vae_k.to(torch.bfloat16)
        vae_v = vae_v.to(torch.bfloat16)

        # Build joint K/V: [kv_cache, text_markers] (replicated across SP ranks)
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            cache_k = past_key_values.key_cache[self.layer_idx]
            cache_v = past_key_values.value_cache[self.layer_idx]
            joint_k = torch.cat([cache_k, text_k], dim=0).unsqueeze(0)
            joint_v = torch.cat([cache_v, text_v], dim=0).unsqueeze(0)
        else:
            joint_k = text_k.unsqueeze(0)
            joint_v = text_v.unsqueeze(0)

        # Reshape to batched (1, S, H, D) for diffusion Attention
        vae_q_4d = vae_q.unsqueeze(0)
        vae_k_4d = vae_k.unsqueeze(0)
        vae_v_4d = vae_v.unsqueeze(0)
        text_q_4d = text_q.unsqueeze(0)

        # Call SP-aware attention: VAE as main, text+cache as joint
        attn_out = self.sp_attn(
            vae_q_4d,
            vae_k_4d,
            vae_v_4d,
            DiffusionAttentionMetadata(
                joint_query=text_q_4d,
                joint_key=joint_k,
                joint_value=joint_v,
                joint_strategy="front",
            ),
        )
        # attn_out: (1, text_len + local_vae_len, H, D)
        text_len = text_q.shape[0]
        attn_out = attn_out.squeeze(0)  # (text_len + local_vae_len, H, D)
        text_attn = attn_out[:text_len].reshape(text_len, self.q_size)
        vae_attn = attn_out[text_len:].reshape(-1, self.q_size)

        # Apply output projections
        text_out, _ = self.o_proj(text_attn)
        vae_out, _ = self.o_proj_moe_gen(vae_attn)

        # Merge back into packed format
        total_len = packed_query_sequence.shape[0]
        full_output = text_out.new_zeros((total_len, self.hidden_size))
        full_output[packed_text_indexes] = text_out
        full_output[packed_vae_token_indexes] = vae_out

        return full_output, past_key_values

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        # SP path for gen-mode denoising (non-causal, no KV update)
        if (
            mode == "gen"
            and not update_past_key_values
            and not is_causal
            and self._is_sp_active()
            and packed_vae_token_indexes is not None
            and packed_text_indexes is not None
        ):
            return self._forward_sp_gen(
                packed_query_sequence=packed_query_sequence,
                packed_query_position_embeddings=packed_query_position_embeddings,
                past_key_values=past_key_values,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )

        if mode == "und":
            qkv, _ = self.qkv_proj(packed_query_sequence)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            packed_query_states = q.view(-1, self.num_heads, self.head_dim)
            packed_key_states = k.view(-1, self.num_kv_heads, self.head_dim)
            packed_value_states = v.view(-1, self.num_kv_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == "gen":
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            # Project text tokens through base qkv
            text_qkv, _ = self.qkv_proj(packed_text_query_sequence)
            text_q, text_k, text_v = text_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Project vae tokens through moe_gen qkv
            vae_qkv, _ = self.qkv_proj_moe_gen(packed_vae_query_sequence)
            vae_q, vae_k, vae_v = vae_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Merge into packed tensors
            total_len = packed_query_sequence.shape[0]
            packed_query_states = packed_query_sequence.new_zeros((total_len, self.q_size))
            packed_key_states = packed_query_sequence.new_zeros((total_len, self.kv_size))
            packed_value_states = packed_query_sequence.new_zeros((total_len, self.kv_size))

            packed_query_states[packed_text_indexes] = text_q
            packed_query_states[packed_vae_token_indexes] = vae_q
            packed_key_states[packed_text_indexes] = text_k
            packed_key_states[packed_vae_token_indexes] = vae_k
            packed_value_states[packed_text_indexes] = text_v
            packed_value_states[packed_vae_token_indexes] = vae_v

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_kv_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_kv_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                packed_query_states[packed_vae_token_indexes]
            )

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                packed_key_states[packed_vae_token_indexes]
            )

        cos, sin = [x[..., : self.head_dim // 2] for x in packed_query_position_embeddings]
        packed_query_states = self.rotary_op(packed_query_states.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
        packed_key_states = self.rotary_op(packed_key_states.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_kv_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_kv_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.q_size)
        if mode == "und":
            packed_attn_output, _ = self.o_proj(packed_attn_output)
        elif mode == "gen":
            text_out, _ = self.o_proj(packed_attn_output[packed_text_indexes])
            vae_out, _ = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])
            full_output = text_out.new_zeros((packed_attn_output.shape[0], self.hidden_size))
            full_output[packed_text_indexes] = text_out
            full_output[packed_vae_token_indexes] = vae_out
            packed_attn_output = full_output

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        attn_module: type[nn.Module] | None = PackedAttentionMoT,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = attn_module(
            config, layer_idx, parallel_config=parallel_config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )

        self.mlp = BagelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.mlp_moe_gen = BagelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_moe_gen",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        packed_query_sequence: torch.Tensor | None = None,
        query_lens: torch.Tensor = None,
        packed_query_position_embeddings: torch.Tensor = None,
        packed_query_indexes: torch.Tensor = None,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        if packed_query_sequence is None:
            packed_query_sequence = hidden_states
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(
                torch.bfloat16
            )

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTModel(Qwen2PreTrainedModel):
    def __init__(
        self,
        config,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = "Mo" in config.layer_module

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen2MoTDecoderLayer(
                    config,
                    layer_idx,
                    attn_module=PackedAttentionMoT,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            packed_query_sequence, past_key_values = decoder_layer(
                hidden_states=packed_query_sequence,
                encoder_hidden_states=None,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                    packed_query_sequence[packed_vae_token_indexes]
                )
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2MoTForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.model = Qwen2MoTModel(
            config, parallel_config=parallel_config, quant_config=quant_config, prefix=f"{prefix}.model"
        )
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for vLLM parallel layers.

        Handles stacked parameter remapping for QKVParallelLinear:
          - q_proj, k_proj, v_proj -> qkv_proj (shard ids: q, k, v)
          - q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen -> qkv_proj_moe_gen
        Other parallel layers (gate_proj, up_proj, down_proj, embed_tokens, etc.)
        keep HF checkpoint names and use weight_loader for TP sharding.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # More specific _moe_gen patterns FIRST to avoid substring
            # ambiguity (`.q_proj` is a substring of `.q_proj_moe_gen`).
            (".qkv_proj_moe_gen", ".q_proj_moe_gen", "q"),
            (".qkv_proj_moe_gen", ".k_proj_moe_gen", "k"),
            (".qkv_proj_moe_gen", ".v_proj_moe_gen", "v"),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # MLP gate/up projections — fused into MergedColumnParallelLinear.
            # HF checkpoints store separate gate_proj / up_proj weights;
            # these entries remap them to the fused gate_up_proj parameter.
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        self.stacked_params_mapping = stacked_params_mapping
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                param = params_dict.get(stacked_name)
                if param is None:
                    break
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                name = stacked_name
                loaded = True
                break

            if not loaded:
                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(torch.zeros(max_num_patch_per_side**2, hidden_size), requires_grad=False)
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


class Bagel(nn.Module):
    config_class = BagelConfig
    base_model_prefix = "bagel"

    def __init__(
        self,
        language_model,
        vit_model,
        config: BagelConfig,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self.parallel_config = parallel_config

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(
                self.vit_hidden_size,
                self.hidden_size,
                config.connector_act,
                quant_config=quant_config,
                prefix=f"{prefix}.connector",
            )
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    @property
    def _sp_size(self) -> int:
        if self.parallel_config is None:
            return 1
        sp = self.parallel_config.sequence_parallel_size
        return sp if sp is not None and sp > 1 else 1

    def _split_vae_for_sp(
        self,
        x_t: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_seqlens: torch.Tensor,
        packed_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split VAE tokens across SP ranks for the denoising loop.

        Returns adjusted (x_t, packed_vae_position_ids, packed_vae_token_indexes,
        packed_text_indexes, packed_seqlens, packed_position_ids) for the local rank.
        """
        sp_size = self._sp_size
        sp_rank = get_sequence_parallel_rank()
        num_vae = x_t.shape[0]
        assert num_vae % sp_size == 0, f"VAE token count {num_vae} not divisible by SP size {sp_size}"
        chunk = num_vae // sp_size
        start = sp_rank * chunk
        end = start + chunk

        local_x_t = x_t[start:end]
        local_vae_pos_ids = packed_vae_position_ids[start:end]

        # Rebuild local packed indices:
        # packed sequence = [start_of_image, local_vae_tokens..., end_of_image]
        # BAGEL always has exactly 2 text markers (start/end_of_image).
        num_text = packed_text_indexes.shape[0]
        assert num_text == 2, f"Expected exactly 2 text markers (start/end_of_image), got {num_text}"
        assert packed_seqlens.numel() == 1, (
            f"SP currently supports single-image batches only, got {packed_seqlens.numel()} sequences"
        )
        local_vae_len = chunk
        local_total = num_text + local_vae_len

        local_text_indexes = torch.tensor([0, local_vae_len + 1], device=packed_text_indexes.device)
        local_vae_indexes = torch.arange(1, 1 + local_vae_len, device=packed_vae_token_indexes.device)

        local_seqlens = torch.tensor([local_total], device=packed_seqlens.device, dtype=packed_seqlens.dtype)

        # Build local position IDs preserving global positions.
        # Text markers keep their original positions; VAE tokens get
        # the global positions for the local chunk.
        text_pos_ids = packed_position_ids[packed_text_indexes]
        vae_pos_ids_full = packed_position_ids[packed_vae_token_indexes]
        local_vae_pos = vae_pos_ids_full[start:end]
        local_position_ids = torch.zeros(
            local_total, device=packed_position_ids.device, dtype=packed_position_ids.dtype
        )
        local_position_ids[local_text_indexes] = text_pos_ids
        local_position_ids[local_vae_indexes] = local_vae_pos

        return local_x_t, local_vae_pos_ids, local_vae_indexes, local_text_indexes, local_seqlens, local_position_ids

    def _gather_vae_for_sp(self, local_v_t: torch.Tensor) -> torch.Tensor:
        """Gather VAE velocity outputs from all SP ranks."""
        sp_size = self._sp_size
        gathered = [torch.zeros_like(local_v_t) for _ in range(sp_size)]
        sp_group = get_sp_group()
        dist.all_gather(gathered, local_v_t.contiguous(), group=sp_group.device_group)
        return torch.cat(gathered, dim=0)

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids["bos_token_id"]] + text_ids + [new_token_ids["eos_token_id"]]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: list,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, : h * p, : w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side,
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_input(self, curr_kvlens, curr_rope, image_sizes, new_token_ids=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_position_ids = self.get_flattened_position_ids(
                H, W, self.latent_downsample, max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_position_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w

            packed_init_noises.append(torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size**2))
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_seqlens.append(num_image_tokens + 2)

            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        # Construct Output
        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        return self.prepare_input(curr_kvlens, curr_rope, image_sizes, new_token_ids)

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @staticmethod
    def _merge_naive_caches(caches: list) -> NaiveCache:
        """Merge multiple NaiveCache objects by concatenating KV tensors per layer."""
        if not caches:
            # Handle empty list case gracefully if desired,
            # though original code also crashed on this.
            return NaiveCache(0)

        num_layers = len(caches[0].key_cache)
        merged = NaiveCache(num_layers)
        for layer_idx in range(num_layers):
            merged.key_cache[layer_idx] = torch.cat([c.key_cache[layer_idx] for c in caches], dim=0)
            merged.value_cache[layer_idx] = torch.cat([c.value_cache[layer_idx] for c in caches], dim=0)
        return merged

    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: tuple[float, float] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: torch.LongTensor | None = None,
        cfg_text_packed_position_ids: torch.LongTensor | None = None,
        cfg_text_past_key_values: NaiveCache | None = None,
        cfg_text_key_values_lens: torch.IntTensor | None = None,
        cfg_text_packed_key_value_indexes: torch.LongTensor | None = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: torch.LongTensor | None = None,
        cfg_img_packed_position_ids: torch.LongTensor | None = None,
        cfg_img_past_key_values: NaiveCache | None = None,
        cfg_img_key_values_lens: torch.IntTensor | None = None,
        cfg_img_packed_key_value_indexes: torch.LongTensor | None = None,
        return_trajectory_latents: bool = False,
        scheduler: object | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        # Optional trajectory recording for RL rollout data collection
        trajectory_latents: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_timesteps: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_log_probs: list[torch.Tensor] | None = (
            [] if (return_trajectory_latents and scheduler is not None) else None
        )
        _sched_kw = scheduler_kwargs or {}

        use_cfg_text = cfg_text_scale > 1.0
        use_cfg_img = cfg_img_scale > 1.0

        # ── Detect CFG parallel mode ──
        cfg_parallel_ready = use_cfg_text and get_classifier_free_guidance_world_size() > 1

        if cfg_parallel_ready:
            return self._generate_image_parallel(
                x_t=x_t,
                timesteps=timesteps,
                dts=dts,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_seqlens=packed_seqlens,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_interval=cfg_interval,
                cfg_text_scale=cfg_text_scale,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                cfg_img_scale=cfg_img_scale,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                return_trajectory_latents=return_trajectory_latents,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
            )

        # ── SP + CFG: sequential single-branch forwards ──
        use_sp = self._sp_size > 1
        if use_sp and use_cfg_text:
            for i, t in enumerate(timesteps):
                timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                in_cfg_window = t > cfg_interval[0] and t <= cfg_interval[1]
                cfg_text_scale_ = cfg_text_scale if in_cfg_window else 1.0
                cfg_img_scale_ = cfg_img_scale if in_cfg_window else 1.0

                common = dict(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_seqlens=packed_seqlens,
                )

                v_t = self.forward_single_branch(
                    **common,
                    packed_indexes=packed_indexes,
                    packed_position_ids=packed_position_ids,
                    key_values_lens=key_values_lens,
                    past_key_values=past_key_values,
                    packed_key_value_indexes=packed_key_value_indexes,
                )

                if cfg_text_scale_ > 1.0:
                    cfg_text_v_t = self.forward_single_branch(
                        **common,
                        packed_indexes=cfg_text_packed_query_indexes,
                        packed_position_ids=cfg_text_packed_position_ids,
                        key_values_lens=cfg_text_key_values_lens,
                        past_key_values=cfg_text_past_key_values,
                        packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                    )
                    cfg_img_v_t = None
                    if cfg_img_scale_ > 1.0:
                        cfg_img_v_t = self.forward_single_branch(
                            **common,
                            packed_indexes=cfg_img_packed_query_indexes,
                            packed_position_ids=cfg_img_packed_position_ids,
                            key_values_lens=cfg_img_key_values_lens,
                            past_key_values=cfg_img_past_key_values,
                            packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                        )
                    v_t = self._combine_cfg(
                        v_t,
                        cfg_text_v_t,
                        cfg_img_v_t,
                        cfg_text_scale_,
                        cfg_img_scale_,
                        cfg_renorm_type,
                        cfg_renorm_min,
                    )

                if scheduler is not None:
                    out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                    x_t = out.prev_sample
                    if trajectory_log_probs is not None and out.log_prob is not None:
                        trajectory_log_probs.append(out.log_prob)
                else:
                    x_t = x_t - v_t.to(x_t.device) * dts[i]
                if return_trajectory_latents:
                    trajectory_latents.append(x_t.clone())
                    trajectory_timesteps.append(timesteps[i] - dts[i])

            unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
            return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

        # ── SP without CFG: direct single-branch loop ──
        if use_sp:
            for i, t in enumerate(timesteps):
                timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                v_t = self.forward_single_branch(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_indexes=packed_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=past_key_values,
                    packed_key_value_indexes=packed_key_value_indexes,
                )
                if scheduler is not None:
                    out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                    x_t = out.prev_sample
                    out_log_prob = getattr(out, "log_prob", None)
                    if trajectory_log_probs is not None and out_log_prob is not None:
                        trajectory_log_probs.append(out_log_prob)
                else:
                    x_t = x_t - v_t.to(x_t.device) * dts[i]
                if return_trajectory_latents:
                    trajectory_latents.append(x_t.clone())
                    trajectory_timesteps.append(timesteps[i] - dts[i])

            unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
            return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

        # ── Batched CFG mode (cfg_parallel_size=1, no SP) ──
        cfg_batched = None

        if use_cfg_text:
            seq_len = int(packed_seqlens.sum())

            # Branch 0: main (gen_context), always present
            branches_qi = [packed_indexes]
            branches_kvi = [packed_key_value_indexes]
            branches_kvl = [key_values_lens]
            branches_pid = [packed_position_ids]
            branches_cache = [past_key_values]

            # Branch 1: cfg_text (unconditional text), always present when use_cfg_text
            branches_qi.append(cfg_text_packed_query_indexes)
            branches_kvi.append(cfg_text_packed_key_value_indexes)
            branches_kvl.append(cfg_text_key_values_lens)
            branches_pid.append(cfg_text_packed_position_ids)
            branches_cache.append(cfg_text_past_key_values)

            # Branch 2: cfg_img (text-only, no image), optional
            if use_cfg_img:
                branches_qi.append(cfg_img_packed_query_indexes)
                branches_kvi.append(cfg_img_packed_key_value_indexes)
                branches_kvl.append(cfg_img_key_values_lens)
                branches_pid.append(cfg_img_packed_position_ids)
                branches_cache.append(cfg_img_past_key_values)

            num_branches = len(branches_cache)

            # Compute per-branch offsets in the merged KV+Q attention tensor
            merged_offsets = [0]
            for b_idx in range(num_branches):
                merged_offsets.append(merged_offsets[-1] + int(branches_kvl[b_idx].sum()) + seq_len)

            cfg_batched = {
                "num_branches": num_branches,
                "seq_len": seq_len,
                "batched_query_lens": packed_seqlens.repeat(num_branches),
                "batched_position_ids": torch.cat(branches_pid),
                "batched_kv_lens": torch.cat(branches_kvl),
                "batched_query_indexes": torch.cat(
                    [qi + merged_offsets[b_idx] for b_idx, qi in enumerate(branches_qi)]
                ),
                "batched_kv_indexes": torch.cat(
                    [kvi + merged_offsets[b_idx] for b_idx, kvi in enumerate(branches_kvi)]
                ),
                "batched_text_indexes": torch.cat(
                    [packed_text_indexes + b_idx * seq_len for b_idx in range(num_branches)]
                ),
                "batched_vae_indexes": torch.cat(
                    [packed_vae_token_indexes + b_idx * seq_len for b_idx in range(num_branches)]
                ),
                "merged_cache": self._merge_naive_caches(branches_cache),
            }

        for i, t in enumerate(timesteps):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self.forward(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale_,
                cfg_img_scale=cfg_img_scale_,
                cfg_batched=cfg_batched,
            )

            if scheduler is not None:
                out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                x_t = out.prev_sample
                if trajectory_log_probs is not None and out.log_prob is not None:
                    trajectory_log_probs.append(out.log_prob)
            else:
                x_t = x_t - v_t.to(x_t.device) * dts[i]  # velocity pointing from data to noise
            if return_trajectory_latents:
                trajectory_latents.append(x_t.clone())
                trajectory_timesteps.append(timesteps[i] - dts[i])

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

    def _generate_image_parallel(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        dts: torch.Tensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        cfg_interval: tuple[float, float],
        cfg_text_scale: float,
        cfg_text_packed_query_indexes: torch.LongTensor | None,
        cfg_text_packed_position_ids: torch.LongTensor | None,
        cfg_text_past_key_values: NaiveCache | None,
        cfg_text_key_values_lens: torch.IntTensor | None,
        cfg_text_packed_key_value_indexes: torch.LongTensor | None,
        cfg_img_scale: float,
        cfg_img_packed_query_indexes: torch.LongTensor | None,
        cfg_img_packed_position_ids: torch.LongTensor | None,
        cfg_img_past_key_values: NaiveCache | None,
        cfg_img_key_values_lens: torch.IntTensor | None,
        cfg_img_packed_key_value_indexes: torch.LongTensor | None,
        return_trajectory_latents: bool = False,
        scheduler: object | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        """CFG parallel denoising loop: each rank computes one CFG branch.

        Rank 0: gen branch (full conditioning)
        Rank 1: text_cfg branch (unconditional text)
        Rank 2: img_cfg branch (no image condition), only when cfg_img_scale > 1.0
        """
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        cfg_world_size = get_classifier_free_guidance_world_size()
        use_cfg_img = cfg_img_scale > 1.0

        # Validate cfg_parallel_size vs cfg_img_scale consistency
        if cfg_world_size == 3 and not use_cfg_img:
            raise ValueError(
                f"cfg_parallel_size=3 requires cfg_img_scale > 1.0, "
                f"but got cfg_img_scale={cfg_img_scale}. "
                f"Use cfg_parallel_size=2 for text-only CFG parallel(text2img), or set cfg_img_scale > 1.0."
            )
        if cfg_world_size == 2 and use_cfg_img:
            raise ValueError(
                f"Image CFG (cfg_img_scale={cfg_img_scale}) requires cfg_parallel_size=3, "
                f"but got cfg_parallel_size=2. "
                f"Use cfg_parallel_size=3 to enable image CFG in parallel mode."
            )

        # Ensure all ranks start with the same x_t (initial noise may differ
        # across ranks when no per-request seed is set).
        x_t = x_t.contiguous()
        cfg_group.broadcast(x_t, src=0)

        # Select this rank's branch inputs
        if cfg_rank == 0:
            # Gen branch: use main inputs directly
            branch_position_ids = packed_position_ids
            branch_indexes = packed_indexes
            branch_past_key_values = past_key_values
            branch_key_values_lens = key_values_lens
            branch_key_value_indexes = packed_key_value_indexes
        elif cfg_rank == 1:
            # Text CFG branch
            branch_position_ids = cfg_text_packed_position_ids
            branch_indexes = cfg_text_packed_query_indexes
            branch_past_key_values = cfg_text_past_key_values
            branch_key_values_lens = cfg_text_key_values_lens
            branch_key_value_indexes = cfg_text_packed_key_value_indexes
        elif cfg_rank == 2:
            # Image CFG branch
            branch_position_ids = cfg_img_packed_position_ids
            branch_indexes = cfg_img_packed_query_indexes
            branch_past_key_values = cfg_img_past_key_values
            branch_key_values_lens = cfg_img_key_values_lens
            branch_key_value_indexes = cfg_img_packed_key_value_indexes
        else:
            raise RuntimeError(f"Unexpected cfg_rank={cfg_rank} for Bagel 3-branch CFG parallel")

        trajectory_latents: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_timesteps: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_log_probs: list[torch.Tensor] | None = (
            [] if (return_trajectory_latents and scheduler is not None) else None
        )
        _sched_kw = scheduler_kwargs or {}

        for i, t in enumerate(timesteps):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            use_cfg_this_step = t > cfg_interval[0] and t <= cfg_interval[1] and cfg_text_scale > 1.0

            if use_cfg_this_step:
                # CFG interval: each rank computes its own branch
                local_v_t = self.forward_single_branch(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_indexes=branch_indexes,
                    packed_position_ids=branch_position_ids,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=branch_key_values_lens,
                    past_key_values=branch_past_key_values,
                    packed_key_value_indexes=branch_key_value_indexes,
                )

                gathered = cfg_group.all_gather(local_v_t, separate_tensors=True)
                v_t = self._combine_cfg(
                    gathered[0],
                    gathered[1],
                    gathered[2] if (use_cfg_img and len(gathered) > 2) else None,
                    cfg_text_scale,
                    cfg_img_scale,
                    cfg_renorm_type,
                    cfg_renorm_min,
                )
            else:
                # Outside CFG interval: all ranks compute with gen inputs, no comm
                v_t = self.forward_single_branch(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_indexes=packed_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=past_key_values,
                    packed_key_value_indexes=packed_key_value_indexes,
                )

            if scheduler is not None:
                out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                x_t = out.prev_sample
                if trajectory_log_probs is not None and out.log_prob is not None:
                    trajectory_log_probs.append(out.log_prob)
            else:
                x_t = x_t - v_t.to(x_t.device) * dts[i]
            if return_trajectory_latents:
                trajectory_latents.append(x_t.clone())
                trajectory_timesteps.append(timesteps[i] - dts[i])

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

    @staticmethod
    def _combine_cfg(
        v_t: torch.Tensor,
        cfg_text_v_t: torch.Tensor,
        cfg_img_v_t: torch.Tensor | None,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_renorm_type: str,
        cfg_renorm_min: float,
    ) -> torch.Tensor:
        """Combine 3-branch CFG predictions with renormalization.

        Args:
            v_t: velocity from gen branch (full conditioning)
            cfg_text_v_t: velocity from text_cfg branch (unconditional text)
            cfg_img_v_t: velocity from img_cfg branch (no image), or None
            cfg_text_scale: text guidance scale
            cfg_img_scale: image guidance scale
            cfg_renorm_type: "text_channel", "global", or "channel"
            cfg_renorm_min: minimum renormalization scale
        """
        if cfg_renorm_type == "text_channel":
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
            scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_t_text = v_t_text_ * scale
            if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
            else:
                v_t = v_t_text
        else:
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

            if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
            else:
                v_t_ = v_t_text_

            # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
            if cfg_renorm_type == "global":
                norm_v_t = torch.norm(v_t)
                norm_v_t_ = torch.norm(v_t_)
            elif cfg_renorm_type == "channel":
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
            else:
                raise NotImplementedError(f"{cfg_renorm_type} is not supported")
            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_t = v_t_ * scale

        return v_t

    def forward_single_branch(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        """Run a single-branch forward pass (no CFG batching).

        Used by CFG parallel mode where each rank computes one branch.
        Returns the velocity v_t for the given branch.
        Supports Ulysses / Ring SP when parallel_config.sequence_parallel_size > 1.
        """
        use_sp = self._sp_size > 1

        if use_sp:
            # Split VAE tokens across SP ranks
            (
                local_x_t,
                local_vae_pos_ids,
                local_vae_indexes,
                local_text_indexes,
                local_seqlens,
                local_position_ids,
            ) = self._split_vae_for_sp(
                x_t,
                packed_vae_position_ids,
                packed_vae_token_indexes,
                packed_text_indexes,
                packed_seqlens,
                packed_position_ids,
            )

            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
            packed_sequence = packed_text_embedding.new_zeros((int(local_seqlens.sum()), self.hidden_size))
            packed_sequence[local_text_indexes] = packed_text_embedding

            assert timestep.unique().shape[0] == 1
            packed_pos_embed = self.latent_pos_embed(local_vae_pos_ids)
            local_timestep = timestep[: local_x_t.shape[0]]
            packed_timestep_embeds = self.time_embedder(local_timestep)
            x_t_emb = self.vae2llm(local_x_t) + packed_timestep_embeds + packed_pos_embed
            if x_t_emb.dtype != packed_sequence.dtype:
                x_t_emb = x_t_emb.to(packed_sequence.dtype)
            packed_sequence[local_vae_indexes] = x_t_emb

            # Build local packed_indexes for KV cache merging.
            # In the denoising loop packed_indexes is always contiguous
            # (arange(kv_len, kv_len + total)), so we can safely build
            # the local slice from scratch.
            local_total = int(local_seqlens.sum())
            kv_len = int(key_values_lens.sum())
            original_total = int(packed_seqlens.sum())
            assert torch.equal(
                packed_indexes,
                torch.arange(kv_len, kv_len + original_total, device=packed_indexes.device, dtype=packed_indexes.dtype),
            ), "packed_indexes must be contiguous for SP; non-contiguous layout not supported"
            local_packed_indexes = torch.arange(
                kv_len,
                kv_len + local_total,
                device=packed_indexes.device,
                dtype=packed_indexes.dtype,
            )

            extra_inputs = {}
            if self.use_moe:
                extra_inputs["mode"] = "gen"
                extra_inputs["packed_vae_token_indexes"] = local_vae_indexes
                extra_inputs["packed_text_indexes"] = local_text_indexes

            output = self.language_model.forward(
                packed_query_sequence=packed_sequence,
                query_lens=local_seqlens,
                packed_query_position_ids=local_position_ids,
                packed_query_indexes=local_packed_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )

            local_v_t = self.llm2vae(output.packed_query_sequence)
            local_v_t = local_v_t[local_vae_indexes]
            return self._gather_vae_for_sp(local_v_t)

        # Original non-SP path
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t_emb = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t_emb.dtype != packed_sequence.dtype:
            x_t_emb = x_t_emb.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t_emb

        extra_inputs = {}
        if self.use_moe:
            extra_inputs["mode"] = "gen"
            extra_inputs["packed_vae_token_indexes"] = packed_vae_token_indexes
            extra_inputs["packed_text_indexes"] = packed_text_indexes

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        return v_t

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_batched: dict | None = None,
    ):
        # Build query sequence (identical for all CFG branches)
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs["mode"] = "gen"

        use_cfg = cfg_text_scale > 1.0
        cfg_text_v_t = None
        cfg_img_v_t = None

        if use_cfg and cfg_batched is not None:
            # ── Batched CFG: single LLM forward for all branches ──
            seq_len = cfg_batched["seq_len"]
            num_branches = cfg_batched["num_branches"]

            batched_sequence = packed_sequence.repeat(num_branches, 1)

            if self.use_moe:
                extra_inputs["packed_text_indexes"] = cfg_batched["batched_text_indexes"]
                extra_inputs["packed_vae_token_indexes"] = cfg_batched["batched_vae_indexes"]

            output = self.language_model.forward(
                packed_query_sequence=batched_sequence,
                query_lens=cfg_batched["batched_query_lens"],
                packed_query_position_ids=cfg_batched["batched_position_ids"],
                packed_query_indexes=cfg_batched["batched_query_indexes"],
                past_key_values=cfg_batched["merged_cache"],
                key_values_lens=cfg_batched["batched_kv_lens"],
                packed_key_value_indexes=cfg_batched["batched_kv_indexes"],
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )

            # Extract per-branch velocities from batched output
            all_hidden = output.packed_query_sequence
            assert all_hidden.shape[0] == seq_len * num_branches, (
                f"Expected packed sequence length {seq_len * num_branches}, but got {all_hidden.shape[0]}"
            )

            v_t = self.llm2vae(all_hidden[:seq_len])[packed_vae_token_indexes]

            branch_idx = 1
            cfg_text_v_t = self.llm2vae(all_hidden[branch_idx * seq_len : (branch_idx + 1) * seq_len])[
                packed_vae_token_indexes
            ]
            branch_idx += 1
            if cfg_img_scale > 1.0:
                cfg_img_v_t = self.llm2vae(all_hidden[branch_idx * seq_len : (branch_idx + 1) * seq_len])[
                    packed_vae_token_indexes
                ]
        else:
            # ── Single forward (no CFG or outside cfg_interval) ──
            if self.use_moe:
                extra_inputs["packed_vae_token_indexes"] = packed_vae_token_indexes
                extra_inputs["packed_text_indexes"] = packed_text_indexes

            output = self.language_model.forward(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=packed_position_ids,
                packed_query_indexes=packed_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            v_t = self.llm2vae(output.packed_query_sequence)
            v_t = v_t[packed_vae_token_indexes]

        # ── CFG combination ──
        if use_cfg:
            v_t = self._combine_cfg(
                v_t,
                cfg_text_v_t,
                cfg_img_v_t,
                cfg_text_scale,
                cfg_img_scale,
                cfg_renorm_type,
                cfg_renorm_min,
            )

        return v_t
