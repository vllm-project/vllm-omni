# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 LLM with Mixture-of-Tokenizers (MoT) for SenseNova-U1.

Ported from the sensenova_u1 package with vllm tensor-parallel support:
- QKVParallelLinear for fused q/k/v projections (both und and gen paths)
- MergedColumnParallelLinear for fused gate+up projections
- RowParallelLinear for o_proj and down_proj
- VocabParallelEmbedding for token embeddings
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention


@dataclass
class SenseNovaU1ModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: DynamicCache | None = None


@dataclass
class SenseNovaU1CausalLMOutput:
    logits: torch.Tensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Mask / Cache utilities
# ---------------------------------------------------------------------------


def create_block_causal_mask(index: torch.Tensor):
    """Block-wise causal mask from 1D time-index. Returns (1, 1, L, L)."""
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)
    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))
    return torch.where(mask[None, None], 0.0, float("-inf"))


def prepare_flash_kv_cache(past_key_values, current_len: int, batch_size: int):
    """Convert prefix cache [B,H,S,D] → flash layout [B,S,H,D] and
    preallocate buffers for [prefix + current] tokens."""
    if past_key_values is None:
        return
    for layer in past_key_values.layers:
        past_k, past_v = layer.keys, layer.values
        if past_k is None or past_v is None:
            layer.flash_prefix_len = 0
            layer.flash_total_len = current_len
            layer.flash_k_cache = None
            layer.flash_v_cache = None
            continue
        past_k_flash = past_k.transpose(1, 2).contiguous()
        past_v_flash = past_v.transpose(1, 2).contiguous()
        prefix_len = past_k_flash.shape[1]
        total_len = prefix_len + current_len
        k_cache = torch.empty(
            (batch_size, total_len, past_k_flash.shape[2], past_k_flash.shape[3]),
            device=past_k_flash.device,
            dtype=past_k_flash.dtype,
        )
        v_cache = torch.empty_like(k_cache)
        k_cache[:, :prefix_len].copy_(past_k_flash)
        v_cache[:, :prefix_len].copy_(past_v_flash)
        layer.flash_prefix_len = prefix_len
        layer.flash_total_len = total_len
        layer.flash_k_cache = k_cache
        layer.flash_v_cache = v_cache


def clear_flash_kv_cache(past_key_values):
    if past_key_values is None:
        return
    for layer in past_key_values.layers:
        for attr in ("flash_prefix_len", "flash_total_len", "flash_k_cache", "flash_v_cache"):
            if hasattr(layer, attr):
                delattr(layer, attr)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _compute_default_rope_parameters(config, device=None, **_kw):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


class Qwen3RotaryEmbedding(nn.Module):
    """Custom RoPE that doubles head_dim then subsamples every-other freq."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config

        def _rope_init_fn(cfg, dev=None):
            inv_freq, attn_scaling = _compute_default_rope_parameters(cfg, dev)
            cfg2 = copy.deepcopy(cfg)
            head_dim = getattr(cfg2, "head_dim", cfg2.hidden_size // cfg2.num_attention_heads)
            cfg2.head_dim = int(head_dim) * 2
            inv_freq_full, _ = _compute_default_rope_parameters(cfg2, dev)
            return inv_freq_full[::2], attn_scaling

        inv_freq, self.attention_scaling = _rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class SenseNovaU1MLP(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu", quant_config=None, prefix: str = ""
    ):
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
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        from transformers.activations import ACT2FN

        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Attention with MoT
# ---------------------------------------------------------------------------


class SenseNovaU1Attention(nn.Module):
    """Multi-head attention with Mixture-of-Tokenizers (MoT) routing.

    Two parallel sets of Q/K/V/O projections:
    - qkv_proj / o_proj: understanding (und) path
    - qkv_proj_mot_gen / o_proj_mot_gen: generation (gen) path

    3D RoPE: time (t), height (h), width (w) with separate frequencies.
    """

    def __init__(self, config, layer_idx: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_kv_groups = self.total_num_heads // self.total_num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        bias = getattr(config, "attention_bias", True)

        # Understanding path
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Generation (MoT) path
        self.qkv_proj_mot_gen = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_mot_gen",
        )
        self.o_proj_mot_gen = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_mot_gen",
        )

        # Local head counts (TP-sharded)
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # QK norms — understanding
        self.q_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        # QK norms — generation
        self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        # 3D RoPE: t uses half head_dim, h/w each use quarter head_dim
        t_config = copy.deepcopy(config)
        t_config.head_dim = config.head_dim // 2
        self.rotary_emb = Qwen3RotaryEmbedding(config=t_config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 4
        hw_config.rope_theta = config.rope_theta_hw
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.scaling,
            num_kv_heads=self.num_heads,
            prefix=f"{prefix}.attn",
        )
        self.attn.attention = self.attn.sdpa_fallback

    @staticmethod
    def _align_mask_dtype(mask: torch.Tensor | None, query: torch.Tensor) -> torch.Tensor | None:
        """SDPA requires float ``attn_mask`` to match query's dtype."""
        if mask is None or not mask.is_floating_point() or mask.dtype == query.dtype:
            return mask
        return mask.to(query.dtype)

    def _run_attn(
        self,
        query_bhsd: torch.Tensor,
        key_bhsd: torch.Tensor,
        value_bhsd: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run unified attention with [B, H, S, D] inputs. Returns [B, S, H, D]."""
        if self.num_kv_groups > 1:
            n = self.num_kv_groups
            key_bhsd = key_bhsd.repeat_interleave(n, dim=1)
            value_bhsd = value_bhsd.repeat_interleave(n, dim=1)
        q = query_bhsd.transpose(1, 2).contiguous()
        k = key_bhsd.transpose(1, 2).contiguous()
        v = value_bhsd.transpose(1, 2).contiguous()
        attention_mask = self._align_mask_dtype(attention_mask, q)
        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        return self.attn(q, k, v, attn_metadata)

    def _run_attn_bshd(
        self,
        query_bshd: torch.Tensor,
        key_bshd: torch.Tensor,
        value_bshd: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run unified attention with [B, S, H, D] inputs. Returns [B, S, H, D]."""
        if self.num_kv_groups > 1:
            n = self.num_kv_groups
            key_bshd = key_bshd.repeat_interleave(n, dim=2)
            value_bshd = value_bshd.repeat_interleave(n, dim=2)
        attention_mask = self._align_mask_dtype(attention_mask, query_bshd)
        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        return self.attn(query_bshd, key_bshd, value_bshd, attn_metadata)

    def _project_and_rope(self, hidden_states, indexes, qkv_proj, q_norm, k_norm, q_norm_hw, k_norm_hw):
        """Project Q/K/V via the given QKVParallelLinear and apply 3D RoPE."""
        input_shape = hidden_states.shape[:-1]  # (B, S)
        qkv, _ = qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(*input_shape, self.num_heads, self.head_dim)
        k = k.view(*input_shape, self.num_kv_heads, self.head_dim)
        v = v.view(*input_shape, self.num_kv_heads, self.head_dim)

        # Split head_dim into t and hw halves
        q_t, q_hw = q.chunk(2, dim=-1)
        k_t, k_hw = k.chunk(2, dim=-1)

        # Norms then transpose to [B, H, S, D/2]
        q_t = q_norm(q_t).transpose(1, 2)
        k_t = k_norm(k_t).transpose(1, 2)
        q_hw = q_norm_hw(q_hw).transpose(1, 2)
        k_hw = k_norm_hw(k_hw).transpose(1, 2)

        # Split hw into h and w quarters
        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        # RoPE for each dimension
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        q_t, k_t = apply_rotary_pos_emb(q_t, k_t, cos_t, sin_t)
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        q_h, k_h = apply_rotary_pos_emb(q_h, k_h, cos_h, sin_h)
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        q_w, k_w = apply_rotary_pos_emb(q_w, k_w, cos_w, sin_w)

        # Reassemble: [B, H, S, head_dim]
        query_states = torch.cat([q_t, q_h, q_w], dim=-1)
        key_states = torch.cat([k_t, k_h, k_w], dim=-1)
        value_states = v.transpose(1, 2)  # [B, H, S, D]
        return query_states, key_states, value_states

    def forward_und(self, hidden_states, indexes, attention_mask, past_key_values=None, **kwargs):
        """Understanding path — unified Attention with explicit 4D mask."""
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states = self._project_and_rope(
            hidden_states,
            indexes,
            self.qkv_proj,
            self.q_norm,
            self.k_norm,
            self.q_norm_hw,
            self.k_norm_hw,
        )
        update_cache = kwargs.get("update_cache", True)
        if past_key_values is not None:
            if update_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            else:
                layer = past_key_values.layers[self.layer_idx]
                if layer.keys is not None:
                    key_states = torch.cat([layer.keys, key_states], dim=2)
                    value_states = torch.cat([layer.values, value_states], dim=2)

        attn_output = self._run_attn(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output, _ = self.o_proj(attn_output)
        return attn_output

    def forward_gen(self, hidden_states, indexes, attention_mask, past_key_values=None, **kwargs):
        """Generation path — unified Attention, bidirectional with optional KV cache."""
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states = self._project_and_rope(
            hidden_states,
            indexes,
            self.qkv_proj_mot_gen,
            self.q_norm_mot_gen,
            self.k_norm_mot_gen,
            self.q_norm_hw_mot_gen,
            self.k_norm_hw_mot_gen,
        )
        update_cache = kwargs.get("update_cache", True)

        if attention_mask is None:
            # Bidirectional path: no causal mask, optionally attend to a prefix.
            q = query_states.transpose(1, 2).contiguous()  # [B,S,H,D]
            k_cur = key_states.transpose(1, 2).contiguous()
            v_cur = value_states.transpose(1, 2).contiguous()

            if past_key_values is not None:
                if update_cache:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                    k = key_states.transpose(1, 2).contiguous()
                    v = value_states.transpose(1, 2).contiguous()
                else:
                    layer = past_key_values.layers[self.layer_idx]
                    if hasattr(layer, "flash_k_cache") and layer.flash_k_cache is not None:
                        prefix_len = layer.flash_prefix_len
                        cur_len = k_cur.shape[1]
                        layer.flash_k_cache[:, prefix_len : prefix_len + cur_len].copy_(k_cur)
                        layer.flash_v_cache[:, prefix_len : prefix_len + cur_len].copy_(v_cur)
                        k = layer.flash_k_cache[:, : prefix_len + cur_len]
                        v = layer.flash_v_cache[:, : prefix_len + cur_len]
                    else:
                        past_k = past_key_values.layers[self.layer_idx].keys
                        past_v = past_key_values.layers[self.layer_idx].values
                        if past_k is not None:
                            k = torch.cat([past_k.transpose(1, 2).contiguous(), k_cur], dim=1)
                            v = torch.cat([past_v.transpose(1, 2).contiguous(), v_cur], dim=1)
                        else:
                            k, v = k_cur, v_cur
            else:
                k, v = k_cur, v_cur

            attn_output = self._run_attn_bshd(q, k, v, None)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output, _ = self.o_proj_mot_gen(attn_output)
            return attn_output

        # Masked fallback with explicit 4D additive mask
        if past_key_values is not None:
            if update_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            else:
                layer = past_key_values.layers[self.layer_idx]
                if layer.keys is not None:
                    key_states = torch.cat([layer.keys, key_states], dim=2)
                    value_states = torch.cat([layer.values, value_states], dim=2)

        attn_output = self._run_attn(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output, _ = self.o_proj_mot_gen(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states,
        image_gen_indicators,
        exist_und,
        exist_gen,
        indexes,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        if exist_und and not exist_gen:
            return self.forward_und(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        if not exist_und and exist_gen:
            return self.forward_gen(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        raise NotImplementedError("Mixed und+gen tokens in a single forward not implemented for initial port")


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class SenseNovaU1DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.self_attn = SenseNovaU1Attention(
            config, layer_idx, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = SenseNovaU1MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.mlp_mot_gen = SenseNovaU1MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_mot_gen",
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def _forward_und(self, hidden_states, indexes, attention_mask, past_key_values, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            image_gen_indicators=None,
            exist_und=True,
            exist_gen=False,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states

    def _forward_gen(self, hidden_states, indexes, attention_mask, past_key_values, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            image_gen_indicators=None,
            exist_und=False,
            exist_gen=True,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp_mot_gen(self.post_attention_layernorm_mot_gen(hidden_states))
        return residual + hidden_states

    def forward(
        self,
        hidden_states,
        image_gen_indicators,
        exist_und,
        exist_gen,
        indexes,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        if exist_und and not exist_gen:
            return self._forward_und(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        if not exist_und and exist_gen:
            return self._forward_gen(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        raise NotImplementedError("Mixed und+gen tokens in a single forward not implemented for initial port")


# ---------------------------------------------------------------------------
# Model (decoder stack)
# ---------------------------------------------------------------------------


class SenseNovaU1Model(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                SenseNovaU1DecoderLayer(config, i, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        image_gen_indicators=None,
        indexes=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if image_gen_indicators is None:
            exist_und, exist_gen = True, False
        else:
            exist_und = (~image_gen_indicators).any().item()
            exist_gen = image_gen_indicators.any().item()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Resolve attention mask. Callers must always provide `indexes`; this
        # keeps the model stateless and safe under concurrent requests.
        assert indexes is not None, "SenseNovaU1Model.forward requires explicit `indexes`."
        if not isinstance(attention_mask, dict):
            past_len = past_key_values.get_seq_length() if past_key_values else 0
            seq_len = inputs_embeds.shape[1]
            total_len = past_len + seq_len
            mask = torch.zeros(1, 1, seq_len, total_len, device=inputs_embeds.device)
            if seq_len > 1:
                causal = torch.tril(torch.ones(seq_len, seq_len, device=inputs_embeds.device))
                mask[:, :, :, past_len:] = torch.where(causal == 1, 0.0, float("-inf"))
            causal_mask_mapping = {"full_attention": mask}
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_und=exist_und,
                exist_gen=exist_gen,
                indexes=indexes,
                attention_mask=causal_mask_mapping.get(layer.attention_type, causal_mask_mapping.get("full_attention")),
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if not exist_gen:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm_mot_gen(hidden_states)

        return SenseNovaU1ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# ForCausalLM wrapper
# ---------------------------------------------------------------------------


class SenseNovaU1ForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.model = SenseNovaU1Model(config, quant_config=quant_config, prefix=f"{prefix}.model")
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=f"{prefix}.lm_head")
        # LogitsProcessor handles the TP all-gather of vocab-sharded ParallelLMHead
        # outputs so callers see full-vocab logits regardless of tp_size.
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids=None,
        indexes=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        embed_only: bool = False,
        compute_logits: bool = True,
        **kwargs,
    ):
        # Routing every access through ``forward`` keeps any externally-attached
        # hooks (e.g. CPU-offload swap) firing for sub-module accesses such as
        # token embedding lookup or hidden-state-only model runs. Callers should
        # prefer this entry point over reaching into ``self.model`` /
        # ``self.lm_head`` directly so that offloaded weights get materialised
        # on the right device first.
        if embed_only:
            if input_ids is None:
                raise ValueError("embed_only=True requires input_ids")
            return SenseNovaU1CausalLMOutput(
                inputs_embeds=self.model.embed_tokens(input_ids),
            )

        outputs = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.logits_processor(self.lm_head, outputs.last_hidden_state) if compute_logits else None
        return SenseNovaU1CausalLMOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens
