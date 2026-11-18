# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 codebook-MTP tower layers.

Each of the 13 layers (4 shared, 9 per-codebook) mixes one code embedding into
the hidden state from below, then runs one gated-attention transformer block
whose causal attention covers the whole sequence. The attention is a regular
paged vLLM ``Attention`` layer: its KV cache lives in the engine's KV pool and
shares the block table of the backbone's full-attention layers, because the
tower has exactly one row per backbone position.

A layer whose code embedding is the only thing below ``enorm`` (every layer
but the first, which also adds the input frame's other codes) folds
``eh_proj``'s embedding half into a per-code table at load:
``eh_proj([enorm(e_c); hnorm(h)]) = T[c] + W_h hnorm(h)`` with
``T = W_e enorm(E)``. That halves the layer's input projection and drops the
embedding read.

The checkpoint uses interleaved (GPT-J) partial RoPE. After loading, every
head's first ``rotary_dim`` query/key channels are permuted from interleaved
to NeoX order (and the q/k norm weights with them): q·k is invariant to a
shared channel permutation, so NeoX RoPE on the permuted channels computes the
same attention, and vLLM's fused QK-norm + RoPE + gate kernel applies.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import CacheConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniAudioDecoderConfig
from vllm_omni.model_executor.models.fish_omni.kernels import (
    eager_qk_rmsnorm_rope_gate,
    gemma_rms_norm,
    gemma_rms_norm_table,
    sigmoid_gate,
    tower_mix_norm,
)


class TowerRotary(nn.Module):
    """NeoX-layout ``[cos | sin]`` cache in fp32 (the reference rotates in fp32)."""

    def __init__(self, config: FishOmniAudioDecoderConfig, max_position: int) -> None:
        super().__init__()
        self.rotary_dim = config.rotary_dim
        inv_freq = 1.0 / config.rope_base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        )
        angles = torch.arange(max_position, dtype=torch.float32).unsqueeze(-1) * inv_freq
        self.register_buffer("cos_sin_cache", torch.cat((angles.cos(), angles.sin()), dim=-1), persistent=False)


def interleaved_to_neox(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Channel order that turns interleaved rotary pairs into NeoX halves."""
    half = rotary_dim // 2
    order = torch.arange(head_dim)
    order[:half] = torch.arange(0, rotary_dim, 2)
    order[half:rotary_dim] = torch.arange(1, rotary_dim, 2)
    return order


class GemmaNorm(nn.Module):
    """Zero-centered RMSNorm weight holder; the math lives in ``kernels``."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.eps = eps


class TowerLayer(nn.Module):
    """``h_out = final_norm(Block(eh_proj([enorm(e); hnorm(h_in)])))``."""

    def __init__(
        self,
        config: FishOmniAudioDecoderConfig,
        rotary: TowerRotary,
        cache_config: CacheConfig | None,
        prefix: str,
        code_table: bool = False,
    ) -> None:
        super().__init__()
        dim = config.dim
        self.num_heads = config.n_head
        self.num_kv_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.norm_eps
        eps = config.norm_eps
        self.token_embedding = nn.Embedding(config.vocab_size, dim)
        self.enorm = GemmaNorm(dim, eps)
        self.hnorm = GemmaNorm(dim, eps)
        self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
        self.input_layernorm = GemmaNorm(dim, eps)
        # Checkpoint rows are [query | gate | key | value]; finalize_weights()
        # regroups them as per-head [query, gate], then key, then value.
        self.qkv_proj = nn.Linear(dim, 2 * self.q_size + 2 * self.kv_size, bias=False)
        self.q_norm = GemmaNorm(self.head_dim, eps)
        self.k_norm = GemmaNorm(self.head_dim, eps)
        self.o_proj = nn.Linear(self.q_size, dim, bias=False)
        self.post_attention_layernorm = GemmaNorm(dim, eps)
        self.gate_up_proj = nn.Linear(dim, 2 * config.intermediate_size, bias=False)
        self.act = SiluAndMul()
        self.down_proj = nn.Linear(config.intermediate_size, dim, bias=False)
        self.final_layernorm = GemmaNorm(dim, eps)
        self.rotary = rotary
        # Set by build_code_table(): [codebook_size, dim], replacing the
        # embedding half of eh_proj.
        self.use_code_table = code_table
        self.code_table: torch.Tensor | None = None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Regroup [q | gate] per head and permute rotary channels to NeoX order."""
        order = interleaved_to_neox(self.head_dim, self.rotary.rotary_dim).to(self.qkv_proj.weight.device)
        weight = self.qkv_proj.weight.data
        q, gate, k, v = weight.split([self.q_size, self.q_size, self.kv_size, self.kv_size], dim=0)
        q = q.reshape(self.num_heads, self.head_dim, -1)[:, order]
        gate = gate.reshape(self.num_heads, self.head_dim, -1)
        k = k.reshape(self.num_kv_heads, self.head_dim, -1)[:, order]
        q_gate = torch.stack((q, gate), dim=1).reshape(2 * self.q_size, -1)
        weight.copy_(torch.cat((q_gate, k.reshape(self.kv_size, -1), v), dim=0))
        self.q_norm.weight.data.copy_(self.q_norm.weight.data[order])
        self.k_norm.weight.data.copy_(self.k_norm.weight.data[order])
        if self.use_code_table:
            self.build_code_table()

    @torch.no_grad()
    def build_code_table(self) -> None:
        """Fold enorm and eh_proj's embedding half into ``T = W_e enorm(E)``.

        The products are taken in fp64 from the same bf16 operands the fused
        path would read, then rounded once to the activation dtype.
        """
        weight = self.eh_proj.weight.data
        dim = weight.shape[0]
        embedding = self.token_embedding.weight.data
        normed = gemma_rms_norm(embedding, self.enorm.weight.data, self.eps)
        table = (normed.double() @ weight[:, :dim].double().t()).to(weight.dtype)
        self.code_table = table
        self.eh_proj.weight.data = weight[:, dim:].contiguous()
        self.eh_proj.in_features = dim
        # The table replaces the embedding and its norm.
        self.token_embedding.weight.data = embedding.new_empty((0, dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        code_is_semantic: torch.Tensor,
        extra_embedding: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.eps
        if self.code_table is not None:
            partial = self.eh_proj(gemma_rms_norm(hidden_states, self.hnorm.weight, eps))
            normed, mixed = gemma_rms_norm_table(
                partial, code_ids, code_is_semantic, self.code_table, self.input_layernorm.weight, eps
            )
        else:
            mixed = self.eh_proj(
                tower_mix_norm(
                    hidden_states,
                    code_ids,
                    code_is_semantic,
                    self.token_embedding.weight,
                    extra_embedding,
                    self.enorm.weight,
                    self.hnorm.weight,
                    eps,
                )
            )
            normed = gemma_rms_norm(mixed, self.input_layernorm.weight, eps)
        qkv = self.qkv_proj(normed)
        q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        qk_rope = eager_qk_rmsnorm_rope_gate if current_platform.device_type == "npu" else fused_qk_rmsnorm_rope_gate
        q, k, gate = qk_rope(
            q_gate,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary.cos_sin_cache,
            positions,
            eps,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary.rotary_dim,
            norm_beta=1.0,
        )
        attn_out = self.attn(q, k, v.contiguous())
        ff_input, residual = gemma_rms_norm(
            self.o_proj(sigmoid_gate(attn_out, gate)), self.post_attention_layernorm.weight, eps, residual=mixed
        )
        normed, _ = gemma_rms_norm(
            self.down_proj(self.act(self.gate_up_proj(ff_input))), self.final_layernorm.weight, eps, residual=residual
        )
        return normed


__all__ = ["TowerLayer", "TowerRotary", "interleaved_to_neox"]
