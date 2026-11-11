# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused-kernel forward for the AudioVAE's transformers Qwen2 stacks.

Run through transformers, a Qwen2 layer launches about 45 kernels: the
RMSNorm, RoPE and SiLU arithmetic as separate elementwise ops and attention
through SDPA with a materialized window mask. Here a layer is one QKV GEMM,
the RoPE kernel, FlashAttention with the sliding window, the output GEMM, a
fused residual-add RMSNorm, one gate/up GEMM, SiLU-and-mul and the down GEMM.
Sequences are packed along the token dimension; with a paged KV cache the
keys of earlier calls are attended as well.

The weights stay in the transformers modules. Results match transformers up
to the attention kernel's accumulation order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm import _custom_ops as ops


@dataclass
class PagedKV:
    """Per-layer paged key/value caches and this call's addressing into them."""

    key_caches: list[torch.Tensor]  # per layer [blocks, block_size, kv_heads, head_dim]
    value_caches: list[torch.Tensor]
    slot_mapping: torch.Tensor  # [tokens] cache slot of each new token; negative skips the write
    seqused_k: torch.Tensor  # [sequences] keys of each sequence after this call
    block_table: torch.Tensor  # [sequences, logical blocks] physical block of each logical block
    max_seqlen_k: int


class PackedQwen2:
    """Causal (sliding-window) Qwen2 forward over packed sequences."""

    def __init__(self, model: nn.Module, max_positions: int):
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

        cfg = model.config
        self.model = model
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * len(model.layers)
        # transformers' sliding mask keeps kv_idx > q_idx - window: the query and window - 1 earlier positions.
        self.windows = [
            [cfg.sliding_window - 1, 0] if kind == "sliding_attention" else [-1, -1]
            for kind in layer_types[: len(model.layers)]
        ]
        self.layers = []
        for layer in model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            if attn.o_proj.bias is not None or mlp.gate_proj.bias is not None:
                raise ValueError("Packed Qwen2 expects bias-free o_proj and MLP projections")
            self.layers.append(
                (
                    layer.input_layernorm,
                    torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]),
                    torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias]),
                    attn.o_proj.weight,
                    layer.post_attention_layernorm,
                    torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]),
                    mlp.down_proj.weight,
                    attn.scaling,
                )
            )
        param = next(model.parameters())
        self.max_positions = int(max_positions)
        positions = torch.arange(self.max_positions, device=param.device)
        # transformers' cos/sin in the model dtype; the neox-style kernel takes the first half of each.
        cos, sin = model.rotary_emb(param.new_empty(1), positions[None])
        half = self.head_dim // 2
        self.cos_sin_cache = torch.cat([cos[0, :, :half], sin[0, :, :half]], dim=-1).contiguous()
        self._unit_scale = torch.ones((), device=param.device, dtype=torch.float32)
        self._attn = flash_attn_varlen_func
        self._fa_version = 3 if is_fa_version_supported(3) else 2

    def new_paged_caches(self, blocks: int, block_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        param = next(self.model.parameters())
        shape = (blocks, block_size, self.num_kv_heads, self.head_dim)
        keys = [torch.zeros(shape, device=param.device, dtype=param.dtype) for _ in self.layers]
        values = [torch.zeros(shape, device=param.device, dtype=param.dtype) for _ in self.layers]
        return keys, values

    def __call__(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        kv: PagedKV | None = None,
    ) -> torch.Tensor:
        """Return the final-norm hidden states [tokens, hidden] of packed ``hidden``."""
        tokens = hidden.shape[0]
        residual = hidden.clone()
        first_norm = self.layers[0][0]
        x = torch.empty_like(hidden)
        ops.rms_norm(x, hidden, first_norm.weight, first_norm.variance_epsilon)
        for index, (_, qkv_w, qkv_b, o_w, post_norm, gate_up_w, down_w, scale) in enumerate(self.layers):
            qkv = F.linear(x, qkv_w, qkv_b)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            ops.rotary_embedding(positions, q, k, self.head_dim, self.cos_sin_cache, True)
            q = q.view(tokens, self.num_heads, self.head_dim)
            k = k.view(tokens, self.num_kv_heads, self.head_dim)
            v = v.view(tokens, self.num_kv_heads, self.head_dim)
            if kv is None:
                attn = self._attn(
                    q,
                    k,
                    v,
                    max_seqlen,
                    cu_seqlens,
                    max_seqlen,
                    cu_seqlens_k=cu_seqlens,
                    softmax_scale=scale,
                    causal=True,
                    window_size=self.windows[index],
                    fa_version=self._fa_version,
                )
            else:
                key_cache, value_cache = kv.key_caches[index], kv.value_caches[index]
                ops.reshape_and_cache_flash(
                    k, v, key_cache, value_cache, kv.slot_mapping, "auto", self._unit_scale, self._unit_scale
                )
                attn = self._attn(
                    q,
                    key_cache,
                    value_cache,
                    max_seqlen,
                    cu_seqlens,
                    kv.max_seqlen_k,
                    seqused_k=kv.seqused_k,
                    block_table=kv.block_table,
                    softmax_scale=scale,
                    causal=True,
                    window_size=self.windows[index],
                    fa_version=self._fa_version,
                    # max_seqlen_k bounds the stream, not the window it reads; no split-KV.
                    num_splits=1,
                )
            x = F.linear(attn.view(tokens, self.q_size), o_w)
            ops.fused_add_rms_norm(x, residual, post_norm.weight, post_norm.variance_epsilon)
            gate_up = F.linear(x, gate_up_w)
            act = gate_up.new_empty(tokens, gate_up.shape[-1] // 2)
            torch.ops._C.silu_and_mul(act, gate_up)
            x = F.linear(act, down_w)
            next_norm = self.layers[index + 1][0] if index + 1 < len(self.layers) else self.model.norm
            ops.fused_add_rms_norm(x, residual, next_norm.weight, next_norm.variance_epsilon)
        return x


__all__ = ["PackedQwen2", "PagedKV"]
