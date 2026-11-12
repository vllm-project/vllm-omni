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

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm import _custom_ops as ops

# vllm_xpu_kernels reads VLLM_XPU_SPEC_DECODE_MAX_QLEN once at import (default 16)
# and sends uniform paged queries in (1, cutoff] through a split-K decode kernel.
# That kernel has no "not compiled" fallback; the ordinary varlen path does.
_XPU_SPEC_DECODE_MAX_QLEN_DEFAULT = 16


@dataclass
class PagedKV:
    """Per-layer paged key/value caches and this call's addressing into them."""

    key_caches: list[torch.Tensor]  # per layer [blocks, block_size, kv_heads, head_dim]
    value_caches: list[torch.Tensor]
    slot_mapping: torch.Tensor  # [tokens] cache slot of each new token; negative skips the write
    seqused_k: torch.Tensor  # [sequences] keys of each sequence after this call
    block_table: torch.Tensor  # [sequences, logical blocks] physical block of each logical block
    max_seqlen_k: int


def _xpu_spec_decode_max_qlen() -> int:
    """Cutoff above which XPU keeps a paged query on the varlen kernel."""
    try:
        return int(os.environ.get("VLLM_XPU_SPEC_DECODE_MAX_QLEN", str(_XPU_SPEC_DECODE_MAX_QLEN_DEFAULT)))
    except ValueError:
        return _XPU_SPEC_DECODE_MAX_QLEN_DEFAULT


def _decode_queries_for_xpu(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Present a short uniform chunk as one q_len=1 sequence per token.

    AudioVAE steps are not speculative decode, but their query length falls in
    the XPU spec-decode window. Each token keeps the parent block table and a
    causal key length of ``seqused_k - (q_len - 1 - position)``, which is the
    same mask the chunk would have used, on the varlen path that can fall back.
    """
    batch = cu_seqlens.shape[0] - 1
    q_len = max_seqlen
    if batch <= 0 or q_len <= 1 or q_len > _xpu_spec_decode_max_qlen() or q.shape[0] != batch * q_len:
        return q, cu_seqlens, max_seqlen, seqused_k, block_table
    total = q.shape[0]
    position = torch.arange(q_len, device=seqused_k.device)
    seqused = (
        (seqused_k.reshape(batch, 1) - (q_len - 1 - position).reshape(1, q_len))
        .clamp_min(1)
        .reshape(total)
        .to(torch.int32)
    )
    cu_seqlens_q = torch.arange(total + 1, device=q.device, dtype=torch.int32)
    return q, cu_seqlens_q, 1, seqused, block_table.repeat_interleave(q_len, dim=0)


class PackedQwen2:
    """Causal (sliding-window) Qwen2 forward over packed sequences."""

    def __init__(self, model: nn.Module, max_positions: int):
        from vllm.platforms import current_platform

        # XPU's wrapper takes cu_seqlens_q before max_seqlen_q; CUDA's dispatcher
        # is the other way around. Callers pass keywords, as diffusion flash
        # attention does. fa_version is only a CUDA dispatcher argument.
        if current_platform.is_xpu():
            from vllm._xpu_ops import xpu_ops

            flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func
            attn_kwargs: dict[str, int] = {}
            self._xpu = True
        else:
            from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

            attn_kwargs = {"fa_version": 3 if is_fa_version_supported(3) else 2}
            self._xpu = False

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
        self._attn_kwargs = attn_kwargs

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
                    q=q,
                    k=k,
                    v=v,
                    max_seqlen_q=max_seqlen,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_k=max_seqlen,
                    cu_seqlens_k=cu_seqlens,
                    softmax_scale=scale,
                    causal=True,
                    window_size=self.windows[index],
                    **self._attn_kwargs,
                )
            else:
                key_cache, value_cache = kv.key_caches[index], kv.value_caches[index]
                ops.reshape_and_cache_flash(
                    k, v, key_cache, value_cache, kv.slot_mapping, "auto", self._unit_scale, self._unit_scale
                )
                query, cu_query, query_len, seqused_k, block_table = (
                    _decode_queries_for_xpu(q, cu_seqlens, max_seqlen, kv.seqused_k, kv.block_table)
                    if self._xpu
                    else (q, cu_seqlens, max_seqlen, kv.seqused_k, kv.block_table)
                )
                attn = self._attn(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    max_seqlen_q=query_len,
                    cu_seqlens_q=cu_query,
                    max_seqlen_k=kv.max_seqlen_k,
                    seqused_k=seqused_k,
                    block_table=block_table,
                    softmax_scale=scale,
                    causal=True,
                    window_size=self.windows[index],
                    # max_seqlen_k bounds the stream, not the window it reads; no split-KV.
                    num_splits=1,
                    **self._attn_kwargs,
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
