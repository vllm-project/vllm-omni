# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Primer-hybrid transformer layers with slot-indexed KV ring buffers for Magenta RealTime 2."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.layernorm import RMSNorm

_R_SOFTPLUS_0 = 1.442695041  # 1.0 / ln(2) so softplus(0) * _R_SOFTPLUS_0 == 1.0
_ATTN_MASK_NEG = -1e9


def _rms_norm_f32(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply vLLM's ``RMSNorm`` directly (promoting to fp32 when requested)."""
    return norm(x)


def _scale_queries(
    q: torch.Tensor,
    per_dim_scale: torch.Tensor | None,
    head_dim: int,
) -> torch.Tensor:
    """Scale queries matching ``SequenceLayers`` ``_scale_queries``."""
    base_scale = 1.0 / math.sqrt(head_dim)
    if per_dim_scale is None:
        return q * base_scale
    scale = (_R_SOFTPLUS_0 * base_scale) * F.softplus(per_dim_scale.to(dtype=q.dtype))
    return q * scale


class PrimerHybridFFN(nn.Module):
    """Primer-hybrid residual FFN block matching ``SLTransformerFFN``."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        *,
        use_bias: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=use_bias)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=use_bias)
        self.post_norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = _rms_norm_f32(self.pre_norm, x)
        h = F.gelu(self.fc1(h), approximate="tanh")
        h = self.fc2(h)
        h = _rms_norm_f32(self.post_norm, h)
        return residual + h


class SlotRingBufferAttention(nn.Module):
    """Primer-hybrid causal sliding-window self/cross attention over per-slot KV ring buffers.

    Maintains persistent fixed-address GPU ring buffers indexed by ``slot_id`` of
    capacity ``max_past_horizon + 1``. Writing at ``step % ring_cap`` in single-frame
    decode mode requires no KV concatenation, preserves O(1) memory per session, and
    executes safely inside vLLM MRv2 FULL CUDA graphs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        max_past_horizon: int,
        *,
        kv_input_dim: int | None = None,
        num_kv_heads: int | None = None,
        num_sinks: int = 1,
        use_per_dim_scale: bool = True,
        norm_eps: float = 1e-6,
        max_slots: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.kv_input_dim = int(kv_input_dim if kv_input_dim is not None else hidden_size)
        self.is_cross_attention = kv_input_dim is not None and self.kv_input_dim != self.hidden_size
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads if num_kv_heads is not None else num_heads)
        self.head_dim = int(head_dim)
        self.max_past_horizon = int(max_past_horizon)
        self.ring_cap = max(1, self.max_past_horizon + 1)
        self.num_sinks = int(num_sinks)
        self.max_slots = max(1, int(max_slots))

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.pre_norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(self.kv_input_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.kv_input_dim, kv_dim, bias=False)
        self.output_proj = nn.Linear(q_dim, self.hidden_size, bias=False)
        self.post_norm = RMSNorm(self.hidden_size, eps=norm_eps)

        if use_per_dim_scale:
            self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        else:
            self.per_dim_scale = None

        if self.num_sinks > 0:
            self.sink_key_embeddings = nn.Parameter(
                torch.zeros(self.num_sinks, self.num_heads, self.head_dim, dtype=dtype)
            )
            self.sink_value_embeddings = nn.Parameter(
                torch.zeros(self.num_sinks, self.num_kv_heads, self.head_dim, dtype=dtype)
            )
        else:
            self.sink_key_embeddings = None
            self.sink_value_embeddings = None

        # Fixed-address GPU KV ring buffers: [num_slots, ring_cap, num_kv_heads, head_dim]
        self.register_buffer(
            "ring_k",
            torch.zeros(self.max_slots, self.ring_cap, self.num_kv_heads, self.head_dim, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "ring_v",
            torch.zeros(self.max_slots, self.ring_cap, self.num_kv_heads, self.head_dim, dtype=dtype),
            persistent=False,
        )

    def ensure_slot_capacity(self, max_slots: int) -> None:
        if max_slots <= self.max_slots:
            return
        device = self.ring_k.device
        dtype = self.ring_k.dtype
        self.max_slots = int(max_slots)
        self.ring_k = torch.zeros(
            self.max_slots, self.ring_cap, self.num_kv_heads, self.head_dim, dtype=dtype, device=device
        )
        self.ring_v = torch.zeros(
            self.max_slots, self.ring_cap, self.num_kv_heads, self.head_dim, dtype=dtype, device=device
        )

    def reset_slot(self, slot_id: int) -> None:
        if 0 <= slot_id < self.max_slots:
            self.ring_k[slot_id].zero_()
            self.ring_v[slot_id].zero_()

    def _compute_single_step_attention(
        self,
        q: torch.Tensor,
        slot_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention for ``[N, H, D]`` queries against indexed ``ring_k/ring_v``.

        Args:
            q: Unscaled queries ``[N, H, D]``.
            slot_ids: ``[N]`` long slot indices.
            attn_mask: ``[N, ring_cap]`` boolean visibility mask (True = valid key).
        """
        n = q.shape[0]
        # Gather slot ring K/V: [N, ring_cap, H_kv, D]
        k = self.ring_k.index_select(0, slot_ids)
        v = self.ring_v.index_select(0, slot_ids)
        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)

        # Transpose to [N, H, ..., D]
        q_nhd = q.unsqueeze(2)  # [N, H, 1, D]
        k_nhkd = k.permute(0, 2, 1, 3)  # [N, H, C, D]
        v_nhkd = v.permute(0, 2, 1, 3)  # [N, H, C, D]

        # Sink logits MUST be computed before per_dim_scale on q.
        sink_logits: torch.Tensor | None = None
        if self.sink_key_embeddings is not None:
            sink_k = self.sink_key_embeddings.to(dtype=q_nhd.dtype)  # [S, H, D]
            sink_k_t = sink_k.permute(1, 2, 0).unsqueeze(0)  # [1, H, D, S]
            sink_logits = torch.matmul(q_nhd, sink_k_t)  # [N, H, 1, S]

        q_scaled = _scale_queries(q_nhd, self.per_dim_scale, self.head_dim)
        logits = torch.matmul(q_scaled, k_nhkd.transpose(-1, -2))  # [N, H, 1, C]

        if self.sink_key_embeddings is not None:
            assert sink_logits is not None and self.sink_value_embeddings is not None
            sink_v = self.sink_value_embeddings.to(dtype=v_nhkd.dtype)  # [S, H_kv, D]
            if num_groups > 1:
                sink_v = sink_v.repeat_interleave(num_groups, dim=1)
            sink_v_t = sink_v.permute(1, 0, 2).unsqueeze(0).expand(n, -1, -1, -1)  # [N, H, S, D]
            v_nhkd = torch.cat([sink_v_t, v_nhkd], dim=2)
            logits = torch.cat([sink_logits, logits], dim=-1)  # [N, H, 1, S + C]
            sink_mask = torch.ones((n, self.num_sinks), dtype=torch.bool, device=attn_mask.device)
            full_mask = torch.cat([sink_mask, attn_mask], dim=-1)
        else:
            full_mask = attn_mask

        neg = torch.full((), _ATTN_MASK_NEG, dtype=logits.dtype, device=logits.device)
        logits = torch.where(full_mask[:, None, None, :], logits, neg)
        probs = F.softmax(logits.float(), dim=-1).to(dtype=v_nhkd.dtype)
        ctx = torch.matmul(probs, v_nhkd).squeeze(2)  # [N, H, D]
        return ctx.reshape(n, self.num_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        write_pos: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one step (CUDA-graph safe when every row writes/reads distinct or scratch slots)."""
        residual = x
        q_in = _rms_norm_f32(self.pre_norm, x)
        kv_in = q_in if kv_input is None else kv_input

        q = self.q_proj(q_in).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(kv_in).view(-1, self.num_kv_heads, self.head_dim).to(dtype=self.ring_k.dtype)
        v = self.v_proj(kv_in).view(-1, self.num_kv_heads, self.head_dim).to(dtype=self.ring_v.dtype)

        # O(1) scatter into persistent slot KV ring buffer before attention.
        self.ring_k[slot_ids, write_pos] = k
        self.ring_v[slot_ids, write_pos] = v

        ctx = self._compute_single_step_attention(q, slot_ids, attn_mask)
        out = self.output_proj(ctx)
        out = _rms_norm_f32(self.post_norm, out)
        return residual + out

    def forward_sequence(
        self,
        x: torch.Tensor,
        *,
        kv_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full sequence forward ``[B, T, D]`` for parity validation and eager prefill."""
        residual = x
        bsz, seq_len, _ = x.shape
        q_in = _rms_norm_f32(self.pre_norm, x)
        kv_in = q_in if kv_input is None else kv_input

        q = self.q_proj(q_in).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).view(bsz, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).view(bsz, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        sink_logits: torch.Tensor | None = None
        if self.sink_key_embeddings is not None:
            sink_k = self.sink_key_embeddings.to(dtype=q.dtype).permute(1, 2, 0).unsqueeze(0)
            sink_logits = torch.matmul(q, sink_k)

        q_scaled = _scale_queries(q, self.per_dim_scale, self.head_dim)
        logits = torch.matmul(q_scaled, k.transpose(-1, -2))

        pos = torch.arange(seq_len, device=x.device)
        diff = pos[:, None] - pos[None, :]
        causal_mask = (diff >= 0) & (diff <= self.max_past_horizon)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

        if self.sink_key_embeddings is not None:
            assert sink_logits is not None and self.sink_value_embeddings is not None
            sink_v = self.sink_value_embeddings.to(dtype=v.dtype)
            if num_groups > 1:
                sink_v = sink_v.repeat_interleave(num_groups, dim=1)
            sink_v = sink_v.permute(1, 0, 2).unsqueeze(0).expand(bsz, -1, -1, -1)
            v = torch.cat([sink_v, v], dim=2)
            logits = torch.cat([sink_logits, logits], dim=-1)
            sink_mask = torch.ones((bsz, 1, seq_len, self.num_sinks), dtype=torch.bool, device=x.device)
            causal_mask = torch.cat([sink_mask, causal_mask], dim=-1)

        neg = torch.full((), _ATTN_MASK_NEG, dtype=logits.dtype, device=logits.device)
        logits = torch.where(causal_mask, logits, neg)
        probs = F.softmax(logits.float(), dim=-1).to(dtype=v.dtype)
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        out = self.output_proj(ctx)
        out = _rms_norm_f32(self.post_norm, out)
        return residual + out


class DepthSelfAttention(nn.Module):
    """Causal self-attention over Depthformer steps (max length ``num_codebooks`` = 12)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        max_past_horizon: int = 12,
        *,
        use_per_dim_scale: bool = True,
        norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_past_horizon = int(max_past_horizon)

        qkv_dim = self.num_heads * self.head_dim
        self.pre_norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, qkv_dim, bias=False)
        self.output_proj = nn.Linear(qkv_dim, self.hidden_size, bias=False)
        self.post_norm = RMSNorm(self.hidden_size, eps=norm_eps)
        if use_per_dim_scale:
            self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        else:
            self.per_dim_scale = None

    def forward_step(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """Decode one depth step ``step_idx in [0, Q)`` using fixed static ``k_cache/v_cache``.

        Args:
            x: ``[B, hidden_size]``
            k_cache: ``[B, Q, H, D]`` updated in place at index ``step_idx``.
            v_cache: ``[B, Q, H, D]`` updated in place at index ``step_idx``.
            step_idx: Static Python step index (enables clean CUDA graph capture).
        """
        residual = x
        h = _rms_norm_f32(self.pre_norm, x)
        bsz = h.shape[0]

        q = self.q_proj(h).view(bsz, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(bsz, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(bsz, self.num_heads, self.head_dim)

        k_cache[:, step_idx].copy_(k.to(dtype=k_cache.dtype))
        v_cache[:, step_idx].copy_(v.to(dtype=v_cache.dtype))

        q_scaled = _scale_queries(q.unsqueeze(2), self.per_dim_scale, self.head_dim)  # [B, H, 1, D]
        k_hist = k_cache[:, : step_idx + 1].permute(0, 2, 1, 3).to(dtype=q_scaled.dtype)  # [B, H, T, D]
        v_hist = v_cache[:, : step_idx + 1].permute(0, 2, 1, 3).to(dtype=q_scaled.dtype)
        logits = torch.matmul(q_scaled, k_hist.transpose(-1, -2))
        probs = F.softmax(logits.float(), dim=-1).to(dtype=v_hist.dtype)
        ctx = torch.matmul(probs, v_hist).squeeze(2).reshape(bsz, self.num_heads * self.head_dim)
        out = _rms_norm_f32(self.post_norm, self.output_proj(ctx))
        return residual + out

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel causal depth attention for ``[B, Q, D]``."""
        residual = x
        bsz, q_len, _ = x.shape
        h = _rms_norm_f32(self.pre_norm, x)
        q = self.q_proj(h).view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(h).view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(h).view(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_scaled = _scale_queries(q, self.per_dim_scale, self.head_dim)
        logits = torch.matmul(q_scaled, k.transpose(-1, -2))
        pos = torch.arange(q_len, device=x.device)
        diff = pos[:, None] - pos[None, :]
        mask = (diff >= 0) & (diff <= self.max_past_horizon)
        neg = torch.full((), _ATTN_MASK_NEG, dtype=logits.dtype, device=logits.device)
        logits = torch.where(mask[None, None, :, :], logits, neg)
        probs = F.softmax(logits.float(), dim=-1).to(dtype=v.dtype)
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        out = _rms_norm_f32(self.post_norm, self.output_proj(ctx))
        return residual + out


class TemporalTransformerBlock(nn.Module):
    """Temporal Primer-hybrid block: SelfAttention -> CrossAttention -> FFN."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        encoder_dim: int,
        self_past_horizon: int,
        cross_past_horizon: int,
        *,
        num_sinks: int = 1,
        norm_eps: float = 1e-6,
        max_slots: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.self_attention = SlotRingBufferAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_past_horizon=self_past_horizon,
            kv_input_dim=None,
            num_sinks=num_sinks,
            use_per_dim_scale=True,
            norm_eps=norm_eps,
            max_slots=max_slots,
            dtype=dtype,
        )
        self.cross_attention = SlotRingBufferAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_past_horizon=cross_past_horizon,
            kv_input_dim=encoder_dim,
            num_sinks=num_sinks,
            use_per_dim_scale=True,
            norm_eps=norm_eps,
            max_slots=max_slots,
            dtype=dtype,
        )
        self.ffn = PrimerHybridFFN(
            hidden_size=hidden_size,
            ffn_dim=ffn_dim,
            use_bias=True,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_hidden: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        self_write_pos: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_write_pos: torch.Tensor,
        cross_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_attention(
            x,
            slot_ids=slot_ids,
            write_pos=self_write_pos,
            attn_mask=self_attn_mask,
        )
        h = self.cross_attention(
            h,
            slot_ids=slot_ids,
            write_pos=cross_write_pos,
            attn_mask=cross_attn_mask,
            kv_input=enc_hidden,
        )
        return self.ffn(h)

    def forward_sequence(self, x: torch.Tensor, enc_hidden: torch.Tensor) -> torch.Tensor:
        h = self.self_attention.forward_sequence(x)
        h = self.cross_attention.forward_sequence(h, kv_input=enc_hidden)
        return self.ffn(h)


class DepthTransformerBlock(nn.Module):
    """Depthformer Primer-hybrid block: DepthSelfAttention -> FFN."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        max_past_horizon: int = 12,
        *,
        norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.self_attention = DepthSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_past_horizon=max_past_horizon,
            use_per_dim_scale=True,
            norm_eps=norm_eps,
            dtype=dtype,
        )
        self.ffn = PrimerHybridFFN(
            hidden_size=hidden_size,
            ffn_dim=ffn_dim,
            use_bias=True,
            norm_eps=norm_eps,
        )

    def forward_step(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        h = self.self_attention.forward_step(x, k_cache, v_cache, step_idx)
        return self.ffn(h)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.self_attention.forward_sequence(x))


def load_flax_attention_weights(
    attn: SlotRingBufferAttention | DepthSelfAttention,
    jax_attn_block: dict[str, Any],
) -> None:
    """Load a Flax Primer-hybrid attention block into PyTorch ``q_proj/k_proj/v_proj/output_proj``."""
    attn.pre_norm.weight.data.copy_(
        torch.as_tensor(jax_attn_block["pre_norm"]["scale"], dtype=attn.pre_norm.weight.dtype)
    )
    attn.post_norm.weight.data.copy_(
        torch.as_tensor(jax_attn_block["post_norm"]["scale"], dtype=attn.post_norm.weight.dtype)
    )

    out_kernel = torch.as_tensor(jax_attn_block["output_projection"]["kernel"], dtype=attn.output_proj.weight.dtype)
    # JAX EinsumDense '...nh,dnh->...d': kernel shape is [d_model, num_heads, head_dim],
    # which directly matches PyTorch nn.Linear [out_features, in_features] after flatten.
    if out_kernel.ndim == 3:
        out_weight = out_kernel.reshape(out_kernel.shape[0], -1)
    else:
        out_weight = out_kernel.t()
    attn.output_proj.weight.data.copy_(out_weight)

    jax_inner = jax_attn_block["attention"]
    if "query_key_value_projection" in jax_inner:
        qkv = torch.as_tensor(jax_inner["query_key_value_projection"]["kernel"])
        q_k, k_k, v_k = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    elif "key_value_projection" in jax_inner:
        q_k = torch.as_tensor(jax_inner["query_projection"]["kernel"])
        kv = torch.as_tensor(jax_inner["key_value_projection"]["kernel"])
        k_k, v_k = kv[:, 0], kv[:, 1]
    elif "shared_key_value_projection" in jax_inner:
        q_k = torch.as_tensor(jax_inner["query_projection"]["kernel"])
        shared = torch.as_tensor(jax_inner["shared_key_value_projection"]["kernel"])
        k_k, v_k = shared, shared
    else:
        q_k = torch.as_tensor(jax_inner["query_projection"]["kernel"])
        k_k = torch.as_tensor(jax_inner["key_projection"]["kernel"])
        v_k = torch.as_tensor(jax_inner["value_projection"]["kernel"])

    def _to_linear_weight(kernel: torch.Tensor, target: nn.Linear) -> torch.Tensor:
        flat = kernel.reshape(kernel.shape[0], -1)
        return flat.t().to(dtype=target.weight.dtype)

    attn.q_proj.weight.data.copy_(_to_linear_weight(q_k, attn.q_proj))
    attn.k_proj.weight.data.copy_(_to_linear_weight(k_k, attn.k_proj))
    attn.v_proj.weight.data.copy_(_to_linear_weight(v_k, attn.v_proj))

    if attn.per_dim_scale is not None and "per_dim_scale" in jax_inner:
        attn.per_dim_scale.data.copy_(torch.as_tensor(jax_inner["per_dim_scale"], dtype=attn.per_dim_scale.dtype))
    sink_k = getattr(attn, "sink_key_embeddings", None)
    if sink_k is not None and "sink_key_embeddings" in jax_inner:
        sink_k.data.copy_(torch.as_tensor(jax_inner["sink_key_embeddings"], dtype=sink_k.dtype))
    sink_v = getattr(attn, "sink_value_embeddings", None)
    if sink_v is not None and "sink_value_embeddings" in jax_inner:
        sink_v.data.copy_(torch.as_tensor(jax_inner["sink_value_embeddings"], dtype=sink_v.dtype))


def load_flax_ffn_weights(ffn: PrimerHybridFFN, jax_ffn_block: dict[str, Any]) -> None:
    """Load a Flax ``SLTransformerFFN`` block into ``PrimerHybridFFN``."""
    ffn.pre_norm.weight.data.copy_(torch.as_tensor(jax_ffn_block["pre_norm"]["scale"], dtype=ffn.pre_norm.weight.dtype))
    ffn.post_norm.weight.data.copy_(
        torch.as_tensor(jax_ffn_block["post_norm"]["scale"], dtype=ffn.post_norm.weight.dtype)
    )
    ffn.fc1.weight.data.copy_(torch.as_tensor(jax_ffn_block["ffn_layer1"]["kernel"], dtype=ffn.fc1.weight.dtype).t())
    if ffn.fc1.bias is not None and "bias" in jax_ffn_block["ffn_layer1"]:
        ffn.fc1.bias.data.copy_(torch.as_tensor(jax_ffn_block["ffn_layer1"]["bias"], dtype=ffn.fc1.bias.dtype))
    ffn.fc2.weight.data.copy_(torch.as_tensor(jax_ffn_block["ffn_layer2"]["kernel"], dtype=ffn.fc2.weight.dtype).t())
    if ffn.fc2.bias is not None and "bias" in jax_ffn_block["ffn_layer2"]:
        ffn.fc2.bias.data.copy_(torch.as_tensor(jax_ffn_block["ffn_layer2"]["bias"], dtype=ffn.fc2.bias.dtype))


__all__ = [
    "DepthSelfAttention",
    "DepthTransformerBlock",
    "PrimerHybridFFN",
    "SlotRingBufferAttention",
    "TemporalTransformerBlock",
    "load_flax_attention_weights",
    "load_flax_ffn_weights",
]
