# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness regression tests for hoisting the RoPE tables and RingKV
position tables out of the per-layer loop.

The frozen oracle below is a verbatim copy of the per-row (``[B]`` offsets,
``active`` row mask) ``_apply_rope`` / ``_RingKV.complete`` / layer ``forward``
/ stack ``step`` from ``main`` at commit 3bc3f1a7d, i.e. after #7670 and
#8192 and before the hoist. It is compared against the live module every
run -- never against a pre-saved fixture, since ``cos``/``sin`` are not
bit-portable across torch versions or hardware -- and must NOT be updated when
the live modules change.

Comparisons use ``torch.equal`` throughout: this is a pure refactor, so any
mismatch is an ordering/accumulation bug, not tolerance noise.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.personaplex.personaplex_depformer import _rms_norm_f32
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import _MimiStreamingTransformer
from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    PersonaPlexTemporalStreaming,
    _apply_rope,
    _RingKV,
    _ringkv_positions,
    _rope_tables,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CPU = torch.device("cpu")


# ===========================================================================
# Frozen oracle: verbatim pre-hoist copies (main @ 3bc3f1a7d). Do not "fix"
# these to match the new code -- they are the reference the new code is
# checked against.
# ===========================================================================


def _legacy_apply_rope(q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor, max_period: float = 10_000.0):
    B, H, T, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts.view(-1, 1, T, 1)

    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(*dims, D), ko.view(*dims, D)


class _LegacyRingKV:
    def __init__(self, batch_size: int, num_heads: int, dim_per_head: int, capacity: int, device, dtype):
        self.capacity = capacity
        self.cache = torch.zeros((2, batch_size, num_heads, capacity, dim_per_head), device=device, dtype=dtype)
        self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.start_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    def reset_slot(self, b: int) -> None:
        self.start_offset[b] = self.end_offset[b]

    def reset_row(self, b: int) -> None:
        self.end_offset[b] = 0
        self.start_offset[b] = 0

    def bump_slot_start(self, b: int) -> None:
        self.start_offset[b] += 1

    def complete(self, k: torch.Tensor, v: torch.Tensor, active: torch.Tensor):
        B, H, T, D = k.shape
        indexes = (
            torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype).view(1, -1)
            + self.end_offset.view(-1, 1)
        ) % self.capacity
        idx4 = indexes.view(B, 1, T, 1).expand(-1, H, -1, D)
        active_view = active.view(B, 1, 1, 1)
        old_k = self.cache[0].gather(2, idx4)
        old_v = self.cache[1].gather(2, idx4)
        k = torch.where(active_view, k, old_k)
        v = torch.where(active_view, v, old_v)
        self.cache[0].scatter_(2, idx4, k)
        self.cache[1].scatter_(2, idx4, v)
        self.end_offset.add_(T * active.to(self.end_offset.dtype))

        idx = torch.arange(self.capacity, device=self.end_offset.device, dtype=torch.long)
        end_offset = self.end_offset.view(-1, 1)
        invalid = idx.view(1, -1) >= end_offset
        end_index = end_offset % self.capacity
        delta = idx.view(1, -1) - end_index
        positions = torch.where(delta <= 0, end_offset + delta, end_offset + delta - self.capacity)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)
        below = positions < self.start_offset.view(-1, 1)
        positions = torch.where(below, torch.full_like(positions, -1), positions)
        return self.cache[0], self.cache[1], positions


def _legacy_temporal_layer_forward(layer, x, kv: _LegacyRingKV, offset, context: int, active):
    """Frozen ``_TemporalLayer.forward``, run on a REAL layer's weights so legacy
    and new only ever differ by the hoist."""
    B, T, _ = x.shape
    h = _rms_norm_f32(x, layer.norm1_alpha, 1e-8)
    qkv = F.linear(h, layer.in_proj_weight)
    qkv = qkv.view(B, T, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = _legacy_apply_rope(q, k, offset)

    keys, values, pos_k = kv.complete(k, v, active)
    pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
    pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
    attn_bias = attn_bias.unsqueeze(1)
    attn = F.scaled_dot_product_attention(q, keys, values, attn_bias, dropout_p=0.0)
    attn = attn.transpose(1, 2).reshape(B, T, layer.dim)
    x = x + F.linear(attn, layer.out_proj_weight)

    h = _rms_norm_f32(x, layer.norm2_alpha, 1e-8)
    a, b = F.linear(h, layer.gating_in).chunk(2, dim=-1)
    return x + F.linear(F.silu(a) * b, layer.gating_out)


def _legacy_mimi_layer_forward(layer, x, kv: _LegacyRingKV, offset, context: int, active):
    """Frozen ``_MimiTransformerLayer.forward`` (personaplex_mimi.py)."""
    B, T, _ = x.shape
    h = layer.norm1(x)
    qkv = F.linear(h, layer.in_proj_weight)
    qkv = qkv.view(B, T, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = _legacy_apply_rope(q, k, offset)
    keys, values, pos_k = kv.complete(k, v, active=active)
    pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
    pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
    attn = F.scaled_dot_product_attention(q, keys, values, attn_bias.unsqueeze(1), dropout_p=0.0)
    attn = attn.transpose(1, 2).reshape(B, T, layer.dim)
    x = x + layer.scale1 * F.linear(attn, layer.out_proj_weight)
    h = layer.norm2(x)
    h = F.linear(F.gelu(F.linear(h, layer.linear1)), layer.linear2)
    return x + layer.scale2 * h


def _init_deterministic(module: nn.Module, generator: torch.Generator) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.1)


def _random_active(batch_size: int, gen: torch.Generator) -> torch.Tensor:
    # Mostly-active rows with occasional idle ones, so per-row offsets drift apart.
    return torch.rand(batch_size, generator=gen) > 0.25


# ===========================================================================
# 1. RoPE bit-exactness with per-row offsets
# ===========================================================================


@pytest.mark.parametrize(
    "offsets,t,head_dim",
    [
        ((0, 0), 1, 64),
        ((5, 0), 2, 64),
        ((2999, 17), 1, 128),
        ((1, 250, 3000), 4, 128),
        ((250, 7), 10, 64),
    ],
)
def test_apply_rope_matches_legacy_per_row(offsets, t, head_dim):
    T = t
    gen = torch.Generator().manual_seed(0)
    B, H = len(offsets), 3
    q = torch.randn(B, H, T, head_dim, generator=gen)
    k = torch.randn(B, H, T, head_dim, generator=gen)
    offset = torch.tensor(offsets, dtype=torch.long)

    ref_q, ref_k = _legacy_apply_rope(q.clone(), k.clone(), offset)
    rotr, roti = _rope_tables(offset, T, head_dim)
    assert rotr.shape == roti.shape == (B, 1, T, head_dim // 2)
    new_q, new_k = _apply_rope(q.clone(), k.clone(), rotr, roti)

    assert torch.equal(ref_q, new_q)
    assert torch.equal(ref_k, new_k)


def test_rope_tables_reused_across_layers_within_step():
    """One table pair applied to several q/k pairs (as every layer of one step
    does) must match the recompute-per-call oracle for each of them."""
    gen = torch.Generator().manual_seed(1)
    head_dim, T = 64, 2
    offset = torch.tensor([42, 3], dtype=torch.long)
    rotr, roti = _rope_tables(offset, T, head_dim)
    for _ in range(5):
        q = torch.randn(2, 3, T, head_dim, generator=gen)
        k = torch.randn(2, 3, T, head_dim, generator=gen)
        ref_q, ref_k = _legacy_apply_rope(q.clone(), k.clone(), offset)
        new_q, new_k = _apply_rope(q.clone(), k.clone(), rotr, roti)
        assert torch.equal(ref_q, new_q)
        assert torch.equal(ref_k, new_k)


# ===========================================================================
# 2. RingKV bit-exactness across ring wraps, with active masks and recycle
# ===========================================================================


@pytest.mark.parametrize(
    "t,capacity",
    [
        (1, 5),
        # T > 1 (codec_chunk_frames: 5 streams multi-position frames into Mimi).
        # capacity is not a multiple of T, so chunks straddle the ring boundary.
        (2, 7),
        (5, 7),
        (5, 12),
    ],
)
def test_ringkv_complete_matches_legacy_across_wrap(t, capacity):
    T = t
    gen = torch.Generator().manual_seed(2)
    B, H, D = 3, 2, 4
    legacy = _LegacyRingKV(B, H, D, capacity, CPU, torch.float32)
    hoisted = _RingKV(B, H, D, capacity, CPU, torch.float32)
    standalone = _RingKV(B, H, D, capacity, CPU, torch.float32)  # no tables passed in
    # The stack-level offset the hoisted tables are built from.
    offset = torch.zeros(B, dtype=torch.long)

    wrapped_mid_chunk = False
    for step in range(2000):
        k = torch.randn(B, H, T, D, generator=gen)
        v = torch.randn(B, H, T, D, generator=gen)
        active = _random_active(B, gen)
        wrapped_mid_chunk |= bool((((offset % capacity) + T > capacity) & active).any())

        ref = legacy.complete(k.clone(), v.clone(), active)
        indexes, positions = _ringkv_positions(offset, T, capacity, active)
        new = hoisted.complete(k.clone(), v.clone(), active, indexes, positions)
        own = standalone.complete(k.clone(), v.clone(), active)
        offset = offset + T * active.to(offset.dtype)

        for name, a, b, c in zip(("keys", "values", "positions"), ref, new, own):
            assert torch.equal(a, b), f"step {step}: hoisted {name} mismatch"
            assert torch.equal(a, c), f"step {step}: standalone {name} mismatch"

        if step % 37 == 0:
            row = step % B
            for ring in (legacy, hoisted, standalone):
                ring.reset_slot(row)
        if step % 53 == 0:
            row = (step + 1) % B
            for ring in (legacy, hoisted, standalone):
                ring.bump_slot_start(row)
        if step % 211 == 0:
            row = (step + 2) % B
            for ring in (legacy, hoisted, standalone):
                ring.reset_row(row)
            offset[row] = 0

    assert wrapped_mid_chunk or T == 1
    for ring in (hoisted, standalone):
        assert torch.equal(legacy.end_offset, ring.end_offset)
        assert torch.equal(legacy.start_offset, ring.start_offset)
        assert torch.equal(legacy.cache, ring.cache)


# ===========================================================================
# 3. End-to-end step(): Helium temporal stack
# ===========================================================================


def test_temporal_streaming_step_matches_legacy_end_to_end():
    dim, num_heads, hidden, num_layers, context, text_card = 32, 4, 64, 3, 16, 17
    B = 3

    real_stack = PersonaPlexTemporalStreaming(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden=hidden,
        context=context,
        text_card=text_card,
    )
    _init_deterministic(real_stack, torch.Generator().manual_seed(3))
    real_stack.eval()
    real_stack.streaming_init(batch_size=B)

    legacy_kvs = [_LegacyRingKV(B, num_heads, dim // num_heads, context, CPU, torch.float32) for _ in range(num_layers)]
    legacy_offset = torch.zeros(B, dtype=torch.long)

    gen = torch.Generator().manual_seed(4)
    for step_idx in range(2000):  # context=16 -> >100 wraps per row
        frame = torch.randn(B, 1, dim, generator=gen)
        active = _random_active(B, gen)

        x = frame.clone()
        for layer, kv in zip(real_stack.layers, legacy_kvs):
            x = _legacy_temporal_layer_forward(layer, x, kv, legacy_offset, context, active)
        legacy_offset.add_(x.shape[1] * active.to(legacy_offset.dtype))
        legacy_out = _rms_norm_f32(x, real_stack.out_norm_alpha, 1e-8)
        legacy_text_logits = F.linear(legacy_out, real_stack.text_linear)[:, None]

        new_out, new_text_logits = real_stack.step(frame.clone(), active)

        assert torch.equal(legacy_out, new_out), f"step {step_idx}: transformer_out mismatch"
        assert torch.equal(legacy_text_logits, new_text_logits), f"step {step_idx}: text_logits mismatch"

        if step_idx % 71 == 0:
            row = step_idx % B
            for kv in legacy_kvs:
                kv.reset_slot(row)
            real_stack.reset_slot(row)
        if step_idx % 97 == 0:
            row = (step_idx + 1) % B
            for kv in legacy_kvs:
                kv.bump_slot_start(row)
            real_stack.bump_slot_start(row)

    assert torch.equal(legacy_offset, real_stack._offset)
    for legacy_kv, real_kv in zip(legacy_kvs, real_stack._kv):
        assert torch.equal(legacy_kv.end_offset, real_kv.end_offset)
        assert torch.equal(legacy_kv.start_offset, real_kv.start_offset)


# ===========================================================================
# 4. End-to-end step(): Mimi encoder/decoder transformer
# ===========================================================================


@pytest.mark.parametrize("t", [1, 2, 10])  # 10 = codec_chunk_frames 5 x 2 positions
def test_mimi_streaming_step_matches_legacy_end_to_end(t):
    T = t
    dim, num_heads, num_layers, context = 16, 2, 2, 12
    B = 3

    real_stack = _MimiStreamingTransformer(num_layers=num_layers, dim=dim, num_heads=num_heads, context=context)
    _init_deterministic(real_stack, torch.Generator().manual_seed(5))
    real_stack.eval()
    real_stack.streaming_init(batch_size=B)

    legacy_kvs = [_LegacyRingKV(B, num_heads, dim // num_heads, context, CPU, torch.float32) for _ in range(num_layers)]
    legacy_offset = torch.zeros(B, dtype=torch.long)

    gen = torch.Generator().manual_seed(6)
    for step_idx in range(1000):
        frame = torch.randn(B, T, dim, generator=gen)
        active = _random_active(B, gen)

        x = frame.clone()
        for layer, kv in zip(real_stack.layers, legacy_kvs):
            x = _legacy_mimi_layer_forward(layer, x, kv, legacy_offset, context, active)
        legacy_offset.add_(x.shape[1] * active.to(legacy_offset.dtype))

        with torch.no_grad():
            new_x = real_stack.step(frame.clone(), active)

        assert torch.equal(x, new_x), f"step {step_idx}: mimi transformer output mismatch"

        if step_idx % 41 == 0:
            # Mimi recycle restarts the row at position 0 (frozen main semantics).
            row = step_idx % B
            for kv in legacy_kvs:
                kv.reset_row(row)
            legacy_offset[row] = 0
            real_stack.reset_slot(row)

    assert torch.equal(legacy_offset, real_stack._offset)
    for legacy_kv, real_kv in zip(legacy_kvs, real_stack._kv):
        assert torch.equal(legacy_kv.end_offset, real_kv.end_offset)
        assert torch.equal(legacy_kv.start_offset, real_kv.start_offset)
