# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""State-slot aware streaming attention for the MOSS v2 codec (CUDA only)."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_ring_kv(
    K,
    V,
    Cache,
    Packed,
    Offsets,
    Slots,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    SLOTS: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, h = tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + b)
    offset = tl.load(Offsets + slot)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos, d = i // D, i % D
    current = (pos - offset % C + C) % C
    is_new = (current < T) & (pos < C)
    base = (slot * H + h) * C * D + i
    old_k = tl.load(Cache + base, (pos < C) & ~is_new, 0)
    old_v = tl.load(Cache + SLOTS * H * C * D + base, (pos < C) & ~is_new, 0)
    new_k = tl.load(K + b * KB + h * KH + current * KT + d, is_new, 0)
    new_v = tl.load(V + b * VB + h * VH + current * VT + d, is_new, 0)
    key = tl.where(is_new, new_k, old_k)
    value = tl.where(is_new, new_v, old_v)
    # Each physical element has one owner. Never read an element another CTA
    # writes: current slots come directly from K/V, historical slots read Cache.
    tl.store(Cache + base, new_k, is_new)
    tl.store(Cache + SLOTS * H * C * D + base, new_v, is_new)
    out = (b * H + h) * C * D + i
    tl.store(Packed + out, key, pos < C)
    tl.store(Packed + B * H * C * D + out, value, pos < C)


@torch.library.custom_op("moss_codec::pack_ring_kv", mutates_args=("cache",), device_types="cuda")
def pack_ring_kv(
    k: torch.Tensor, v: torch.Tensor, cache: torch.Tensor, offsets: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    """Bit-preserving KV insertion + compact gather, with no full-ring writeback."""
    batch, heads, frames, dim = k.shape
    capacity = cache.shape[3]
    if frames > capacity:
        # Preserve legacy duplicate-scatter semantics for oversized chunks.
        packed = cache.index_select(1, slots)
        indexes = (offsets.index_select(0, slots)[:, None] + torch.arange(frames, device=k.device)) % capacity
        indexes = indexes[:, None, :, None].expand(batch, heads, frames, dim)
        packed[0].scatter_(2, indexes, k)
        packed[1].scatter_(2, indexes, v)
        cache.index_copy_(1, slots, packed)
        return packed
    assert k.stride(-1) == v.stride(-1) == 1 and cache.is_contiguous()
    packed = torch.empty((2, batch, heads, capacity, dim), device=k.device, dtype=cache.dtype)
    _pack_ring_kv[(triton.cdiv(capacity * dim, 256), batch, heads)](
        k,
        v,
        cache,
        packed,
        offsets,
        slots,
        batch,
        heads,
        frames,
        dim,
        capacity,
        cache.shape[1],
        *k.stride()[:3],
        *v.stride()[:3],
        256,
    )
    return packed


@pack_ring_kv.register_fake
def _pack_ring_kv_fake(k, v, cache, offsets, slots):
    return torch.empty((2, k.shape[0], k.shape[1], cache.shape[3], k.shape[3]), device=k.device, dtype=cache.dtype)
