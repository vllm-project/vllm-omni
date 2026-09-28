# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Direct-slot ring attention for MOSS streaming codec.

Inputs q/k/v are (execution_batch, heads, padded_T, head_dim). cache is
contiguous (2, state_capacity, heads, ring_capacity, head_dim). end_offset
contains absolute *pre-call* positions per state slot. slot_ids and valid_lengths
contain one entry per execution row. valid_lengths=0 is a scratch/padding row.

This op writes only valid tokens, attends to the resulting ring with the same
position semantics as RingKVCache.complete, then advances only active offsets.
Slots must be unique and in bounds; 0 <= valid_lengths <= padded_T. Chunks
longer than the ring retain only their final ring_capacity tokens.
These device-value preconditions are caller-owned to avoid host synchronization.
The op deliberately preserves chunk-complete ring overwrite semantics; it does
not recover entries overwritten by later tokens of the same input chunk.
"""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit
def _rotate(value, partner, dim, position, scale: tl.constexpr):
    angle = libdevice.exp((dim // 2).to(tl.float32) * scale) * position.to(tl.float32)
    cosine = libdevice.cos(angle)
    sine = libdevice.sin(angle)
    signed_partner = tl.where(dim % 2 == 0, -partner.to(tl.float32), partner.to(tl.float32))
    return libdevice.add_rn(libdevice.mul_rn(value.to(tl.float32), cosine), libdevice.mul_rn(signed_partner, sine)).to(
        value.dtype
    )


@triton.jit
def _prepare_rope(
    q_ptr,
    k_ptr,
    offset_ptr,
    qo_ptr,
    ko_ptr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    qs3: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    ks3: tl.constexpr,
    offset_stride: tl.constexpr,
    heads: tl.constexpr,
    frames: tl.constexpr,
    dim: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    batch = tl.program_id(0)
    x = tl.program_id(1) * block + tl.arange(0, block)
    mask = x < heads * frames * dim
    d = x % dim
    t = (x // dim) % frames
    h = x // (dim * frames)
    qbase = q_ptr + batch * qs0 + h * qs1 + t * qs2
    kbase = k_ptr + batch * ks0 + h * ks1 + t * ks2
    q = tl.load(qbase + d * qs3, mask, 0)
    qp = tl.load(qbase + (d ^ 1) * qs3, mask, 0)
    k = tl.load(kbase + d * ks3, mask, 0)
    kp = tl.load(kbase + (d ^ 1) * ks3, mask, 0)
    position = tl.load(offset_ptr + batch * offset_stride) + t
    qo = _rotate(q, qp, d, position, scale)
    ko = _rotate(k, kp, d, position, scale)
    tl.store(qo_ptr + batch * heads * frames * dim + x, qo, mask)
    tl.store(ko_ptr + batch * heads * frames * dim + x, ko, mask)


@triton.jit
def _write(
    k_ptr,
    v_ptr,
    rope_ptr,
    cache_ptr,
    end_ptr,
    slots_ptr,
    lengths_ptr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    ks3: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    vs3: tl.constexpr,
    end_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    length_stride: tl.constexpr,
    state_capacity: tl.constexpr,
    num_heads: tl.constexpr,
    num_frames: tl.constexpr,
    ring_capacity: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    rope_scale: tl.constexpr,
    rope_stride: tl.constexpr,
    rotate: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row * slot_stride)
    length = tl.load(lengths_ptr + row * length_stride)
    start = tl.load(end_ptr + slot * end_stride)
    x = tl.program_id(1) * block_size + tl.arange(0, block_size)
    dim = x % head_dim
    token = (x // head_dim) % num_frames
    head = x // (num_frames * head_dim)
    # For chunks longer than the ring, write only the final C tokens. This
    # avoids duplicate physical addresses (and write races) in one launch.
    skip = tl.maximum(length - ring_capacity, 0)
    source_token = token + skip
    mask = (head < num_heads) & (source_token < length)
    pos = (start + source_token) % ring_capacity
    dst = ((slot * num_heads + head) * ring_capacity + pos) * head_dim + dim
    kval = tl.load(k_ptr + row * ks0 + head * ks1 + source_token * ks2 + dim * ks3, mask, 0)
    if rotate:
        partner = tl.load(k_ptr + row * ks0 + head * ks1 + source_token * ks2 + (dim ^ 1) * ks3, mask, 0)
        position = tl.load(rope_ptr + row * rope_stride) + source_token
        kval = _rotate(kval, partner, dim, position, rope_scale)
    vval = tl.load(v_ptr + row * vs0 + head * vs1 + source_token * vs2 + dim * vs3, mask, 0)
    tl.store(cache_ptr + dst, kval, mask)
    tl.store(cache_ptr + state_capacity * num_heads * ring_capacity * head_dim + dst, vval, mask)


@triton.jit
def _attend(
    q_ptr,
    k_ptr,
    v_ptr,
    rope_ptr,
    cache_ptr,
    end_ptr,
    slots_ptr,
    lengths_ptr,
    out_ptr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    qs3: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    ks3: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    vs3: tl.constexpr,
    end_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    length_stride: tl.constexpr,
    state_capacity: tl.constexpr,
    num_heads: tl.constexpr,
    num_frames: tl.constexpr,
    ring_capacity: tl.constexpr,
    head_dim: tl.constexpr,
    context_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    fused_write: tl.constexpr,
    rope_scale: tl.constexpr,
    rope_stride: tl.constexpr,
    rotate: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    bh = tl.program_id(1)
    batch, head = bh // num_heads, bh % num_heads
    slot = tl.load(slots_ptr + batch * slot_stride)
    length = tl.load(lengths_ptr + batch * length_stride)
    start = tl.load(end_ptr + slot * end_stride)
    end = start + length
    dims = tl.arange(0, head_dim)
    if rotate:
        position = tl.load(rope_ptr + batch * rope_stride) + rows[:, None]
    if fused_write:
        # Exactly one CTA owns every query and KV entry of this slot/head.
        # The barrier makes all of its writes visible before attention loads.
        keep = (rows < length) & (rows >= tl.maximum(length - ring_capacity, 0))
        dst = ((slot * num_heads + head) * ring_capacity + (start + rows) % ring_capacity)[:, None] * head_dim
        kval = tl.load(k_ptr + batch * ks0 + head * ks1 + rows[:, None] * ks2 + dims[None, :] * ks3, keep[:, None], 0)
        if rotate:
            partner = tl.load(
                k_ptr + batch * ks0 + head * ks1 + rows[:, None] * ks2 + (dims[None, :] ^ 1) * ks3,
                keep[:, None],
                0,
            )
            kval = _rotate(kval, partner, dims[None, :], position, rope_scale)
        vval = tl.load(v_ptr + batch * vs0 + head * vs1 + rows[:, None] * vs2 + dims[None, :] * vs3, keep[:, None], 0)
        tl.store(cache_ptr + dst + dims[None, :], kval, keep[:, None])
        tl.store(
            cache_ptr + state_capacity * num_heads * ring_capacity * head_dim + dst + dims[None, :], vval, keep[:, None]
        )
        tl.debug_barrier()
    q = tl.load(
        q_ptr + batch * qs0 + head * qs1 + rows[:, None] * qs2 + dims[None, :] * qs3, rows[:, None] < num_frames, 0
    )
    if rotate:
        partner = tl.load(
            q_ptr + batch * qs0 + head * qs1 + rows[:, None] * qs2 + (dims[None, :] ^ 1) * qs3,
            rows[:, None] < num_frames,
            0,
        )
        q = _rotate(q, partner, dims[None, :], position, rope_scale)
    acc = tl.full((block_m, head_dim), 0.0, tl.float32)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0.0, tl.float32)
    # Physical-ring order matches the existing masked_attention accumulation.
    last = end - 1
    last_index = last % ring_capacity
    for first in range(tl.cdiv(ring_capacity, block_n)):
        cols = first * block_n + tl.arange(0, block_n)
        delta = cols - last_index
        positions = tl.where(delta <= 0, last + delta, last + delta - ring_capacity)
        distance = start + rows[:, None] - positions[None, :]
        allowed = (
            (rows[:, None] < length)
            & (cols[None, :] < ring_capacity)
            & (positions[None, :] >= 0)
            & (positions[None, :] < end)
            & (distance >= 0)
        )
        if context_size > 0:
            allowed = allowed & (distance < context_size)
        base = (slot * num_heads + head) * ring_capacity * head_dim
        k = tl.load(cache_ptr + base + cols[None, :] * head_dim + dims[:, None], cols[None, :] < ring_capacity, 0)
        score = tl.dot(q, k).to(tl.float32) * (head_dim**-0.5)
        score = tl.where(allowed, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        rescale = tl.exp(maximum - safe_max)
        p = tl.exp(score - safe_max[:, None])
        denominator = denominator * rescale + tl.sum(p, 1)
        v = tl.load(
            cache_ptr
            + state_capacity * num_heads * ring_capacity * head_dim
            + base
            + cols[:, None] * head_dim
            + dims[None, :],
            cols[:, None] < ring_capacity,
            0,
        )
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        maximum = next_max
    out = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        out_ptr + ((batch * num_heads + head) * num_frames + rows[:, None]) * head_dim + dims[None, :],
        out,
        rows[:, None] < num_frames,
    )


@triton.jit
def _advance(
    end_ptr,
    slots_ptr,
    lengths_ptr,
    end_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    length_stride: tl.constexpr,
    batch_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.arange(0, block_size)
    slot = tl.load(slots_ptr + row * slot_stride, row < batch_size, 0)
    length = tl.load(lengths_ptr + row * length_stride, row < batch_size, 0)
    active = (row < batch_size) & (length > 0)
    start = tl.load(end_ptr + slot * end_stride, active, 0)
    tl.store(end_ptr + slot * end_stride, start + length, active)


@torch.library.custom_op("vllm_omni::moss_slot_ring_attention", mutates_args=("cache", "end_offset"))
def slot_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    end_offset: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    rope_offset: torch.Tensor | None = None,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Fuse KV write into attention when one CTA owns the entire slot/head.

    With rope_offset, q/k are unrotated GPT-J inputs. The caller still advances its
    separate MHA/RoPE offset state; this op advances only end_offset in place.
    context=-1 means unbounded causal attention over available ring entries.
    Outputs beyond each row's valid_lengths are exactly zero.
    """
    b, h, t, d = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q/k/v must have identical shapes")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must share bf16 or fp16 dtype")
    if d not in (32, 64, 128) or t < 1:
        raise ValueError("requires head_dim 32/64/128 and positive padded_T")
    if cache.ndim != 5 or cache.shape[0] != 2 or cache.shape[2] != h or cache.shape[4] != d:
        raise ValueError("cache must be (2, state_capacity, heads, capacity, head_dim)")
    s, c = cache.shape[1], cache.shape[3]
    if not cache.is_contiguous() or cache.dtype != q.dtype:
        raise ValueError("cache must be contiguous and share q dtype")
    if end_offset.shape != (s,) or slot_ids.shape != (b,) or valid_lengths.shape != (b,):
        raise ValueError("invalid metadata shapes")
    if any(x.dtype not in (torch.int32, torch.int64) for x in (end_offset, slot_ids, valid_lengths)):
        raise ValueError("metadata must be int32/int64")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v, cache, end_offset, slot_ids, valid_lengths)):
        raise ValueError("all tensors must share one CUDA device")
    if context == 0 or context < -1:
        raise ValueError("context must be -1 or positive")
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if b == 0:
        return out
    rotate = rope_offset is not None
    if rotate:
        if rope_offset.shape != (b,) or rope_offset.device != q.device:
            raise ValueError("rope_offset must contain one CUDA position per execution row")
        if rope_offset.dtype not in (torch.int32, torch.int64) or max_period <= 0:
            raise ValueError("invalid RoPE offset dtype or period")
    rope_ptr = rope_offset if rotate else end_offset
    rope_stride = rope_ptr.stride(0)
    rope_scale = -math.log(max_period) * 2 / d
    if rotate and t > 64:
        # Large query tiles benefit from contiguous, head-major rotated Q/K.
        # Keeping this preparation separate avoids repeating rotation in the
        # attention CTAs and preserves the original memory access pattern.
        qr, kr = torch.empty_like(out), torch.empty_like(out)
        _prepare_rope[(b, triton.cdiv(h * t * d, 256))](
            q,
            k,
            rope_ptr,
            qr,
            kr,
            *q.stride(),
            *k.stride(),
            rope_stride,
            h,
            t,
            d,
            rope_scale,
            256,
        )
        q, k = qr, kr
        rotate = False
    strides = (end_offset.stride(0), slot_ids.stride(0), valid_lengths.stride(0))
    fused_write = t <= (16 if t <= 32 else 64)
    if not fused_write:
        _write[(b, triton.cdiv(h * t * d, 256))](
            k,
            v,
            rope_ptr,
            cache,
            end_offset,
            slot_ids,
            valid_lengths,
            *k.stride(),
            *v.stride(),
            *strides,
            s,
            h,
            t,
            c,
            d,
            256,
            rope_scale,
            rope_stride,
            rotate,
        )
    bm = 16 if t <= 32 else 64
    _attend[(triton.cdiv(t, bm), b * h)](
        q,
        k,
        v,
        rope_ptr,
        cache,
        end_offset,
        slot_ids,
        valid_lengths,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *strides,
        s,
        h,
        t,
        c,
        d,
        context,
        bm,
        64,
        fused_write,
        rope_scale,
        rope_stride,
        rotate,
        num_warps=8 if c <= 256 else 4,
        num_stages=2,
    )
    _advance[(1,)](end_offset, slot_ids, valid_lengths, *strides, b, triton.next_power_of_2(b))
    return out


@slot_ring_attention.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    end_offset: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    rope_offset: torch.Tensor | None = None,
    max_period: float = 10000.0,
) -> torch.Tensor:
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


slot_ring_attention.supports_rope = True
