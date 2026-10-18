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

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .triton_launch import cuda_pdl_launch_kwargs

_gdc_launch = tl.extra.cuda.gdc_launch_dependents
_gdc_wait = tl.extra.cuda.gdc_wait

# Hopper+ supports programmatic dependent launch. The kernel waits for its
# predecessor before any global access and lets its successor launch at once; results are unchanged.
_USE_PDL = current_platform.is_cuda() and current_platform.has_device_capability(90)
# PTX exp and pointer-select loads are CUDA Triton. HIP rejects the asm, and
# tl.where on pointers crashes AMD pointer canonicalization.
_CUDA_TRITON = current_platform.is_cuda() and not current_platform.is_rocm()


def _bthd_out(q: torch.Tensor) -> torch.Tensor:
    """(B, H, T, D) output backed by (B, T, H, D) storage.

    The codec immediately transposes attention output back to (B, T, H*D), so
    this layout makes that reshape a view instead of a copy kernel.
    """
    b, h, t, d = q.shape
    return torch.empty((b, t, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)


@triton.jit
def _write(
    k_ptr,
    v_ptr,
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
    row_starts: tl.constexpr = False,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row * slot_stride)
    length = tl.load(lengths_ptr + row * length_stride)
    # end_ptr holds per-slot end offsets, or with row_starts per-row starts.
    start = tl.load(end_ptr + (row if row_starts else slot) * end_stride)
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
    vval = tl.load(v_ptr + row * vs0 + head * vs1 + source_token * vs2 + dim * vs3, mask, 0)
    tl.store(cache_ptr + dst, kval, mask)
    tl.store(cache_ptr + state_capacity * num_heads * ring_capacity * head_dim + dst, vval, mask)


@triton.jit
def _attend(
    q_ptr,
    cache_ptr,
    end_ptr,
    slots_ptr,
    lengths_ptr,
    out_ptr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    qs3: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    os2: tl.constexpr,
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
    skip_empty_tiles: tl.constexpr = False,
    row_starts: tl.constexpr = False,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    bh = tl.program_id(1)
    batch, head = bh // num_heads, bh % num_heads
    slot = tl.load(slots_ptr + batch * slot_stride)
    length = tl.load(lengths_ptr + batch * length_stride)
    start = tl.load(end_ptr + (batch if row_starts else slot) * end_stride)
    end = start + length
    dims = tl.arange(0, head_dim)
    q = tl.load(
        q_ptr + batch * qs0 + head * qs1 + rows[:, None] * qs2 + dims[None, :] * qs3, rows[:, None] < num_frames, 0
    )
    acc = tl.full((block_m, head_dim), 0.0, tl.float32)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0.0, tl.float32)
    # Padding rows own scratch slots and must emit zeros without reading KV.
    # Keep valid-row accumulation and the ring overwrite semantics unchanged.
    if skip_empty_tiles:
        block_start = tl.program_id(0) * block_m
        live = (length > 0) & (block_start < length)
        live = live & (block_start + block_m > tl.maximum(length - ring_capacity, 0))
    else:
        live = length > 0
    if live:
        # Physical-ring order matches the existing masked_attention accumulation.
        last = end - 1
        last_index = last % ring_capacity
        if skip_empty_tiles:
            # Before the first wrap, only physical keys [0, end) exist.
            # Omitted blocks have zero probability for every query; keep the
            # physical order and arithmetic of all remaining blocks unchanged.
            key_blocks = tl.cdiv(tl.minimum(end, ring_capacity), block_n)
        else:
            key_blocks = tl.cdiv(ring_capacity, block_n)
        for first in range(key_blocks):
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
        out_ptr + batch * os0 + head * os1 + rows[:, None] * os2 + dims[None, :],
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


@triton.jit
def _write_attend(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    starts_ptr,
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
    os0: tl.constexpr,
    os1: tl.constexpr,
    os2: tl.constexpr,
    start_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    length_stride: tl.constexpr,
    state_capacity: tl.constexpr,
    num_heads: tl.constexpr,
    num_frames: tl.constexpr,
    ring_capacity: tl.constexpr,
    head_dim: tl.constexpr,
    context_size: tl.constexpr,
    write_frames: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_w: tl.constexpr,
    skip_empty_tiles: tl.constexpr = False,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    bh = tl.program_id(1)
    batch, head = bh // num_heads, bh % num_heads
    slot = tl.load(slots_ptr + batch * slot_stride)
    length = tl.load(lengths_ptr + batch * length_stride)
    start = tl.load(starts_ptr + batch * start_stride)
    end = start + length
    dims = tl.arange(0, head_dim)
    base = (slot * num_heads + head) * ring_capacity * head_dim
    v_base = state_capacity * num_heads * ring_capacity * head_dim + base
    # Every program of this (row, head) writes the whole chunk for its head,
    # so it only reads ring entries it wrote itself (duplicate writes across
    # query blocks store identical values). Chunks longer than the ring keep
    # only their final ring_capacity tokens, as in _write.
    skip = tl.maximum(length - ring_capacity, 0)
    for first in range(tl.cdiv(write_frames, block_w)):
        token = first * block_w + tl.arange(0, block_w)
        source = token + skip
        mask = (source < length)[:, None]
        pos = (start + source) % ring_capacity
        kval = tl.load(k_ptr + batch * ks0 + head * ks1 + source[:, None] * ks2 + dims[None, :] * ks3, mask, 0)
        vval = tl.load(v_ptr + batch * vs0 + head * vs1 + source[:, None] * vs2 + dims[None, :] * vs3, mask, 0)
        dst = pos[:, None] * head_dim + dims[None, :]
        tl.store(cache_ptr + base + dst, kval, mask)
        tl.store(cache_ptr + v_base + dst, vval, mask)
    tl.debug_barrier()
    q = tl.load(
        q_ptr + batch * qs0 + head * qs1 + rows[:, None] * qs2 + dims[None, :] * qs3, rows[:, None] < num_frames, 0
    )
    acc = tl.full((block_m, head_dim), 0.0, tl.float32)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0.0, tl.float32)
    # Same exact tile skipping as _attend: omitted tiles are wholly masked, so
    # their online-softmax update is the identity.
    if skip_empty_tiles:
        block_start = tl.program_id(0) * block_m
        live = (length > 0) & (block_start < length)
        live = live & (block_start + block_m > tl.maximum(length - ring_capacity, 0))
    else:
        live = length > 0
    if live:
        last = end - 1
        last_index = last % ring_capacity
        if skip_empty_tiles:
            key_blocks = tl.cdiv(tl.minimum(end, ring_capacity), block_n)
        else:
            key_blocks = tl.cdiv(ring_capacity, block_n)
        for block in range(key_blocks):
            cols = block * block_n + tl.arange(0, block_n)
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
            k = tl.load(cache_ptr + base + cols[None, :] * head_dim + dims[:, None], cols[None, :] < ring_capacity, 0)
            score = tl.dot(q, k).to(tl.float32) * (head_dim**-0.5)
            score = tl.where(allowed, score, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(score, 1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            rescale = tl.exp(maximum - safe_max)
            p = tl.exp(score - safe_max[:, None])
            denominator = denominator * rescale + tl.sum(p, 1)
            v = tl.load(
                cache_ptr + v_base + cols[:, None] * head_dim + dims[None, :],
                cols[:, None] < ring_capacity,
                0,
            )
            acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
            maximum = next_max
    out = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        out_ptr + batch * os0 + head * os1 + rows[:, None] * os2 + dims[None, :],
        out,
        rows[:, None] < num_frames,
    )


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
    skip_empty_tiles: bool = False,
    advance_offsets: bool = True,
) -> torch.Tensor:
    """Fused address/mask path, with three ordered launches and no KV gather.

    q/k are already rotary-position encoded. The caller still advances its
    separate MHA/RoPE offset state. ``advance_offsets=False`` leaves the shared
    resolution group's offsets to its owner; the default advances end_offset.
    context=-1 means unbounded causal attention over available ring entries.
    Outputs beyond each row's valid_lengths are exactly zero.
    skip_empty_tiles skips wholly masked tiles for small (T <= 32) chunks.
    Larger chunks keep the static loop, which benchmarks faster after wrap.
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
    out = _bthd_out(q)
    if b == 0:
        return out
    strides = (end_offset.stride(0), slot_ids.stride(0), valid_lengths.stride(0))
    _write[(b, triton.cdiv(h * t * d, 256))](
        k, v, cache, end_offset, slot_ids, valid_lengths, *k.stride(), *v.stride(), *strides, s, h, t, c, d, 256
    )
    bm = 16 if t <= 32 else 64
    _attend[(triton.cdiv(t, bm), b * h)](
        q,
        cache,
        end_offset,
        slot_ids,
        valid_lengths,
        out,
        *q.stride(),
        *out.stride()[:3],
        *strides,
        s,
        h,
        t,
        c,
        d,
        context,
        bm,
        64,
        skip_empty_tiles=skip_empty_tiles and t <= 32,
        num_warps=8 if c <= 256 else 4,
        num_stages=2,
    )
    if advance_offsets:
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
    skip_empty_tiles: bool = False,
    advance_offsets: bool = True,
) -> torch.Tensor:
    return _bthd_out(q)


def _check_inputs(q, k, v, cache, metadata, context):
    b, h, t, d = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q/k/v must have identical shapes")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must share bf16 or fp16 dtype")
    if d not in (32, 64, 128) or t < 1:
        raise ValueError("requires head_dim 32/64/128 and positive padded_T")
    if cache.ndim != 5 or cache.shape[0] != 2 or cache.shape[2] != h or cache.shape[4] != d:
        raise ValueError("cache must be (2, state_capacity, heads, capacity, head_dim)")
    if not cache.is_contiguous() or cache.dtype != q.dtype:
        raise ValueError("cache must be contiguous and share q dtype")
    if any(x.dtype not in (torch.int32, torch.int64) for x in metadata):
        raise ValueError("metadata must be int32/int64")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v, cache, *metadata)):
        raise ValueError("all tensors must share one CUDA device")
    if context == 0 or context < -1:
        raise ValueError("context must be -1 or positive")


# Measured crossover on H200 (codec stages, B=8..32): the single fused launch
# wins up to T=120 frames, the write + attend pair from T=240.
_FUSED_ROWS_MAX_FRAMES = 128


@torch.library.custom_op("vllm_omni::moss_slot_ring_attention_rows", mutates_args=("cache",))
def slot_ring_attention_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    starts: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    skip_empty_tiles: bool = False,
) -> torch.Tensor:
    """``slot_ring_attention`` as one launch, with per-row start positions.

    ``starts`` holds each execution row's pre-call absolute position (the
    caller's gathered offset). The ring is written and attended exactly as by
    ``slot_ring_attention``; no offset is advanced here, so the caller owns
    every offset update and can fuse them with its own.
    """
    b, h, t, d = q.shape
    _check_inputs(q, k, v, cache, (starts, slot_ids, valid_lengths), context)
    if starts.shape != (b,) or slot_ids.shape != (b,) or valid_lengths.shape != (b,):
        raise ValueError("invalid metadata shapes")
    out = _bthd_out(q)
    if b == 0:
        return out
    s, c = cache.shape[1], cache.shape[3]
    bm = 16 if t <= 32 else 64
    if t > _FUSED_ROWS_MAX_FRAMES:
        # Long chunks: every fused program would rewrite the whole chunk for
        # its head. One write launch, then the attention launch, is faster
        # there and computes the identical ring and output.
        strides = (starts.stride(0), slot_ids.stride(0), valid_lengths.stride(0))
        _write[(b, triton.cdiv(h * t * d, 256))](
            k,
            v,
            cache,
            starts,
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
            row_starts=True,
        )
        _attend[(triton.cdiv(t, bm), b * h)](
            q,
            cache,
            starts,
            slot_ids,
            valid_lengths,
            out,
            *q.stride(),
            *out.stride()[:3],
            *strides,
            s,
            h,
            t,
            c,
            d,
            context,
            bm,
            64,
            skip_empty_tiles=False,
            row_starts=True,
            num_warps=8 if c <= 256 else 4,
            num_stages=2,
        )
        return out
    _write_attend[(triton.cdiv(t, bm), b * h)](
        q,
        k,
        v,
        cache,
        starts,
        slot_ids,
        valid_lengths,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride()[:3],
        starts.stride(0),
        slot_ids.stride(0),
        valid_lengths.stride(0),
        s,
        h,
        t,
        c,
        d,
        context,
        min(t, c),
        bm,
        64,
        16,
        skip_empty_tiles=skip_empty_tiles and t <= 32,
        num_warps=8 if c <= 256 else 4,
        num_stages=2,
    )
    return out


@slot_ring_attention_rows.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    starts: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    skip_empty_tiles: bool = False,
) -> torch.Tensor:
    return _bthd_out(q)


@triton.jit
def _exp_ftz(x, USE_PTX_EXP: tl.constexpr):  # noqa: N803
    # tl.exp(x) lowers to ex2.approx.f32(x * log2e); .ftz only drops the
    # subnormal-result fixup, so results differ only below 2**-126.
    # The inline asm is PTX, so HIP uses tl.exp.
    if USE_PTX_EXP:
        return tl.inline_asm_elementwise(
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            [x * 1.4426950408889634],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return tl.exp(x)


@triton.jit
def _ring_flash(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    starts_ptr,
    slots_ptr,
    lengths_ptr,
    out_ptr,
    tok_starts_ptr,
    entry_seg_ptr,
    entry_blk_ptr,
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
    os0: tl.constexpr,
    os1: tl.constexpr,
    os2: tl.constexpr,
    start_stride: tl.constexpr,
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
    row_starts: tl.constexpr,
    skip_interior_mask: tl.constexpr,
    fma_scale: tl.constexpr,
    packed: tl.constexpr,
    USE_PDL: tl.constexpr = False,  # noqa: N803
    USE_PTX_EXP: tl.constexpr = False,  # noqa: N803
    SELECT_PTR: tl.constexpr = False,  # noqa: N803
):
    """Ring write + attention in one pass, without redundant work.

    Each program owns ``block_m`` query rows of one (row, head) and writes
    only those rows' K/V into the ring. Keys of the current chunk (positions
    >= start) are read straight from k/v, older keys from the ring. A chunk
    only overwrites slots holding positions < end - capacity, which no query
    attends to, so no program reads a slot that another program writes.

    The visited tiles retain ``_attend``'s physical ring order and online
    softmax update. Tested configurations match exactly; this is not a
    universal bitwise guarantee, since FTZ drops subnormal weights and Triton
    may change reduction layouts. Differences from ``_attend``:
    - only key tiles holding an allowed (query, key) pair of this block are
      visited; a wholly masked tile's update is the identity;
    - exp uses ex2.approx.ftz (equal unless a weight is below 2**-126);
    - skip_interior_mask: interior tiles (every row valid, every key allowed)
      skip the mask;
    - fma_scale: a power-of-two softmax scale is applied by an FMA after the
      row max (exact: the product is exact, so fma rounds like mul + sub).
    The last two change the surrounding code, and Triton may then choose a
    different MMA/reduction layout for multi-warpgroup tiles; the launcher
    enables them only for the single-warpgroup 64-row tile.

    ``packed``: q/k/v/out hold one (1, H, N, D) sequence of segments, one per
    stream. Program ``(e, head)`` takes query block ``entry_blk[e]`` of
    segment ``entry_seg[e]``, whose tokens start at ``tok_starts[segment]``;
    per-row arithmetic is that of the batched launch with the same tile.
    """
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    if packed:
        head = bh
        batch = tl.load(entry_seg_ptr + pid_m).to(tl.int32)
        pid_m = tl.load(entry_blk_ptr + pid_m).to(tl.int32)
        tok0 = tl.load(tok_starts_ptr + batch).to(tl.int64)
    else:
        batch, head = bh // num_heads, bh % num_heads
        tok0 = 0
    slot = tl.load(slots_ptr + batch * slot_stride)
    length = tl.load(lengths_ptr + batch * length_stride).to(tl.int32)
    start = tl.load(starts_ptr + (batch if row_starts else slot) * start_stride)
    end = start + length
    # Rows past the segment belong to the next stream in packed mode.
    row_limit = length if packed else num_frames
    rows = pid_m * block_m + tl.arange(0, block_m)
    dims = tl.arange(0, head_dim)
    base = (slot * num_heads + head) * ring_capacity * head_dim
    v_base = state_capacity * num_heads * ring_capacity * head_dim + base
    if packed:
        q_ptr += head * qs1 + tok0 * qs2
        out_ptr += head * os1 + tok0 * os2
        k_row = k_ptr + head * ks1 + tok0 * ks2
        v_row = v_ptr + head * vs1 + tok0 * vs2
    else:
        q_ptr += batch * qs0 + head * qs1
        out_ptr += batch * os0 + head * os1
        k_row = k_ptr + batch * ks0 + head * ks1
        v_row = v_ptr + batch * vs0 + head * vs1
    scale: tl.constexpr = head_dim**-0.5

    # Ring write of this block's rows. Chunks longer than the ring keep only
    # their final ring_capacity tokens (no duplicate addresses in a launch).
    keep = (rows >= tl.maximum(length - ring_capacity, 0)) & (rows < length)
    dst = ((start + rows) % ring_capacity)[:, None] * head_dim + dims[None, :]
    kval = tl.load(k_row + rows[:, None] * ks2 + dims[None, :] * ks3, keep[:, None], 0)
    vval = tl.load(v_row + rows[:, None] * vs2 + dims[None, :] * vs3, keep[:, None], 0)
    tl.store(cache_ptr + base + dst, kval, keep[:, None])
    tl.store(cache_ptr + v_base + dst, vval, keep[:, None])

    q = tl.load(q_ptr + rows[:, None] * qs2 + dims[None, :] * qs3, rows[:, None] < row_limit, 0)
    acc = tl.full((block_m, head_dim), 0.0, tl.float32)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0.0, tl.float32)
    # [lo, hi]: union of the allowed key positions of this block's valid rows.
    row_lo = pid_m * block_m
    row_hi = tl.minimum(row_lo + block_m, length) - 1
    lo = tl.maximum(end - ring_capacity, 0)
    if context_size > 0:
        lo = tl.maximum(lo, start + row_lo - context_size + 1)
    hi = start + row_hi
    if (row_lo < length) & (lo <= hi):
        # Physical column c holds token (position - start)
        # c - last_index + newest, minus the capacity past the newest column.
        last_index = ((end - 1) % ring_capacity).to(tl.int32)
        newest = length - 1
        # position >= 0 <=> token >= -start (start may exceed int32).
        min_token = (-tl.minimum(start, ring_capacity)).to(tl.int32)
        tile_lo = ((lo % ring_capacity) // block_n).to(tl.int32)
        tile_hi = ((hi % ring_capacity) // block_n).to(tl.int32)
        num_tiles: tl.constexpr = (ring_capacity + block_n - 1) // block_n
        # A wrapped range visits physical tiles [0, tile_hi] then
        # [tile_lo, num_tiles), keeping the ascending physical order.
        wrapped = lo % ring_capacity > hi % ring_capacity
        first_start = tl.where(wrapped, 0, tile_lo)
        first_count = tl.where(wrapped, tile_hi + 1, tile_hi - tile_lo + 1)
        second_start = tl.maximum(tile_lo, tile_hi + 1)
        second_count = tl.where(wrapped, num_tiles - second_start, 0)
        row_key = tl.where(rows < length, rows, -(1 << 30))
        for i in range(first_count + second_count):
            block = tl.where(i < first_count, first_start + i, second_start + i - first_count)
            cols = block * block_n + tl.arange(0, block_n)
            token = cols - last_index + newest - tl.where(cols > last_index, ring_capacity, 0)
            in_ring = cols < ring_capacity
            fresh = in_ring & (token >= 0)
            k_fresh_ptr = k_row + token[None, :] * ks2 + dims[:, None] * ks3
            k_cache_ptr = cache_ptr + base + cols[None, :] * head_dim + dims[:, None]
            if SELECT_PTR:
                k_src = tl.where(fresh[None, :], k_fresh_ptr, k_cache_ptr)
                k = tl.load(k_src, in_ring[None, :], 0)
            else:
                k_fresh = tl.load(k_fresh_ptr, fresh[None, :] & in_ring[None, :], 0)
                k_cache = tl.load(k_cache_ptr, (~fresh)[None, :] & in_ring[None, :], 0)
                k = tl.where(fresh[None, :], k_fresh, k_cache)
            score = tl.dot(q, k).to(tl.float32)
            if not fma_scale:
                score = score * scale
            interior = False
            if skip_interior_mask:
                tile_first = block * block_n
                first_token = tile_first - last_index + newest - tl.where(tile_first > last_index, ring_capacity, 0)
                interior = (row_lo + block_m <= length) & (tile_first + block_n <= ring_capacity)
                interior &= (tile_first > last_index) | (tile_first + block_n - 1 <= last_index)
                interior &= (first_token >= min_token) & (first_token + block_n - 1 <= row_lo)
                if context_size > 0:
                    interior &= row_lo + block_m - 1 - first_token < context_size
            if not interior:
                # distance = start + row - position = row - token; invalid
                # rows and keys get distances outside [0, context).
                key_token = tl.where(in_ring & (token >= min_token), token, 1 << 30)
                distance = row_key[:, None] - key_token[None, :]
                if context_size > 0:
                    allowed = distance.to(tl.uint32, bitcast=True) < context_size
                else:
                    allowed = distance >= 0
                score = tl.where(allowed, score, -float("inf"))
            if fma_scale:
                # max(score * scale) == max(score) * scale for a power-of-two
                # scale, and fma(score, scale, -m) rounds once like the exact
                # product minus m.
                next_max = tl.maximum(maximum, tl.max(score, 1) * scale)
            else:
                next_max = tl.maximum(maximum, tl.max(score, 1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            rescale = tl.exp(maximum - safe_max)
            if fma_scale:
                p = _exp_ftz(tl.fma(score, scale, -safe_max[:, None]), USE_PTX_EXP)
            else:
                p = _exp_ftz(score - safe_max[:, None], USE_PTX_EXP)
            denominator = denominator * rescale + tl.sum(p, 1)
            v_fresh_ptr = v_row + token[:, None] * vs2 + dims[None, :] * vs3
            v_cache_ptr = cache_ptr + v_base + cols[:, None] * head_dim + dims[None, :]
            if SELECT_PTR:
                v_src = tl.where(fresh[:, None], v_fresh_ptr, v_cache_ptr)
                v = tl.load(v_src, in_ring[:, None], 0)
            else:
                v_fresh = tl.load(v_fresh_ptr, fresh[:, None] & in_ring[:, None], 0)
                v_cache = tl.load(v_cache_ptr, (~fresh)[:, None] & in_ring[:, None], 0)
                v = tl.where(fresh[:, None], v_fresh, v_cache)
            acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
            maximum = next_max
    out = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(out_ptr + rows[:, None] * os2 + dims[None, :], out, rows[:, None] < row_limit)


def _flash_config(t: int, c: int) -> tuple[int, int, int]:
    """(block_m, num_warps, num_stages); block_m/num_warps match ``_attend``.

    The MMA instruction and reduction layout follow block_m and num_warps, so
    keeping the existing choice keeps every row's arithmetic identical.
    """
    return (16 if t <= 32 else 64), (8 if c <= 256 else 4), 2


def _launch_flash(q, k, v, cache, starts, slot_ids, valid_lengths, context, row_starts, config=None):
    b, h, t, d = q.shape
    s, c = cache.shape[1], cache.shape[3]
    out = _bthd_out(q)
    if b == 0:
        return out
    bm, nw, ns = config or _flash_config(t, c)
    one_warpgroup = bm == 64 and nw == 4
    _ring_flash[(triton.cdiv(t, bm), b * h)](
        q,
        k,
        v,
        cache,
        starts,
        slot_ids,
        valid_lengths,
        out,
        slot_ids,  # packed-only pointers, unused
        slot_ids,
        slot_ids,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride()[:3],
        starts.stride(0),
        slot_ids.stride(0),
        valid_lengths.stride(0),
        s,
        h,
        t,
        c,
        d,
        context,
        bm,
        64,
        row_starts,
        one_warpgroup,
        one_warpgroup and d == 64,
        False,
        USE_PDL=_USE_PDL,
        USE_PTX_EXP=_CUDA_TRITON,
        SELECT_PTR=_CUDA_TRITON,
        num_warps=nw,
        num_stages=ns,
        **cuda_pdl_launch_kwargs(_USE_PDL),
    )
    return out


def packed_block_entries(lengths: torch.Tensor, block_m: int, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(entry_seg, entry_blk)`` of ``capacity`` program entries for packed segments.

    Segment ``s`` owns ``cdiv(lengths[s], block_m)`` query blocks, listed in
    segment order; spare entries get block index ``capacity`` (no valid row).
    ``capacity`` must be at least the number of blocks: ``N // block_m + S``
    for ``N`` packed tokens in ``S`` segments always is.
    """
    blocks = (lengths + block_m - 1) // block_m
    ends = torch.cumsum(blocks, 0)
    entry = torch.arange(capacity, device=lengths.device, dtype=ends.dtype)
    seg = torch.searchsorted(ends, entry, right=True).clamp_(max=lengths.shape[0] - 1)
    blk = entry - (ends[seg] - blocks[seg])
    blk = torch.where(entry < ends[-1], blk, capacity)
    return seg.to(torch.int32), blk.to(torch.int32)


def _launch_flash_packed(
    q, k, v, cache, starts, slot_ids, lengths, tok_starts, entry_seg, entry_blk, context, block_m, launch
):
    _, h, _, d = q.shape
    s, c = cache.shape[1], cache.shape[3]
    out = _bthd_out(q)
    block_n, nw, ns = launch
    nw = nw or (8 if c <= 256 else 4)
    one_warpgroup = block_m == 64 and nw == 4
    # Strides and sizes are constexpr: drop the N-dependent ones (the batch
    # stride and the frame count, unused when packed) so every packed size
    # shares one compiled kernel per stage. Token-major q/k/v keep the rest fixed.
    _ring_flash[(entry_seg.shape[0], h)](
        q,
        k,
        v,
        cache,
        starts,
        slot_ids,
        lengths,
        out,
        tok_starts,
        entry_seg,
        entry_blk,
        0,
        *q.stride()[1:],
        0,
        *k.stride()[1:],
        0,
        *v.stride()[1:],
        0,
        *out.stride()[1:3],
        starts.stride(0),
        slot_ids.stride(0),
        lengths.stride(0),
        s,
        h,
        1,
        c,
        d,
        context,
        block_m,
        block_n or 64,
        True,
        one_warpgroup,
        one_warpgroup and d == 64,
        True,
        USE_PDL=_USE_PDL,
        USE_PTX_EXP=_CUDA_TRITON,
        SELECT_PTR=_CUDA_TRITON,
        num_warps=nw,
        num_stages=ns or 2,
        **cuda_pdl_launch_kwargs(_USE_PDL),
    )
    return out


@torch.library.custom_op("vllm_omni::moss_slot_ring_flash_attention_packed", mutates_args=("cache",))
def slot_ring_flash_attention_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    starts: torch.Tensor,
    slot_ids: torch.Tensor,
    lengths: torch.Tensor,
    tok_starts: torch.Tensor,
    entry_seg: torch.Tensor,
    entry_blk: torch.Tensor,
    context: int,
    block_m: int,
    block_n: int = 0,
    num_warps: int = 0,
    num_stages: int = 0,
) -> torch.Tensor:
    """Ring write + attention for streams packed along time.

    q/k/v are (1, H, N, D): segment ``s`` is tokens ``[tok_starts[s],
    tok_starts[s] + lengths[s])`` of state slot ``slot_ids[s]``, whose
    pre-call position is ``starts[s]``. Rows of a segment get the arithmetic
    of a batched ``slot_ring_flash_attention_rows`` launch with tile
    ``block_m``. Slots must be unique; the caller advances the offsets.
    Output rows outside every segment are left unwritten. ``block_n``,
    ``num_warps`` and ``num_stages`` (0: the batched launch's) retile the
    key loop: same math, rounding-level differences.
    """
    _check_inputs(q, k, v, cache, (starts, slot_ids, lengths, tok_starts, entry_seg, entry_blk), context)
    if q.shape[0] != 1 or block_m not in (16, 32, 64, 128):
        raise ValueError("packed attention takes one (1, H, N, D) sequence and block_m 16/32/64/128")
    return _launch_flash_packed(
        q,
        k,
        v,
        cache,
        starts,
        slot_ids,
        lengths,
        tok_starts,
        entry_seg,
        entry_blk,
        context,
        block_m,
        (block_n, num_warps, num_stages),
    )


@slot_ring_flash_attention_packed.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    starts: torch.Tensor,
    slot_ids: torch.Tensor,
    lengths: torch.Tensor,
    tok_starts: torch.Tensor,
    entry_seg: torch.Tensor,
    entry_blk: torch.Tensor,
    context: int,
    block_m: int,
    block_n: int = 0,
    num_warps: int = 0,
    num_stages: int = 0,
) -> torch.Tensor:
    return _bthd_out(q)


@torch.library.custom_op("vllm_omni::moss_slot_ring_flash_attention", mutates_args=("cache", "end_offset"))
def slot_ring_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    end_offset: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    skip_empty_tiles: bool = False,
) -> torch.Tensor:
    """``slot_ring_attention`` with the single-pass flash ring kernel.

    Same state and masking contract; one write+attend launch plus the offset
    advance. FTZ and reduction-layout choices prevent a universal bitwise
    output guarantee. ``skip_empty_tiles`` is accepted for signature
    compatibility: this kernel always skips wholly masked tiles. Non-CUDA
    runtimes use the portable ring path to preserve exact output parity.
    """
    if not _CUDA_TRITON:
        # HIP cannot use the CUDA pointer-select path. Its separate masked
        # loads change the reduction layout for larger chunks, so preserve the
        # established exact-output contract with the portable ring kernels.
        return slot_ring_attention(
            q,
            k,
            v,
            cache,
            end_offset,
            slot_ids,
            valid_lengths,
            context,
            skip_empty_tiles,
        )
    b = q.shape[0]
    _check_inputs(q, k, v, cache, (end_offset, slot_ids, valid_lengths), context)
    if end_offset.shape != (cache.shape[1],) or slot_ids.shape != (b,) or valid_lengths.shape != (b,):
        raise ValueError("invalid metadata shapes")
    out = _launch_flash(q, k, v, cache, end_offset, slot_ids, valid_lengths, context, row_starts=False)
    if b:
        strides = (end_offset.stride(0), slot_ids.stride(0), valid_lengths.stride(0))
        _advance[(1,)](end_offset, slot_ids, valid_lengths, *strides, b, triton.next_power_of_2(b))
    return out


@slot_ring_flash_attention.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    end_offset: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    skip_empty_tiles: bool = False,
) -> torch.Tensor:
    return _bthd_out(q)


@torch.library.custom_op("vllm_omni::moss_slot_ring_flash_attention_rows", mutates_args=("cache",))
def slot_ring_flash_attention_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    starts: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    skip_empty_tiles: bool = False,
) -> torch.Tensor:
    """``slot_ring_attention_rows`` with the single-pass flash ring kernel.

    One launch per chunk on CUDA, preserving ring updates and causal masking;
    see ``_ring_flash`` for the numerical limitations. Non-CUDA runtimes use
    the portable ring path to preserve exact output parity.
    """
    if not _CUDA_TRITON:
        return slot_ring_attention_rows(
            q,
            k,
            v,
            cache,
            starts,
            slot_ids,
            valid_lengths,
            context,
            skip_empty_tiles,
        )
    b = q.shape[0]
    _check_inputs(q, k, v, cache, (starts, slot_ids, valid_lengths), context)
    if starts.shape != (b,) or slot_ids.shape != (b,) or valid_lengths.shape != (b,):
        raise ValueError("invalid metadata shapes")
    return _launch_flash(q, k, v, cache, starts, slot_ids, valid_lengths, context, row_starts=True)


@slot_ring_flash_attention_rows.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    starts: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    context: int = -1,
    skip_empty_tiles: bool = False,
) -> torch.Tensor:
    return _bthd_out(q)
