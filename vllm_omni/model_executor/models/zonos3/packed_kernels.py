# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Zonos3 packed tensor-ring kernels; no page allocator or scheduler.

RoPE is applied by the Codec using Zonos3's adjacent pairs and FP32 phase
arithmetic before casting to QKV dtype. These kernels do not import Qwen's
half-rotation or its probability-rounding convention.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _segment_layout(
    OFFSETS, REQUEST, LOCAL, VALID, N: tl.constexpr, B: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), B, tl.int32)
    # Upper bound handles empty segments and the single global capacity tail.
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        end = tl.load(OFFSETS + tl.minimum(mid + 1, B))
        right = (mid < B) & (row >= end)
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    request = tl.minimum(lo, B - 1)
    start = tl.load(OFFSETS + request)
    total = tl.load(OFFSETS + B)
    tl.store(REQUEST + row, request, row < N)
    tl.store(LOCAL + row, row - start, row < N)
    tl.store(VALID + row, row < total, row < N)


def segment_layout(offsets, rows):
    """Map the packed slab to segments in one launch, including empty rows."""
    request = torch.empty(rows, device=offsets.device, dtype=torch.long)
    local = torch.empty_like(request)
    valid = torch.empty(rows, device=offsets.device, dtype=torch.bool)
    _segment_layout[(triton.cdiv(rows, 128),)](
        offsets, request, local, valid, rows, offsets.numel() - 1, offsets.numel().bit_length(), 128
    )
    return request, local, valid


@triton.jit
def _commit_conv_history(
    X, HISTORY, SLOTS, OFFSETS, C: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr, BLOCK: tl.constexpr
):
    request = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)[:, None]
    tap = tl.arange(0, 8)[None, :]
    slot = tl.load(SLOTS + request)
    start = tl.load(OFFSETS + request)
    length = tl.load(OFFSETS + request + 1) - start
    source = length + tap
    # Each program owns every history tap for its channels; read before writing.
    previous = tl.load(HISTORY + (slot * C + channel) * 6 + source, (channel < C) & (tap < 6) & (source < 6), other=0)
    fresh = tl.load(X + (start + source - 6) * XR + channel * XC, (channel < C) & (tap < 6) & (source >= 6), other=0)
    tl.store(HISTORY + (slot * C + channel) * 6 + tap, tl.where(source < 6, previous, fresh), (channel < C) & (tap < 6))


def commit_conv_history(x, history, slots, offsets):
    _commit_conv_history[(slots.numel(), triton.cdiv(x.shape[1], 128))](
        x, history, slots, offsets, x.shape[1], x.stride(0), x.stride(1), 128
    )


@triton.jit
def _ring_write(K, V, ADDRESS, KR, VR, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BD)
    address = tl.load(ADDRESS + row)
    src = (row * H + head) * D + d
    dst = (address * H + head) * D + d
    tl.store(KR + dst, tl.load(K + src, d < D, other=0), d < D)
    tl.store(VR + dst, tl.load(V + src, d < D, other=0), d < D)


@triton.jit
def _ring_attention(
    Q,
    KR,
    VR,
    SLOT,
    POSITION,
    ORIGIN,
    VALID,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    live = tl.load(VALID + row)
    slot = tl.load(SLOT + row)
    position = tl.load(POSITION + row)
    origin = tl.load(ORIGIN + row)
    d = tl.arange(0, BD)
    q = tl.load(Q + (row * H + head) * D + d, d < D, other=0).to(tl.float32)
    # Zonos3 calls torch SDPA. Its actual CUDA efficient backend stores the
    # *unnormalized* exponentials as scalar_t before P@V, and divides by the
    # FP32 denominator in the output epilogue. See PyTorch kernel_forward.h
    # iterative_softmax/accumToSmem and epilogue_rescale_output.h. Controlled
    # injection of released Q/K/V verifies this boundary; this is not Qwen's
    # normalized-probability rounding convention.
    # Match the observed 64-key efficient-SDPA tiles, aligned to the chunk's
    # history origin rather than shifting the reduction tree for each Q.
    # All rows still execute together in one launch; no per-request dispatch.
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    numerator = tl.full((BD,), 0, tl.float32)
    first = tl.maximum(0, (position - origin) // BK)
    last = tl.minimum(tl.cdiv(R, BK), tl.cdiv(position - origin + W, BK))
    for block in range(first, last):
        j = block * BK + tl.arange(0, BK)
        p = origin - (W - 1) + j
        mask = live & (j < R) & (p >= 0) & (p <= position) & (p > position - W)
        safe_p = tl.where(mask, p, 0)
        address = ((slot * R + safe_p % R) * H + head) * D
        k = tl.load(KR + address[:, None] + d[None, :], mask[:, None] & (d < D)[None, :], other=0).to(tl.float32)
        logits = (tl.sum(k * q[None, :], 1) * (D**-0.5)) * 1.4426950408889634
        logits = tl.where(mask, logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 0))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        rescale = tl.exp2(maximum - safe_maximum)
        probability = tl.where(mask, tl.exp2(logits - safe_maximum), 0.0)
        denominator = denominator * rescale + tl.sum(probability, 0)
        probability = probability.to(Q.dtype.element_ty).to(tl.float32)
        v = tl.load(VR + address[:, None] + d[None, :], mask[:, None] & (d < D)[None, :], other=0).to(tl.float32)
        numerator = numerator * rescale + tl.sum(v * probability[:, None], 0)
        maximum = next_maximum
    result = numerator / tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + (row * H + head) * D + d, result, d < D)


def packed_ring_attention(q, k, v, address, keys, values, slots, positions, origins, valid, ring, window):
    """One KV write launch and one attention launch across all packed Q."""
    rows, heads, dim = q.shape
    out = torch.empty_like(q)
    grid = (rows, heads)
    _ring_write[grid](k.contiguous(), v.contiguous(), address, keys, values, heads, dim, triton.next_power_of_2(dim))
    _ring_attention[grid](
        q,
        keys,
        values,
        slots,
        positions,
        origins,
        valid,
        out,
        heads,
        dim,
        ring,
        window,
        triton.next_power_of_2(dim),
        64,
    )
    return out


@triton.jit
def _segmented_fir(
    A,
    B,
    WEIGHT,
    SLOTS,
    REQUEST,
    VALID,
    POSITION,
    TOTAL,
    OUT,
    N: tl.constexpr,
    RING: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    live = tl.load(VALID + row, row < N, other=0)
    req = tl.load(REQUEST + row, row < N, other=0)
    slot = tl.load(SLOTS + req)
    pos = tl.load(POSITION + row, row < N, other=0)
    total = tl.load(TOTAL + req)
    tap = tl.arange(0, BK)
    source = pos[:, None] + tap[None, :] - K // 2
    mask = live[:, None] & (tap < K)[None, :] & (source >= 0) & (source < total[:, None])
    address = slot[:, None] * RING + (source + RING) % RING
    a = tl.load(A + address, mask, other=0).to(tl.float64)
    b = tl.load(B + address, mask, other=0).to(tl.float64)
    wa = tl.load(WEIGHT + tap, tap < K, other=0).to(tl.float64)
    wb = tl.load(WEIGHT + K + tap, tap < K, other=0).to(tl.float64)
    # Match the prototype/reference branch sums and the official FP32 boundary.
    value = (tl.sum(a * wa[None, :], 1) + tl.sum(b * wb[None, :], 1)).to(tl.float32).to(tl.float64)
    tl.store(OUT + row, value, row < N)


@triton.jit
def _segmented_resample(
    X,
    WEIGHT,
    SLOTS,
    REQUEST,
    VALID,
    POSITION,
    TOTAL,
    OUT,
    N: tl.constexpr,
    RING: tl.constexpr,
    ORIG: tl.constexpr,
    NEW: tl.constexpr,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    live = tl.load(VALID + row, row < N, other=0)
    req = tl.load(REQUEST + row, row < N, other=0)
    slot = tl.load(SLOTS + req)
    pos = tl.load(POSITION + row, row < N, other=0)
    total = tl.load(TOTAL + req)
    tap = tl.arange(0, BK)
    source = (pos // NEW)[:, None] * ORIG - WIDTH + tap[None, :]
    mask = live[:, None] & (tap < K)[None, :] & (source >= 0) & (source < total[:, None])
    address = slot[:, None] * RING + (source + RING) % RING
    x = tl.load(X + address, mask, other=0).to(tl.float64)
    w = tl.load(WEIGHT + (pos % NEW)[:, None] * K + tap[None, :], (tap < K)[None, :], other=0).to(tl.float64)
    value = tl.sum(x * w, 1).to(tl.float32)
    tl.store(OUT + row, value, row < N)


def segmented_fir(waves, weight, slots, request, valid, position, total, ring):
    """Tap workspace is register-local; only a 1D output is allocated."""
    n = position.numel()
    out = torch.empty(n, dtype=torch.float64, device=position.device)
    k = weight.shape[-1]
    _segmented_fir[(triton.cdiv(n, 4),)](
        *waves,
        weight,
        slots,
        request,
        valid,
        position,
        total,
        out,
        n,
        ring,
        k,
        triton.next_power_of_2(k),
        4,
        enable_fp_fusion=False,
    )
    return out


def segmented_resample(x, weight, slots, request, valid, position, total, ring, orig, new, width):
    """Direct finite-history polyphase filtering without a global tap stencil."""
    n = position.numel()
    out = torch.empty(n, dtype=torch.float32, device=position.device)
    k = weight.shape[-1]
    _segmented_resample[(triton.cdiv(n, 4),)](
        x,
        weight,
        slots,
        request,
        valid,
        position,
        total,
        out,
        n,
        ring,
        orig,
        new,
        width,
        k,
        triton.next_power_of_2(k),
        4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _segmented_depthwise(
    X,
    HISTORY,
    WEIGHT,
    BIAS,
    SLOTS,
    REQUEST,
    LOCAL,
    VALID,
    OUT,
    C: tl.constexpr,
    BC: tl.constexpr,
    XR: tl.constexpr,
    XC: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    live = tl.load(VALID + row)
    req = tl.load(REQUEST + row)
    slot = tl.load(SLOTS + req)
    local = tl.load(LOCAL + row)
    # The profiled legacy operator is PyTorch's native generic depthwise CUDA
    # kernel (not cuDNN). DepthwiseConv2d.cu at the installed torch revision
    # initializes the accumulator with bias, then visits taps in order. Preserve
    # that order and explicit FP32 FMA rather than adding bias after the sum.
    acc = tl.load(BIAS + channel, channel < C, other=0).to(tl.float32)
    for tap in tl.static_range(7):
        from_history = local + tap < 6
        old = tl.load(
            HISTORY + (slot * C + channel) * 6 + local + tap,
            live & (channel < C) & from_history,
            other=0,
        ).to(tl.float32)
        new = tl.load(
            X + (row + tap - 6) * XR + channel * XC,
            live & (channel < C) & ~from_history,
            other=0,
        ).to(tl.float32)
        weight = tl.load(WEIGHT + channel * 7 + tap, channel < C, other=0).to(tl.float32)
        acc = tl.fma(tl.where(from_history, old, new), weight, acc)
    tl.store(OUT + row * C + channel, acc, channel < C)


def segmented_depthwise(x, history, weight, bias, slots, request, local, valid):
    """Seven-tap convolution reads boundaries directly; no im2col allocation."""
    rows, channels = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _segmented_depthwise[(rows, triton.cdiv(channels, 128))](
        x,
        history,
        weight,
        bias,
        slots,
        request,
        local,
        valid,
        out,
        channels,
        128,
        x.stride(0),
        x.stride(1),
        enable_fp_fusion=False,
    )
    return out
