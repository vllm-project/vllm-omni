# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Direct physical-ring attention with deferred KV append."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _attend_ring(
    Q,
    K,
    V,
    KP,
    VP,
    S,
    P,
    OUTPUT,
    QB: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    SS: tl.constexpr,
    PS: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    frame = tl.program_id(2)
    slot = tl.load(S + batch * SS)
    start = tl.load(P + batch * PS)
    j = tl.arange(0, BW)
    d = tl.arange(0, BD)
    absolute = start + frame - W + 1 + j
    valid = (j < W) & (absolute >= 0)
    old = absolute < start
    ring_index = tl.maximum(absolute, 0) % W
    new_index = tl.maximum(absolute - start, 0)
    pool_offset = ((slot * W + ring_index[:, None]) * H + head) * D + d[None, :]
    old_mask = valid[:, None] & old[:, None] & (d[None, :] < D)
    new_mask = valid[:, None] & ~old[:, None] & (new_index[:, None] < T) & (d[None, :] < D)
    old_key = tl.load(KP + pool_offset, old_mask, 0)
    new_key = tl.load(K + batch * KB + new_index[:, None] * KT + head * KH + d[None, :] * KD, new_mask, 0)
    keys = tl.where(old[:, None], old_key, new_key).to(tl.float32)
    query = tl.load(Q + batch * QB + frame * QT + head * QH + d * QD, d < D, 0).to(tl.float32)
    scores = tl.sum(keys * query[None, :], 1) * (D**-0.5)
    scores = tl.where(valid, scores, -float("inf"))
    probabilities = tl.exp(scores - tl.max(scores, 0))
    probabilities = probabilities / tl.sum(probabilities, 0)
    old_value = tl.load(VP + pool_offset, old_mask, 0)
    new_value = tl.load(V + batch * VB + new_index[:, None] * VT + head * VH + d[None, :] * VD, new_mask, 0)
    values = tl.where(old[:, None], old_value, new_value).to(tl.float32)
    output = tl.sum(probabilities[:, None] * values, 0)
    tl.store(OUTPUT + ((batch * T + frame) * H + head) * D + d, output, d < D)


@triton.jit
def _append_ring(
    K,
    V,
    KP,
    VP,
    S,
    P,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    SS: tl.constexpr,
    PS: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    RETAINED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(S + batch * SS)
    start = tl.load(P + batch * PS)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    frame = T - RETAINED + offsets // D
    dim = offsets % D
    mask = offsets < RETAINED * D
    key = tl.load(K + batch * KB + frame * KT + head * KH + dim * KD, mask, 0)
    value = tl.load(V + batch * VB + frame * VT + head * VH + dim * VD, mask, 0)
    target = ((slot * W + (start + frame) % W) * H + head) * D + dim
    tl.store(KP + target, key, mask)
    tl.store(VP + target, value, mask)


def ring_attention_cuda(query, key, value, key_pool, value_pool, slots, positions):
    batch, frames, heads, dim = query.shape
    window = key_pool.shape[1]
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    _attend_ring[(batch, heads, frames)](
        query,
        key,
        value,
        key_pool,
        value_pool,
        slots,
        positions,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        slots.stride(0),
        positions.stride(0),
        frames,
        heads,
        dim,
        window,
        triton.next_power_of_2(window),
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    # All queries must finish reading the old ring before any entry is overwritten.
    retained = min(frames, window)
    _append_ring[(batch, heads, triton.cdiv(retained * dim, 256))](
        key,
        value,
        key_pool,
        value_pool,
        slots,
        positions,
        *key.stride(),
        *value.stride(),
        slots.stride(0),
        positions.stride(0),
        frames,
        heads,
        dim,
        window,
        retained,
        256,
    )
    return output
