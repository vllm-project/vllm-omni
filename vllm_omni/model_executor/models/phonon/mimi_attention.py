# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Reference and Triton implementations of Mimi's streaming ring attention."""

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton


def ring_attention_chunk(query, key, value, key_pool, value_pool, slots, positions, window=None):
    """Attend a complete chunk against the physical ring and current keys."""
    capacity = key_pool.shape[1]
    window = capacity if window is None else window
    if not 0 < window <= capacity:
        raise ValueError("Attention window must fit inside the ring")
    if query.is_cuda:
        return ring_attention_cuda(query, key, value, key_pool, value_pool, slots, positions, window)
    frames = query.shape[1]
    offsets = torch.arange(frames, device=positions.device)
    physical = torch.arange(capacity, device=positions.device)
    old_positions = positions[:, None] - 1 - (positions[:, None] - 1 - physical).remainder(capacity)
    current_positions = positions[:, None] + offsets
    delta = current_positions[:, :, None] - torch.cat((old_positions, current_positions), -1)[:, None, :]
    key_positions = torch.cat((old_positions, current_positions), -1)
    mask = (key_positions[:, None, :] >= 0) & (delta >= 0) & (delta < window)
    valid_history = (old_positions >= 0)[:, :, None, None]
    keys = torch.cat((torch.where(valid_history, key_pool[slots], 0), key), 1)
    values = torch.cat((torch.where(valid_history, value_pool[slots], 0), value), 1)
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        keys.transpose(1, 2),
        values.transpose(1, 2),
        attn_mask=mask[:, None],
    ).transpose(1, 2)
    retained = min(frames, capacity)
    indices = current_positions[:, -retained:].remainder(capacity)
    key_pool[slots[:, None], indices] = key[:, -retained:]
    value_pool[slots[:, None], indices] = value[:, -retained:]
    return output


@triton.jit
def _attend_ring(
    Q,
    KP,
    VP,
    S,
    P,
    OUTPUT,
    QB: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    SS: tl.constexpr,
    PS: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    frame = tl.program_id(2)
    slot = tl.load(S + batch * SS)
    position = tl.load(P + batch * PS) + frame
    j = tl.arange(0, BW)
    d = tl.arange(0, BD)
    absolute = position - j
    valid = (j < WINDOW) & (absolute >= 0)
    ring_index = tl.maximum(absolute, 0) % RING
    offset = ((slot * RING + ring_index[:, None]) * H + head) * D + d[None, :]
    mask = valid[:, None] & (d[None, :] < D)
    keys = tl.load(KP + offset, mask, 0).to(tl.float32)
    query = tl.load(Q + batch * QB + frame * QT + head * QH + d * QD, d < D, 0).to(tl.float32)
    scores = tl.sum(keys * query[None, :], 1) * (D**-0.5)
    scores = tl.where(valid, scores, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, 0))
    weights = weights / tl.sum(weights, 0)
    values = tl.load(VP + offset, mask, 0).to(tl.float32)
    output = tl.sum(weights[:, None] * values, 0)
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


def ring_attention_cuda(query, key, value, key_pool, value_pool, slots, positions, window):
    batch, frames, heads, dim = query.shape
    capacity = key_pool.shape[1]
    max_chunk = capacity - window + 1
    if not 0 < window <= capacity:
        raise ValueError("Attention window must fit inside the ring")
    if frames > max_chunk:
        outputs = []
        for start in range(0, frames, max_chunk):
            stop = start + max_chunk
            outputs.append(
                ring_attention_cuda(
                    query[:, start:stop],
                    key[:, start:stop],
                    value[:, start:stop],
                    key_pool,
                    value_pool,
                    slots,
                    positions + start,
                    window,
                )
            )
        return torch.cat(outputs, 1)
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    # Headroom preserves every old key needed by the earliest query in this chunk.
    _append_ring[(batch, heads, triton.cdiv(frames * dim, 256))](
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
        capacity,
        frames,
        256,
    )
    _attend_ring[(batch, heads, frames)](
        query,
        key_pool,
        value_pool,
        slots,
        positions,
        output,
        *query.stride(),
        slots.stride(0),
        positions.stride(0),
        frames,
        heads,
        dim,
        capacity,
        window,
        triton.next_power_of_2(window),
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output
