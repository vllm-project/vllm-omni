# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused request-isolated packed pre-lookahead convolution.

One program owns a request/output tile. It recomputes the small conv1 halo
rather than synchronizing programs, and reads/writes distinct cache buffers.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prelookahead_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    meta_ptr,
    out_ptr,
    IN: tl.constexpr,
    HIDDEN: tl.constexpr,
    LOOK: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    source = tl.load(meta_ptr + row * 5)
    destination = tl.load(meta_ptr + row * 5 + 1)
    length = tl.load(meta_ptr + row * 5 + 2)
    if tile * BT >= length:
        return
    old = tl.load(meta_ptr + row * 5 + 3).to(tl.pointer_type(x_ptr.dtype.element_ty))
    new = tl.load(meta_ptr + row * 5 + 4).to(tl.pointer_type(x_ptr.dtype.element_ty))
    m = tile * BT - 2 + tl.arange(0, BM)
    t = tile * BT + tl.arange(0, BT)
    o = tl.arange(0, BO)
    accumulator = tl.full((BT, BO), 0, tl.float32)
    for h_start in range(0, HIDDEN, BH):
        h = h_start + tl.arange(0, BH)
        first = tl.full((BM, BH), 0, tl.float32)
        for tap in tl.static_range(LOOK + 1):
            for c_start in tl.static_range(0, IN, BK):
                c = c_start + tl.arange(0, BK)
                values = tl.load(
                    x_ptr + (source + m[:, None] + tap) * IN + c[None, :],
                    mask=(m[:, None] >= 0) & (m[:, None] < length) & (m[:, None] < tile * BT + BT) & (c[None, :] < IN),
                    other=0,
                )
                weights = tl.load(
                    w1_ptr + h[None, :] * IN * (LOOK + 1) + c[:, None] * (LOOK + 1) + tap,
                    mask=(c[:, None] < IN) & (h[None, :] < HIDDEN),
                    other=0,
                )
                first += tl.dot(values, weights, input_precision="ieee")
        # Match the two rounding boundaries of Conv1d then LeakyReLU.
        first = (first + tl.load(b1_ptr + h, mask=h < HIDDEN, other=0)[None, :]).to(x_ptr.dtype.element_ty)
        first = tl.where(first >= 0, first, first.to(tl.float32) * 0.01).to(x_ptr.dtype.element_ty)
        history = tl.load(
            old + h[None, :] * 2 + m[:, None] + 2,
            mask=(m[:, None] >= -2) & (m[:, None] < 0) & (h[None, :] < HIDDEN),
            other=0,
        )
        first = tl.where(m[:, None] < 0, history, first)
        for tap in tl.static_range(3):
            indices = tl.broadcast_to((tl.arange(0, BT) + tap)[:, None], (BT, BH))
            activations = tl.gather(first, indices, axis=0)
            weights = tl.load(
                w2_ptr + o[None, :] * HIDDEN * 3 + h[:, None] * 3 + tap,
                mask=(h[:, None] < HIDDEN) & (o[None, :] < IN),
                other=0,
            )
            accumulator += tl.dot(activations, weights, input_precision="ieee")
        # Each cache cell has one writer, including a one-token update where
        # the first retained activation still comes from the old cache.
        owns = ((m >= tile * BT) | ((tile == 0) & (m < 0))) & (m < tile * BT + BT)
        tl.store(
            new + h[None, :] * 2 + m[:, None] - length + 2,
            first,
            mask=owns[:, None] & (m[:, None] >= length - 2) & (m[:, None] < length) & (h[None, :] < HIDDEN),
        )
    value = (accumulator + tl.load(b2_ptr + o, mask=o < IN, other=0)[None, :]).to(x_ptr.dtype.element_ty)
    residual = tl.load(
        x_ptr + (source + t[:, None]) * IN + o[None, :],
        mask=(t[:, None] < length) & (o[None, :] < IN),
        other=0,
    )
    tl.store(
        out_ptr + (destination + t[:, None]) * IN + o[None, :],
        value.to(tl.float32) + residual.to(tl.float32),
        mask=(t[:, None] < length) & (o[None, :] < IN),
    )


def packed_prelookahead(layer, inputs, old_caches, new_caches):
    """Inputs include each row's real right context; no host tensor-value reads."""
    lengths = [value.shape[1] - layer.pre_lookahead_len for value in inputs]
    packed = torch.cat([value[0] for value in inputs], dim=0).contiguous()
    output = packed.new_empty(sum(lengths), layer.in_channels)
    records = []
    source = destination = 0
    for value, length, old, new in zip(inputs, lengths, old_caches, new_caches):
        records.append([source, destination, length, old.data_ptr(), new.data_ptr()])
        source += value.shape[1]
        destination += length
    metadata = torch.tensor(records, dtype=torch.int64, pin_memory=True).to(packed.device, non_blocking=True)
    _prelookahead_kernel[(triton.cdiv(max(lengths), 16), len(inputs))](
        packed,
        layer.conv1.weight,
        layer.conv1.bias,
        layer.conv2.weight,
        layer.conv2.bias,
        metadata,
        output,
        IN=layer.in_channels,
        HIDDEN=layer.channels,
        LOOK=layer.pre_lookahead_len,
        BT=16,
        BM=32,
        BH=64,
        BK=32,
        BO=triton.next_power_of_2(layer.in_channels),
        num_warps=4,
    )
    return [value.unsqueeze(0) for value in torch.split(output, lengths, dim=0)]
