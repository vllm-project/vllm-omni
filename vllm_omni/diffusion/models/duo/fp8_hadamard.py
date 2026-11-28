# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP32 butterfly preconditioning around official FA3 FP8 attention."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def butterfly(
    x,
    signs,
    center,
    count,
    output,
    rows: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    inverse: tl.constexpr,
    centered: tl.constexpr,
    bounded: tl.constexpr,
    block_rows: tl.constexpr,
):
    r = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    head = tl.program_id(1)
    d = tl.arange(0, dim)
    valid = r < rows
    if bounded:
        valid = valid & (r < tl.load(count))
    value = tl.load(x + r[:, None] * stride0 + head * stride1 + d[None, :], valid[:, None], 0).to(tl.float32)
    sign = tl.load(signs + d)
    if not inverse:
        if centered:
            mean = tl.load(center + head * dim + d)
            value = tl.where(valid[:, None], value - mean[None, :], 0.0)
        value *= sign[None, :]
    for step in tl.static_range(7):
        bit = 1 << step
        indices = tl.broadcast_to((d ^ bit)[None, :], (block_rows, dim))
        partner = tl.gather(value, indices, axis=1)
        value = tl.where((d[None, :] & bit) == 0, value + partner, partner - value)
    value *= dim**-0.5
    if inverse:
        value *= sign[None, :]
        if centered:
            value += tl.load(center + head * dim + d)[None, :]
    tl.store(output + r[:, None] * out_stride0 + head * out_stride1 + d[None, :], value, (r < rows)[:, None])


def rotate(x, signs, *, center=None, count=None, inverse=False, out=None):
    rows, heads, dim = x.shape
    if out is None:
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
    else:
        output = out
    butterfly[(triton.cdiv(rows, 4), heads)](
        x,
        signs,
        center,
        count,
        output,
        rows,
        heads,
        dim,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        inverse,
        center is not None,
        count is not None,
        4,
        num_warps=4,
    )
    return output
