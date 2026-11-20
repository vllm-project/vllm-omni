# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shape-gated exact BF16 rotary embedding for MiniCPM-o compact MRv2."""

from vllm.triton_utils import tl, triton


@triton.jit
def rotate_bf16(
    q, k, table, positions, qo, ko, qs: tl.constexpr, ks: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr
):
    r = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 32)[None, :]
    pos = tl.load(positions + r)
    c = tl.load(table + pos * dim + d).to(tl.float32)
    s = tl.load(table + pos * dim + dim // 2 + d).to(tl.float32)
    mask = h < heads
    offset = h * dim + d
    for branch in tl.static_range(2):
        src = q if branch == 0 else k
        step = qs if branch == 0 else ks
        first = tl.load(src + r * step + offset, mask, 0).to(tl.float32)
        second = tl.load(src + r * step + offset + dim // 2, mask, 0).to(tl.float32)
        neg_rot = ((-second) * s).to(tl.bfloat16).to(tl.float32)
        pos_rot = (first * s).to(tl.bfloat16).to(tl.float32)
        left = (first * c).to(tl.bfloat16).to(tl.float32) + neg_rot
        right = (second * c).to(tl.bfloat16).to(tl.float32) + pos_rot
        o = (qo if branch == 0 else ko) + r * heads * dim + offset
        tl.store(o, left, mask)
        tl.store(o + dim // 2, right, mask)


def fused(q, k, positions, table):
    rows, width = q.shape
    qo, ko = (q.new_empty((rows, width)) for _ in range(2))
    rotate_bf16[(rows,)](q, k, table, positions, qo, ko, q.stride(0), k.stride(0), 12, 64)
    return qo, ko
