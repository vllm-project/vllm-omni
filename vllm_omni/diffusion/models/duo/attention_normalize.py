# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-KV attention with FP32 partials and a separate normalization launch.

Production uses one partition, preserving the original KV reduction order.
The multi-partition prototype remains disabled pending trajectory validation.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def split_kv_partial(
    q,
    k,
    v,
    pk,
    pv,
    slots,
    lengths,
    output_ptr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    h: tl.constexpr,
    kh: tl.constexpr,
    d: tl.constexpr,
    start: tl.constexpr,
    end: tl.constexpr,
    group: tl.constexpr,
    cap: tl.constexpr,
    scale: tl.constexpr,
    bq: tl.constexpr,
    bk: tl.constexpr,
    bd: tl.constexpr,
    partial_max,
    partial_den,
    splits: tl.constexpr,
    rows: tl.constexpr,
):
    row = start + tl.program_id(0) * bq + tl.arange(0, bq)
    head = tl.program_id(1)
    kv_head = head // (h // kh)
    dim = tl.arange(0, bd)
    q = tl.load(q + row[:, None] * qs0 + head * qs1 + dim[None, :], (row[:, None] < end) & (dim[None, :] < d), 0)
    maximum = tl.full((bq,), -float("inf"), tl.float32)
    denominator = tl.zeros((bq,), tl.float32)
    accumulator = tl.zeros((bq, bd), tl.float32)
    length = tl.load(lengths + group)
    split = tl.program_id(2)
    tiles_per_split = tl.cdiv(tl.cdiv(length, bk), splits)
    for base in range(split * tiles_per_split, tl.minimum((split + 1) * tiles_per_split, tl.cdiv(length, bk))):
        col = base * bk + tl.arange(0, bk)
        slot = tl.load(slots + group * cap + col, col < length, 0)
        past = slot >= 0
        current = -slot - 1
        kpast = tl.load(
            pk + slot[:, None] * kh * d + kv_head * d + dim[None, :],
            (col[:, None] < length) & past[:, None] & (dim[None, :] < d),
            0,
        )
        know = tl.load(
            k + current[:, None] * ks0 + kv_head * ks1 + dim[None, :],
            (col[:, None] < length) & ~past[:, None] & (dim[None, :] < d),
            0,
        )
        key = tl.where(past[:, None], kpast, know)
        vpast = tl.load(
            pv + slot[:, None] * kh * d + kv_head * d + dim[None, :],
            (col[:, None] < length) & past[:, None] & (dim[None, :] < d),
            0,
        )
        vnow = tl.load(
            v + current[:, None] * vs0 + kv_head * vs1 + dim[None, :],
            (col[:, None] < length) & ~past[:, None] & (dim[None, :] < d),
            0,
        )
        value = tl.where(past[:, None], vpast, vnow)
        score = tl.dot(q, tl.trans(key)).to(tl.float32) * scale
        score = tl.where(col[None, :] < length, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        weight = tl.exp(score - new_max[:, None])
        correction = tl.exp(maximum - new_max)
        accumulator = accumulator * correction[:, None] + tl.dot(weight.to(value.dtype), value)
        denominator = denominator * correction + tl.sum(weight, 1)
        maximum = new_max
    tl.store(partial_max + (split * rows + row) * h + head, maximum, row < end)
    tl.store(partial_den + (split * rows + row) * h + head, denominator, row < end)
    result = accumulator
    tl.store(
        output_ptr + ((split * rows + row[:, None]) * h + head) * d + dim[None, :],
        result,
        (row[:, None] < end) & (dim[None, :] < d),
    )


@triton.jit
def merge_kv_partial(
    partial,
    maxima,
    denominators,
    output,
    rows: tl.constexpr,
    h: tl.constexpr,
    d: tl.constexpr,
    start: tl.constexpr,
    end: tl.constexpr,
    splits: tl.constexpr,
    bs: tl.constexpr,
    bd: tl.constexpr,
):
    row = start + tl.program_id(0)
    head = tl.program_id(1)
    ss = tl.arange(0, bs)
    dd = tl.arange(0, bd)
    ix = (ss * rows + row) * h + head
    mm = tl.load(maxima + ix, ss < splits, -float("inf"))
    ll = tl.load(denominators + ix, ss < splits, 0)
    aa = tl.load(partial + ix[:, None] * d + dd[None, :], (ss[:, None] < splits) & (dd[None, :] < d), 0)
    maximum = tl.max(mm, 0)
    correction = tl.exp(mm - maximum)
    denom = tl.sum(ll * correction, 0)
    accum = tl.sum(aa * correction[:, None], 0)
    tl.store(output + (row * h + head) * d + dd, accum / denom, dd < d)
