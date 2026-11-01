# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Masked streaming attention preserving ring-cache position semantics."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _attention(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
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
    ms0: tl.constexpr,
    ms2: tl.constexpr,
    ms3: tl.constexpr,
    num_heads: tl.constexpr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    bh = tl.program_id(1)
    batch, head = bh // num_heads, bh % num_heads
    dims = tl.arange(0, head_dim)
    q = tl.load(
        q_ptr + batch * qs0 + head * qs1 + rows[:, None] * qs2 + dims[None, :] * qs3,
        mask=rows[:, None] < q_len,
        other=0,
    )
    acc = tl.full((block_m, head_dim), 0.0, tl.float32)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0.0, tl.float32)
    for first in range(tl.cdiv(kv_len, block_n)):
        cols = first * block_n + tl.arange(0, block_n)
        k = tl.load(
            k_ptr + batch * ks0 + head * ks1 + cols[None, :] * ks2 + dims[:, None] * ks3,
            mask=cols[None, :] < kv_len,
            other=0,
        )
        allowed = tl.load(
            mask_ptr + batch * ms0 + rows[:, None] * ms2 + cols[None, :] * ms3,
            mask=(rows[:, None] < q_len) & (cols[None, :] < kv_len),
            other=0,
        )
        score = tl.dot(q, k).to(tl.float32) * (head_dim**-0.5)
        score = tl.where(allowed & (cols[None, :] < kv_len), score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        rescale = tl.exp(maximum - safe_max)
        p = tl.exp(score - safe_max[:, None])
        denominator = denominator * rescale + tl.sum(p, axis=1)
        v = tl.load(
            v_ptr + batch * vs0 + head * vs1 + cols[:, None] * vs2 + dims[None, :] * vs3,
            mask=cols[:, None] < kv_len,
            other=0,
        )
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        maximum = next_max
    out = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        out_ptr + ((batch * num_heads + head) * q_len + rows[:, None]) * head_dim + dims[None, :],
        out,
        mask=rows[:, None] < q_len,
    )


@torch.library.custom_op("moss_streaming::masked_attention", mutates_args=())
def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, h, t, d = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    # Small query tiles expose parallelism for short streaming chunks. Larger
    # tiles reuse K/V for steady chunks. Keep 64-key softmax blocks to retain
    # the same accumulation granularity across both paths.
    bm = 16 if t <= 32 else 64
    num_warps = 8 if k.shape[2] <= 256 else 4
    _attention[(triton.cdiv(t, bm), b * h)](
        q,
        k,
        v,
        mask,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        mask.stride(0),
        mask.stride(2),
        mask.stride(3),
        h,
        t,
        k.shape[2],
        d,
        bm,
        64,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


@masked_attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)
