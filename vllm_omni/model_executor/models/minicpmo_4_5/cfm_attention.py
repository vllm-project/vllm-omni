# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiled FP32 attention for the short streaming CFM windows on NVIDIA CUDA."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _attention(
    q,
    k,
    v,
    mask,
    kv_rows,
    out,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qt: tl.constexpr,
    qd: tl.constexpr,
    kb: tl.constexpr,
    kh: tl.constexpr,
    kt: tl.constexpr,
    kd: tl.constexpr,
    vb: tl.constexpr,
    vh: tl.constexpr,
    vt: tl.constexpr,
    vd: tl.constexpr,
    mb: tl.constexpr,
    mt: tl.constexpr,
    mk: tl.constexpr,
    heads: tl.constexpr,
    nq: tl.constexpr,
    nk: tl.constexpr,
    dim: tl.constexpr,
    has_mask: tl.constexpr,
    has_kv_rows: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Keys and values of a pooled cache live in row ``kv_rows[b]`` of the pool.
    kv_b = b
    if has_kv_rows:
        kv_b = tl.load(kv_rows + b).to(tl.int64)
    rows = tile * block_m + tl.arange(0, block_m)
    ds = tl.arange(0, block_d)
    cols = tl.arange(0, block_n)
    queries = tl.load(
        q + b * qb + h * qh + rows[:, None] * qt + ds[None, :] * qd, (rows[:, None] < nq) & (ds[None, :] < dim), 0
    )
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0, tl.float32)
    accumulator = tl.full((block_m, block_d), 0, tl.float32)
    for start in range(tl.cdiv(nk, block_n)):
        keys = start * block_n + cols
        kval = tl.load(
            k + kv_b * kb + h * kh + keys[None, :] * kt + ds[:, None] * kd,
            (keys[None, :] < nk) & (ds[:, None] < dim),
            0,
        )
        score = tl.dot(queries, kval, input_precision="tf32x3") * scale
        valid = (rows[:, None] < nq) & (keys[None, :] < nk)
        if has_mask:
            keep = tl.load(mask + b * mb + rows[:, None] * mt + keys[None, :] * mk, valid, 0)
            valid = valid & (keep != 0)
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        # Fully masked rows produce zero, matching PyTorch SDPA.
        # Keep the running maximum at -inf until a valid tile arrives. A
        # permanent zero would underflow a later tile with very negative logits.
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        correction = tl.exp(maximum - safe_max)
        prob = tl.exp(score - safe_max[:, None])
        denominator = denominator * correction + tl.sum(prob, 1)
        values = tl.load(
            v + kv_b * vb + h * vh + keys[:, None] * vt + ds[None, :] * vd,
            (keys[:, None] < nk) & (ds[None, :] < dim),
            0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(prob, values, input_precision="tf32x3")
        maximum = new_max
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        out + ((b * nq + rows[:, None]) * heads + h) * dim + ds[None, :],
        result,
        (rows[:, None] < nq) & (ds[None, :] < dim),
    )


@torch.library.custom_op("vllm_omni::cfm_tiled_attention", mutates_args=())
def cfm_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    kv_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 Q/K/V, bool [B,Q,K] mask, [B,H,Q,D] result (BQHD storage).

    Uses three TF32 products per FP32 dot; no BF16 casts or cache changes.
    Numerical equivalence is tolerance-based, not bitwise. With ``kv_rows``
    (``[B]`` integers) K/V are ``[R,H,K,D]`` pools and query row ``b``
    attends to pool row ``kv_rows[b]``.
    """
    b, heads, nq, dim = q.shape
    nk = k.shape[2]
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("CFM tiled attention requires float32 Q/K/V")
    # Strided FP32 tiles can exceed the 99 KiB per-block limit on L4.
    # Measured on A800 (SM80) across the deployed shapes (2..32 rows, Q<=150,
    # K<=524, D=64): two stages beat three everywhere, e.g. 265 vs 356 us at
    # (16, 8, 150, 524); the extra stage only paid for SRAM capacity, not
    # latency.
    num_stages = 2
    out = torch.empty((b, nq, heads, dim), device=q.device, dtype=q.dtype)
    _attention[(b, heads, triton.cdiv(nq, 32))](
        q,
        k,
        v,
        mask,
        kv_rows,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *(mask.stride() if mask is not None else (0, 0, 0)),
        heads,
        nq,
        nk,
        dim,
        mask is not None,
        kv_rows is not None,
        dim**-0.5,
        32,
        64,
        triton.next_power_of_2(dim),
        num_warps=4,
        num_stages=num_stages,
    )
    return out.transpose(1, 2)


@cfm_attention.register_fake
def _cfm_attention_fake(q, k, v, mask=None, kv_rows=None):
    b, heads, nq, dim = q.shape
    return q.new_empty((b, nq, heads, dim)).transpose(1, 2)
