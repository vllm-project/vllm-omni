# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Short causal attention for the scratch KV cache inside a code predictor."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _short_kv_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    positions_ptr,
    out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_n: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    heads: tl.constexpr,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    head_dim: tl.constexpr,
    queries_per_kv: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    qi = tl.program_id(2)
    kvh = h // queries_per_kv
    d = tl.arange(0, block_d)
    n = tl.arange(0, block_n)
    last = tl.load(positions_ptr + qi)
    q = tl.load(q_ptr + b * q_stride_b + h * q_stride_h + qi * q_stride_n + d, d < head_dim, other=0).to(tl.float32)
    k = tl.load(
        k_ptr + b * k_stride_b + kvh * k_stride_h + n[:, None] * k_stride_n + d[None, :],
        (n[:, None] < num_keys) & (n[:, None] <= last) & (d[None, :] < head_dim),
        other=0,
    ).to(tl.float32)
    score = tl.sum(k * q[None, :], 1) * scale
    score = tl.where((n < num_keys) & (n <= last), score, -float("inf"))
    p = tl.exp(score - tl.max(score, 0))
    p = p / tl.sum(p, 0)
    v = tl.load(
        v_ptr + b * v_stride_b + kvh * v_stride_h + n[:, None] * v_stride_n + d[None, :],
        (n[:, None] < num_keys) & (n[:, None] <= last) & (d[None, :] < head_dim),
        other=0,
    ).to(tl.float32)
    out = tl.sum(p[:, None] * v, 0)
    tl.store(out_ptr + ((b * num_queries + qi) * heads + h) * head_dim + d, out, d < head_dim)


def short_kv_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor, scale: float
) -> torch.Tensor:
    """One/two-query GQA over a frame-local scratch cache with contiguous head dimensions.

    The returned [B, H, Q, D] tensor has a contiguous [B, Q, H, D] view,
    avoiding a separate transpose-copy before the output projection.
    """
    b, h, nq, d = q.shape
    nk = k.shape[2]
    out = torch.empty((b, nq, h, d), device=q.device, dtype=q.dtype)
    _short_kv_attention[(b, h, nq)](
        q,
        k,
        v,
        positions,
        out,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        h,
        nq,
        nk,
        d,
        h // k.shape[1],
        scale,
        triton.next_power_of_2(nk),
        triton.next_power_of_2(d),
        num_warps=4,
    )
    return out.transpose(1, 2)
