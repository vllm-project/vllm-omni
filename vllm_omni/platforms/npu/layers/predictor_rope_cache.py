# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Predictor RoPE and frame-local KV stores with eager rounding semantics."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rope_store(
    q,
    k,
    v,
    cos,
    sin,
    positions,
    out,
    kc,
    vc,
    hq: tl.constexpr,
    hk: tl.constexpr,
    steps: tl.constexpr,
    dim: tl.constexpr,
    cap: tl.constexpr,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qs: tl.constexpr,
    kb: tl.constexpr,
    kh: tl.constexpr,
    ks: tl.constexpr,
    vb: tl.constexpr,
    vh: tl.constexpr,
    vs: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // ((hq + hk) * steps)
    head = row // steps % (hq + hk)
    step = row % steps
    d = tl.arange(0, block)
    other = (d + dim // 2) % dim
    sign = tl.where(d < dim // 2, -1.0, 1.0)
    c = tl.load(cos + (batch * steps + step) * dim + d, d < dim, other=0).to(tl.float32)
    sin_value = tl.load(sin + (batch * steps + step) * dim + d, d < dim, other=0).to(tl.float32)
    if head < hq:
        base = batch * qb + head * qh + step * qs
        x = tl.load(q + base + d, d < dim, other=0)
        y = tl.load(q + base + other, d < dim, other=0).to(tl.float32) * sign
        # Eager multiplies round to the input dtype before the addition.
        a = (x.to(tl.float32) * c).to(x.dtype).to(tl.float32)
        b = (y * sin_value).to(x.dtype).to(tl.float32)
        tl.store(out + ((batch * hq + head) * steps + step) * dim + d, a + b, d < dim)
    else:
        head = head - hq
        base = batch * kb + head * kh + step * ks
        x = tl.load(k + base + d, d < dim, other=0)
        y = tl.load(k + base + other, d < dim, other=0).to(tl.float32) * sign
        # Eager multiplies round to the input dtype before the addition.
        a = (x.to(tl.float32) * c).to(x.dtype).to(tl.float32)
        b = (y * sin_value).to(x.dtype).to(tl.float32)
        pos = tl.load(positions + step)
        target = ((batch * hk + head) * cap + pos) * dim + d
        tl.store(kc + target, a + b, d < dim)
        value = tl.load(v + batch * vb + head * vh + step * vs + d, d < dim, other=0)
        tl.store(vc + target, value, d < dim)


def rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    """Inputs use BNSD; only selected cache positions are modified."""
    batch, heads, steps, dim = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _rope_store[(batch * (heads + k.shape[1]) * steps,)](
        q,
        k,
        v,
        cos,
        sin,
        positions,
        out,
        k_cache,
        v_cache,
        heads,
        k.shape[1],
        steps,
        dim,
        k_cache.shape[2],
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        triton.next_power_of_2(dim),
        enable_fp_fusion=False,
    )
    return out
