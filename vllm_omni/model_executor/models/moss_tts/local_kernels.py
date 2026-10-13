# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Inference-only Triton kernels for the MOSS single-layer local decoder.

Imported lazily on CUDA. No request state or random generator is owned here;
callers provide frame-local KV buffers and PyTorch-generated uniform samples.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _attention(
    QKV,
    Tokens,
    Embedding,
    K,
    V,
    Out,
    Residual,
    H: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CAPACITY: tl.constexpr,
    POSITION: tl.constexpr,
    LOOKUP: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Fused gather, KV insertion and attention for GROUP heads per program."""
    batch, group = tl.program_id(0), tl.program_id(1)
    h = group * GROUP + tl.arange(0, GROUP)
    d = tl.arange(0, BD)
    dmask = (h[:, None] < HEADS) & (d[None, :] < D)
    row = tl.load(Tokens + batch) if LOOKUP else batch
    offset = row * (3 * H) + h[:, None] * D + d[None, :]
    q = tl.load(QKV + offset, dmask, 0).to(tl.float32)
    k = tl.load(QKV + offset + H, dmask, 0).to(tl.float32)
    v = tl.load(QKV + offset + 2 * H, dmask, 0).to(tl.float32)
    base = (batch * HEADS + h[:, None]) * CAPACITY * D + d[None, :]
    tl.store(K + base + POSITION * D, k, dmask)
    tl.store(V + base + POSITION * D, v, dmask)
    if POSITION == 0:
        result = v
    else:
        s = tl.arange(0, BS)
        scores = tl.full((GROUP, BS), -float("inf"), tl.float32)
        for p in tl.static_range(POSITION):
            kp = tl.load(K + base + p * D, dmask, 0).to(tl.float32)
            sp = tl.sum(kp * q, axis=1) * (D**-0.5)
            scores = tl.where(s[None, :] == p, sp[:, None], scores)
        sp = tl.sum(k * q, axis=1) * (D**-0.5)
        scores = tl.where(s[None, :] == POSITION, sp[:, None], scores)
        probs = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        probs = probs / tl.sum(probs, axis=1)[:, None]
        result = v * tl.sum(tl.where(s[None, :] == POSITION, probs, 0.0), axis=1)[:, None]
        for p in tl.static_range(POSITION):
            vp = tl.load(V + base + p * D, dmask, 0).to(tl.float32)
            result += vp * tl.sum(tl.where(s[None, :] == p, probs, 0.0), axis=1)[:, None]
    tl.store(Out + batch * H + h[:, None] * D + d[None, :], result, dmask)
    if LOOKUP:
        residual = tl.load(Embedding + row * H + h[:, None] * D + d[None, :], dmask, 0)
        tl.store(Residual + batch * H + h[:, None] * D + d[None, :], residual, dmask)


def lookup_attention(
    qkv: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: int,
    *,
    tokens: torch.Tensor | None = None,
    embedding: torch.Tensor | None = None,
    group: int = 8,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gather, KV insertion, attention and optional residual gather.

    Each program handles ``group`` heads as (group, head_dim) tiles.
    """
    batch, heads, capacity, dim = key.shape
    hidden = heads * dim
    out = torch.empty((batch, hidden), dtype=key.dtype, device=key.device)
    residual = torch.empty_like(out) if tokens is not None else out
    _attention[(batch, triton.cdiv(heads, group))](
        qkv,
        tokens,
        embedding,
        key,
        value,
        out,
        residual,
        hidden,
        heads,
        dim,
        capacity,
        position,
        tokens is not None,
        triton.next_power_of_2(dim),
        triton.next_power_of_2(position + 1),
        group,
        num_warps=num_warps,
    )
    return out, residual


@triton.jit
def _pack_logits(values, ids):
    # Sort by logit, then by decreasing (UINT_MAX - token_id). This gives
    # deterministic smaller-ID-first tie breaking, including BF16 ties.
    bits = values.to(tl.float32).to(tl.uint32, bitcast=True)
    bits = tl.where(values == 0, 0, bits).to(tl.uint32)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint32)).to(tl.uint64)


@triton.jit
def _unpack_logits(packed):
    ordered = (packed >> 32).to(tl.uint32)
    bits = ordered ^ tl.where((ordered & 0x80000000) != 0, 0x80000000, 0xFFFFFFFF).to(tl.uint32)
    values = bits.to(tl.float32, bitcast=True)
    ids = (0xFFFFFFFF - packed.to(tl.uint32)).to(tl.int32)
    return values, ids


@triton.jit
def _skinny_linear(
    X,
    W,
    Bias,
    Residual,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    SR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mi = tl.arange(0, BM)
    ni = tl.program_id(0) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN, BK), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ki
        a = tl.load(X + mi[:, None] * SX + kk[None, :], (mi[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + ni[:, None] * SW + kk[None, :], (ni[:, None] < N) & (kk[None, :] < K), 0)
        acc += a[:, None, :].to(tl.float32) * w[None, :, :].to(tl.float32)
    result = tl.sum(acc, axis=2)
    if HAS_BIAS:
        result += tl.load(Bias + ni, ni < N, 0)[None, :].to(tl.float32)
    result = result.to(X.dtype.element_ty).to(tl.float32)
    if HAS_RESIDUAL:
        result += tl.load(Residual + mi[:, None] * SR + ni[None, :], (mi[:, None] < M) & (ni[None, :] < N), 0).to(
            tl.float32
        )
    tl.store(Out + mi[:, None] * N + ni[None, :], result, (mi[:, None] < M) & (ni[None, :] < N))


def fused_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tiny-batch projection with the original BF16 linear/residual boundary.

    Only the two Hopper shapes measured faster than cuBLAS are dispatched here.
    Larger batches and the MLP up projection keep their existing linear ops.
    """
    batch, k = x.shape
    n = weight.shape[0]
    if batch in (1, 2) and n == k == 2560:
        bk = 512
    elif batch == 1 and (n, k) == (2560, 9728):
        bk = 1024
    else:
        raise ValueError("Unsupported MOSS tiny-batch projection shape")
    if x.dtype != torch.bfloat16 or x.stride(1) != 1 or weight.stride(1) != 1:
        raise ValueError("MOSS tiny-batch projection requires BF16 and contiguous inner dimensions")
    if residual is not None and residual.stride(1) != 1:
        raise ValueError("MOSS tiny-batch projection requires a contiguous residual inner dimension")
    out = torch.empty((batch, n), device=x.device, dtype=x.dtype)
    _skinny_linear[(triton.cdiv(n, 4),)](
        x,
        weight,
        bias,
        residual,
        out,
        batch,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        residual.stride(0) if residual is not None else n,
        bias is not None,
        residual is not None,
        triton.next_power_of_2(batch),
        4,
        bk,
        num_warps=4,
    )
    return out


@triton.jit
def _sample(
    Input,
    Uniform,
    Out,
    Width: tl.constexpr,
    STRIDE_U: tl.constexpr,
    TOP_K: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    TOP_P: tl.constexpr,
    BLOCK: tl.constexpr,
    KEEP: tl.constexpr,
):
    batch = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    values = tl.load(Input + batch * Width + i, i < Width, -float("inf")).to(tl.float32)
    if Input.dtype.element_ty == tl.bfloat16 and BLOCK <= 65536:
        # BF16 has only 16 meaningful bits; append the token ID tie-break in
        # a uint32 key. This retains exactly the same ordering as uint64
        # float32 keys without widening every top-k compare/shuffle.
        bits = values.to(tl.uint32, bitcast=True) >> 16
        bits = tl.where(values == 0, 0, bits).to(tl.uint32)
        ordered = bits ^ tl.where((bits & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint32)
        packed = (ordered << 16) | (0xFFFF - i.to(tl.uint32))
        selected = tl.topk(packed, KEEP)
        ordered = selected >> 16
        bits = ordered ^ tl.where((ordered & 0x8000) != 0, 0x8000, 0xFFFF).to(tl.uint32)
        logits = (bits << 16).to(tl.float32, bitcast=True)
        ids = (0xFFFF - (selected & 0xFFFF)).to(tl.int32)
    else:
        packed = _pack_logits(values, i)
        selected = tl.topk(packed, KEEP)
        logits, ids = _unpack_logits(selected)
    ranks = tl.arange(0, KEEP)
    logits = tl.where(ranks < TOP_K, logits / TEMPERATURE, -float("inf"))
    probs = tl.exp(logits - tl.max(logits, 0))
    probs = probs / tl.sum(probs, 0)
    cdf = tl.cumsum(probs)
    # Keep the first token crossing top_p, matching shifted nucleus masking.
    probs = tl.where((cdf - probs <= TOP_P) | (ranks == 0), probs, 0.0)
    cdf = tl.cumsum(probs)
    u = tl.load(Uniform + batch * STRIDE_U)
    target = u * tl.sum(probs, 0)
    index = tl.min(tl.where(cdf > target, ranks, KEEP - 1), 0)
    token = tl.sum(tl.where(ranks == index, ids, 0), 0)
    tl.store(Out + batch, token)


def fused_sample(
    logits: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    top_k: int = 25,
    temperature: float = 1.7,
    top_p: float = 0.8,
) -> torch.Tensor:
    """Top-k/top-p inverse-CDF sampling; RNG is supplied by the caller.

    Ties prefer lower token IDs. Distribution matches top-k/top-p for untied
    boundaries; RNG consumption/mapping differs from torch.multinomial.
    """
    batch, width = logits.shape
    if width < 2 or not 0 < top_k <= min(32, width) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError(
            "Fused local sampling requires width >= 2, 1 <= top_k <= 32, temperature > 0 and 0 < top_p <= 1"
        )
    out = torch.empty(batch, dtype=torch.long, device=logits.device)
    _sample[(batch,)](
        logits,
        uniforms,
        out,
        width,
        uniforms.stride(0),
        top_k,
        temperature,
        top_p,
        triton.next_power_of_2(width),
        max(2, triton.next_power_of_2(top_k)),
        num_warps=4,
    )
    return out
