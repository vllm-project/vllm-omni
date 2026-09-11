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
):
    head, batch = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    row = tl.load(Tokens + batch) if LOOKUP else batch
    offset = row * (3 * H) + head * D + d
    q = tl.load(QKV + offset, d < D, 0).to(tl.float32)
    k = tl.load(QKV + offset + H, d < D, 0).to(tl.float32)
    v = tl.load(QKV + offset + 2 * H, d < D, 0).to(tl.float32)
    base = (batch * HEADS + head) * CAPACITY * D
    tl.store(K + base + POSITION * D + d, k, d < D)
    tl.store(V + base + POSITION * D + d, v, d < D)
    if POSITION == 0:
        result = v
    else:
        s = tl.arange(0, BS)
        offsets = base + s[:, None] * D + d[None, :]
        mask = (s[:, None] < POSITION) & (d[None, :] < D)
        keys = tl.load(K + offsets, mask, 0).to(tl.float32)
        values = tl.load(V + offsets, mask, 0).to(tl.float32)
        # Never read the just-written slot or unwritten future slots.
        keys = tl.where(s[:, None] == POSITION, k[None, :], keys)
        values = tl.where(s[:, None] == POSITION, v[None, :], values)
        scores = tl.sum(keys * q[None, :], axis=1) * (D**-0.5)
        scores = tl.where(s <= POSITION, scores, -float("inf"))
        probs = tl.exp(scores - tl.max(scores, axis=0))
        probs = probs / tl.sum(probs, axis=0)
        result = tl.sum(values * probs[:, None], axis=0)
    tl.store(Out + batch * H + head * D + d, result, d < D)
    if LOOKUP:
        residual = tl.load(Embedding + row * H + head * D + d, d < D, 0)
        tl.store(Residual + batch * H + head * D + d, residual, d < D)


def lookup_attention(qkv, key, value, position, *, tokens=None, embedding=None):
    """Fused gather, KV insertion, attention and optional residual gather."""
    batch, heads, capacity, dim = key.shape
    hidden = heads * dim
    out = torch.empty((batch, hidden), dtype=key.dtype, device=key.device)
    residual = torch.empty_like(out) if tokens is not None else out
    _attention[(heads, batch)](
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
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, residual


@triton.jit
def _pack_logits(values, ids):
    # Sort by logit, then by decreasing (UINT_MAX - token_id). This gives
    # deterministic smaller-ID-first tie breaking, including BF16 ties.
    bits = values.to(tl.float32).to(tl.uint32, bitcast=True)
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
def _linear(
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
    EPILOGUE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ki
        a = tl.load(X + mi[:, None] * SX + kk[None, :], (mi[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + ni[None, :] * SW + kk[:, None], (ni[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(Bias + ni, ni < N, 0)[None, :].to(tl.float32)
    # Preserve the original Linear -> BF16 -> activation/residual boundary.
    acc = acc.to(X.dtype.element_ty).to(tl.float32)
    if EPILOGUE == "silu":
        acc = acc * tl.sigmoid(acc)
    elif EPILOGUE == "residual":
        acc += tl.load(Residual + mi[:, None] * SR + ni[None, :], (mi[:, None] < M) & (ni[None, :] < N), 0).to(
            tl.float32
        )
    tl.store(Out + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N))


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
    EPILOGUE: tl.constexpr,
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
    if EPILOGUE == "silu":
        result = result * tl.sigmoid(result)
    elif EPILOGUE == "residual":
        result += tl.load(Residual + mi[:, None] * SR + ni[None, :], (mi[:, None] < M) & (ni[None, :] < N), 0).to(
            tl.float32
        )
    tl.store(Out + mi[:, None] * N + ni[None, :], result, (mi[:, None] < M) & (ni[None, :] < N))


def fused_linear(x, weight, bias=None, residual=None, *, activation=False, config=None):
    """BF16/FP16 GEMM with rounded bias/SiLU/residual epilogue."""
    batch, k = x.shape
    n = weight.shape[0]
    if x.dtype not in (torch.bfloat16, torch.float16) or x.stride(1) != 1 or weight.stride(1) != 1:
        raise ValueError("Fused local GEMM requires BF16/FP16 inputs with contiguous inner dimensions")
    if residual is not None and residual.stride(1) != 1:
        raise ValueError("Fused local GEMM requires a contiguous residual inner dimension")
    bn = 64
    bm = 16 if batch <= 16 else 32
    bk, warps = 64, 4
    if config is None:
        if batch <= 2 and n == k == 2560:
            config = (0, 4, 512, 4)
        elif batch == 1 and (n, k) == (2560, 9728):
            config = (0, 4, 1024, 4)
        elif n == 9728 and batch >= 8:
            config = (16, 64, 128, 4)
        else:
            config = (16, 32, 128, 4)
    if config is not None:
        bm, bn, bk, warps = config
    out = torch.empty((batch, n), device=x.device, dtype=x.dtype)
    if bm == 0:
        _skinny_linear[(triton.cdiv(n, bn),)](
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
            "silu" if activation else "residual" if residual is not None else "none",
            triton.next_power_of_2(batch),
            bn,
            bk,
            num_warps=warps,
        )
        return out
    _linear[(triton.cdiv(batch, bm), triton.cdiv(n, bn))](
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
        "silu" if activation else "residual" if residual is not None else "none",
        bm,
        bn,
        bk,
        num_warps=warps,
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


def fused_sample(logits, uniforms, *, top_k=25, temperature=1.7, top_p=0.8):
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
