# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Inference-only Triton kernels for the MOSS single-layer local decoder.

Imported lazily on CUDA. No request state or random generator is owned here;
callers provide frame-local KV buffers and PyTorch-generated uniform samples.
"""

import math

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
    TOKEN_STRIDE: tl.constexpr,
):
    head, batch = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    # Tokens may be a strided column view of the frame's code matrix.
    row = tl.load(Tokens + batch * TOKEN_STRIDE) if LOOKUP else batch
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


def lookup_attention(
    qkv: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: int,
    *,
    tokens: torch.Tensor | None = None,
    embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        tokens.stride(0) if tokens is not None else 1,
        # One warp per (head, row) is ~5x faster than four for these tiny
        # tiles (H200, B=128); only the fp32 reduction order differs.
        num_warps=1,
        enable_fp_fusion=False,
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
def _sample(
    Input,
    Uniform,
    Out,
    Width: tl.constexpr,
    STRIDE_U: tl.constexpr,
    STRIDE_OUT: tl.constexpr,
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
    tl.store(Out + batch * STRIDE_OUT, token)


def fused_sample(
    logits: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    top_k: int = 25,
    temperature: float = 1.7,
    top_p: float = 0.8,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k/top-p inverse-CDF sampling; RNG is supplied by the caller.

    Ties prefer lower token IDs. Distribution matches top-k/top-p for untied
    boundaries; RNG consumption/mapping differs from torch.multinomial.
    """
    batch, width = logits.shape
    if (
        width < 2
        or not 0 < top_k <= min(32, width)
        or not math.isfinite(temperature)
        or temperature <= 0
        or not 0 < top_p <= 1
    ):
        raise ValueError(
            "Fused local sampling requires width >= 2, 1 <= top_k <= 32, temperature > 0 and 0 < top_p <= 1"
        )
    if not logits.is_cuda or not logits.is_contiguous() or uniforms.shape != (batch,):
        raise ValueError("fused_sample requires contiguous CUDA logits and one uniform per row")
    if uniforms.device != logits.device:
        raise ValueError("fused_sample uniforms must be on the logits device")
    if out is None:
        out = torch.empty(batch, dtype=torch.long, device=logits.device)
    elif out.shape != (batch,) or out.dtype != torch.long or out.device != logits.device:
        raise ValueError("fused_sample out must be a (batch,) int64 view")
    _sample[(batch,)](
        logits,
        uniforms,
        out,
        width,
        uniforms.stride(0),
        out.stride(0),
        top_k,
        temperature,
        top_p,
        triton.next_power_of_2(width),
        max(2, triton.next_power_of_2(top_k)),
        num_warps=4,
    )
    return out
