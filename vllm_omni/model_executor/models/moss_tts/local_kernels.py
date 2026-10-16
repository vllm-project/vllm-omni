# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Inference-only Triton kernels for the MOSS single-layer local decoder.

Imported lazily on CUDA. No request state or random generator is owned here;
callers provide frame-local KV buffers and PyTorch-generated uniform samples.
"""

import numpy as np
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_gdc_launch = tl.extra.cuda.gdc_launch_dependents
_gdc_wait = tl.extra.cuda.gdc_wait
libdevice = tl.extra.cuda.libdevice

# Hopper+ supports programmatic dependent launch. The kernel waits for its
# predecessor before any global access and lets its successor launch at once; results are unchanged.
_USE_PDL = current_platform.is_cuda() and current_platform.has_device_capability(90)


@triton.jit
def _attend(
    QKV,
    row,
    batch,
    group,
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
    LATE: tl.constexpr = False,
):
    """KV insertion and attention for GROUP heads of one row (shared body).

    ``LATE`` only reorders memory traffic (same arithmetic): the new K/V are
    stored after the cached K/V are read, so those loads need not wait for
    the token-dependent QKV gather; the frame-local cache is kept in L2 with
    ``evict_last`` (demoted on its final read) and the never-read K/V of the
    last position is not stored.
    """
    LAST: tl.constexpr = POSITION + 1 == CAPACITY
    CACHE_LOAD: tl.constexpr = ("evict_first" if LAST else "evict_last") if LATE else ""
    h = group * GROUP + tl.arange(0, GROUP)
    d = tl.arange(0, BD)
    dmask = (h[:, None] < HEADS) & (d[None, :] < D)
    offset = row * (3 * H) + h[:, None] * D + d[None, :]
    q = tl.load(QKV + offset, dmask, 0).to(tl.float32)
    k = tl.load(QKV + offset + H, dmask, 0).to(tl.float32)
    v = tl.load(QKV + offset + 2 * H, dmask, 0).to(tl.float32)
    base = (batch * HEADS + h[:, None]) * CAPACITY * D + d[None, :]
    if not LATE:
        tl.store(K + base + POSITION * D, k, dmask)
        tl.store(V + base + POSITION * D, v, dmask)
    if POSITION == 0:
        result = v
    else:
        s = tl.arange(0, BS)
        scores = tl.full((GROUP, BS), -float("inf"), tl.float32)
        for p in tl.static_range(POSITION):
            kp = tl.load(K + base + p * D, dmask, 0, eviction_policy=CACHE_LOAD).to(tl.float32)
            sp = tl.sum(kp * q, axis=1) * (D**-0.5)
            scores = tl.where(s[None, :] == p, sp[:, None], scores)
        sp = tl.sum(k * q, axis=1) * (D**-0.5)
        scores = tl.where(s[None, :] == POSITION, sp[:, None], scores)
        probs = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        probs = probs / tl.sum(probs, axis=1)[:, None]
        result = v * tl.sum(tl.where(s[None, :] == POSITION, probs, 0.0), axis=1)[:, None]
        for p in tl.static_range(POSITION):
            vp = tl.load(V + base + p * D, dmask, 0, eviction_policy=CACHE_LOAD).to(tl.float32)
            result += vp * tl.sum(tl.where(s[None, :] == p, probs, 0.0), axis=1)[:, None]
    if LATE and not LAST:
        tl.store(K + base + POSITION * D, k, dmask, eviction_policy="evict_last")
        tl.store(V + base + POSITION * D, v, dmask, eviction_policy="evict_last")
    tl.store(Out + batch * H + h[:, None] * D + d[None, :], result, dmask)
    if LOOKUP:
        residual = tl.load(Embedding + row * H + h[:, None] * D + d[None, :], dmask, 0)
        tl.store(Residual + batch * H + h[:, None] * D + d[None, :], residual, dmask)


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
    LATE: tl.constexpr = False,
):
    """Fused gather, KV insertion and attention for GROUP heads per program."""
    batch, group = tl.program_id(0), tl.program_id(1)
    row = tl.load(Tokens + batch) if LOOKUP else batch
    _attend(
        QKV,
        row,
        batch,
        group,
        Embedding,
        K,
        V,
        Out,
        Residual,
        H,
        HEADS,
        D,
        CAPACITY,
        POSITION,
        LOOKUP,
        BD,
        BS,
        GROUP,
        LATE,
    )


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
    late_store: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gather, KV insertion, attention and optional residual gather.

    Each program handles ``group`` heads as (group, head_dim) tiles.
    ``late_store`` selects the L2-resident cache variant (same results).
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
        late_store,
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
def _sample_row(
    Input,
    Uniform,
    batch,
    Width: tl.constexpr,
    STRIDE_U: tl.constexpr,
    TOP_K: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    TOP_P: tl.constexpr,
    BLOCK: tl.constexpr,
    KEEP: tl.constexpr,
):
    """Top-k/top-p inverse-CDF sample of one logits row (shared body)."""
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
    return tl.sum(tl.where(ranks == index, ids, 0), 0)


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
    token = _sample_row(Input, Uniform, batch, Width, STRIDE_U, TOP_K, TEMPERATURE, TOP_P, BLOCK, KEEP)
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


@triton.jit
def _sample_attend(
    Logits,
    Uniform,
    Codes,
    QKV,
    Embedding,
    K,
    V,
    Out,
    Residual,
    Width: tl.constexpr,
    STRIDE_U: tl.constexpr,
    STRIDE_C: tl.constexpr,
    TOP_K: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    TOP_P: tl.constexpr,
    BLOCK: tl.constexpr,
    KEEP: tl.constexpr,
    H: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CAPACITY: tl.constexpr,
    POSITION: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    GROUP: tl.constexpr,
    ATTEND: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    """Sample one codebook, store its code, then run the next depth position.

    Every head-group program samples its row redundantly (deterministic, so
    all agree) instead of waiting for a separate sampling launch. The sample
    and attention bodies are the ones used by ``_sample`` / ``_attention``.
    A larger head group only adds warps along the head axis, so every per-head
    reduction keeps the unfused order and results are bit-identical (the
    tests compare against ``fused_sample`` + ``lookup_attention``).
    """
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    batch, group = tl.program_id(0), tl.program_id(1)
    token = _sample_row(Logits, Uniform, batch, Width, STRIDE_U, TOP_K, TEMPERATURE, TOP_P, BLOCK, KEEP)
    if ATTEND:
        _attend(
            QKV,
            token,
            batch,
            group,
            Embedding,
            K,
            V,
            Out,
            Residual,
            H,
            HEADS,
            D,
            CAPACITY,
            POSITION,
            True,
            BD,
            BS,
            GROUP,
            True,
        )
    if group == 0:
        tl.store(Codes + batch * STRIDE_C, token.to(tl.int64))


def sample_attend(
    logits: torch.Tensor,
    uniforms: torch.Tensor,
    codes: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    top_p: float,
    qkv: torch.Tensor | None = None,
    embedding: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    position: int = 0,
    group: int = 32,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``fused_sample`` + code store + ``lookup_attention`` in one launch.

    ``codes`` is the (strided) destination column for the sampled code.
    With ``qkv`` (the ``[vocab, 3H]`` lookup table of the sampled codebook),
    also inserts the next position's K/V and returns ``(attn_out, residual)``
    exactly as ``lookup_attention(qkv, key, value, position, tokens=...,
    embedding=embedding)`` would; otherwise returns ``None``.
    """
    batch, width = logits.shape
    if width < 2 or not 0 < top_k <= min(32, width) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError(
            "Fused local sampling requires width >= 2, 1 <= top_k <= 32, temperature > 0 and 0 < top_p <= 1"
        )
    if logits.stride(1) != 1 or logits.stride(0) != width:
        raise ValueError("Fused local sampling requires contiguous logits")
    attend = qkv is not None
    if attend:
        assert key is not None and value is not None and embedding is not None
        _, heads, capacity, dim = key.shape
        hidden = heads * dim
        out = torch.empty((batch, hidden), dtype=key.dtype, device=key.device)
        residual = torch.empty_like(out)
        groups = triton.cdiv(heads, group)
    else:
        heads = capacity = dim = hidden = 1
        out = residual = None
        groups = 1
    _sample_attend[(batch, groups)](
        logits,
        uniforms,
        codes,
        qkv,
        embedding,
        key,
        value,
        out,
        residual,
        width,
        uniforms.stride(0),
        codes.stride(0),
        top_k,
        temperature,
        top_p,
        triton.next_power_of_2(width),
        max(2, triton.next_power_of_2(top_k)),
        hidden,
        heads,
        dim,
        capacity,
        position,
        triton.next_power_of_2(dim),
        triton.next_power_of_2(position + 1),
        group,
        attend,
        USE_PDL=_USE_PDL,
        launch_pdl=_USE_PDL,
        num_warps=num_warps,
    )
    return (out, residual) if attend else None


@triton.jit
def _binary_continue(
    Logits,
    Exponential,
    Out,
    B,
    STRIDE_L: tl.constexpr,
    INV_TEMPERATURE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Two-way softmax + exponential-race multinomial, matching ATen.

    ``torch.softmax`` (warp softmax: ``exp(x - max) / sum``, accurate expf),
    then ``torch.multinomial(probs, 1)`` = ``argmax(probs / Exp(1))`` with the
    first index winning ties. IEEE division and libdevice ``exp`` keep the
    exact ATen float32 results. Writes ``choice == 0`` (continue).
    """
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < B
    x0 = tl.load(Logits + rows * STRIDE_L, mask, 0.0).to(tl.float32) * INV_TEMPERATURE
    x1 = tl.load(Logits + rows * STRIDE_L + 1, mask, 0.0).to(tl.float32) * INV_TEMPERATURE
    q0 = tl.load(Exponential + rows * 2, mask, 1.0)
    q1 = tl.load(Exponential + rows * 2 + 1, mask, 1.0)
    m = tl.maximum(x0, x1)
    e0 = libdevice.exp(x0 - m)
    e1 = libdevice.exp(x1 - m)
    total = e0 + e1
    p0 = tl.math.div_rn(e0, total)
    p1 = tl.math.div_rn(e1, total)
    r0 = tl.math.div_rn(p0, q0)
    r1 = tl.math.div_rn(p1, q1)
    # argmax: NaN counts as the maximum, first index wins ties.
    pick_one = (r1 > r0) | ((r1 != r1) & (r0 == r0))
    tl.store(Out + rows, (~pick_one).to(tl.int8), mask)


def binary_continue(
    logits: torch.Tensor,
    exponential: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Continue/stop gate equal to ``_sample_token(logits.float(), T, k, 1.0, True).eq(0)``.

    Valid for width-2 logits when top-k/top-p filtering are no-ops (k == 0 or
    k >= 2, top_p >= 1). ``exponential`` is ``torch.empty((B, 2)).exponential_()``
    drawn where multinomial would draw it, so the RNG stream is unchanged.
    """
    batch, width = logits.shape
    if width != 2 or temperature <= 0 or logits.stride(1) != 1:
        raise ValueError("binary_continue requires contiguous (B, 2) logits and temperature > 0")
    if exponential.shape != (batch, 2) or exponential.dtype != torch.float32 or not exponential.is_contiguous():
        raise ValueError("binary_continue requires contiguous float32 (B, 2) exponential draws")
    # ATen divides by a CPU scalar as a multiply by its float32 reciprocal.
    inv_temperature = float(np.float32(1.0) / np.float32(max(temperature, 1e-6)))
    out = torch.empty(batch, dtype=torch.bool, device=logits.device)
    block = 128
    _binary_continue[(triton.cdiv(batch, block),)](
        logits,
        exponential,
        out.view(torch.int8),
        batch,
        logits.stride(0),
        inv_temperature,
        block,
        num_warps=4,
    )
    return out


@triton.jit
def _mtp_epilogue(
    Codes,
    Continue,
    TextStep,
    Embeds,
    Table,
    OutEmbeds,
    OutCodes,
    H: tl.constexpr,
    N_VQ: tl.constexpr,
    VOCAB: tl.constexpr,
    PAD: tl.constexpr,
    STRIDE_C: tl.constexpr,
    STRIDE_T: tl.constexpr,
    STRIDE_E: tl.constexpr,
    BH: tl.constexpr,
):
    """Talker MTP output: masked audio-embedding sum, emit gate, output codes."""
    batch, block = tl.program_id(0), tl.program_id(1)
    cols = block * BH + tl.arange(0, BH)
    cmask = cols < H
    emit = (tl.load(TextStep + batch * STRIDE_T) > 0) & (tl.load(Continue + batch) != 0)
    acc = tl.zeros((BH,), tl.float32)
    for c in tl.static_range(N_VQ):
        code = tl.load(Codes + batch * STRIDE_C + c)
        valid = code != PAD
        safe = tl.minimum(tl.maximum(tl.where(valid, code, 0), 0), VOCAB - 1)
        row = tl.load(Table + (c * VOCAB + safe) * H + cols, cmask, 0.0)
        # bf16 row * bool mask -> bf16 (exact), accumulated in float32.
        acc += (row.to(tl.float32) * valid.to(tl.float32)).to(row.dtype).to(tl.float32)
        if block == 0:
            tl.store(OutCodes + batch * N_VQ + c, tl.where(emit, code, PAD))
    audio = acc.to(OutEmbeds.dtype.element_ty).to(tl.float32)
    gated = (audio * emit.to(tl.float32)).to(OutEmbeds.dtype.element_ty).to(tl.float32)
    base = tl.load(Embeds + batch * STRIDE_E + cols, cmask, 0.0).to(tl.float32)
    tl.store(OutEmbeds + batch * H + cols, base + gated, cmask)


def mtp_epilogue(
    codes: torch.Tensor,
    should_continue: torch.Tensor,
    text_step: torch.Tensor,
    input_embeds: torch.Tensor,
    stacked_embeddings: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch equivalent of the MOSS Local ``talker_mtp`` output path.

    ``emit = (text_step[:, 0] > 0) & should_continue``;
    ``out = input_embeds + audio_embed(codes) * emit`` (pad codes contribute
    zero, per-codebook float32 sum rounded once, as ``torch.sum`` does);
    ``codes_out = where(emit, codes, pad_id)``.
    """
    batch, n_vq = codes.shape
    n_tables, vocab, hidden = stacked_embeddings.shape
    if n_tables != n_vq or input_embeds.shape != (batch, hidden) or input_embeds.dtype != stacked_embeddings.dtype:
        raise ValueError("mtp_epilogue shape/dtype mismatch")
    if codes.stride(1) != 1 or input_embeds.stride(1) != 1 or not stacked_embeddings.is_contiguous():
        raise ValueError("mtp_epilogue requires contiguous inner dimensions")
    out_embeds = torch.empty((batch, hidden), dtype=input_embeds.dtype, device=input_embeds.device)
    out_codes = torch.empty((batch, n_vq), dtype=codes.dtype, device=codes.device)
    block = 512
    _mtp_epilogue[(batch, triton.cdiv(hidden, block))](
        codes,
        should_continue.view(torch.int8) if should_continue.dtype == torch.bool else should_continue,
        text_step,
        input_embeds,
        stacked_embeddings,
        out_embeds,
        out_codes,
        hidden,
        n_vq,
        vocab,
        pad_id,
        codes.stride(0),
        text_step.stride(0),
        input_embeds.stride(0),
        block,
        num_warps=4,
    )
    return out_embeds, out_codes
