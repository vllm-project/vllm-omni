# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Fused residual-codebook predictor for Qwen3-TTS (CUDA, BF16).

Same computation as the frame-local-KV predictor with per-call top-k Gumbel
sampling, restructured to issue few kernels per residual step, since at serving
batch sizes each of the 15 dependent steps is launch- and latency-bound:

- Per layer: one QKV GEMM straight from the residual stream (the RMSNorm
  weight is folded into it), one attention kernel (the row's RMSNorm scale,
  q/k RMSNorm, RoPE, this call's K/V write and attention over the frame's
  keys), O projection accumulated into the residual stream (``addmm_``),
  gate/up GEMM (folded likewise), SiLU-mul with the row scale, down
  projection accumulated into the stream.
- The embedding of each sampled code followed by ``small_to_mtp_projection``
  is linear, so it folds at load into one ``[vocab, hidden]`` table per
  codebook; the sampling kernel writes the next step's input row directly.
- Sampling finds the top-k threshold by bisection over the BF16 order keys
  (exact k-th largest, ties kept) instead of a sort.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _rms_rows_kernel(x_ptr, x_stride, w_ptr, out_ptr, eps, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    dt = out_ptr.dtype.element_ty
    y = (x * tl.rsqrt(var + eps)).to(dt).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (w * y).to(dt))


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _head_norm_rope(x1, x2, w1, w2, c1, c2, s1, s2, eps, HD: tl.constexpr):
    """HF RMSNorm (``weight * normed.to(bf16)``) then RoPE in BF16 elementwise steps, on the two halves."""
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / HD
    r = tl.rsqrt(var + eps)
    n1 = _bf(w1 * _bf(x1 * r))
    n2 = _bf(w2 * _bf(x2 * r))
    # rotate_half(n) = [-n2, n1]
    o1 = _bf(_bf(n1 * c1) + _bf(-n2 * s1))
    o2 = _bf(_bf(n2 * c2) + _bf(n1 * s2))
    return o1, o2


@triton.jit
def _row_rsqrt(x_ptr, row, x_stride, eps, HID: tl.constexpr):
    """rsqrt(mean(x[row]^2) + eps): the RMSNorm scale of a residual-stream row."""
    x = tl.load(x_ptr + row * x_stride + tl.arange(0, HID)).to(tl.float32)
    return tl.rsqrt(tl.sum(x * x, axis=0) / HID + eps)


@triton.jit
def _cp_attention_kernel(
    qkv_ptr,  # [B * NQ, (H + 2 * KVH) * HD]
    kc_ptr,  # [Bmax, KVH, MAXP, HD]
    vc_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,  # [MAXP, HD] bf16
    sin_ptr,
    out_ptr,  # [B * NQ, H * HD]
    first,
    eps,
    scale,
    NQ: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    HD: tl.constexpr,
    MAXP: tl.constexpr,
    BK: tl.constexpr,
    PDL: tl.constexpr = False,
    x_ptr=None,  # ROW_RMS: [B * NQ, HID] residual rows the QKV projection read
    x_stride=0,
    row_eps=0.0,
    HID: tl.constexpr = 1,
    ROW_RMS: tl.constexpr = False,
):
    """With ROW_RMS the QKV rows come from the un-normalized residual stream through
    RMSNorm-weight-folded weights; each row is scaled by its RMSNorm factor here."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    qi = tl.program_id(2)
    kvh = h // (H // KVH)
    W: tl.constexpr = (H + 2 * KVH) * HD
    HALF: tl.constexpr = HD // 2
    d = tl.arange(0, HALF)
    j = tl.arange(0, BK)
    p = first + qi
    row = (b * NQ + qi).to(tl.int64)
    cache = ((b * KVH + kvh) * MAXP).to(tl.int64) * HD
    # Each program is one short dependent chain, so issue every load that does
    # not depend on another up front: one memory round trip instead of several.
    # Weights and RoPE tables come first: under PDL they load before the wait.
    qw1 = tl.load(qw_ptr + d)
    qw2 = tl.load(qw_ptr + HALF + d)
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    c1 = tl.load(cos_ptr + p * HD + d)
    c2 = tl.load(cos_ptr + p * HD + HALF + d)
    s1 = tl.load(sin_ptr + p * HD + d)
    s2 = tl.load(sin_ptr + p * HD + HALF + d)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    old = (j < first)[:, None]
    kp = kc_ptr + cache + j[:, None] * HD + d[None, :]
    vp = vc_ptr + cache + j[:, None] * HD + d[None, :]
    k1 = tl.load(kp, mask=old, other=0.0)
    k2 = tl.load(kp + HALF, mask=old, other=0.0)
    v1 = tl.load(vp, mask=old, other=0.0)
    v2 = tl.load(vp + HALF, mask=old, other=0.0)
    qb = qkv_ptr + row * W + h * HD
    qx1 = tl.load(qb + d).to(tl.float32)
    qx2 = tl.load(qb + HALF + d).to(tl.float32)
    if ROW_RMS:
        rq = _row_rsqrt(x_ptr, row, x_stride, row_eps, HID)
        qx1 = qx1 * rq
        qx2 = qx2 * rq
    k1 = k1.to(tl.float32)
    k2 = k2.to(tl.float32)
    v1 = v1.to(tl.float32)
    v2 = v2.to(tl.float32)
    for n in tl.static_range(NQ):
        # this call's position first + n, computed from its own QKV row
        on = first + n
        rb = qkv_ptr + (b * NQ + n).to(tl.int64) * W
        kx1 = tl.load(rb + (H + kvh) * HD + d).to(tl.float32)
        kx2 = tl.load(rb + (H + kvh) * HD + HALF + d).to(tl.float32)
        vn1 = tl.load(rb + (H + KVH + kvh) * HD + d).to(tl.float32)
        vn2 = tl.load(rb + (H + KVH + kvh) * HD + HALF + d).to(tl.float32)
        if ROW_RMS:
            rk = _row_rsqrt(x_ptr, (b * NQ + n).to(tl.int64), x_stride, row_eps, HID)
            kx1 = kx1 * rk
            kx2 = kx2 * rk
            vn1 = (vn1 * rk).to(kc_ptr.dtype.element_ty).to(tl.float32)
            vn2 = (vn2 * rk).to(kc_ptr.dtype.element_ty).to(tl.float32)
        if NQ == 1:
            kc1, kc2, ks1, ks2 = c1, c2, s1, s2
        else:
            kc1 = tl.load(cos_ptr + on * HD + d)
            kc2 = tl.load(cos_ptr + on * HD + HALF + d)
            ks1 = tl.load(sin_ptr + on * HD + d)
            ks2 = tl.load(sin_ptr + on * HD + HALF + d)
        kn1, kn2 = _head_norm_rope(
            kx1, kx2, kw1, kw2,
            kc1.to(tl.float32), kc2.to(tl.float32), ks1.to(tl.float32), ks2.to(tl.float32), eps, HD,
        )  # fmt: skip
        sel = (j == on)[:, None]
        k1 = tl.where(sel, kn1[None, :], k1)
        k2 = tl.where(sel, kn2[None, :], k2)
        v1 = tl.where(sel, vn1[None, :], v1)
        v2 = tl.where(sel, vn2[None, :], v2)
        wm = (d < HALF) & (qi == n) & (h % (H // KVH) == 0)
        dtc = kc_ptr.dtype.element_ty
        tl.store(kc_ptr + cache + on * HD + d, kn1.to(dtc), mask=wm)
        tl.store(kc_ptr + cache + on * HD + HALF + d, kn2.to(dtc), mask=wm)
        tl.store(vc_ptr + cache + on * HD + d, vn1.to(dtc), mask=wm)
        tl.store(vc_ptr + cache + on * HD + HALF + d, vn2.to(dtc), mask=wm)
    q1, q2 = _head_norm_rope(
        qx1, qx2, qw1.to(tl.float32), qw2.to(tl.float32),
        c1.to(tl.float32), c2.to(tl.float32), s1.to(tl.float32), s2.to(tl.float32), eps, HD,
    )  # fmt: skip
    km = j <= p
    s = (tl.sum(k1 * q1[None, :], axis=1) + tl.sum(k2 * q2[None, :], axis=1)) * scale
    s = tl.where(km, s, -float("inf"))
    e = tl.exp(s - tl.max(s, axis=0))
    e = tl.where(km, e, 0.0)
    w = e / tl.sum(e, axis=0)
    ob = out_ptr + row * (H * HD) + h * HD
    dto = out_ptr.dtype.element_ty
    tl.store(ob + d, tl.sum(v1 * w[:, None], axis=0).to(dto))
    tl.store(ob + HALF + d, tl.sum(v2 * w[:, None], axis=0).to(dto))


@triton.jit
def _silu_mul_rms_kernel(gu_ptr, x_ptr, x_stride, out_ptr, eps, N: tl.constexpr, BN: tl.constexpr, HID: tl.constexpr):
    """SiLU(gate) * up for gate/up rows projected from the un-normalized residual stream through
    RMSNorm-weight-folded weights: each row is first scaled by its RMSNorm factor."""
    row = tl.program_id(0).to(tl.int64)
    r = _row_rsqrt(x_ptr, row, x_stride, eps, HID)
    offs = tl.program_id(1) * BN + tl.arange(0, BN)
    m = offs < N
    g = tl.load(gu_ptr + row * 2 * N + offs, mask=m).to(tl.float32) * r
    u = tl.load(gu_ptr + row * 2 * N + N + offs, mask=m).to(tl.float32) * r
    dt = out_ptr.dtype.element_ty
    sg = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (sg * u).to(dt), mask=m)


@triton.jit
def _cp_sample_kernel(
    logits_ptr,  # [B, V] bf16
    u_ptr,  # uniforms, row stride u_stride, contiguous over V
    u_stride,
    codes_ptr,  # [B, G] int64
    step,
    G: tl.constexpr,
    table_ptr,  # [V, HID] next-step input table, or dummy
    next_ptr,  # [B, HID]
    inv_temperature,
    V: tl.constexpr,
    TOPK: tl.constexpr,
    HID: tl.constexpr,
    HAS_NEXT: tl.constexpr,
    ss_ptr=None,  # [B] fp32: sum of squares of the next-step input row (PDL chain only)
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, V)
    # The uniforms come from before the predictor: load them ahead of the wait.
    u = tl.load(u_ptr + row * u_stride + idx)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    value = tl.load(logits_ptr + row * V + idx)
    scaled = (value.to(tl.float32) * inv_temperature).to(value.dtype)
    if TOPK > 0:
        # order-preserving 16-bit keys of the BF16 values
        bits = scaled.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        # Top-k keeps all numeric ties at the cutoff, including both zeros.
        bits = tl.where((bits & 0x7FFF) == 0, 0, bits)
        key = tl.where(bits >= 0x8000, 0xFFFF - bits, bits | 0x8000)
        the = 0
        for bit in tl.static_range(15, -1, -1):
            cand = the | (1 << bit)
            cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
            the = tl.where(cnt >= TOPK, cand, the)
        sf = tl.where(key >= the, scaled.to(tl.float32), -float("inf"))
    else:
        sf = scaled.to(tl.float32)
    score = sf - tldevice.log(-tldevice.log(u))
    max_score = tl.max(score, 0)
    chosen = tl.min(tl.where(score == max_score, idx, V), 0)
    first_nan = tl.min(tl.where(score != score, idx, V), 0)
    code = tl.where(first_nan < V, first_nan, chosen)
    tl.store(codes_ptr + row * G + step, code.to(tl.int64))
    if HAS_NEXT:
        c = tl.arange(0, HID)
        e = tl.load(table_ptr + code.to(tl.int64) * HID + c)
        tl.store(next_ptr + row * HID + c, e)
        if PDL:
            ef = e.to(tl.float32)
            tl.store(ss_ptr + row, tl.sum(ef * ef, axis=0))


# ---------------------------------------------------------------------------
# PDL chain GEMMs: out = A[M, K] @ W[N, K]^T, computed as W @ A^T so the weight
# rows fill the MMA's M dimension at small M. Every kernel of the chain is
# launched with programmatic dependent launch; it reads only constants before
# ``gdc_wait`` and lets the next kernel launch right after it, so each GEMM
# fetches its weights while the previous kernel is still running.
#
# Epilogues (EPI):
#   0: out = bf16(acc * r[row])                 qkv projection, lm head
#   1: out = SiLU(g) * u from the tile halves   gate/up (rows interleaved per tile)
#   2: x[row] += acc in place, row sum of squares per N tile    o / down
# where r = rsqrt(mean(x^2) + eps) is rebuilt from the previous EPI 2's per-tile
# sums of squares; the RMSNorm weight is folded into W at load.
# Split-K partials go to an FP32 workspace and the last CTA of each tile adds
# them in a fixed order, so results do not depend on CTA timing.
# ---------------------------------------------------------------------------


@triton.jit
def _w_chunk(w_ptr, n, k0, K: tl.constexpr, BK: tl.constexpr):
    return tl.load(w_ptr + n[:, None] * K + (k0 + tl.arange(0, BK))[None, :])


@triton.jit
def _a_dot(acc, w, a_ptr, arow, rmask, k0, K: tl.constexpr, BK: tl.constexpr):
    a = tl.load(a_ptr + arow[:, None] * K + (k0 + tl.arange(0, BK))[None, :], mask=rmask[:, None], other=0.0)
    return tl.dot(w, tl.trans(a), acc)


@triton.jit
def _prefetch_l2(ptrs):
    return tl.inline_asm_elementwise(
        "prefetch.global.L2::evict_last [$1];\n\tmov.u32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32, is_pure=False,
        pack=1,
    )  # fmt: skip


@triton.jit
def _cp_gemm_epilogue(
    acc, m, rmask, arow, n, pid_n, out_ptr, ss_in_ptr, ss_out_ptr, eps,
    N: tl.constexpr, HID: tl.constexpr, RMAX: tl.constexpr, BN: tl.constexpr, EPI: tl.constexpr,
    NT_IN: tl.constexpr,
):  # fmt: skip
    dt = out_ptr.dtype.element_ty
    if EPI == 2:
        # this CTA owns columns n of x: add in place, then their share of each row's sum of squares
        xp = out_ptr + m[None, :] * N + n[:, None]
        xn = (tl.load(xp, mask=rmask[None, :], other=0.0).to(tl.float32) + acc).to(dt)
        tl.store(xp, xn, mask=rmask[None, :])
        xf = xn.to(tl.float32)
        tl.store(ss_out_ptr + pid_n * RMAX + m, tl.sum(xf * xf, axis=0), mask=rmask)
    else:
        ss = tl.zeros([acc.shape[1]], dtype=tl.float32)
        for t in tl.static_range(NT_IN):
            ss += tl.load(ss_in_ptr + t * RMAX + arow, mask=rmask, other=0.0)
        y = acc * tl.rsqrt(ss / HID + eps)[None, :]
        if EPI == 0:
            tl.store(out_ptr + m[None, :] * N + n[:, None], y.to(dt), mask=rmask[None, :])
        else:
            HALF: tl.constexpr = BN // 2
            g, u = tl.split(tl.permute(tl.reshape(y, [2, HALF, y.shape[1]]), (1, 2, 0)))
            g = g.to(dt).to(tl.float32)
            u = u.to(dt).to(tl.float32)
            sg = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
            j = pid_n * HALF + tl.arange(0, HALF)
            tl.store(out_ptr + m[None, :] * (N // 2) + j[:, None], (sg * u).to(dt), mask=rmask[None, :])


@triton.jit
def _cp_gemm_reduce(
    ws_ptr, cnt_ptr, pid_n, pid_m, MT, M, a_rs, a_ro, n, out_ptr, ss_in_ptr, ss_out_ptr, eps,
    N: tl.constexpr, HID: tl.constexpr, RMAX: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr,
    SK: tl.constexpr, EPI: tl.constexpr, NT_IN: tl.constexpr,
):  # fmt: skip
    """Split-K arrival: the last CTA of tile (pid_n, pid_m) adds the partials in order and runs the epilogue."""
    tl.debug_barrier()
    cnt = cnt_ptr + pid_n * MT + pid_m
    arrived = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
    if arrived == SK - 1:
        for m0 in range(pid_m * BM, M, BM * MT):
            m = m0 + tl.arange(0, BM)
            rmask = m < M
            acc = tl.zeros([BN, BM], dtype=tl.float32)
            for s in tl.static_range(SK):
                acc += tl.load(
                    ws_ptr + s * RMAX * N + m[None, :] * N + n[:, None], mask=rmask[None, :], other=0.0,
                    cache_modifier=".cg",
                )  # fmt: skip
            _cp_gemm_epilogue(
                acc, m, rmask, m * a_rs + a_ro, n, pid_n, out_ptr, ss_in_ptr, ss_out_ptr, eps,
                N, HID, RMAX, BN, EPI, NT_IN,
            )  # fmt: skip
        tl.atomic_xchg(cnt, 0)


@triton.jit
def _cp_gemm_kernel(
    a_ptr, a_rs, a_ro,  # A row m is physical row m * a_rs + a_ro
    w_ptr,  # [N, K]
    ws_ptr, cnt_ptr,  # split-K FP32 workspace [SK, RMAX, N] and per-tile arrival counters (zero at rest)
    out_ptr, ss_in_ptr, ss_out_ptr, M, eps,
    K: tl.constexpr, N: tl.constexpr, HID: tl.constexpr, RMAX: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, NCH: tl.constexpr, BM: tl.constexpr,
    EPI: tl.constexpr, NT_IN: tl.constexpr, MSTAGES: tl.constexpr,
):  # fmt: skip
    """Small-M variant: the CTA's [BN, NCH * BK] weight panel is loaded once, before the wait, and reused for
    every M tile of the CTA (M tiles pid_m, pid_m + MT, ...)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    MT = tl.num_programs(2)
    SK: tl.constexpr = K // (BK * NCH)
    n = pid_n * BN + tl.arange(0, BN)
    kb = pid_k * (NCH * BK)
    w0 = _w_chunk(w_ptr, n, kb, K, BK)
    w1 = w0
    w2 = w0
    w3 = w0
    if NCH > 1:
        w1 = _w_chunk(w_ptr, n, kb + BK, K, BK)
    if NCH > 2:
        w2 = _w_chunk(w_ptr, n, kb + 2 * BK, K, BK)
        w3 = _w_chunk(w_ptr, n, kb + 3 * BK, K, BK)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    for m0 in tl.range(pid_m * BM, M, BM * MT, num_stages=MSTAGES):
        m = m0 + tl.arange(0, BM)
        rmask = m < M
        arow = m * a_rs + a_ro
        acc = tl.zeros([BN, BM], dtype=tl.float32)
        acc = _a_dot(acc, w0, a_ptr, arow, rmask, kb, K, BK)
        if NCH > 1:
            acc = _a_dot(acc, w1, a_ptr, arow, rmask, kb + BK, K, BK)
        if NCH > 2:
            acc = _a_dot(acc, w2, a_ptr, arow, rmask, kb + 2 * BK, K, BK)
            acc = _a_dot(acc, w3, a_ptr, arow, rmask, kb + 3 * BK, K, BK)
        if SK == 1:
            _cp_gemm_epilogue(
                acc, m, rmask, arow, n, pid_n, out_ptr, ss_in_ptr, ss_out_ptr, eps, N, HID, RMAX, BN, EPI, NT_IN
            )
        else:
            tl.store(ws_ptr + pid_k * RMAX * N + m[None, :] * N + n[:, None], acc, mask=rmask[None, :])
    if SK > 1:
        _cp_gemm_reduce(
            ws_ptr, cnt_ptr, pid_n, pid_m, MT, M, a_rs, a_ro, n, out_ptr, ss_in_ptr, ss_out_ptr, eps,
            N, HID, RMAX, BN, BM, SK, EPI, NT_IN,
        )  # fmt: skip


@triton.jit
def _cp_gemm_stream_kernel(
    a_ptr, a_rs, a_ro, w_ptr, ws_ptr, cnt_ptr, out_ptr, ss_in_ptr, ss_out_ptr, M, eps,
    K: tl.constexpr, N: tl.constexpr, HID: tl.constexpr, RMAX: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
    EPI: tl.constexpr, NT_IN: tl.constexpr, STAGES: tl.constexpr, NLP: tl.constexpr,
):  # fmt: skip
    """Large-M variant: one M tile per CTA and a pipelined K loop; the CTA's weight slice is prefetched into
    L2 before the wait instead of being held in registers."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    KS: tl.constexpr = K // SK
    n = pid_n * BN + tl.arange(0, BN)
    kb = pid_k * KS
    # one prefetch per 128-byte line of the slice (index clamped to pad to a power of two)
    li = tl.minimum(tl.arange(0, NLP), KS // 64 - 1)
    _prefetch_l2(w_ptr + n[:, None] * K + kb + (li * 64)[None, :])
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    m = pid_m * BM + tl.arange(0, BM)
    rmask = m < M
    arow = m * a_rs + a_ro
    kk = tl.arange(0, BK)
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for k0 in tl.range(kb, kb + KS, BK, num_stages=STAGES):
        w = tl.load(w_ptr + n[:, None] * K + (k0 + kk)[None, :])
        a = tl.load(a_ptr + arow[:, None] * K + (k0 + kk)[None, :], mask=rmask[:, None], other=0.0)
        acc = tl.dot(w, tl.trans(a), acc)
    if SK == 1:
        _cp_gemm_epilogue(
            acc, m, rmask, arow, n, pid_n, out_ptr, ss_in_ptr, ss_out_ptr, eps, N, HID, RMAX, BN, EPI, NT_IN
        )
    else:
        tl.store(ws_ptr + pid_k * RMAX * N + m[None, :] * N + n[:, None], acc, mask=rmask[None, :])
        _cp_gemm_reduce(
            ws_ptr, cnt_ptr, pid_n, pid_m, tl.num_programs(2), M, a_rs, a_ro, n, out_ptr, ss_in_ptr, ss_out_ptr,
            eps, N, HID, RMAX, BN, BM, SK, EPI, NT_IN,
        )  # fmt: skip


# Chain GEMM launch configs by the largest batch they serve (tuned on H200 over the
# whole predictor graph). Register-panel kernel: (BN, BK, NCH, BM, warps, MT, MSTAGES);
# streaming kernel: ("s", BN, BK, SK, BM, warps, STAGES). Batches above the
# last entry take the cuBLAS path (the chain was ~5% slower at 96 and 128). The
# gate/up tile width is fixed by its weight layout.
_PDL_GU_BN = 64
_PDL_CONFIGS: tuple[tuple[int, dict[str, tuple]], ...] = (
    (1, {
        "qkv": (64, 256, 2, 16, 4, 1, 1), "o": (32, 256, 1, 16, 4, 1, 2), "gu": (64, 256, 2, 16, 4, 1, 1),
        "down": (32, 128, 4, 16, 4, 1, 1), "lm": (64, 256, 2, 16, 8, 1, 1),
    }),
    (16, {
        "qkv": (128, 256, 2, 16, 8, 1, 1), "o": (64, 256, 2, 16, 4, 1, 1), "gu": (64, 256, 2, 16, 4, 1, 2),
        "down": (64, 256, 2, 16, 4, 1, 2), "lm": (128, 256, 2, 16, 8, 1, 1),
    }),
    (32, {
        "qkv": (128, 256, 2, 32, 8, 2, 1), "o": ("s", 64, 128, 2, 32, 4, 4), "gu": ("s", 64, 128, 2, 64, 4, 2),
        "down": (64, 256, 4, 32, 4, 2, 1), "lm": (128, 256, 2, 16, 8, 2, 1),
    }),
    (64, {
        "qkv": (128, 256, 2, 32, 8, 2, 1), "o": (64, 128, 4, 32, 4, 2, 3), "gu": (64, 256, 2, 64, 4, 1, 1),
        "down": (64, 256, 4, 32, 4, 2, 1), "lm": (128, 256, 2, 32, 8, 2, 1),
    }),
)  # fmt: skip


def _pdl_supported(device: torch.device) -> bool:
    """Programmatic dependent launch (``gdc_wait``) needs Hopper or newer and a Triton that exposes it."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 9:
        return False
    return hasattr(tl.extra.cuda, "gdc_wait")


def _pdl_split(cfg: tuple, k: int) -> int:
    return cfg[3] if cfg[0] == "s" else k // (cfg[1] * cfg[2])


def _pdl_fits(cfg: tuple, n: int, k: int) -> bool:
    """Whether a launch config tiles an [n, k] weight exactly (the configs are tuned for one model size)."""
    if cfg[0] == "s":
        _, bn, bk, sk = cfg[:4]
        return n % bn == 0 and k % (bk * sk) == 0 and (k // sk) % 64 == 0
    bn, bk, nch = cfg[:3]
    return n % bn == 0 and k >= bk * nch and k % (bk * nch) == 0 and nch in (1, 2, 4)


class FusedCodePredictor:
    """Execution object built from a loaded ``Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM``."""

    def __init__(self, predictor, max_batch: int) -> None:
        model = predictor.model
        weight = next(model.parameters())
        dev, dt = weight.device, weight.dtype
        if dt != torch.bfloat16:
            raise ValueError("fused code predictor requires BF16 weights")
        cfg = predictor.config
        self.groups = int(cfg.num_code_groups)
        self.vocab = int(cfg.vocab_size)
        self.hidden = int(cfg.hidden_size)
        attn0 = model.layers[0].self_attn
        self.heads, self.kv_heads, self.head_dim = attn0.num_heads, attn0.num_kv_heads, attn0.head_dim
        self.scale = attn0.scaling
        self.eps = float(model.norm.variance_epsilon)
        self.max_pos = attn0.max_seq
        # Position p attends keys 0..p and the last residual step sits at G - 1.
        self._attn_keys = triton.next_power_of_2(self.groups)
        self.max_batch = max_batch
        f32 = lambda t: t.detach().to(device=dev, dtype=torch.float32)  # noqa: E731
        c = lambda t: t.to(dt).contiguous()  # noqa: E731
        proj = predictor.small_to_mtp_projection
        if isinstance(proj, torch.nn.Identity):
            pw, pb = None, None
        else:
            pw, pb = f32(proj.weight), f32(proj.bias)
        self.proj_wt = c(pw.t()) if pw is not None else None
        self.proj_b = c(pb) if pb is not None else None
        tables = []
        for emb in model.codec_embedding:
            e = f32(emb.weight)
            tables.append(c(e @ pw.t() + pb) if pw is not None else c(e))
        self.tables = tables
        self.layers = []
        for layer in model.layers:
            a = layer.self_attn
            self.layers.append(
                dict(
                    index=len(self.layers),
                    ln1=c(f32(layer.input_layernorm.weight)),
                    ln2=c(f32(layer.post_attention_layernorm.weight)),
                    eps1=float(layer.input_layernorm.variance_epsilon),
                    eps2=float(layer.post_attention_layernorm.variance_epsilon),
                    qkv=c(f32(a.qkv_proj.weight).t()),
                    qn=c(f32(a.q_norm.weight)),
                    kn=c(f32(a.k_norm.weight)),
                    head_eps=float(a.q_norm.variance_epsilon),
                    o=c(f32(a.o_proj.weight).t()),
                    gu=c(f32(layer.mlp.gate_up_proj.weight).t()),
                    down=c(f32(layer.mlp.down_proj.weight).t()),
                )
            )
            if a.qkv_proj.bias is not None:
                raise ValueError("fused code predictor assumes bias-free attention")
        self.norm_w = c(f32(model.norm.weight))
        self.lm_heads = [c(f32(h.weight).t()) for h in predictor.lm_head]
        rot = model.rotary_emb
        self.cos = rot.cos_cached.to(device=dev, dtype=dt).contiguous()
        self.sin = rot.sin_cached.to(device=dev, dtype=dt).contiguous()
        kv = (len(self.layers), 2, max_batch, self.kv_heads, self.max_pos, self.head_dim)
        self.cache = torch.zeros(kv, device=dev, dtype=dt)
        self._dummy = torch.zeros(1, device=dev, dtype=dt)
        self._pdl_configs = _PDL_CONFIGS if _pdl_supported(dev) and self._pdl_shapes_fit() else ()
        if self._pdl_configs:
            self._init_pdl(max_batch)
        if self.layers[0]["qkv"] is not None:
            # The cuBLAS path reads the residual stream directly: fold the
            # RMSNorm weights into the projections (as the PDL chain does) and
            # apply each row's RMSNorm scale in the consuming kernels instead of
            # a separate normalization kernel per layer and step.
            for L in self.layers:
                L["qkv"] = c(L["qkv"].float() * L["ln1"].float()[:, None])
                L["gu"] = c(L["gu"].float() * L["ln2"].float()[:, None])

    def _pdl_shapes_fit(self) -> bool:
        L = self.layers[0]
        inter = L["gu"].shape[1] // 2
        shapes = {
            "qkv": L["qkv"].shape[::-1], "o": L["o"].shape[::-1], "gu": L["gu"].shape[::-1],
            "down": L["down"].shape[::-1], "lm": self.lm_heads[0].shape[::-1],
        }  # fmt: skip
        if inter % (_PDL_GU_BN // 2) or self.hidden % 64:
            return False
        for _, cfg in _PDL_CONFIGS:
            gu = cfg["gu"]
            if (gu[1] if gu[0] == "s" else gu[0]) != _PDL_GU_BN:
                return False
            if not all(_pdl_fits(cfg[name], n, k) for name, (n, k) in shapes.items()):
                return False
        return True

    def _init_pdl(self, max_batch: int) -> None:
        """Weights and buffers of the PDL chain (see the GEMM section above)."""
        dev, hid = self.norm_w.device, self.hidden

        def fold(wt: torch.Tensor, ln: torch.Tensor) -> torch.Tensor:
            # [K, N] transposed weight -> [N, K] with the RMSNorm weight folded into its columns
            return (wt.float().t() * ln.float()[None, :]).to(torch.bfloat16).contiguous()

        half = _PDL_GU_BN // 2
        self.pdl_w = []
        for L in self.layers:
            gu = fold(L["gu"], L["ln2"])
            inter = gu.shape[0] // 2
            # each BN-row tile holds BN/2 gate rows then the matching BN/2 up rows
            gu = torch.stack((gu[:inter].reshape(-1, half, hid), gu[inter:].reshape(-1, half, hid)), 1)
            self.pdl_w.append(
                dict(
                    qkv=fold(L["qkv"], L["ln1"]),
                    o=L["o"].t().contiguous(),
                    gu=gu.reshape(-1, hid).contiguous(),
                    down=L["down"].t().contiguous(),
                )
            )
        self.pdl_lm = [fold(h, self.norm_w) for h in self.lm_heads]
        inter = self.pdl_w[0]["gu"].shape[0] // 2
        rows = 2 * max_batch
        self._rows = rows
        bf = torch.bfloat16
        self.pdl_x = torch.empty(rows, hid, device=dev, dtype=bf)
        self.pdl_qkv = torch.empty(rows, (self.heads + 2 * self.kv_heads) * self.head_dim, device=dev, dtype=bf)
        self.pdl_att = torch.empty(rows, self.heads * self.head_dim, device=dev, dtype=bf)
        self.pdl_act = torch.empty(rows, inter, device=dev, dtype=bf)
        self.pdl_logits = torch.empty(max_batch, self.vocab, device=dev, dtype=bf)
        widest = max(self.pdl_qkv.shape[1], 2 * inter, self.vocab, hid)
        ks = {name: w.shape[1] for name, w in self.pdl_w[0].items()} | {"lm": hid}
        splits = max(_pdl_split(c[name], ks[name]) for _, c in self._pdl_configs for name in ks)
        self.pdl_ws = torch.empty(splits * rows * widest, device=dev, dtype=torch.float32)
        self.pdl_cnt = torch.zeros(16384, device=dev, dtype=torch.int32)
        # per-row sums of squares of the residual stream, [N tiles, rows], ping-ponged between producers
        self.pdl_ss = [torch.zeros(hid // 16, rows, device=dev, dtype=torch.float32) for _ in range(2)]
        if max_batch <= self._pdl_configs[-1][0]:
            # Every batch takes the PDL chain: drop the weights only the other path reads.
            for L in self.layers:
                for key in ("qkv", "o", "gu", "down"):
                    L[key] = None
            self.lm_heads = None

    def warmup_batches(self) -> list[int]:
        """Batch sizes that between them compile every kernel variant ``__call__`` can launch."""
        # Triton specializes integer arguments on == 1 and on divisibility by 16: per config bucket run
        # its largest batch and one below it, so row counts of both kinds are compiled.
        batches = {1, min(2, self.max_batch), self.max_batch, max(1, self.max_batch - 1)}
        for limit, _ in self._pdl_configs:
            limit = min(limit, self.max_batch)
            batches.update((limit, max(1, limit - 1)))
        return sorted(batches)

    def _pdl_config(self, batch: int) -> dict | None:
        for limit, cfg in self._pdl_configs:
            if batch <= limit:
                return cfg
        return None

    def _gemm(self, cfg, a, a_rs, a_ro, w, out, ss_in, nt_in, ss_out, M, epi, eps=0.0) -> int:
        """Launch one chain GEMM; returns the number of sum-of-squares tiles it wrote (EPI 2)."""
        N, K = w.shape
        common = (a, a_rs, a_ro, w, self.pdl_ws, self.pdl_cnt, out, ss_in, ss_out, M, eps)
        if cfg[0] == "s":
            _, BN, BK, SK, BM, warps, stages = cfg
            _cp_gemm_stream_kernel[(N // BN, SK, triton.cdiv(M, BM))](
                *common, K=K, N=N, HID=self.hidden, RMAX=self._rows, BN=BN, BK=BK, SK=SK, BM=BM, EPI=epi,
                NT_IN=nt_in, STAGES=stages, NLP=triton.next_power_of_2(K // SK // 64),
                num_warps=warps, launch_pdl=True,
            )  # fmt: skip
        else:
            BN, BK, NCH, BM, warps, MT, mstages = cfg
            _cp_gemm_kernel[(N // BN, K // (BK * NCH), max(1, min(MT, triton.cdiv(M, BM))))](
                *common, K=K, N=N, HID=self.hidden, RMAX=self._rows, BN=BN, BK=BK, NCH=NCH, BM=BM, EPI=epi,
                NT_IN=nt_in, MSTAGES=mstages, num_warps=warps, launch_pdl=True,
            )  # fmt: skip
        return self.hidden // BN if epi == 2 else 0

    def _call_pdl(self, cfg, B, layer0_code, inp, inv_temperature, top_k, sample_uniforms) -> torch.Tensor:
        x = self.pdl_x
        if self.proj_wt is not None:
            torch.addmm(self.proj_b, inp, self.proj_wt, out=x[: 2 * B])
        else:
            x[: 2 * B].copy_(inp)
        ss_cur, ss_nxt = self.pdl_ss
        torch.sum(x[: 2 * B].float().square(), dim=1, out=ss_cur[0, : 2 * B])
        nt = 1
        codes = torch.empty(B, self.groups, device=x.device, dtype=torch.int64)
        codes[:, 0] = layer0_code.reshape(B)
        nq, first = 2, 0
        for step in range(1, self.groups):
            rows = B * nq
            for L, W in zip(self.layers, self.pdl_w):
                self._gemm(cfg["qkv"], x, 1, 0, W["qkv"], self.pdl_qkv, ss_cur, nt, None, rows, 0, L["eps1"])
                _cp_attention_kernel[(B, self.heads, nq)](
                    self.pdl_qkv, self.cache[L["index"], 0], self.cache[L["index"], 1], L["qn"], L["kn"],
                    self.cos, self.sin, self.pdl_att, first, L["head_eps"], self.scale,
                    NQ=nq, H=self.heads, KVH=self.kv_heads, HD=self.head_dim, MAXP=self.max_pos,
                    BK=self._attn_keys, PDL=True, num_warps=1, launch_pdl=True,
                )  # fmt: skip
                nt = self._gemm(cfg["o"], self.pdl_att, 1, 0, W["o"], x, None, 0, ss_nxt, rows, 2)
                ss_cur, ss_nxt = ss_nxt, ss_cur
                self._gemm(cfg["gu"], x, 1, 0, W["gu"], self.pdl_act, ss_cur, nt, None, rows, 1, L["eps2"])
                nt = self._gemm(cfg["down"], self.pdl_act, 1, 0, W["down"], x, None, 0, ss_nxt, rows, 2)
                ss_cur, ss_nxt = ss_nxt, ss_cur
            # the last row of each request feeds the head
            self._gemm(
                cfg["lm"], x, nq, nq - 1, self.pdl_lm[step - 1], self.pdl_logits, ss_cur, nt, None, B, 0, self.eps
            )
            has_next = step < self.groups - 1
            us = sample_uniforms[:, step - 1]
            _cp_sample_kernel[(B,)](
                self.pdl_logits, us, us.stride(0), codes, step, self.groups,
                self.tables[step - 1] if has_next else self._dummy, x, inv_temperature,
                V=self.vocab, TOPK=top_k, HID=self.hidden, HAS_NEXT=has_next, ss_ptr=ss_nxt, PDL=True,
                num_warps=4, launch_pdl=True,
            )  # fmt: skip
            ss_cur, ss_nxt = ss_nxt, ss_cur
            nt = 1
            first = step + 1
            nq = 1
        return codes

    def _layer(self, x: torch.Tensor, li: int, B: int, nq: int, first: int) -> None:
        L = self.layers[li]
        rows = x.shape[0]
        # qkv / gate-up weights carry the RMSNorm weights; the consuming kernels apply the row scales.
        qkv = torch.mm(x, L["qkv"])
        att = torch.empty(rows, self.heads * self.head_dim, device=x.device, dtype=x.dtype)
        _cp_attention_kernel[(B, self.heads, nq)](
            qkv, self.cache[li, 0], self.cache[li, 1], L["qn"], L["kn"], self.cos, self.sin, att,
            first, L["head_eps"], self.scale, x_ptr=x, x_stride=x.stride(0), row_eps=L["eps1"],
            NQ=nq, H=self.heads, KVH=self.kv_heads, HD=self.head_dim, MAXP=self.max_pos,
            BK=self._attn_keys, HID=self.hidden, ROW_RMS=True, num_warps=1,
        )  # fmt: skip
        x.addmm_(att, L["o"])
        gu = torch.mm(x, L["gu"])
        inter = gu.shape[1] // 2
        a = torch.empty(rows, inter, device=x.device, dtype=x.dtype)
        _silu_mul_rms_kernel[(rows, triton.cdiv(inter, 1024))](
            gu, x, x.stride(0), a, L["eps2"], N=inter, BN=1024, HID=self.hidden
        )
        x.addmm_(a, L["down"])

    @torch.inference_mode()
    def __call__(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        inv_temperature: float,
        top_k: int,
        sample_uniforms: torch.Tensor,
    ) -> torch.Tensor:
        B = int(layer0_code.shape[0])
        if B > self.max_batch:
            raise ValueError("fused code predictor batch exceeds its capacity")
        dev, dt = layer0_embed.device, self.norm_w.dtype
        inp = torch.stack([last_talker_hidden.reshape(B, -1), layer0_embed.reshape(B, -1)], 1).to(dt)
        inp = inp.reshape(2 * B, -1)
        cfg = self._pdl_config(B)
        if cfg is not None:
            return self._call_pdl(cfg, B, layer0_code, inp, inv_temperature, top_k, sample_uniforms)
        x = torch.addmm(self.proj_b, inp, self.proj_wt) if self.proj_wt is not None else inp.clone()
        codes = torch.empty(B, self.groups, device=dev, dtype=torch.int64)
        codes[:, 0] = layer0_code.reshape(B)
        u = sample_uniforms
        nq, first = 2, 0
        for step in range(1, self.groups):
            for li in range(len(self.layers)):
                self._layer(x, li, B, nq, first)
            last = x[nq - 1 :: nq] if nq > 1 else x
            hf = torch.empty(B, self.hidden, device=dev, dtype=dt)
            _rms_rows_kernel[(B,)](last, last.stride(0), self.norm_w, hf, self.eps, N=self.hidden)
            logits = torch.mm(hf, self.lm_heads[step - 1])
            has_next = step < self.groups - 1
            nxt = torch.empty(B, self.hidden, device=dev, dtype=dt) if has_next else self._dummy
            us = u[:, step - 1]
            _cp_sample_kernel[(B,)](
                logits, us, us.stride(0), codes, step, self.groups,
                self.tables[step - 1] if has_next else self._dummy, nxt, inv_temperature,
                V=self.vocab, TOPK=top_k, HID=self.hidden, HAS_NEXT=has_next, num_warps=4,
            )  # fmt: skip
            x = nxt
            first = step + 1
            nq = 1
        return codes
