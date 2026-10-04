# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Stateful, fused streaming decoder for the Qwen3-TTS 12 Hz codec.

Every layer of the codec decoder is causal, so what a new frame needs from the
past is small and fixed: each causal conv's last ``(k - 1) * dilation`` input
rows, each overlapping transposed conv's last GEMM row, and each attention
layer's last ``sliding_window`` keys and values. A request owns a slot of
per-layer state holding exactly that, and a call decodes ``T`` new frames for
each of ``B`` requests, reading and advancing the state. The output equals the
whole-utterance decode (``Qwen3TTSTokenizerV2Decoder._forward_exact``) frame by
frame, up to floating point rounding: nothing is recomputed and nothing is
truncated.

Layout and fusion:

- Activations are time-major ``[rows, C]``; every dense conv is a tap-major
  im2col (one Triton kernel that also applies the pending bias and SnakeBeta,
  and reads/writes the history) followed by one cuBLAS GEMM.
- Linear chains fold at load: the RVQ output projections, ``pre_conv`` and the
  transformer's ``input_proj`` become one ``k = 3`` conv over the concatenated
  codebook embeddings; LayerScale folds into ``o_proj``/``down_proj``; ConvNeXt
  ``gamma`` folds into ``pwconv2``.
- A bias that feeds a residual stream is carried to that stream's consumer
  (the next im2col or transposed-conv GEMM) instead of being added in place.
- Residual branches accumulate into their stream with in-place ``addmm_``.

History buffers are double-buffered per slot: a call reads half ``parity[slot]``
and writes the other half, then flips the parity, so no kernel overwrites rows
another program of the same launch still reads. A row whose first frame index
``pos`` is 0 reads zeros, which are the convs' causal padding, and attention
only looks back ``min(pos + 1, window)`` positions, so a new request needs no
state reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from vllm.triton_utils import tl, triton

from vllm_omni.platforms import current_omni_platform

_KERNEL = 7


# --------------------------------------------------------------------------- kernels


@triton.jit
def _rvq_gather_kernel(codes_ptr, books_ptr, out_ptr, NQ: tl.constexpr, CB: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, HALF)
    c0 = tl.load(codes_ptr + row * NQ).to(tl.int64)
    c0 = tl.where((c0 >= 0) & (c0 < CB), c0, 0)  # EOS/special rows decode as code 0; callers drop them
    first = tl.load(books_ptr + c0 * HALF + offs).to(tl.float32)
    acc = tl.zeros([HALF], dtype=tl.float32)
    for q in range(1, NQ):
        c = tl.load(codes_ptr + row * NQ + q).to(tl.int64)
        c = tl.where((c >= 0) & (c < CB), c, 0)
        acc += tl.load(books_ptr + (q * CB + c) * HALF + offs).to(tl.float32)
    dt = out_ptr.dtype.element_ty
    tl.store(out_ptr + row * 2 * HALF + offs, first.to(dt))
    tl.store(out_ptr + row * 2 * HALF + HALF + offs, acc.to(dt))


@triton.jit
def _fast_sin(y):
    # Two-step reduction to [-pi, pi], then the hardware approximation
    # (absolute error ~4e-7 there): accurate sin's instruction count, not
    # memory, bounds every kernel that applies SnakeBeta.
    k = tl.extra.cuda.libdevice.rint(y * 0.15915494309189535)
    r = tl.fma(-k, -1.7484555314695172e-07, tl.fma(-k, 6.2831854820251465, y))
    return tl.inline_asm_elementwise("sin.approx.f32 $0, $1;", "=r,r", [r], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _snake(v, a, ib):
    s = _fast_sin(v * a)
    return v + ib * s * s


@triton.jit
def _im2col_kernel(
    x_ptr,  # [B * T, C] layer input rows
    bias_ptr,  # [C] pending bias (added to x rows only) or dummy
    alpha_ptr,  # [C] exp(alpha) or dummy
    ibeta_ptr,  # [C] 1 / (exp(beta) + eps) or dummy
    hist_ptr,  # [2, S, H, C] transformed history rows
    slots_ptr,
    pos_ptr,
    par_ptr,
    col_ptr,  # [B * T, K * C]
    T,
    S,
    C: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SNAKE: tl.constexpr,
    BU: tl.constexpr,
    BC: tl.constexpr,
    EXT: tl.constexpr = False,
):
    """Source row ``u`` of ``[history (H) | x (T)]``: scatter to every tap that reads it, save the last H."""
    b = tl.program_id(0)
    u = tl.program_id(1) * BU + tl.arange(0, BU)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    cm = c < C
    n_src = H + T
    um = u < n_src
    slot = tl.load(slots_ptr + b).to(tl.int64)
    live = tl.load(pos_ptr + b) != 0
    par = tl.load(par_ptr + slot).to(tl.int64)
    # history rows
    from_hist = u < H
    hrow = tl.where(from_hist, u, 0)
    hv = tl.load(
        hist_ptr + ((par * S + slot) * H + hrow)[:, None] * C + c[None, :],
        mask=(from_hist & um & live)[:, None] & cm[None, :],
        other=0.0,
    ).to(tl.float32)
    xrow = tl.where(from_hist, 0, u - H)
    xv = tl.load(
        x_ptr + (b * T + xrow).to(tl.int64)[:, None] * C + c[None, :],
        mask=((~from_hist) & um)[:, None] & cm[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        xv += tl.load(bias_ptr + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    if HAS_SNAKE:
        a = tl.load(alpha_ptr + c, mask=cm, other=0.0).to(tl.float32)[None, :]
        ib = tl.load(ibeta_ptr + c, mask=cm, other=0.0).to(tl.float32)[None, :]
        xv = _snake(xv, a, ib)
    dt = col_ptr.dtype.element_ty
    v = tl.where(from_hist[:, None], hv, xv.to(dt).to(tl.float32)).to(dt)
    if EXT:
        tl.store(col_ptr + (b * n_src + u).to(tl.int64)[:, None] * C + c[None, :], v, mask=um[:, None] & cm[None, :])
    else:
        for j in tl.static_range(K):
            t = u - j * D
            tm = (t >= 0) & (t < T) & um
            tl.store(
                col_ptr + (b * T + tl.where(tm, t, 0)).to(tl.int64)[:, None] * (K * C) + j * C + c[None, :],
                v,
                mask=tm[:, None] & cm[None, :],
            )
    if H > 0:
        hm = (u >= T) & um
        tl.store(
            hist_ptr + (((1 - par) * S + slot) * H + tl.where(hm, u - T, 0))[:, None] * C + c[None, :],
            v,
            mask=hm[:, None] & cm[None, :],
        )


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, eps, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    dt = out_ptr.dtype.element_ty
    y = (x * tl.rsqrt(var + eps)).to(dt).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (w * y).to(dt))


@triton.jit
def _rope_kv_write_kernel(
    qkv_ptr,  # [B * T, 3 * NH * HD]
    q_ptr,  # [B * T, NH * HD]
    k_ring_ptr,  # [S, RING, NH * HD]
    v_ring_ptr,
    slots_ptr,
    pos_ptr,
    T,
    theta,
    NH: tl.constexpr,
    HD: tl.constexpr,
    RING: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    b = r // T
    p = tl.load(pos_ptr + b) + r % T
    slot = tl.load(slots_ptr + b).to(tl.int64)
    half: tl.constexpr = HD // 2
    i = tl.arange(0, half)
    inv_freq = tl.exp(-(2.0 * i.to(tl.float32) / HD) * tl.log(theta))
    ang = p.to(tl.float32) * inv_freq
    dt = q_ptr.dtype.element_ty
    cos = tl.cos(ang).to(dt).to(tl.float32)
    sin = tl.sin(ang).to(dt).to(tl.float32)
    W: tl.constexpr = NH * HD
    base = qkv_ptr + r.to(tl.int64) * 3 * W + h * HD
    q1 = tl.load(base + i).to(tl.float32)
    q2 = tl.load(base + half + i).to(tl.float32)
    k1 = tl.load(base + W + i).to(tl.float32)
    k2 = tl.load(base + W + half + i).to(tl.float32)
    vv = tl.arange(0, HD)
    v = tl.load(base + 2 * W + vv)
    qo = q_ptr + r.to(tl.int64) * W + h * HD
    tl.store(qo + i, (q1 * cos - q2 * sin).to(dt))
    tl.store(qo + half + i, (q2 * cos + q1 * sin).to(dt))
    ring = (slot * RING + p % RING) * W + h * HD
    tl.store(k_ring_ptr + ring + i, (k1 * cos - k2 * sin).to(dt))
    tl.store(k_ring_ptr + ring + half + i, (k2 * cos + k1 * sin).to(dt))
    tl.store(v_ring_ptr + ring + vv, v)


@triton.jit
def _attend_kernel(
    q_ptr,  # [B * T, NH * HD]
    k_ring_ptr,
    v_ring_ptr,
    out_ptr,  # [B * T, NH * HD]
    slots_ptr,
    pos_ptr,
    T,
    scale,
    NH: tl.constexpr,
    HD: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    b = r // T
    p = tl.load(pos_ptr + b) + r % T
    slot = tl.load(slots_ptr + b).to(tl.int64)
    W: tl.constexpr = NH * HD
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + r.to(tl.int64) * W + h * HD + d).to(tl.float32)
    j = tl.arange(0, BK)
    n = tl.minimum(p + 1, WINDOW)
    km = j < n
    kp = p - j  # key positions p, p-1, ...
    idx = (slot * RING + tl.where(km, kp, 0) % RING) * W + h * HD
    k = tl.load(k_ring_ptr + idx[:, None] + d[None, :], mask=km[:, None], other=0.0).to(tl.float32)
    s = tl.sum(k * q[None, :], axis=1) * scale
    s = tl.where(km, s, -float("inf"))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(km, e, 0.0)
    w = e / tl.sum(e, axis=0)
    dt = out_ptr.dtype.element_ty
    w = w.to(dt).to(tl.float32)
    v = tl.load(v_ring_ptr + idx[:, None] + d[None, :], mask=km[:, None], other=0.0).to(tl.float32)
    o = tl.sum(v * w[:, None], axis=0)
    tl.store(out_ptr + r.to(tl.int64) * W + h * HD + d, o.to(dt))


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, N: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BN + tl.arange(0, BN)
    m = offs < N
    g = tl.load(gu_ptr + row * 2 * N + offs, mask=m).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * N + N + offs, mask=m).to(tl.float32)
    dt = out_ptr.dtype.element_ty
    sg = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (sg * u).to(dt), mask=m)


@triton.jit
def _dwconv_ln_kernel(
    z_ptr,  # [B * T, C] ConvNeXt input rows (true values)
    w_ptr,  # [K, C]
    b_ptr,  # [C]
    lnw_ptr,
    lnb_ptr,
    hist_ptr,  # [2, S, K - 1, C]
    slots_ptr,
    pos_ptr,
    par_ptr,
    out_ptr,  # [B * T, C] LayerNorm output
    T,
    S,
    eps,
    C: tl.constexpr,
    K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # 0 .. max(T, K - 1)
    H: tl.constexpr = K - 1
    c = tl.arange(0, C)
    slot = tl.load(slots_ptr + b).to(tl.int64)
    live = tl.load(pos_ptr + b) != 0
    par = tl.load(par_ptr + slot).to(tl.int64)
    dt = out_ptr.dtype.element_ty
    hbase = hist_ptr + (par * S + slot) * H * C
    acc = tl.load(b_ptr + c).to(tl.float32)
    for j in tl.static_range(K):
        u = t + j  # index into [hist (H) | z (T)]
        fh = u < H
        hv = tl.load(hbase + tl.where(fh, u, 0) * C + c, mask=(c < C) & fh & live, other=0.0).to(tl.float32)
        zrow = (b * T + tl.where(fh, 0, u - H)).to(tl.int64)
        zv = tl.load(z_ptr + zrow * C + c, mask=(c < C) & (~fh) & (t < T), other=0.0).to(tl.float32)
        acc += tl.where(fh, hv, zv) * tl.load(w_ptr + j * C + c).to(tl.float32)
    h = acc.to(dt).to(tl.float32)
    mean = tl.sum(h, axis=0) / C
    hc = h - mean
    var = tl.sum(hc * hc, axis=0) / C
    y = hc * tl.rsqrt(var + eps) * tl.load(lnw_ptr + c).to(tl.float32) + tl.load(lnb_ptr + c).to(tl.float32)
    tl.store(out_ptr + (b * T + t).to(tl.int64) * C + c, y.to(dt), mask=(c < C) & (t < T))
    # new history row t: source row T + t of [hist | z]
    u = T + t
    fh = u < H
    hv = tl.load(hbase + tl.where(fh, u, 0) * C + c, mask=(c < C) & fh & live & (t < H), other=0.0)
    zrow = (b * T + tl.where(fh, 0, u - H)).to(tl.int64)
    zv = tl.load(z_ptr + zrow * C + c, mask=(c < C) & (~fh) & (t < H), other=0.0)
    tl.store(
        hist_ptr + (((1 - par) * S + slot) * H + tl.where(t < H, t, 0)) * C + c,
        tl.where(fh, hv, zv).to(dt),
        mask=(c < C) & (t < H),
    )


@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, N: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BN + tl.arange(0, BN)
    m = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=m).to(tl.float32) + tl.load(b_ptr + offs, mask=m).to(tl.float32)
    dt = x_ptr.dtype.element_ty
    x = x.to(dt).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(x_ptr + row * N + offs, y.to(dt), mask=m)


@triton.jit
def _col2im_kernel(
    z_ptr,  # [B * T, 2 * R, C] transposed-conv GEMM rows
    bias_ptr,  # [C]
    prev_ptr,  # [2, S, R, C] previous call's last GEMM row, second half
    slots_ptr,
    pos_ptr,
    par_ptr,
    out_ptr,  # [B * T * R, C]
    T,
    S,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(0)  # b * T + t
    j = tl.program_id(1)  # 0 .. R - 1
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    cm = c < C
    b = row // T
    t = row % T
    slot = tl.load(slots_ptr + b).to(tl.int64)
    live = tl.load(pos_ptr + b) != 0
    par = tl.load(par_ptr + slot).to(tl.int64)
    zr = z_ptr + row.to(tl.int64) * 2 * R * C
    cur = tl.load(zr + j * C + c, mask=cm).to(tl.float32)
    prev_in = tl.load(zr - 2 * R * C + (R + j) * C + c, mask=cm & (t > 0), other=0.0).to(tl.float32)
    prev_st = tl.load(prev_ptr + ((par * S + slot) * R + j) * C + c, mask=cm & (t == 0) & live, other=0.0).to(
        tl.float32
    )
    y = cur + tl.where(t > 0, prev_in, prev_st) + tl.load(bias_ptr + c, mask=cm).to(tl.float32)
    dt = out_ptr.dtype.element_ty
    tl.store(out_ptr + (row.to(tl.int64) * R + j) * C + c, y.to(dt), mask=cm)
    last = tl.load(zr + (R + j) * C + c, mask=cm & (t == T - 1))
    tl.store(prev_ptr + (((1 - par) * S + slot) * R + j) * C + c, last, mask=cm & (t == T - 1))


@triton.jit
def _conv_out_kernel(
    ext_ptr,  # [B, H + T, C] transformed rows: history then this call's
    w_ptr,  # [K, C]
    wb,  # conv bias (float)
    out_ptr,  # [B, T] float32 waveform
    T,
    C: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    CP: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.arange(0, CP)
    cm = c < C
    tm = t < T
    base = ext_ptr + b.to(tl.int64) * (T + K - 1) * C
    acc = tl.zeros([BT], dtype=tl.float32)
    for j in tl.static_range(K):
        v = tl.load(base + (t + j)[:, None] * C + c[None, :], mask=tm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(v * tl.load(w_ptr + j * C + c, mask=cm, other=0.0).to(tl.float32)[None, :], axis=1)
    dt = ext_ptr.dtype.element_ty
    y = acc.to(dt).to(tl.float32) + wb
    y = tl.minimum(tl.maximum(y.to(dt).to(tl.float32), -1.0), 1.0)
    tl.store(out_ptr + b.to(tl.int64) * T + t, y, mask=tm)


@triton.jit
def _flip_kernel(par_ptr, slots_ptr):
    slot = tl.load(slots_ptr + tl.program_id(0)).to(tl.int64)
    tl.store(par_ptr + slot, 1 - tl.load(par_ptr + slot))


# --------------------------------------------------------------------------- module


@dataclass
class _ConvSpec:
    weight: torch.Tensor  # [K * C_in, C_out] tap-major
    c_in: int
    k: int
    d: int
    hist: torch.Tensor | None = None  # [2, S, H, C_in]


def _conv_taps(w: torch.Tensor) -> torch.Tensor:
    """``[C_out, C_in, K]`` -> ``[K * C_in, C_out]``."""
    c_out, c_in, k = w.shape
    return w.permute(2, 1, 0).reshape(k * c_in, c_out)


def _snake_params(module) -> tuple[torch.Tensor, torch.Tensor]:
    # SnakeBeta: x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) * x) ** 2
    alpha = module.alpha.detach().float()
    beta = module.beta.detach().float()
    return torch.exp(alpha).contiguous(), (1.0 / (torch.exp(beta) + 1e-9)).contiguous()


class StreamingCodecDecoder(nn.Module):
    """Frame-exact stateful decoder built from a loaded ``Qwen3TTSTokenizerV2Decoder``."""

    RING = 128  # KV ring positions; a call may carry up to RING - window + 1 frames

    def __init__(self, decoder: nn.Module, num_slots: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        cfg = decoder.config
        dev = next(decoder.parameters()).device
        dtype = dtype or next(decoder.parameters()).dtype
        self.dtype, self.device = dtype, dev
        self.num_slots = num_slots + 1  # last slot: scratch for padding rows
        self.scratch_slot = num_slots
        S = self.num_slots
        self.nq = int(cfg.num_quantizers)
        self.cb = int(cfg.codebook_size)
        self.half = int(cfg.codebook_dim) // 2
        self.hidden = int(cfg.hidden_size)
        self.heads = int(cfg.num_attention_heads)
        self.head_dim = int(cfg.head_dim)
        self.window = int(cfg.sliding_window)
        self.theta = float(cfg.rope_theta)
        self.eps = float(cfg.rms_norm_eps)
        self.max_frames = self.RING - self.window + 1
        self.spf = int(decoder.total_upsample)
        if self.heads != int(cfg.num_key_value_heads):
            raise ValueError("streaming codec decoder assumes no grouped KV")

        f32 = lambda t: t.detach().to(device=dev, dtype=torch.float32)  # noqa: E731
        cast = lambda t: t.to(dtype).contiguous()  # noqa: E731

        def hist(h: int, c: int) -> torch.Tensor:
            return torch.zeros(2, S, h, c, device=dev, dtype=dtype)

        # RVQ books, pre-divided by usage: [NQ * CB, HALF]
        q = decoder.quantizer
        books = []
        for vq in [q.rvq_first.vq.layers[0]] + list(q.rvq_rest.vq.layers):
            cbk = vq._codebook
            books.append(f32(cbk.embedding_sum) / f32(cbk.cluster_usage).clamp(min=cbk.epsilon)[:, None])
        self.books = cast(torch.cat(books))
        w_rvq = torch.cat(
            [f32(q.rvq_first.output_proj.weight)[:, :, 0], f32(q.rvq_rest.output_proj.weight)[:, :, 0]], 1
        )
        # pre_conv (k=3) and input_proj folded onto the embedding concat
        pre_w, pre_b = f32(decoder.pre_conv.conv.weight), f32(decoder.pre_conv.conv.bias)
        pt = decoder.pre_transformer
        in_w, in_b = f32(pt.input_proj.weight), f32(pt.input_proj.bias)
        taps = torch.stack([in_w @ pre_w[:, :, j] @ w_rvq for j in range(pre_w.shape[2])], dim=2)  # [hid, 2H, 3]
        self.pre = _ConvSpec(cast(_conv_taps(taps)), 2 * self.half, 3, 1, hist(2, 2 * self.half))
        self.pre_b = cast(in_w @ pre_b + in_b)

        # transformer
        self.layers = []
        qd = self.heads * self.head_dim
        for layer in pt.layers:
            a = layer.self_attn
            qkv = torch.cat([f32(a.q_proj.weight), f32(a.k_proj.weight), f32(a.v_proj.weight)], 0)
            o = f32(a.o_proj.weight) * f32(layer.self_attn_layer_scale.scale)[:, None]
            gu = torch.cat([f32(layer.mlp.gate_proj.weight), f32(layer.mlp.up_proj.weight)], 0)
            down = f32(layer.mlp.down_proj.weight) * f32(layer.mlp_layer_scale.scale)[:, None]
            self.layers.append(
                dict(
                    ln1=cast(f32(layer.input_layernorm.weight)),
                    ln2=cast(f32(layer.post_attention_layernorm.weight)),
                    qkv=cast(qkv.t()),
                    o=cast(o.t()),
                    gu=cast(gu.t()),
                    down=cast(down.t()),
                    k_ring=torch.zeros(S, self.RING, qd, device=dev, dtype=dtype),
                    v_ring=torch.zeros(S, self.RING, qd, device=dev, dtype=dtype),
                )
            )
        self.norm_w = cast(f32(pt.norm.weight))
        self.out_w = cast(f32(pt.output_proj.weight).t())
        self.out_b = cast(f32(pt.output_proj.bias))

        # upsample: transposed conv (kernel = stride, no overlap) + ConvNeXt
        self.ups = []
        latent = int(cfg.latent_dim)
        pending = torch.zeros(latent, device=dev, dtype=torch.float32)
        for trans, block in decoder.upsample:
            tw = f32(trans.conv.weight)  # [C_in, C_out, r]
            r = tw.shape[2]
            w = tw.permute(0, 2, 1).reshape(latent, r * latent)  # [C_in, r * C_out]
            bias = (pending @ w).view(r, latent) + f32(trans.conv.bias)[None, :]
            dw = block.dwconv.conv
            gamma = f32(block.gamma)
            self.ups.append(
                dict(
                    r=r,
                    w=cast(w),
                    b=cast(bias.reshape(-1)),
                    dw_w=cast(f32(dw.weight)[:, 0, :].t()),
                    dw_b=cast(f32(dw.bias)),
                    ln_w=cast(f32(block.norm.weight)),
                    ln_b=cast(f32(block.norm.bias)),
                    ln_eps=float(block.norm.eps),
                    pw1=cast(f32(block.pwconv1.weight).t()),
                    pw1_b=cast(f32(block.pwconv1.bias)),
                    pw2=cast((f32(block.pwconv2.weight) * gamma[:, None]).t()),
                    hist=hist(_KERNEL - 1, latent),
                )
            )
            pending = f32(block.pwconv2.bias) * gamma

        # conv_in (pending bias of the last ConvNeXt added on load)
        dec = decoder.decoder
        cin = dec[0].conv
        self.conv_in = _ConvSpec(cast(_conv_taps(f32(cin.weight))), latent, _KERNEL, 1, hist(_KERNEL - 1, latent))
        self.conv_in_pb = cast(pending)
        pending = f32(cin.bias)

        # decoder blocks
        self.blocks = []
        for i, blk in enumerate(dec[1:-2]):
            snake, trans, *units = blk.block
            c_in = trans.conv.in_channels
            c_out = trans.conv.out_channels
            rate = trans.conv.stride[0]
            tw = f32(trans.conv.weight)  # [C_in, C_out, 2 * rate]
            entry = dict(
                pb=cast(pending),
                snake=_snake_params(snake),
                rate=rate,
                c_in=c_in,
                c_out=c_out,
                w=cast(tw.permute(0, 2, 1).reshape(c_in, 2 * rate * c_out)),
                b=cast(f32(trans.conv.bias)),
                prev=torch.zeros(2, S, rate, c_out, device=dev, dtype=dtype),
                units=[],
            )
            pending = torch.zeros(c_out, device=dev, dtype=torch.float32)
            for u in units:
                c1, c2 = u.conv1.conv, u.conv2.conv
                entry["units"].append(
                    dict(
                        pb=cast(pending),
                        act1=_snake_params(u.act1),
                        conv1=_ConvSpec(
                            cast(_conv_taps(f32(c1.weight))),
                            c_out,
                            _KERNEL,
                            c1.dilation[0],
                            hist((_KERNEL - 1) * c1.dilation[0], c_out),
                        ),
                        b1=cast(f32(c1.bias)),
                        conv1_w4=(
                            cast(f32(c1.weight)).unsqueeze(2).contiguous(memory_format=torch.channels_last)
                            if c1.dilation[0] == 1
                            else None
                        ),
                        act2=_snake_params(u.act2),
                        w2=cast(f32(c2.weight)[:, :, 0].t()),
                    )
                )
                pending = pending + f32(c2.bias)
            self.blocks.append(entry)
        self.out_pb = cast(pending)
        self.out_snake = _snake_params(dec[-2])
        co = dec[-1].conv
        self.out_c = co.in_channels
        self.conv_out_w = cast(f32(co.weight)[0].t())  # [K, C]
        self.conv_out_b = float(f32(co.bias)[0])
        self.conv_out = _ConvSpec(self.conv_out_w, self.out_c, _KERNEL, 1, hist(_KERNEL - 1, self.out_c))
        self.parity = torch.zeros(S, device=dev, dtype=torch.int32)
        self._dummy = torch.zeros(1, device=dev, dtype=dtype)

    def _slot_state(self, slot: int) -> list[torch.Tensor]:
        state = [self.parity[slot : slot + 1]]
        for layer in self.layers:
            state.extend((layer["k_ring"][slot], layer["v_ring"][slot]))
        for spec in (self.pre, self.conv_in, self.conv_out):
            assert spec.hist is not None
            state.append(spec.hist[:, slot])
        for up in self.ups:
            state.append(up["hist"][:, slot])
        for block in self.blocks:
            state.append(block["prev"][:, slot])
            for unit in block["units"]:
                state.append(unit["conv1"].hist[:, slot])
        return state

    def save_slot(self, slot: int) -> list[torch.Tensor]:
        """Offload a preempted request before its runner slot is recycled."""
        return [tensor.to(device="cpu", copy=True) for tensor in self._slot_state(slot)]

    def restore_slot(self, slot: int, state: list[torch.Tensor]) -> None:
        for target, saved in zip(self._slot_state(slot), state, strict=True):
            target.copy_(saved)

    # ------------------------------------------------------------------ helpers
    def _im2col(self, x, spec: _ConvSpec, T, slots, pos, bias=None, snake=None, ext=False):
        k = spec.k
        d = spec.d
        c = x.shape[1]
        h = (k - 1) * d
        B = slots.shape[0]
        if ext:
            col = torch.empty(B * (h + T), c, device=x.device, dtype=self.dtype)
        else:
            col = torch.empty(B * T, k * c, device=x.device, dtype=self.dtype)
        bu = 32
        bc = min(128, triton.next_power_of_2(c))
        grid = (B, triton.cdiv(h + T, bu), triton.cdiv(c, bc))
        a, ib = snake if snake is not None else (self._dummy, self._dummy)
        _im2col_kernel[grid](
            x, bias if bias is not None else self._dummy, a, ib,
            spec.hist if h > 0 else self._dummy, slots, pos, self.parity, col,
            T, self.num_slots, C=c, K=k, D=d, H=h,
            HAS_BIAS=bias is not None, HAS_SNAKE=snake is not None, BU=bu, BC=bc, EXT=ext,
        )  # fmt: skip
        return col

    def _conv_cudnn(self, x, u, T, slots, pos):
        """Undilated unit conv: one source-row pass, then cuDNN on its channels-last view.

        The im2col path writes a ``K``-times patch matrix; cuDNN reads the
        ``[history | new]`` rows directly (same result up to summation order).
        """
        spec = u["conv1"]
        ext = self._im2col(x, spec, T, slots, pos, bias=u["pb"], snake=u["act1"], ext=True)
        b = slots.shape[0]
        c = x.shape[1]
        x4 = ext.view(b, -1, c).permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, H + T], channels-last memory
        y = torch.nn.functional.conv2d(x4, u["conv1_w4"])  # [B, N, 1, T], channels-last memory
        return y.permute(0, 2, 3, 1).reshape(b * T, -1)

    def _snake(self, x, snake, bias):
        n, c = x.shape
        out = torch.empty_like(x)
        bu, bc = 32, min(128, triton.next_power_of_2(c))
        one = torch.zeros(1, device=x.device, dtype=torch.int32)
        _im2col_kernel[(1, triton.cdiv(n, bu), triton.cdiv(c, bc))](
            x, bias, snake[0], snake[1], self._dummy, one, one, self.parity, out,
            n, self.num_slots, C=c, K=1, D=1, H=0, HAS_BIAS=True, HAS_SNAKE=True, BU=bu, BC=bc,
        )  # fmt: skip
        return out

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, codes: torch.Tensor, slots: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``codes`` [B, T, NQ] int, ``slots``/``pos`` [B] int32 -> waveform [B, T * spf] float32.

        ``pos[b]`` is the index of the row's first frame in its stream; 0 starts
        a new stream in ``slots[b]``. Frames beyond ``max_frames`` per call must
        be split by the caller.
        """
        B, T, _ = codes.shape
        if T > self.max_frames:
            raise ValueError(f"at most {self.max_frames} frames per call, got {T}")
        dev, dt = codes.device, self.dtype
        codes = codes.reshape(B * T, self.nq).to(torch.int32).contiguous()
        emb = torch.empty(B * T, 2 * self.half, device=dev, dtype=dt)
        _rvq_gather_kernel[(B * T,)](codes, self.books, emb, NQ=self.nq, CB=self.cb, HALF=self.half)
        x = torch.addmm(self.pre_b, self._im2col(emb, self.pre, T, slots, pos), self.pre.weight)

        hid, qd = self.hidden, self.heads * self.head_dim
        rows = B * T
        h = torch.empty(rows, hid, device=dev, dtype=dt)
        q = torch.empty(rows, qd, device=dev, dtype=dt)
        att = torch.empty(rows, qd, device=dev, dtype=dt)
        a = torch.empty(rows, self.layers[0]["down"].shape[0], device=dev, dtype=dt)
        inter = a.shape[1]
        scale = self.head_dim**-0.5
        for L in self.layers:
            _rms_norm_kernel[(rows,)](x, L["ln1"], h, self.eps, N=hid)
            qkv = torch.mm(h, L["qkv"])
            _rope_kv_write_kernel[(rows, self.heads)](
                qkv, q, L["k_ring"], L["v_ring"], slots, pos, T, self.theta,
                NH=self.heads, HD=self.head_dim, RING=self.RING,
            )  # fmt: skip
            _attend_kernel[(rows, self.heads)](
                q, L["k_ring"], L["v_ring"], att, slots, pos, T, scale,
                NH=self.heads, HD=self.head_dim, RING=self.RING, WINDOW=self.window,
                BK=triton.next_power_of_2(self.window),
            )  # fmt: skip
            x.addmm_(att, L["o"])
            _rms_norm_kernel[(rows,)](x, L["ln2"], h, self.eps, N=hid)
            gu = torch.mm(h, L["gu"])
            _silu_mul_kernel[(rows, triton.cdiv(inter, 1024))](gu, a, N=inter, BN=1024)
            x.addmm_(a, L["down"])
        _rms_norm_kernel[(rows,)](x, self.norm_w, h, self.eps, N=hid)
        y = torch.addmm(self.out_b, h, self.out_w)  # [rows, latent]

        t_rows = T
        for U in self.ups:
            r = U["r"]
            z = torch.addmm(U["b"], y, U["w"]).view(rows * r, -1)
            rows, t_rows = rows * r, t_rows * r
            c = z.shape[1]
            n = torch.empty_like(z)
            _dwconv_ln_kernel[(B, max(t_rows, _KERNEL - 1))](
                z, U["dw_w"], U["dw_b"], U["ln_w"], U["ln_b"], U["hist"], slots, pos, self.parity, n,
                t_rows, self.num_slots, U["ln_eps"], C=c, K=_KERNEL,
            )  # fmt: skip
            p1 = torch.mm(n, U["pw1"])
            _bias_gelu_kernel[(rows, triton.cdiv(p1.shape[1], 1024))](p1, U["pw1_b"], N=p1.shape[1], BN=1024)
            z.addmm_(p1, U["pw2"])
            y = z

        x = torch.mm(self._im2col(y, self.conv_in, t_rows, slots, pos, bias=self.conv_in_pb), self.conv_in.weight)
        for blk in self.blocks:
            s = self._snake(x, blk["snake"], blk["pb"])
            zt = torch.mm(s, blk["w"])  # [rows, 2 * rate * c_out]
            r, c_out = blk["rate"], blk["c_out"]
            x = torch.empty(rows * r, c_out, device=dev, dtype=dt)
            bc = min(128, triton.next_power_of_2(c_out))
            _col2im_kernel[(rows, r, triton.cdiv(c_out, bc))](
                zt, blk["b"], blk["prev"], slots, pos, self.parity, x, t_rows, self.num_slots,
                R=r, C=c_out, BC=bc,
            )  # fmt: skip
            rows, t_rows = rows * r, t_rows * r
            for u in blk["units"]:
                if u.get("conv1_w4") is not None:
                    c1 = self._conv_cudnn(x, u, t_rows, slots, pos)
                else:
                    col = self._im2col(x, u["conv1"], t_rows, slots, pos, bias=u["pb"], snake=u["act1"])
                    c1 = torch.mm(col, u["conv1"].weight)
                s2 = self._snake(c1, u["act2"], u["b1"])
                x.addmm_(s2, u["w2"])

        ext = self._im2col(x, self.conv_out, t_rows, slots, pos, bias=self.out_pb, snake=self.out_snake, ext=True)
        wav = torch.empty(B, t_rows, device=dev, dtype=torch.float32)
        _conv_out_kernel[(B, triton.cdiv(t_rows, 256))](
            ext, self.conv_out_w, self.conv_out_b, wav, t_rows,
            C=self.out_c, K=_KERNEL, BT=256, CP=triton.next_power_of_2(self.out_c),
        )  # fmt: skip
        _flip_kernel[(B,)](self.parity, slots)
        return wav


class StreamingDecodeGraphs:
    """CUDA graphs of ``StreamingCodecDecoder`` for a fixed frame count per batch bucket."""

    def __init__(self, sd: StreamingCodecDecoder, batch_sizes: list[int], frames: int = 1) -> None:
        self.sd, self.frames = sd, frames
        self.graphs: dict[int, tuple] = {}
        dev = sd.device
        pool = torch.cuda.graph_pool_handle()
        for bsz in sorted(set(batch_sizes)):
            codes = torch.zeros(bsz, frames, sd.nq, dtype=torch.int32, device=dev)
            slots = torch.full((bsz,), sd.scratch_slot, dtype=torch.int32, device=dev)
            pos = torch.zeros(bsz, dtype=torch.int32, device=dev)
            sd(codes, slots, pos)
            current_omni_platform.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = sd(codes, slots, pos)
            self.graphs[bsz] = (graph, codes, slots, pos, out)
        self.sizes = sorted(self.graphs)

    def __call__(self, codes: torch.Tensor, slots: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``codes`` [n, frames, NQ], ``slots``/``pos`` [n] on device -> waveform view [n, frames * spf]."""
        n = int(codes.shape[0])
        size = next((b for b in self.sizes if b >= n), None)
        if size is None:
            return self.sd(codes, slots.to(torch.int32), pos.to(torch.int32))
        graph, s_codes, s_slots, s_pos, out = self.graphs[size]
        s_codes[:n].copy_(codes)
        s_slots[:n].copy_(slots)
        s_pos[:n].copy_(pos)
        if n < size:
            s_slots[n:].fill_(self.sd.scratch_slot)
            s_pos[n:].zero_()
        graph.replay()
        return out[:n]
