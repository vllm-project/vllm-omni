# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/codec_triton/kernels.py @ d3612b0.
# The codec implements the VibeVoice acoustic tokenizer architecture (Copyright (c) 2025 Microsoft, MIT License).
# ruff: noqa: N803
"""Triton kernels for the VibeVoice codec layers, in bfloat16 on frame-major activations ``[B, T, C]``.

Frame-major memory keeps every per-frame access contiguous over channels: the
seven depthwise taps are whole-row shifts, the MLP is a plain ``[T, C] x [C, H]``
GEMM, and the window of a strided convolution is one contiguous run of frames.
Each layer reads its left context from ``state`` and writes the context of the
next call to ``new_state`` (two distinct tensors ``[B, context, C_in]``).

Arithmetic follows the PyTorch modules: float32 accumulation, rounded to
bfloat16 wherever the modules store a tensor. Three deliberate differences
stay within rounding noise: the fused block kernel uses the tanh form of GELU
and keeps its hidden activations in float32 up to the GELU output, and a
transposed convolution adds both overlapping frames and the bias before
rounding once.

``block``           RMSNorm -> causal depthwise conv -> scaled residual -> RMSNorm -> MLP -> scaled residual.
                    Up to ``FUSED_MAX_CHANNELS`` channels one kernel does everything after the first RMSNorm,
                    with the 4C hidden activations never leaving the chip; wider blocks run the two linears in cuBLAS.
``conv``            dense causal conv (stem, downsampling, head) as a GEMM over contiguous windows.
``conv_transpose``  causal transposed conv (kernel 2S, stride S) as a stride-1 conv over the windows
                    ``(x[t-1], x[t])`` with ``S * C_out`` outputs per input frame.

Every layer function takes ``next_norm`` (the first RMSNorm weight of a following block) and returns
``(out, normed)``; ``normed`` is that RMSNorm of ``out`` when the layer could compute it in the same
kernel, else ``None``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

FUSED_MAX_CHANNELS = 128  # widest block whose MLP runs inside the fused kernel
CUBLAS_MIN_CHANNELS = 1024  # convs with at least this many input channels run as cuBLAS GEMMs
NORM_MAX_CHANNELS = 256  # widest output a conv kernel holds in one tile, as the fused RMSNorm needs

Params = dict[str, torch.Tensor | int | float]


@triton.jit
def _rnd(x):
    """Round float32 to bfloat16 and back, where the PyTorch module would store a tensor."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rmsnorm_rows(x, weight, eps, C: tl.constexpr):
    """x [R, C] float32 -> weight * x * rsqrt(mean(x^2) + eps), rounded like the PyTorch module."""
    ms = tl.sum(x * x, axis=1) / C
    return _rnd(_rnd(x * tl.math.rsqrt(ms + eps)[:, None]) * weight[None, :])


@triton.jit
def _gelu(x):
    """Tanh-form GELU of float32 x on the hardware tanh approximation."""
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    th = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [u], dtype=tl.float32, is_pure=True, pack=1)
    return 0.5 * x * (1.0 + th)


# ---------------------------------------------------------------- RMSNorm


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, R, eps, C: tl.constexpr, BC: tl.constexpr, BR: tl.constexpr):
    """RMSNorm of the rows of x [R, C]; when BC < C in two passes over BC-channel slices."""
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    m = (r < R)[:, None]
    if BC == C:
        c = tl.arange(0, C)
        x = tl.load(x_ptr + r[:, None] * C + c[None, :], mask=m, other=0.0).to(tl.float32)
        h = _rmsnorm_rows(x, tl.load(w_ptr + c).to(tl.float32), eps, C)
        tl.store(out_ptr + r[:, None] * C + c[None, :], h.to(tl.bfloat16), mask=m)
    else:
        ss = tl.zeros([BR], tl.float32)
        for c0 in range(0, C, BC):
            c = c0 + tl.arange(0, BC)
            x = tl.load(x_ptr + r[:, None] * C + c[None, :], mask=m, other=0.0).to(tl.float32)
            ss += tl.sum(x * x, axis=1)
        inv = tl.math.rsqrt(ss / C + eps)
        for c0 in range(0, C, BC):
            c = c0 + tl.arange(0, BC)
            x = tl.load(x_ptr + r[:, None] * C + c[None, :], mask=m, other=0.0).to(tl.float32)
            h = _rnd(_rnd(x * inv[:, None]) * tl.load(w_ptr + c).to(tl.float32)[None, :])
            tl.store(out_ptr + r[:, None] * C + c[None, :], h.to(tl.bfloat16), mask=m)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension of x ``[B, T, C]``."""
    C = x.shape[-1]
    R = x.numel() // C
    out = torch.empty_like(x)
    bc = C if C <= 256 else 128
    br = min(128, max(8, 4096 // bc))
    _rmsnorm_kernel[(triton.cdiv(R, br),)](x, weight, out, R, eps, C=C, BC=bc, BR=br, num_warps=4)
    return out


# ---------------------------------------------------------------- depthwise taps (shared)


@triton.jit
def _depthwise_taps(x_ptr, state_ptr, w_ptr, b, t, c, t0, T, C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    """Causal depthwise conv of x [B, T, C] at frames t, channels c -> float32 [BT, len(c)].

    Frames before the chunk come from state [B, K-1, C]; tiles that need neither the state nor a row
    mask take a fast path of plain shifted-row loads. w: [K, C]."""
    PAD: tl.constexpr = K - 1
    ptr = x_ptr + b * T * C + t[:, None] * C + c[None, :]
    acc = tl.zeros([BT, c.shape[0]], tl.float32)
    if (t0 >= PAD) & (t0 + BT <= T):
        for k in tl.static_range(K):
            acc += tl.load(w_ptr + k * C + c).to(tl.float32)[None, :] * tl.load(ptr + (k - PAD) * C).to(tl.float32)
    else:
        tmask = t < T
        for k in tl.static_range(K):
            p = t + (k - PAD)
            before = p < 0
            h = tl.load(ptr + (k - PAD) * C, mask=(tmask & ~before)[:, None], other=0.0)
            hs = tl.load(
                state_ptr + b * PAD * C + (p + PAD)[:, None] * C + c[None, :], mask=(tmask & before)[:, None], other=0.0
            )
            h = tl.where(before[:, None], hs, h)
            acc += tl.load(w_ptr + k * C + c).to(tl.float32)[None, :] * h.to(tl.float32)
    return acc


@triton.jit
def _store_new_state(x_ptr, new_state_ptr, b, t, c, t0, T, C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    """new_state [B, K-1, C] = the last K-1 frames of x (when T < K-1 the host shifts the older frames)."""
    PAD: tl.constexpr = K - 1
    if t0 + BT > T - PAD:
        j = t - T + PAD
        m = ((t < T) & (j >= 0))[:, None]
        h = tl.load(x_ptr + b * T * C + t[:, None] * C + c[None, :], mask=m, other=0.0)
        tl.store(new_state_ptr + b * PAD * C + j[:, None] * C + c[None, :], h, mask=m)


# ---------------------------------------------------------------- narrow block: one fused kernel


@triton.jit
def _block_fused_kernel(
    x_ptr,
    hn_ptr,
    state_ptr,
    new_state_ptr,
    out_ptr,
    dw_w,
    dw_b,
    gamma,
    ffn_norm_w,
    w1t,
    b1,
    w2t,
    b2,
    ffn_gamma,
    T,
    eps,
    next_norm_w,
    hn_next_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    NEXT_NORM: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    """A block after its first RMSNorm (hn): taps, residual, RMSNorm, MLP in BH-wide hidden slices, residual.

    NEXT_NORM: also write the next block's first RMSNorm of the output to hn_next."""
    b = tl.program_id(1).to(tl.int64)
    t0 = tl.program_id(0) * BT
    t = t0 + tl.arange(0, BT)
    c = tl.arange(0, C)
    acc = _depthwise_taps(hn_ptr, state_ptr, dw_w, b, t, c, t0, T, C, K, BT)
    _store_new_state(hn_ptr, new_state_ptr, b, t, c, t0, T, C, K, BT)
    offs = b * T * C + t[:, None] * C + c[None, :]
    m = (t < T)[:, None]
    x0 = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    conv = _rnd(acc + tl.load(dw_b + c).to(tl.float32)[None, :])
    x1 = (x0 + _rnd(conv * tl.load(gamma + c).to(tl.float32)[None, :])).to(tl.bfloat16)
    a = _rmsnorm_rows(x1.to(tl.float32), tl.load(ffn_norm_w + c).to(tl.float32), eps, C).to(tl.bfloat16)
    y = tl.zeros([BT, C], tl.float32)
    for h0 in range(0, H, BH):
        n = h0 + tl.arange(0, BH)
        hid = tl.dot(a, tl.load(w1t + c[:, None] * H + n[None, :])) + tl.load(b1 + n).to(tl.float32)[None, :]
        y += tl.dot(_gelu(hid).to(tl.bfloat16), tl.load(w2t + n[:, None] * C + c[None, :]))
    y = _rnd(y + tl.load(b2 + c).to(tl.float32)[None, :])
    out = (x1.to(tl.float32) + _rnd(y * tl.load(ffn_gamma + c).to(tl.float32)[None, :])).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=m)
    if NEXT_NORM:
        hn2 = _rmsnorm_rows(out.to(tl.float32), tl.load(next_norm_w + c).to(tl.float32), eps, C)
        tl.store(hn_next_ptr + offs, hn2.to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------- wide block: kernels around cuBLAS linears


@triton.jit
def _mixer_kernel(
    x_ptr,
    hn_ptr,
    state_ptr,
    new_state_ptr,
    out_ptr,
    xn_ptr,
    dw_w,
    dw_b,
    gamma,
    ffn_norm_w,
    T,
    eps,
    C: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    """Mixer half of a block on full-width rows: out = x + gamma * depthwise(hn), and xn = RMSNorm(out)."""
    b = tl.program_id(1).to(tl.int64)
    c = tl.arange(0, C)
    t0 = tl.program_id(0) * BT
    t = t0 + tl.arange(0, BT)
    acc = _depthwise_taps(hn_ptr, state_ptr, dw_w, b, t, c, t0, T, C, K, BT)
    _store_new_state(hn_ptr, new_state_ptr, b, t, c, t0, T, C, K, BT)
    offs = b * T * C + t[:, None] * C + c[None, :]
    m = (t < T)[:, None]
    x0 = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    conv = _rnd(acc + tl.load(dw_b + c).to(tl.float32)[None, :])
    x1 = _rnd(x0 + _rnd(conv * tl.load(gamma + c).to(tl.float32)[None, :]))
    tl.store(out_ptr + offs, x1.to(tl.bfloat16), mask=m)
    xn = _rmsnorm_rows(x1, tl.load(ffn_norm_w + c).to(tl.float32), eps, C)
    tl.store(xn_ptr + offs, xn.to(tl.bfloat16), mask=m)


@triton.jit
def _residual_kernel(
    y_ptr,
    x1_ptr,
    gamma,
    out_ptr,
    next_norm_w,
    hn_next_ptr,
    R,
    eps,
    C: tl.constexpr,
    BR: tl.constexpr,
    NEXT_NORM: tl.constexpr,
):
    """End of a wide block on [BR, C] rows: out = x1 + gamma * y; NEXT_NORM: also the next block's RMSNorm."""
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    c = tl.arange(0, C)
    offs = r[:, None] * C + c[None, :]
    m = (r < R)[:, None]
    y = tl.load(y_ptr + offs, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offs, mask=m, other=0.0).to(tl.float32)
    out = (x1 + _rnd(y * tl.load(gamma + c).to(tl.float32)[None, :])).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=m)
    if NEXT_NORM:
        hn2 = _rmsnorm_rows(out.to(tl.float32), tl.load(next_norm_w + c).to(tl.float32), eps, C)
        tl.store(hn_next_ptr + offs, hn2.to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------- block


def block_params(module: torch.nn.Module) -> Params:
    """Kernel parameters of a codec ``Block``; the MLP weights are stored in the form its path reads."""

    def w(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(torch.bfloat16).contiguous()

    p: Params = dict(
        norm_w=w(module.norm.weight),
        eps=float(module.norm.eps),
        dw_w=w(module.conv.weight[:, 0].t()),
        dw_b=w(module.conv.bias),
        gamma=w(module.gamma),
        ffn_norm_w=w(module.ffn_norm.weight),
        b1=w(module.fc1.bias),
        b2=w(module.fc2.bias),
        ffn_gamma=w(module.ffn_gamma),
    )
    if module.gamma.numel() <= FUSED_MAX_CHANNELS:
        p.update(w1t=w(module.fc1.weight.t()), w2t=w(module.fc2.weight.t()))  # [C, H], [H, C]
    else:
        p.update(w1=w(module.fc1.weight), w2=w(module.fc2.weight))  # [H, C], [C, H]
    return p


def block(
    p: Params,
    x: torch.Tensor,
    state: torch.Tensor,
    new_state: torch.Tensor,
    normed: torch.Tensor | None,
    next_norm: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """x ``[B, T, C]`` -> block output ``[B, T, C]``; state / new_state ``[B, K-1, C]``.

    ``normed`` is the block's first RMSNorm of x when the previous kernel already produced it."""
    B, T, C = x.shape
    K = p["dw_w"].shape[0]
    pad = K - 1
    if T < pad:  # frames older than this chunk that stay in the state
        new_state[:, : pad - T].copy_(state[:, T:])
    if normed is None:
        normed = rmsnorm(x, p["norm_w"], p["eps"])
    fuse = next_norm is not None
    out = torch.empty_like(x)
    normed_next = torch.empty_like(x) if fuse else out
    norm_w = next_norm if fuse else p["norm_w"]
    if C <= FUSED_MAX_CHANNELS:
        _block_fused_kernel[(triton.cdiv(T, 16), B)](
            x,
            normed,
            state,
            new_state,
            out,
            p["dw_w"],
            p["dw_b"],
            p["gamma"],
            p["ffn_norm_w"],
            p["w1t"],
            p["b1"],
            p["w2t"],
            p["b2"],
            p["ffn_gamma"],
            T,
            p["eps"],
            norm_w,
            normed_next,
            C=C,
            H=p["b1"].numel(),
            K=K,
            NEXT_NORM=fuse,
            BT=16,
            BH=128,
            num_warps=8,
            num_stages=2,
        )
        return out, (normed_next if fuse else None)
    bt = max(2, 4096 // C)
    warps = 4 if C <= 512 else 8
    x1, xn = torch.empty_like(x), torch.empty_like(x)
    _mixer_kernel[(triton.cdiv(T, bt), B)](
        x,
        normed,
        state,
        new_state,
        x1,
        xn,
        p["dw_w"],
        p["dw_b"],
        p["gamma"],
        p["ffn_norm_w"],
        T,
        p["eps"],
        C=C,
        K=K,
        BT=bt,
        num_warps=warps,
    )
    y = F.linear(F.gelu(F.linear(xn.view(B * T, C), p["w1"], p["b1"])), p["w2"], p["b2"])
    _residual_kernel[(triton.cdiv(B * T, bt),)](
        y, x1, p["ffn_gamma"], out, norm_w, normed_next, B * T, p["eps"], C=C, BR=bt, NEXT_NORM=fuse, num_warps=warps
    )
    return out, (normed_next if fuse else None)


# ---------------------------------------------------------------- dense causal conv


@triton.jit
def _stem_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    next_norm_w,
    hn_ptr,
    T,
    eps,
    C_OUT: tl.constexpr,
    K: tl.constexpr,
    NEXT_NORM: tl.constexpr,
    BT: tl.constexpr,
):
    """Stride-1 causal conv of one input channel: x [B, T] -> out [B, T, C_OUT]; state [B, K-1]; w [K, C_OUT].

    NEXT_NORM: also write the next block's first RMSNorm of the output to hn."""
    PAD: tl.constexpr = K - 1
    b = tl.program_id(1).to(tl.int64)
    o = tl.arange(0, C_OUT)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    tmask = t < T
    acc = tl.zeros([BT, C_OUT], tl.float32)
    for k in tl.static_range(K):
        p = t + (k - PAD)
        before = p < 0
        xv = tl.load(x_ptr + b * T + p, mask=tmask & ~before, other=0.0)
        xs = tl.load(state_ptr + b * PAD + p + PAD, mask=tmask & before, other=0.0)
        xv = tl.where(before, xs, xv).to(tl.float32)
        acc += xv[:, None] * tl.load(w_ptr + k * C_OUT + o).to(tl.float32)[None, :]
    out = (acc + tl.load(b_ptr + o).to(tl.float32)[None, :]).to(tl.bfloat16)
    offs = b * T * C_OUT + t[:, None] * C_OUT + o[None, :]
    tl.store(out_ptr + offs, out, mask=tmask[:, None])
    if NEXT_NORM:
        hn = _rmsnorm_rows(out.to(tl.float32), tl.load(next_norm_w + o).to(tl.float32), eps, C_OUT)
        tl.store(hn_ptr + offs, hn.to(tl.bfloat16), mask=tmask[:, None])


@triton.jit
def _window_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    next_norm_w,
    hn_ptr,
    T_IN,
    T_OUT,
    eps,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    PAD: tl.constexpr,
    NEXT_NORM: tl.constexpr,
    NC: tl.constexpr,
    BT: tl.constexpr,
    BO: tl.constexpr,
    BK: tl.constexpr,
):
    """Strided causal conv as one GEMM with reduction K * C_IN: the window of output frame t, input frames
    [t*S - PAD, t*S - PAD + K), is contiguous in x [B, T_IN, C_IN], so operand rows have stride S * C_IN.
    w: [K * C_IN, C_OUT], BO divides C_OUT. Only tiles reaching before the chunk read state [B, PAD, C_IN].

    NEXT_NORM: also write the next block's first RMSNorm of the output to hn, over groups of NC channels
    (NC = C_OUT for a conv; C_OUT / stride for a transposed conv, whose output row holds `stride` frames)."""
    KC: tl.constexpr = K * C_IN
    b = tl.program_id(2).to(tl.int64)
    o = tl.program_id(1) * BO + tl.arange(0, BO)
    t0 = tl.program_id(0) * BT
    t = t0 + tl.arange(0, BT)
    tmask = t < T_OUT
    start = t * S - PAD
    xb = x_ptr + b * T_IN * C_IN
    sb = state_ptr + b * PAD * C_IN
    acc = tl.zeros([BT, BO], tl.float32)
    # The fast / masked choice is made once per tile, so that the pipelined reduction loops stay branch-free.
    if (t0 * S >= PAD) & (t0 + BT <= T_OUT):
        for k0 in range(0, KC, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(xb + start[:, None] * C_IN + kk[None, :])
            acc += tl.dot(a, tl.load(w_ptr + kk[:, None] * C_OUT + o[None, :]))
    else:
        for k0 in range(0, KC, BK):
            kk = k0 + tl.arange(0, BK)
            f = start[:, None] + kk[None, :] // C_IN
            i = kk[None, :] % C_IN
            a = tl.load(xb + f * C_IN + i, mask=(f >= 0) & tmask[:, None], other=0.0)
            a_state = tl.load(sb + (f + PAD) * C_IN + i, mask=(f < 0) & tmask[:, None], other=0.0)
            a = tl.where(f < 0, a_state, a)
            acc += tl.dot(a, tl.load(w_ptr + kk[:, None] * C_OUT + o[None, :]))
    m = tmask[:, None]
    out = (acc + tl.load(b_ptr + o).to(tl.float32)[None, :]).to(tl.bfloat16)
    offs = b * T_OUT * C_OUT + t[:, None] * C_OUT + o[None, :]
    tl.store(out_ptr + offs, out, mask=m)
    if NEXT_NORM:
        if NC == BO:
            hn = _rmsnorm_rows(out.to(tl.float32), tl.load(next_norm_w + o % NC).to(tl.float32), eps, NC)
        else:  # BO // NC output frames per row
            v = tl.reshape(out.to(tl.float32), [BT * (BO // NC), NC])
            hn = tl.reshape(_rmsnorm_rows(v, tl.load(next_norm_w + tl.arange(0, NC)).to(tl.float32), eps, NC), [BT, BO])
        tl.store(hn_ptr + offs, hn.to(tl.bfloat16), mask=m)


def _window(
    x: torch.Tensor,
    state: torch.Tensor,
    p: Params,
    out: torch.Tensor,
    norm_w: torch.Tensor,
    normed: torch.Tensor,
    eps: float,
    *,
    K: int,
    S: int,
    NEXT_NORM: bool,
    NC: int,
) -> None:
    """Launch ``_window_kernel`` for x ``[B, T_in, C_in]`` -> out ``[B, T_out, C_out]`` with its tile sizes."""
    (B, t_in, c_in), (_, t_out, c_out) = x.shape, out.shape
    bo = max(NC, 64) if NEXT_NORM else 64  # a fused RMSNorm needs whole groups of NC channels in a tile
    bk = next(n for n in (128, 64, 32) if (K * c_in) % n == 0 and n * bo <= 16384)  # operand tiles fit shared memory
    _window_kernel[(triton.cdiv(t_out, 64), triton.cdiv(c_out, bo), B)](
        x,
        state,
        p["w"],
        p["b"],
        out,
        norm_w,
        normed,
        t_in,
        t_out,
        eps,
        C_IN=c_in,
        C_OUT=c_out,
        K=K,
        S=S,
        PAD=K - S,
        NEXT_NORM=NEXT_NORM,
        NC=NC,
        BT=64,
        BO=bo,
        BK=bk,
        num_warps=8,
        num_stages=3,
    )


@triton.jit
def _head_kernel(x_ptr, state_ptr, w_ptr, b_ptr, out_ptr, T, C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    """Stride-1 causal conv to one output channel: out[t] = sum_{k, c} w[k, c] x[t - K + 1 + k, c] + b.

    The taps are shifted whole rows as in the depthwise conv, followed by a sum over channels.
    w: [K, C]; out [B, T]."""
    b = tl.program_id(1).to(tl.int64)
    t0 = tl.program_id(0) * BT
    t = t0 + tl.arange(0, BT)
    acc = _depthwise_taps(x_ptr, state_ptr, w_ptr, b, t, tl.arange(0, C), t0, T, C, K, BT)
    out = tl.sum(acc, axis=1) + tl.load(b_ptr).to(tl.float32)
    tl.store(out_ptr + b * T + t, out.to(tl.bfloat16), mask=t < T)


def conv_params(module: torch.nn.Module) -> Params:
    """Kernel parameters of a dense ``CausalConv1d``: weights ``[K * C_in, C_out]``, or ``[C_out, K * C_in]``
    for the cuBLAS path."""
    c_out, c_in, kernel = module.weight.shape
    weight = module.weight.detach().to(torch.bfloat16)
    if c_in >= CUBLAS_MIN_CHANNELS:
        w = weight.permute(0, 2, 1).reshape(c_out, kernel * c_in)
    else:
        w = weight.permute(2, 1, 0).reshape(kernel * c_in, c_out)
    return dict(
        w=w.contiguous(),
        b=module.bias.detach().to(torch.bfloat16).contiguous(),
        stride=module.stride,
        kernel=kernel,
        c_in=c_in,
        c_out=c_out,
    )


def _keep_context(x: torch.Tensor, state: torch.Tensor, new_state: torch.Tensor) -> None:
    """new_state = the last ``context`` frames of ``state`` followed by ``x``."""
    T, pad = x.shape[1], state.shape[1]
    new_state.copy_(x[:, T - pad :] if T >= pad else torch.cat([state, x], dim=1)[:, T:])


def conv(
    p: Params, x: torch.Tensor, state: torch.Tensor, new_state: torch.Tensor, next_norm: torch.Tensor | None, eps: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """x ``[B, T_in, C_in]`` -> ``[B, T_in // stride, C_out]``; state / new_state ``[B, kernel - stride, C_in]``.

    ``eps`` is the epsilon of the RMSNorm ``next_norm`` belongs to."""
    B, t_in, c_in = x.shape
    K, S, c_out = p["kernel"], p["stride"], p["c_out"]
    t_out = t_in // S
    _keep_context(x, state, new_state)
    if c_in >= CUBLAS_MIN_CHANNELS:
        xp = torch.cat([state, x], dim=1)
        windows = xp.as_strided((B, t_out, K * c_in), (xp.shape[1] * c_in, S * c_in, 1))
        return F.linear(windows, p["w"], p["b"]), None
    if c_out == 1:
        out = torch.empty(B, t_out, 1, device=x.device, dtype=x.dtype)
        _head_kernel[(triton.cdiv(t_out, 64), B)](x, state, p["w"], p["b"], out, t_out, C=c_in, K=K, BT=64, num_warps=8)
        return out, None
    fuse = next_norm is not None and c_out <= NORM_MAX_CHANNELS
    out = torch.empty(B, t_out, c_out, device=x.device, dtype=x.dtype)
    normed = torch.empty_like(out) if fuse else out
    norm_w = next_norm if fuse else p["b"]
    if c_in == 1:
        _stem_kernel[(triton.cdiv(t_out, 256), B)](
            x,
            state,
            p["w"],
            p["b"],
            out,
            norm_w,
            normed,
            t_out,
            eps,
            C_OUT=c_out,
            K=K,
            NEXT_NORM=fuse,
            BT=256,
            num_warps=4,
        )
    else:
        _window(x, state, p, out, norm_w, normed, eps, K=K, S=S, NEXT_NORM=fuse, NC=c_out)
    return out, (normed if fuse else None)


# ---------------------------------------------------------------- causal transposed conv


def conv_transpose_params(module: torch.nn.Module) -> Params:
    """Kernel parameters of a ``CausalConvTranspose1d`` (kernel 2S, stride S) as a two-frame window conv.

    Output frame ``t*S + r`` is ``x[t] W[:, :, r] + x[t-1] W[:, :, r + S] + bias``, so with output rows of S
    frames the window ``(x[t-1], x[t])`` maps to ``S * C_out`` outputs ordered ``(r, channel)``."""
    c_in, c_out, kernel = module.weight.shape
    S = module.stride
    w = module.weight.detach().to(torch.bfloat16).view(c_in, c_out, 2, S)
    w = torch.stack([w[:, :, 1], w[:, :, 0]]).permute(0, 1, 3, 2).reshape(2 * c_in, S * c_out)
    if c_in >= CUBLAS_MIN_CHANNELS:
        w = w.t()
    return dict(
        w=w.contiguous(),
        b=module.bias.detach().to(torch.bfloat16).repeat(S).contiguous(),
        stride=S,
        c_in=c_in,
        c_out=c_out,
    )


def conv_transpose(
    p: Params, x: torch.Tensor, state: torch.Tensor, new_state: torch.Tensor, next_norm: torch.Tensor | None, eps: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """x ``[B, T, C_in]`` -> ``[B, T * stride, C_out]``; state / new_state ``[B, 1, C_in]`` (the last input frame)."""
    B, T, c_in = x.shape
    S, c_out = p["stride"], p["c_out"]
    _keep_context(x, state, new_state)
    if c_in >= CUBLAS_MIN_CHANNELS:
        xp = torch.cat([state, x], dim=1)
        windows = xp.as_strided((B, T, 2 * c_in), (xp.shape[1] * c_in, c_in, 1))
        return F.linear(windows, p["w"], p["b"]).view(B, T * S, c_out), None
    fuse = next_norm is not None and c_out <= NORM_MAX_CHANNELS
    out = torch.empty(B, T, S * c_out, device=x.device, dtype=x.dtype)
    normed = torch.empty_like(out) if fuse else out
    _window(x, state, p, out, next_norm if fuse else p["b"], normed, eps, K=2, S=1, NEXT_NORM=fuse, NC=c_out)
    return out.view(B, T * S, c_out), (normed.view(B, T * S, c_out) if fuse else None)
