# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803
"""Normalize directly into a causal convolution's temporal input and next cache.

The lossless path keeps ATen's reduction and mirrors triton_rms_norm's exact
epilogue. The channels-last path mirrors triton_rms_norm_cl's existing math
and tile shape. Only new frames are normalized: history is already normalized,
and missing history is zero-filled without applying bias or normalization.

Spatial padding remains the responsibility of the caller's bitwise-verified
convolution path. Unsupported inputs decline without changing any cache state.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

from . import triton_data_movement as dm
from . import triton_rms_norm as rn
from . import triton_rms_norm_cl as cl

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

if HAS_TRITON:

    @triton.jit
    def _norm_scale_cat_time_kernel(
        X,
        DENOM,
        GAMMA,
        CACHE,
        OUT,
        KEEP,
        scale,
        C: tl.constexpr,
        T: tl.constexpr,
        S: tl.constexpr,
        PAD: tl.constexpr,
        TC: tl.constexpr,
        TK: tl.constexpr,
        X0: tl.constexpr,
        X1: tl.constexpr,
        X2: tl.constexpr,
        K0: tl.constexpr,
        K1: tl.constexpr,
        K2: tl.constexpr,
        SILU: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tiles: tl.constexpr = triton.cdiv(S, BLOCK)
        plane = tl.program_id(0).to(tl.int64) // tiles
        s = tl.program_id(0).to(tl.int64) % tiles * BLOCK + tl.arange(0, BLOCK)
        mask = s < S
        frame = plane % (PAD + T)
        c = plane // (PAD + T) % C
        b = plane // ((PAD + T) * C)
        dtype = X.dtype.element_ty
        value = tl.full((BLOCK,), 0, dtype)
        if frame >= PAD:
            t = frame - PAD
            x = tl.load(X + b * X0 + c * X1 + t * X2 + s, mask=mask, other=0).to(tl.float32)
            d = tl.load(DENOM + (b * T + t) * S + s, mask=mask, other=1)
            g = tl.load(GAMMA + c).to(tl.float32)
            # Same operations and materialization boundaries as rms_norm_scale.
            v = tl.math.div_rn(x, d).to(dtype).to(tl.float32)
            v = (v * scale).to(dtype).to(tl.float32)
            v = (v * g).to(dtype).to(tl.float32)
            v = (v + 0.0).to(dtype)
            if SILU:
                v32 = v.to(tl.float32)
                v = tl.math.div_rn(v32, 1.0 + tldevice.exp(-v32)).to(dtype)
            value = v
        elif TC > 0:
            cached_frame = frame - (PAD - TC)
            if cached_frame >= 0:
                value = tl.load(CACHE + b * K0 + c * K1 + cached_frame * K2 + s, mask=mask, other=0)
        tl.store(OUT + plane * S + s, value, mask=mask)
        if frame >= PAD + T - TK:
            keep_plane = (b * C + c) * TK + frame - (PAD + T - TK)
            tl.store(KEEP + keep_plane * S + s, value, mask=mask)

    @triton.jit
    def _norm_cl_cat_time_kernel(
        X,
        GAMMA,
        BIAS,
        CACHE,
        OUT,
        KEEP,
        scale,
        eps,
        C: tl.constexpr,
        T: tl.constexpr,
        S: tl.constexpr,
        PAD: tl.constexpr,
        TC: tl.constexpr,
        TK: tl.constexpr,
        K0: tl.constexpr,
        K2: tl.constexpr,
        SILU: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        tiles: tl.constexpr = triton.cdiv(S, BLOCK_P)
        plane = tl.program_id(0).to(tl.int64) // tiles
        row = tl.program_id(0).to(tl.int64) % tiles * BLOCK_P + tl.arange(0, BLOCK_P)
        c = tl.arange(0, BLOCK_C)
        mask = (row[:, None] < S) & (c[None, :] < C)
        frame = plane % (PAD + T)
        b = plane // (PAD + T)
        offset = row[:, None] * C + c[None, :]
        dtype = X.dtype.element_ty
        value = tl.full((BLOCK_P, BLOCK_C), 0, dtype)
        if frame >= PAD:
            x = tl.load(X + (b * T + frame - PAD) * S * C + offset, mask=mask, other=0).to(tl.float32)
            if HAS_BIAS:
                bias = tl.load(BIAS + c, mask=c < C, other=0).to(tl.float32)
                x = (x + bias[None, :]).to(dtype).to(tl.float32)
            # Share the standalone epilogue, including BF16 intermediate rounding.
            gamma = tl.load(GAMMA + c, mask=c < C, other=0).to(tl.float32)
            value = cl._normalize_channels_last(x, gamma, scale, eps, dtype, SILU)
        elif TC > 0:
            cached_frame = frame - (PAD - TC)
            if cached_frame >= 0:
                value = tl.load(CACHE + b * K0 + cached_frame * K2 + offset, mask=mask, other=0)
        tl.store(OUT + plane * S * C + offset, value, mask=mask)
        if frame >= PAD + T - TK:
            keep_plane = b * TK + frame - (PAD + T - TK)
            tl.store(KEEP + keep_plane * S * C + offset, value, mask=mask)


def norm_act_cat_time(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    cache: torch.Tensor | None,
    pad_front: int,
    *,
    channels_last: bool,
    silu: bool,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``[zeros | history | norm_act(x)]`` and its last two frames.

    ``channels_last`` selects the existing approximate normalization, not a
    layout conversion. Lossless supports dense channels-first/frame-major
    inputs. Channels-last requires dense NDHWC; both accept sliced history.
    """
    if (
        not HAS_TRITON
        or not x.is_cuda
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or x.ndim != 5
        or x.numel() == 0
        or x.dtype not in _DTYPES
        or pad_front < 0
        or gamma.dtype != x.dtype
        or gamma.device != x.device
        or not gamma.is_contiguous()
    ):
        return None
    b, c, t, h, w = x.shape
    if gamma.numel() != c:
        return None
    if channels_last:
        if c < 2 or c > cl._MAX_CHANNELS or not x.is_contiguous(memory_format=torch.channels_last_3d):
            return None
        if bias is not None and (
            bias.ndim != 1
            or bias.numel() != c
            or bias.device != x.device
            or bias.dtype not in _DTYPES
            or not bias.is_contiguous()
        ):
            return None
    elif bias is not None or not (x.is_contiguous() or x.permute(0, 2, 1, 3, 4).is_contiguous()):
        return None
    tc = 0
    cache_strides = (0, 0, 0)
    if cache is not None:
        if (
            cache.ndim != 5
            or cache.device != x.device
            or cache.dtype != x.dtype
            or cache.shape[:2] != x.shape[:2]
            or cache.shape[3:] != x.shape[3:]
        ):
            return None
        layout = dm._plane_layout(cache)
        expected = (1, h * w * c) if channels_last else (c, h * w)
        if layout is None or layout[:2] != expected:
            return None
        tc = cache.shape[2]
        cache_strides = (cache.stride(0), layout[2], cache.stride(2))
    if tc > pad_front:
        return None

    out_t = pad_front + t
    keep_t = min(2, out_t)
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    out = torch.empty((b, c, out_t, h, w), dtype=x.dtype, device=x.device, memory_format=fmt)
    keep = torch.empty((b, c, keep_t, h, w), dtype=x.dtype, device=x.device, memory_format=fmt)
    with torch.get_device_module().device(x.device):
        if channels_last:
            block_c = triton.next_power_of_2(c)
            block_p = max(1, cl._TILE_ELEMENTS // block_c)
            _norm_cl_cat_time_kernel[(b * out_t * triton.cdiv(h * w, block_p),)](
                x,
                gamma,
                x if bias is None else bias,
                x if cache is None else cache,
                out,
                keep,
                float(scale),
                1e-12,
                c,
                t,
                h * w,
                pad_front,
                tc,
                keep_t,
                cache_strides[0],
                cache_strides[2],
                SILU=silu,
                HAS_BIAS=bias is not None,
                BLOCK_C=block_c,
                BLOCK_P=block_p,
                num_warps=cl._NUM_WARPS,
                enable_fp_fusion=False,
            )
        else:
            denom = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)
            if not denom.is_contiguous():
                denom = denom.contiguous()
            _norm_scale_cat_time_kernel[(b * c * out_t * triton.cdiv(h * w, rn._BLOCK),)](
                x,
                denom,
                gamma,
                x if cache is None else cache,
                out,
                keep,
                float(scale),
                c,
                t,
                h * w,
                pad_front,
                tc,
                keep_t,
                *x.stride()[:3],
                *cache_strides,
                SILU=silu,
                BLOCK=rn._BLOCK,
                num_warps=rn._NUM_WARPS,
                enable_fp_fusion=False,
            )
    return out, keep
