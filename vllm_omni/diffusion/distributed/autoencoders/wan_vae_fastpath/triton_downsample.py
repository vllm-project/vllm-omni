# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803
"""Spatial input assembly and residual average-downsampling for the Wan encoder.

The pad kernel only moves bytes. The shortcut kernel changes the reduction
order and is consequently used only by the ``channels_last`` level. Both read
strided 5D inputs directly, including temporal slices and spatial tiles.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK = 1024
_WARPS = 4


if HAS_TRITON:

    @triton.jit
    def _spatial_pad_kernel(
        X,
        OUT,
        total,
        C: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        S2: tl.constexpr,
        S3: tl.constexpr,
        S4: tl.constexpr,
        CL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        if CL:
            c = i % C
            w = i // C % (W + 1)
            h = i // (C * (W + 1)) % (H + 1)
            bt = i // (C * (W + 1) * (H + 1))
        else:
            w = i % (W + 1)
            h = i // (W + 1) % (H + 1)
            c = i // ((W + 1) * (H + 1)) % C
            bt = i // ((W + 1) * (H + 1) * C)
        offset = (bt // T) * S0 + c * S1 + (bt % T) * S2 + h * S3 + w * S4
        x = tl.load(X + offset, mask=(i < total) & (h < H) & (w < W), other=0)
        tl.store(OUT + i, x, mask=i < total)

    @triton.jit
    def _avg_down_add_kernel(
        X,
        MAIN,
        OUT,
        total,
        C: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        INPUT_T: tl.constexpr,
        FT: tl.constexpr,
        FS: tl.constexpr,
        GROUP: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        S2: tl.constexpr,
        S3: tl.constexpr,
        S4: tl.constexpr,
        M0: tl.constexpr,
        M1: tl.constexpr,
        M2: tl.constexpr,
        M3: tl.constexpr,
        M4: tl.constexpr,
        CL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        if CL:
            c = i % C
            w = i // C % W
            h = i // (C * W) % H
            t = i // (C * W * H) % T
        else:
            w = i % W
            h = i // W % H
            t = i // (W * H) % T
            c = i // (W * H * T) % C
        b = i // (C * T * H * W)
        pad_t = (FT - INPUT_T % FT) % FT
        acc = tl.full((BLOCK,), 0, tl.float32)
        for g in tl.static_range(GROUP):
            # AvgDown3D flattens (input_channel, factor_t, factor_h, factor_w)
            # before grouping channels. Its channel order differs from patchify.
            channel = c * GROUP + g
            sw = w * FS + channel % FS
            sh = h * FS + channel // FS % FS
            st = t * FT + channel // (FS * FS) % FT - pad_t
            sc = channel // (FT * FS * FS)
            offset = b * S0 + sc * S1 + st * S2 + sh * S3 + sw * S4
            value = tl.load(X + offset, mask=(i < total) & (st >= 0) & (st < INPUT_T), other=0)
            acc += value.to(tl.float32)
        # Keep the materialization boundary between mean and residual addition.
        mean = (acc / GROUP).to(X.dtype.element_ty).to(tl.float32)
        main = tl.load(MAIN + b * M0 + c * M1 + t * M2 + h * M3 + w * M4, mask=i < total, other=0)
        out = (main.to(tl.float32) + mean).to(OUT.dtype.element_ty)
        tl.store(OUT + i, out, mask=i < total)


def _supported(x: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and x.is_cuda
        and x.dtype in _DTYPES
        and x.ndim == 5
        and x.numel() > 0
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


def spatial_downsample_input(x: torch.Tensor) -> torch.Tensor | None:
    """Merge batch/frames and pad right/bottom by one, without an intermediate copy.

    Channel-contiguous inputs produce canonical NHWC, including singleton
    batches and strided tiles. The lossless caller retains the upstream path
    for such inputs because reshape can change its suggested memory format.
    """
    if not _supported(x):
        return None
    b, c, t, h, w = x.shape
    # A spatial slice of NDHWC still suggests channels-last to ZeroPad2d even
    # though the slice itself is not contiguous.
    channels_last = c > 1 and x.stride(1) == 1
    out = torch.empty(
        (b * t, c, h + 1, w + 1),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last if channels_last else torch.contiguous_format,
    )
    with torch.accelerator.device_index(x.device.index):
        _spatial_pad_kernel[(triton.cdiv(out.numel(), _BLOCK),)](
            x,
            out,
            out.numel(),
            c,
            t,
            h,
            w,
            *x.stride(),
            CL=channels_last,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    return out


def avg_down3d_add(
    main: torch.Tensor, source: torch.Tensor, factor_t: int, factor_s: int, group_size: int
) -> torch.Tensor | None:
    """``main + AvgDown3D(source)`` with a fused fp32 reduction; tolerance-based only."""
    if not _supported(source) or not _supported(main) or source.device != main.device or source.dtype != main.dtype:
        return None
    if factor_t not in (1, 2) or factor_s not in (1, 2) or group_size not in (1, 4):
        return None
    b, c_in, t_in, h_in, w_in = source.shape
    if h_in % factor_s or w_in % factor_s or (c_in * factor_t * factor_s**2) % group_size:
        return None
    c = c_in * factor_t * factor_s**2 // group_size
    t, h, w = (t_in + factor_t - 1) // factor_t, h_in // factor_s, w_in // factor_s
    if main.shape != (b, c, t, h, w):
        return None
    channels_last = c > 1 and main.is_contiguous(memory_format=torch.channels_last_3d)
    out = torch.empty(
        main.shape,
        dtype=main.dtype,
        device=main.device,
        memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format,
    )
    with torch.accelerator.device_index(source.device.index):
        _avg_down_add_kernel[(triton.cdiv(out.numel(), _BLOCK),)](
            source,
            main,
            out,
            out.numel(),
            c,
            t,
            h,
            w,
            t_in,
            factor_t,
            factor_s,
            group_size,
            *source.stride(),
            *main.stride(),
            CL=channels_last,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
            enable_fp_fusion=False,
        )
    return out
