# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Spatial input assembly and residual average-downsampling for the Wan encoder.

The pad kernel only moves bytes. The shortcut kernel changes the reduction
order and is consequently used only by the ``channels_last`` level. Both read
strided 5D inputs directly, including temporal slices and spatial tiles.

Spatial sizes and strides are runtime arguments, so a new resolution, batch
size or tile edge reuses the compiled kernels. Per-model values (channels,
chunk frames, downsampling factors) and the layout stay specialized. A 3D grid
(row block, row, plane) leaves only constant divisions per element.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK = 1024
# Channels-first rows hold only ``width`` (+1) elements; a smaller block wastes fewer lanes.
_ROW_BLOCK = 256
# Pad kernel channels-first tile width (pixels); rows fill the rest of ``_BLOCK``.
_PAD_BLOCK_W = 128
_WARPS = 4
# CUDA limit for grid axes 1 and 2 (rows and planes); larger inputs decline.
_MAX_GRID_YZ = 65535


if HAS_TRITON:

    @triton.jit
    def _spatial_pad_kernel(
        X,
        OUT,
        H,
        W,
        S0,
        S1,
        S2,
        S3,
        S4,
        C: tl.constexpr,
        T: tl.constexpr,
        CL: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """One ``(BLOCK_R, BLOCK_I)`` tile of the padded output, contiguous along its inner axis.

        Channels-last: rows are pixels of one output row and the inner axis is
        channels; grid ``(w_blocks * C // BLOCK_I, H + 1, b * T)``.
        Channels-first: rows are output rows and the inner axis is pixels; grid
        ``(w_blocks, h_blocks, b * T * C)``.
        """
        plane = tl.program_id(2).to(tl.int64)
        rows = tl.arange(0, BLOCK_R)[:, None]
        inner = tl.arange(0, BLOCK_I)[None, :]
        if CL:
            channel_blocks: tl.constexpr = C // BLOCK_I
            bt = plane
            c = tl.program_id(0) % channel_blocks * BLOCK_I + inner
            w = (tl.program_id(0) // channel_blocks * BLOCK_R + rows).to(tl.int64)
            h = tl.program_id(1).to(tl.int64)
            out_mask = w < W + 1
            out_offset = ((bt * (H + 1) + h) * (W + 1) + w) * C + c
        else:
            bt = plane // C
            c = plane % C
            w = tl.program_id(0).to(tl.int64) * BLOCK_I + inner
            h = tl.program_id(1).to(tl.int64) * BLOCK_R + rows
            out_mask = (h < H + 1) & (w < W + 1)
            out_offset = (plane * (H + 1) + h) * (W + 1) + w
        offset = (bt // T) * S0 + c * S1 + (bt % T) * S2 + h * S3 + w * S4
        x = tl.load(X + offset, mask=out_mask & (h < H) & (w < W), other=0)
        tl.store(OUT + out_offset, x, mask=out_mask)

    @triton.jit
    def _avg_down_add_kernel(
        X,
        MAIN,
        OUT,
        H,
        W,
        S0,
        S1,
        S2,
        S3,
        S4,
        M0,
        M1,
        M2,
        M3,
        M4,
        C: tl.constexpr,
        T: tl.constexpr,
        INPUT_T: tl.constexpr,
        FT: tl.constexpr,
        FS: tl.constexpr,
        GROUP: tl.constexpr,
        CL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Axis 0: block of one output row; axis 1: output row h; axis 2: plane
        # (b * T + t for channels-last, (b * C + c) * T + t for channels-first).
        h = tl.program_id(1).to(tl.int64)
        plane = tl.program_id(2).to(tl.int64)
        j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        if CL:
            t = plane % T
            b = plane // T
            w = j // C
            c = j % C
            row_mask = j < W * C
            out_offset = (plane * H + h) * W * C + j
        else:
            t = plane % T
            c = plane // T % C
            b = plane // (T * C)
            w = j
            row_mask = j < W
            out_offset = (plane * H + h) * W + j
        pad_t: tl.constexpr = (FT - INPUT_T % FT) % FT
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
            value = tl.load(X + offset, mask=row_mask & (st >= 0) & (st < INPUT_T), other=0)
            acc += value.to(tl.float32)
        # Keep the materialization boundary between mean and residual addition.
        mean = (acc / GROUP).to(X.dtype.element_ty).to(tl.float32)
        main = tl.load(MAIN + b * M0 + c * M1 + t * M2 + h * M3 + w * M4, mask=row_mask, other=0)
        out = (main.to(tl.float32) + mean).to(OUT.dtype.element_ty)
        tl.store(OUT + out_offset, out, mask=row_mask)


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
    planes = b * t if channels_last else b * t * c
    if h + 1 > _MAX_GRID_YZ or planes > _MAX_GRID_YZ:
        return None
    out = torch.empty(
        (b * t, c, h + 1, w + 1),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last if channels_last else torch.contiguous_format,
    )
    with torch.accelerator.device_index(x.device.index):
        if channels_last:
            # Largest power-of-two channel block (<= 64) dividing C: no idle lanes.
            block_i = min(64, c & -c)
            block_r = _BLOCK // block_i
            grid = (triton.cdiv(w + 1, block_r) * (c // block_i), h + 1, planes)
        else:
            block_i, block_r = _PAD_BLOCK_W, _BLOCK // _PAD_BLOCK_W
            grid = (triton.cdiv(w + 1, block_i), triton.cdiv(h + 1, block_r), planes)
        _spatial_pad_kernel[grid](
            x,
            out,
            h,
            w,
            *x.stride(),
            C=c,
            T=t,
            CL=channels_last,
            BLOCK_R=block_r,
            BLOCK_I=block_i,
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
    planes = b * t if channels_last else b * c * t
    if h > _MAX_GRID_YZ or planes > _MAX_GRID_YZ:
        return None
    out = torch.empty(
        main.shape,
        dtype=main.dtype,
        device=main.device,
        memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format,
    )
    with torch.accelerator.device_index(source.device.index):
        block = _BLOCK if channels_last else _ROW_BLOCK
        row = w * c if channels_last else w
        _avg_down_add_kernel[(triton.cdiv(row, block), h, planes)](
            source,
            main,
            out,
            h,
            w,
            *source.stride(),
            *main.stride(),
            C=c,
            T=t,
            INPUT_T=t_in,
            FT=factor_t,
            FS=factor_s,
            GROUP=group_size,
            CL=channels_last,
            BLOCK=block,
            num_warps=_WARPS,
            enable_fp_fusion=False,
        )
    return out
