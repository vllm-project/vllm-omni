# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Bit-exact nearest-neighbour 2x spatial upsampling for the Wan VAE decoder.

``WanResample`` doubles the height and width of every frame with
``nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")`` before its
``Conv2d``. With an integer factor of two both ``nearest`` and ``nearest-exact``
map output index ``o`` to input index ``o // 2``, so the operation is a pure
gather: every input element lands in a 2x2 block of the output and no
arithmetic happens. ATen reads channels-first input through a contiguous copy
(the decoder hands it the strided frame-major view ``WanResample`` builds) and
runs its channels-last kernel at about a third of memory bandwidth on GB200; the
kernels here read the input once, in place, and write every output element once.

Channels-first: one program owns ``ROWS`` input rows, loads each as a
contiguous span, interleaves it with itself (``[a, a, b, b, ...]``) and stores
the result to the two identical output rows. Channels-last: one program owns
``BLOCK_W`` pixels x ``BLOCK_C`` channels of one input row and stores each
pixel's channel vector to its four output pixels as contiguous spans.

The wrapper returns ``None`` for anything it does not handle (rank, dtype,
device, layout) so the caller runs ``nn.Upsample.forward``.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_context import in_multi_gpu_spatial_decode

from ._utils import ROW_TILE_WIDTH, _pick_block_width, on_kernel_device, suggests_channels_last

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Input elements loaded per program iteration; each is stored four times, so a
# 2048-element load tile keeps 8192 stores in flight per program.
_TILE_ELEMENTS = 2048
_NUM_WARPS = 4
_MAX_BLOCK_C = 1024
_HAS_INTERLEAVE = HAS_TRITON and hasattr(tl, "interleave")

if HAS_TRITON:

    @triton.jit
    def _upsample2x_rows_kernel(
        x_ptr,
        out_ptr,
        rows,
        channels,
        height,
        width,
        out_width,
        stride_x_batch,
        stride_x_channel,
        stride_x_height,
        ROWS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Channels-first: ``ROWS`` input rows -> two identical, width-doubled output rows each.

        Input rows are contiguous spans (``stride(3) == 1``, ``stride(2) == width``)
        but batch and channel strides are arbitrary, so the frame-major view the
        decoder produces is read without a copy. The output is contiguous, so
        flattened input row ``r`` maps to output rows ``2r`` and ``2r + 1``.
        """
        in_rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_mask = in_rows < rows
        in_height = in_rows % height
        rest = in_rows // height
        channel = rest % channels
        batch = rest // channels
        x_row_offsets = batch * stride_x_batch + channel * stride_x_channel + in_height * stride_x_height
        out_row_offsets = (in_rows * 2) * out_width
        for w0 in range(0, width, BLOCK_W):
            cols = w0 + tl.arange(0, BLOCK_W)
            mask = row_mask[:, None] & (cols < width)[None, :]
            values = tl.load(x_ptr + (x_row_offsets[:, None] + cols[None, :]), mask=mask, other=0)
            doubled = tl.interleave(values, values)
            out_cols = 2 * w0 + tl.arange(0, 2 * BLOCK_W)
            out_mask = row_mask[:, None] & (out_cols < out_width)[None, :]
            out_offsets = out_row_offsets[:, None] + out_cols[None, :]
            tl.store(out_ptr + out_offsets, doubled, mask=out_mask)
            tl.store(out_ptr + (out_offsets + out_width), doubled, mask=out_mask)

    _spatial_upsample2x_rows_kernel = triton.jit(
        _upsample2x_rows_kernel.fn,
        do_not_specialize=[
            "rows",
            "height",
            "width",
            "out_width",
            "stride_x_height",
            "stride_x_channel",
            "stride_x_batch",
        ],
    )

    @triton.jit(
        do_not_specialize=[
            "channels",
            "height",
            "width",
            "width_blocks",
            "stride_x_batch",
            "stride_x_height",
            "stride_x_width",
        ]
    )
    def _upsample2x_pixels_kernel(
        x_ptr,
        out_ptr,
        channels,
        height,
        width,
        width_blocks,
        stride_x_batch,
        stride_x_height,
        stride_x_width,
        PADDED: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Channels-last: ``BLOCK_W`` pixels of one input row -> their 2x2 output pixels.

        Channels are the contiguous dimension, so every pixel's channel vector is
        one contiguous load and four contiguous stores.
        """
        pid = tl.program_id(0).to(tl.int64)
        width_block = pid % width_blocks
        rest = pid // width_blocks
        rows = height + PADDED
        in_height = rest % rows
        batch = rest // rows
        cols = width_block * BLOCK_W + tl.arange(0, BLOCK_W)
        x_row = batch * stride_x_batch + in_height * stride_x_height
        out_row_stride = (2 * (width + PADDED)).to(tl.int64) * channels
        top = 2 * in_height + PADDED
        left = 2 * cols + PADDED
        bottom = top + 1
        right = left + 1
        if PADDED:
            # The extra tile coordinates cover both edges, including the corners.
            top = tl.where(in_height == height, 0, top)
            bottom = tl.where(in_height == height, 2 * height + 1, bottom)
            left = tl.where(cols == width, 0, left)
            right = tl.where(cols == width, 2 * width + 1, right)
        batch_offset = batch * 2 * rows * out_row_stride
        for c0 in range(0, channels, BLOCK_C):
            chans = c0 + tl.arange(0, BLOCK_C)
            mask = (cols < width + PADDED)[:, None] & (chans < channels)[None, :]
            load_mask = mask
            if PADDED:
                load_mask = (in_height < height) & (cols < width)[:, None] & (chans < channels)[None, :]
            values = tl.load(x_ptr + (x_row + cols[:, None] * stride_x_width + chans[None, :]), mask=load_mask, other=0)
            first = batch_offset + top * out_row_stride + left[:, None] * channels + chans[None, :]
            dx = (right - left)[:, None] * channels
            dy = (bottom - top) * out_row_stride
            tl.store(out_ptr + first, values, mask=mask)
            tl.store(out_ptr + first + dx, values, mask=mask)
            tl.store(out_ptr + first + dy, values, mask=mask)
            tl.store(out_ptr + first + dy + dx, values, mask=mask)


def _launch_pixel_upsample(x: torch.Tensor, out: torch.Tensor, *, padded: bool) -> None:
    batch, channels, height, width = x.shape
    block_c = min(triton.next_power_of_2(channels), _MAX_BLOCK_C)
    block_w = max(1, _TILE_ELEMENTS // block_c)
    width_blocks = triton.cdiv(width + padded, block_w)
    grid = (batch * (height + padded) * width_blocks,)
    with torch.get_device_module().device(x.device):
        _upsample2x_pixels_kernel[grid](
            x,
            out,
            channels,
            height,
            width,
            width_blocks,
            x.stride(0),
            x.stride(2),
            x.stride(3),
            PADDED=padded,
            BLOCK_W=block_w,
            BLOCK_C=block_c,
            num_warps=_NUM_WARPS,
        )


def upsample_nearest_2x_padded(x: torch.Tensor, out: torch.Tensor) -> bool:
    """Write nearest-2x BF16 input into an NHWC buffer with one-cell borders.

    Border values are zero; the spatial wrapper fills received halos afterward.
    Returns False without writing when the kernel cannot handle the tensors.
    """
    if (
        not HAS_TRITON
        or not x.is_cuda
        or x.ndim != 4
        or x.dtype != torch.bfloat16
        or x.numel() == 0
        or not x.is_contiguous(memory_format=torch.channels_last)
        or out.device != x.device
        or out.dtype != x.dtype
        or out.shape != (x.shape[0], x.shape[1], 2 * x.shape[2] + 2, 2 * x.shape[3] + 2)
        or not out.is_contiguous(memory_format=torch.channels_last)
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return False
    _launch_pixel_upsample(x, out, padded=True)
    return True


def upsample_nearest_2x(x: torch.Tensor) -> torch.Tensor | None:
    """``F.interpolate(x, scale_factor=(2.0, 2.0), mode="nearest-exact")`` for 4D ``x``, or ``None``.

    Identical to ``mode="nearest"`` as well (both select input index ``o // 2``
    for a factor of two). The output takes the memory format ATen would choose
    (``x.suggest_memory_format()``): contiguous for channels-first input
    (including the strided frame-major view ``WanResample`` produces),
    channels-last for channels-last input. Inputs whose strides are
    channels-last contiguous but that ATen does not treat as channels-last
    decline.
    """
    if not _HAS_INTERLEAVE or x.dim() != 4 or not on_kernel_device(x) or x.dtype not in _SUPPORTED_DTYPES:
        return None
    batch, channels, height, width = x.shape
    if x.numel() == 0:
        return None
    out_shape = (batch, channels, 2 * height, 2 * width)
    channels_last = suggests_channels_last(x)

    if not channels_last and x.stride(3) == 1 and x.stride(2) == width:
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        rows = batch * channels * height
        spatial = in_multi_gpu_spatial_decode()
        block_w = ROW_TILE_WIDTH if spatial else _pick_block_width(width)
        rows_per_program = max(1, _TILE_ELEMENTS // block_w)
        grid = (triton.cdiv(rows, rows_per_program),)
        with torch.get_device_module().device(x.device):
            kernel = _spatial_upsample2x_rows_kernel if spatial else _upsample2x_rows_kernel
            kernel[grid](
                x,
                out,
                rows,
                channels,
                height,
                width,
                2 * width,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                ROWS=rows_per_program,
                BLOCK_W=block_w,
                num_warps=_NUM_WARPS,
            )
        return out

    if channels_last and channels > 1 and x.is_contiguous(memory_format=torch.channels_last):
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        _launch_pixel_upsample(x, out, padded=False)
        return out

    return None


__all__ = ["upsample_nearest_2x", "upsample_nearest_2x_padded"]
