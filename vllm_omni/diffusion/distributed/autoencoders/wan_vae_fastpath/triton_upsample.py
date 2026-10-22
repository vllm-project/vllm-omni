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

from ._utils import _pick_block_width, suggests_channels_last

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

    @triton.jit
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
        in_height = rest % height
        batch = rest // height
        cols = width_block * BLOCK_W + tl.arange(0, BLOCK_W)
        col_mask = cols < width
        x_row = batch * stride_x_batch + in_height * stride_x_height
        out_row_stride = (2 * width) * channels
        out_row = (batch * (2 * height) + 2 * in_height) * out_row_stride
        out_pixels = out_row + (2 * cols) * channels
        for c0 in range(0, channels, BLOCK_C):
            chans = c0 + tl.arange(0, BLOCK_C)
            mask = col_mask[:, None] & (chans < channels)[None, :]
            values = tl.load(x_ptr + (x_row + cols[:, None] * stride_x_width + chans[None, :]), mask=mask, other=0)
            first = out_pixels[:, None] + chans[None, :]
            tl.store(out_ptr + first, values, mask=mask)
            tl.store(out_ptr + (first + channels), values, mask=mask)
            tl.store(out_ptr + (first + out_row_stride), values, mask=mask)
            tl.store(out_ptr + (first + out_row_stride + channels), values, mask=mask)

    @triton.jit
    def _upsample2x_padded_kernel(
        X, Y, N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr
    ):
        # Flat NHWC indexing uses int64 for large video batches.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        c = i % C
        row = i // C
        w = row % (2 * W + 2)
        h = row // (2 * W + 2) % (2 * H + 2)
        n = row // ((2 * W + 2) * (2 * H + 2))
        valid = (n < N) & (h > 0) & (h <= 2 * H) & (w > 0) & (w <= 2 * W)
        source = ((n * H + (h - 1) // 2) * W + (w - 1) // 2) * C + c
        value = tl.load(X + source, valid, 0)
        tl.store(Y + i, value, i < N * C * (2 * H + 2) * (2 * W + 2))


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
    n, c, h, w = x.shape
    with torch.get_device_module().device(x.device):
        _upsample2x_padded_kernel[(triton.cdiv(out.numel(), 4096),)](
            x, out, N=n, C=c, H=h, W=w, BLOCK=4096, num_warps=4
        )
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
    if not _HAS_INTERLEAVE or x.dim() != 4 or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return None
    batch, channels, height, width = x.shape
    if x.numel() == 0:
        return None
    out_shape = (batch, channels, 2 * height, 2 * width)
    channels_last = suggests_channels_last(x)

    if not channels_last and x.stride(3) == 1 and x.stride(2) == width:
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        rows = batch * channels * height
        block_w = _pick_block_width(width)
        rows_per_program = max(1, _TILE_ELEMENTS // block_w)
        grid = (triton.cdiv(rows, rows_per_program),)
        with torch.get_device_module().device(x.device):
            _upsample2x_rows_kernel[grid](
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
        block_c = min(triton.next_power_of_2(channels), _MAX_BLOCK_C)
        block_w = max(1, _TILE_ELEMENTS // block_c)
        width_blocks = triton.cdiv(width, block_w)
        grid = (batch * height * width_blocks,)
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
                BLOCK_W=block_w,
                BLOCK_C=block_c,
                num_warps=_NUM_WARPS,
            )
        return out

    return None


__all__ = ["upsample_nearest_2x", "upsample_nearest_2x_padded"]
