# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.triton_utils import tldevice as libdevice

if HAS_TRITON:

    @triton.jit
    def _reflect_index(index, size):
        index = tl.where(index < 0, -index, index)
        return tl.where(index >= size, 2 * size - 2 - index, index)

    @triton.jit
    def _causal_reflect_pad_kernel(
        output_ptr,
        source_ptr,
        channels,
        height,
        width,
        source_stride_b,
        source_stride_c,
        source_stride_t,
        source_stride_h,
        source_stride_w,
        output_frames,
        output_height,
        output_width,
        front,
        top,
        left,
        right,
        column_blocks,
        plane_tiles,
        apply_silu: tl.constexpr,
        block_h: tl.constexpr,
        block_w: tl.constexpr,
        border_w: tl.constexpr,
    ):
        program = tl.program_id(0)
        plane = program // plane_tiles
        tile = program - plane * plane_tiles
        row_block = tile // column_blocks
        column_block = tile - row_block * column_blocks
        frame = plane % output_frames - front
        batch_channel = plane // output_frames
        live = frame >= 0
        source_plane = (
            (batch_channel // channels).to(tl.int64) * source_stride_b
            + (batch_channel % channels).to(tl.int64) * source_stride_c
            + tl.maximum(frame, 0).to(tl.int64) * source_stride_t
        )

        output_rows = row_block * block_h + tl.arange(0, block_h)
        row_mask = output_rows < output_height
        source_rows = source_plane + _reflect_index(output_rows - top, height).to(tl.int64) * source_stride_h
        output_rows = plane.to(tl.int64) * output_height * output_width + output_rows.to(tl.int64) * output_width

        columns = column_block * block_w + tl.arange(0, block_w)
        mask = row_mask[:, None] & (columns < width)[None, :]
        load_mask = mask & live
        values = tl.load(
            source_ptr + source_rows[:, None] + columns[None, :] * source_stride_w,
            mask=load_mask,
            other=0.0,
        )
        if apply_silu:
            values = tl.div_rn(values, 1.0 + libdevice.exp(-values))
        tl.store(
            output_ptr + output_rows[:, None] + left + columns[None, :],
            tl.where(load_mask, values, 0.0),
            mask=mask,
        )

        if column_block == 0:
            border = tl.arange(0, border_w)
            output_columns = tl.where(border < left, border, width + border)
            source_columns = _reflect_index(output_columns - left, width)
            border_mask = row_mask[:, None] & (border < left + right)[None, :]
            border_load_mask = border_mask & live
            border_values = tl.load(
                source_ptr + source_rows[:, None] + source_columns[None, :] * source_stride_w,
                mask=border_load_mask,
                other=0.0,
            )
            if apply_silu:
                border_values = tl.div_rn(border_values, 1.0 + libdevice.exp(-border_values))
            tl.store(
                output_ptr + output_rows[:, None] + output_columns[None, :],
                tl.where(border_load_mask, border_values, 0.0),
                mask=border_mask,
            )


def try_causal_reflect_pad_exact(
    source: torch.Tensor,
    *,
    front: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    apply_silu: bool,
) -> torch.Tensor | None:
    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and source.is_cuda
        and source.device.index == torch.accelerator.current_device_index()
        and source.dtype == torch.float32
        and source.ndim == 5
        and source.numel() > 0
        and all(type(pad) is int and pad >= 0 for pad in (front, top, bottom, left, right))
    ):
        return None
    batch, channels, frames, height, width = source.shape
    if max(top, bottom) >= height or max(left, right) >= width or (width - 1) * source.stride(4) >= 2**31:
        return None

    output_frames = frames + front
    output_height = height + top + bottom
    output_width = width + left + right
    planes = batch * channels * output_frames
    block_w = min(256, triton.next_power_of_2(width))
    block_h = min(triton.next_power_of_2(output_height), max(1, 1024 // block_w))
    column_blocks = triton.cdiv(width, block_w)
    plane_tiles = triton.cdiv(output_height, block_h) * column_blocks
    if planes * plane_tiles >= 2**31:
        return None

    output = source.new_empty((batch, channels, output_frames, output_height, output_width))
    _causal_reflect_pad_kernel[(planes * plane_tiles,)](
        output,
        source,
        channels,
        height,
        width,
        *source.stride(),
        output_frames,
        output_height,
        output_width,
        front,
        top,
        left,
        right,
        column_blocks,
        plane_tiles,
        apply_silu=apply_silu,
        block_h=block_h,
        block_w=block_w,
        border_w=max(2, triton.next_power_of_2(left + right)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["try_causal_reflect_pad_exact"]
