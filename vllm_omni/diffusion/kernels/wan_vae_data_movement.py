# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact data-movement kernels for the Wan causal VAE decoder.

Adapted from SGLang PR #34125:
https://github.com/sgl-project/sglang/pull/34125

The kernels in this module only move values, fill zeros, or perform the same
two-operand addition as the PyTorch reference path. They preserve the input
layout instead of selecting a new Conv3d layout, so enabling them does not
change the decoder's cuDNN accumulation order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_INT32 = 2**31 - 1
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _cat_pad_5d_kernel(
    x_ptr,
    cache_ptr,
    out_ptr,
    keep_ptr,
    total,
    channels,
    frames,
    height,
    width,
    cache_frames,
    out_frames,
    out_height,
    out_width,
    zero_front_frames,
    pad_height,
    pad_width,
    stride_x_batch,
    stride_x_channel,
    stride_x_frame,
    stride_x_height,
    stride_x_width,
    stride_cache_batch,
    stride_cache_channel,
    stride_cache_frame,
    stride_cache_height,
    stride_cache_width,
    has_cache: tl.constexpr,
    keep_frames_const: tl.constexpr,
    channels_inner: tl.constexpr,
    index_64bit: tl.constexpr,
    block_size_const: tl.constexpr,
):
    if index_64bit:
        offsets = tl.program_id(0).to(tl.int64) * block_size_const + tl.arange(0, block_size_const).to(tl.int64)
    else:
        offsets = tl.program_id(0) * block_size_const + tl.arange(0, block_size_const)
    mask = offsets < total

    if channels_inner:
        out_channel = offsets % channels
        remaining = offsets // channels
        out_width_idx = remaining % out_width
        remaining = remaining // out_width
        out_height_idx = remaining % out_height
        remaining = remaining // out_height
        out_frame = remaining % out_frames
        out_batch = remaining // out_frames
    else:
        out_width_idx = offsets % out_width
        remaining = offsets // out_width
        out_height_idx = remaining % out_height
        remaining = remaining // out_height
        out_frame = remaining % out_frames
        remaining = remaining // out_frames
        out_channel = remaining % channels
        out_batch = remaining // channels

    input_width = out_width_idx - pad_width
    input_height = out_height_idx - pad_height
    input_frame = out_frame - zero_front_frames

    spatial_valid = (input_width >= 0) & (input_width < width) & (input_height >= 0) & (input_height < height)
    from_cache = spatial_valid & (input_frame >= 0) & (input_frame < cache_frames)
    from_x = spatial_valid & (input_frame >= cache_frames) & (input_frame < cache_frames + frames)

    x_frame = input_frame - cache_frames
    x_offset = (
        out_batch * stride_x_batch
        + out_channel * stride_x_channel
        + x_frame * stride_x_frame
        + input_height * stride_x_height
        + input_width * stride_x_width
    )
    values = tl.load(x_ptr + x_offset, mask=mask & from_x, other=0.0)
    if has_cache:
        cache_offset = (
            out_batch * stride_cache_batch
            + out_channel * stride_cache_channel
            + input_frame * stride_cache_frame
            + input_height * stride_cache_height
            + input_width * stride_cache_width
        )
        cache_values = tl.load(cache_ptr + cache_offset, mask=mask & from_cache, other=0.0)
        values = tl.where(from_cache, cache_values, values)
    tl.store(out_ptr + offsets, values, mask=mask)

    if keep_frames_const > 0:
        keep_frame = out_frame - (out_frames - keep_frames_const)
        keep_mask = mask & spatial_valid & (keep_frame >= 0)
        if channels_inner:
            keep_offset = (
                ((out_batch * keep_frames_const + keep_frame) * height + input_height) * width + input_width
            ) * channels + out_channel
        else:
            keep_offset = (
                ((out_batch * channels + out_channel) * keep_frames_const + keep_frame) * height + input_height
            ) * width + input_width
        tl.store(keep_ptr + keep_offset, values, mask=keep_mask)


def _uses_channels_last_3d(tensor: torch.Tensor) -> bool:
    return tensor.shape[1] > 1 and tensor.stride(1) == 1


def cat_pad_5d(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    padding: list[int] | tuple[int, ...],
    keep_cache_frames: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse temporal concat, constant padding, and optional cache refresh.

    ``padding`` follows the Wan causal Conv3d convention
    ``(w_left, w_right, h_top, h_bottom, t_front, t_back)``. The output uses
    the same dense layout that the reference ``cat``/``pad`` path selects.
    Unsupported inputs return ``None`` so callers can execute the reference
    implementation.
    """
    if len(padding) != 6:
        return None
    pad_width_left, pad_width_right, pad_height_top, pad_height_bottom, pad_front, pad_back = padding
    if any(value < 0 for value in padding):
        return None
    if pad_width_left != pad_width_right or pad_height_top != pad_height_bottom or pad_back != 0:
        return None
    if x.dim() != 5 or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return None
    if keep_cache_frames < 0:
        return None

    channels_inner = _uses_channels_last_3d(x)
    cache_frames = 0
    if cache_x is not None:
        if (
            cache_x.dim() != 5
            or cache_x.dtype != x.dtype
            or cache_x.device != x.device
            or cache_x.shape[0] != x.shape[0]
            or cache_x.shape[1] != x.shape[1]
            or cache_x.shape[3:] != x.shape[3:]
            or _uses_channels_last_3d(cache_x) != channels_inner
        ):
            return None
        cache_frames = cache_x.shape[2]

    zero_front_frames = pad_front - cache_frames
    if zero_front_frames < 0:
        return None

    batch, channels, frames, height, width = x.shape
    out_frames = pad_front + frames
    out_height = height + 2 * pad_height_top
    out_width = width + 2 * pad_width_left
    keep_frames = min(keep_cache_frames, out_frames)
    memory_format = torch.channels_last_3d if channels_inner else torch.contiguous_format
    out = torch.empty(
        (batch, channels, out_frames, out_height, out_width),
        device=x.device,
        dtype=x.dtype,
        memory_format=memory_format,
    )
    total = out.numel()
    if total == 0 or total > _MAX_INT32 * 4:
        return None

    if keep_frames:
        keep_arg = torch.empty(
            (batch, channels, keep_frames, height, width),
            device=x.device,
            dtype=x.dtype,
            memory_format=memory_format,
        )
    else:
        keep_arg = out

    if cache_x is None:
        cache_arg = x
        cache_strides = (0, 0, 0, 0, 0)
    else:
        cache_arg = cache_x
        cache_strides = cache_x.stride()

    block_size = 512
    grid = (triton.cdiv(total, block_size),)
    with torch.get_device_module().device(x.device):
        _cat_pad_5d_kernel[grid](
            x,
            cache_arg,
            out,
            keep_arg,
            total,
            channels,
            frames,
            height,
            width,
            cache_frames,
            out_frames,
            out_height,
            out_width,
            zero_front_frames,
            pad_height_top,
            pad_width_left,
            *x.stride(),
            *cache_strides,
            has_cache=cache_x is not None,
            keep_frames_const=keep_frames,
            channels_inner=channels_inner,
            index_64bit=total >= _MAX_INT32,
            block_size_const=block_size,
        )

    if keep_cache_frames:
        return out, keep_arg
    return out


@triton.jit
def _dup_up3d_add_kernel(
    main_ptr,
    source_ptr,
    out_ptr,
    total,
    out_channels,
    out_frames,
    out_height,
    out_width,
    frame_offset,
    stride_main_batch,
    stride_main_channel,
    stride_main_frame,
    stride_main_height,
    stride_main_width,
    stride_source_batch,
    stride_source_channel,
    stride_source_frame,
    stride_source_height,
    stride_source_width,
    stride_out_batch,
    stride_out_channel,
    stride_out_frame,
    stride_out_height,
    stride_out_width,
    factor_temporal_const: tl.constexpr,
    factor_spatial_const: tl.constexpr,
    repeats_const: tl.constexpr,
    channels_inner: tl.constexpr,
    index_64bit: tl.constexpr,
    block_size_const: tl.constexpr,
):
    if index_64bit:
        offsets = tl.program_id(0).to(tl.int64) * block_size_const + tl.arange(0, block_size_const).to(tl.int64)
    else:
        offsets = tl.program_id(0) * block_size_const + tl.arange(0, block_size_const)
    mask = offsets < total

    if channels_inner:
        out_channel = offsets % out_channels
        remaining = offsets // out_channels
        out_width_idx = remaining % out_width
        remaining = remaining // out_width
        out_height_idx = remaining % out_height
        remaining = remaining // out_height
        out_frame = remaining % out_frames
        out_batch = remaining // out_frames
    else:
        out_width_idx = offsets % out_width
        remaining = offsets // out_width
        out_height_idx = remaining % out_height
        remaining = remaining // out_height
        out_frame = remaining % out_frames
        remaining = remaining // out_frames
        out_channel = remaining % out_channels
        out_batch = remaining // out_channels

    unsliced_frame = out_frame + frame_offset
    source_frame = unsliced_frame // factor_temporal_const
    temporal_remainder = unsliced_frame % factor_temporal_const
    source_height = out_height_idx // factor_spatial_const
    height_remainder = out_height_idx % factor_spatial_const
    source_width = out_width_idx // factor_spatial_const
    width_remainder = out_width_idx % factor_spatial_const
    repeated_channel = (
        (out_channel * factor_temporal_const + temporal_remainder) * factor_spatial_const + height_remainder
    ) * factor_spatial_const + width_remainder
    source_channel = repeated_channel // repeats_const

    main_offset = (
        out_batch * stride_main_batch
        + out_channel * stride_main_channel
        + out_frame * stride_main_frame
        + out_height_idx * stride_main_height
        + out_width_idx * stride_main_width
    )
    source_offset = (
        out_batch * stride_source_batch
        + source_channel * stride_source_channel
        + source_frame * stride_source_frame
        + source_height * stride_source_height
        + source_width * stride_source_width
    )
    out_offset = (
        out_batch * stride_out_batch
        + out_channel * stride_out_channel
        + out_frame * stride_out_frame
        + out_height_idx * stride_out_height
        + out_width_idx * stride_out_width
    )
    main_value = tl.load(main_ptr + main_offset, mask=mask, other=0.0)
    source_value = tl.load(source_ptr + source_offset, mask=mask, other=0.0)
    value = main_value.to(tl.float32) + source_value.to(tl.float32)
    tl.store(out_ptr + out_offset, value, mask=mask)


def dup_up3d_add(
    main: torch.Tensor,
    source: torch.Tensor,
    factor_temporal: int,
    factor_spatial: int,
    repeats: int,
    drop_first_frames: bool,
) -> torch.Tensor | None:
    """Evaluate ``main + DupUp3D(source)`` without materializing the shortcut."""
    if main.dim() != 5 or source.dim() != 5:
        return None
    if factor_temporal <= 0 or factor_spatial <= 0 or repeats <= 0:
        return None
    if factor_temporal & (factor_temporal - 1) or factor_spatial & (factor_spatial - 1):
        return None
    if repeats & (repeats - 1):
        return None
    if not main.is_cuda or not source.is_cuda:
        return None
    if main.dtype != source.dtype or main.dtype not in _SUPPORTED_DTYPES or main.device != source.device:
        return None

    batch, source_channels, frames, height, width = source.shape
    frame_offset = factor_temporal - 1 if drop_first_frames else 0
    numerator = source_channels * repeats
    denominator = factor_temporal * factor_spatial * factor_spatial
    if numerator % denominator:
        return None
    expected_shape = (
        batch,
        numerator // denominator,
        frames * factor_temporal - frame_offset,
        height * factor_spatial,
        width * factor_spatial,
    )
    if tuple(main.shape) != expected_shape:
        return None

    out = torch.empty_like(main)
    total = out.numel()
    if total == 0 or total > _MAX_INT32 * 4:
        return None

    block_size = 512
    grid = (triton.cdiv(total, block_size),)
    with torch.get_device_module().device(main.device):
        _dup_up3d_add_kernel[grid](
            main,
            source,
            out,
            total,
            expected_shape[1],
            expected_shape[2],
            expected_shape[3],
            expected_shape[4],
            frame_offset,
            *main.stride(),
            *source.stride(),
            *out.stride(),
            factor_temporal_const=factor_temporal,
            factor_spatial_const=factor_spatial,
            repeats_const=repeats,
            channels_inner=out.stride(1) == 1 and expected_shape[1] > 1,
            index_64bit=total >= _MAX_INT32,
            block_size_const=block_size,
        )
    return out
