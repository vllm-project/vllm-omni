# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Write decoded Lance VAE chunks into the final video buffer."""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton


def can_use_fused_output(chunk: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and chunk.is_cuda
        and chunk.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
    )


@triton.jit
def _write_unpatchified_kernel(
    source,
    output,
    frames,
    total_frames,
    frame_offset,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    batch: tl.constexpr,
    strides: tl.constexpr,
    clamp: tl.constexpr,
    block_size: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    count = tl.cast(frames, tl.int64) * batch * channels * height * width * 4
    out_x = index % (width * 2)
    remaining = index // (width * 2)
    out_y = remaining % (height * 2)
    remaining = remaining // (height * 2)
    frame = remaining % frames
    remaining = remaining // frames
    channel = remaining % channels
    sample = remaining // channels
    # Match the (c r q) channel order in _unpatchify, where q selects the row.
    source_channel = channel * 4 + (out_x % 2) * 2 + out_y % 2
    source_offset = (
        sample * strides[0]
        + source_channel * strides[1]
        + frame * strides[2]
        + (out_y // 2) * strides[3]
        + (out_x // 2) * strides[4]
    )
    output_offset = (((sample * channels + channel) * total_frames + frame_offset + frame) * (height * 2) + out_y) * (
        width * 2
    ) + out_x
    value = tl.load(source + source_offset, index < count, other=0)
    if clamp:
        low = tl.full((block_size,), -1.0, source.dtype.element_ty)
        high = tl.full((block_size,), 1.0, source.dtype.element_ty)
        bits: tl.constexpr = tl.int32 if source.dtype.element_ty == tl.float32 else tl.int16
        raw = value.to(bits, bitcast=True)
        result = tl.where(
            value < low, low.to(bits, bitcast=True), tl.where(value > high, high.to(bits, bitcast=True), raw)
        )
        # Integer selection keeps the input NaN payload and signed zero.
        result = tl.where(value != value, raw, result)
        tl.store(output.to(tl.pointer_type(bits)) + output_offset, result, index < count)
    else:
        tl.store(output + output_offset, value, index < count)


def write_unpatchified(chunk: torch.Tensor, output: torch.Tensor, frame_offset: int, *, clamp: bool) -> None:
    """Write a chunk to a separate, contiguous output buffer. Autograd is not supported."""
    batch, packed_channels, frames, height, width = chunk.shape
    _write_unpatchified_kernel[(triton.cdiv(chunk.numel(), 512),)](
        chunk,
        output,
        frames,
        output.shape[2],
        frame_offset,
        packed_channels // 4,
        height,
        width,
        batch,
        chunk.stride(),
        clamp,
        512,
    )
