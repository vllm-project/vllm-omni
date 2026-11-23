# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.

"""Portable Triton 3D neighborhood attention for LTX-2.5 DiffVAE.

Adapted from ``fallback_na/triton_na.py`` in Lightricks/LTX-2 commit
7954dcb0d986bdc36ef272564a9789ade07fcc65 (Apache-2.0), which vendors the
Comfy Kitchen ``na3d`` fallback. One program handles a run of queries along
the width axis with online softmax and FP32 accumulation, so no full attention
matrix or FlexAttention block mask is materialized.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_NEG_INF = tl.constexpr(-3.0e38)


@triton.jit
def _na3d_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    frames,
    height,
    width,
    heads,
    stride_batch,
    stride_frames,
    stride_height,
    stride_width,
    stride_heads,
    scale,
    kernel_frames: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    is_fp32: tl.constexpr,
):
    width_program = tl.program_id(0)
    frame_height_program = tl.program_id(1)
    batch_head_program = tl.program_id(2)

    query_frame = frame_height_program // height
    query_height = frame_height_program % height
    base = (batch_head_program // heads) * stride_batch + (batch_head_program % heads) * stride_heads

    query_width = width_program * query_block + tl.arange(0, query_block)
    valid_query = query_width < width
    head_offset = tl.arange(0, padded_head_dim)
    valid_head = head_offset < head_dim

    query_offsets = (
        base
        + query_frame * stride_frames
        + query_height * stride_height
        + query_width[:, None] * stride_width
        + head_offset[None, :]
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=valid_query[:, None] & valid_head[None, :],
        other=0.0,
    )

    frame_start = tl.minimum(
        tl.maximum(query_frame - kernel_frames // 2, 0),
        frames - kernel_frames,
    )
    frame_end = frame_start + kernel_frames
    height_start = tl.minimum(
        tl.maximum(query_height - kernel_height // 2, 0),
        height - kernel_height,
    )
    height_end = height_start + kernel_height

    clamped_query_width = tl.where(valid_query, query_width, width - 1)
    width_start = tl.minimum(
        tl.maximum(clamped_query_width - kernel_width // 2, 0),
        width - kernel_width,
    )
    width_end = width_start + kernel_width
    first_query = tl.minimum(width_program * query_block, width - 1)
    last_query = tl.minimum(width_program * query_block + query_block - 1, width - 1)
    key_width_start = tl.minimum(
        tl.maximum(first_query - kernel_width // 2, 0),
        width - kernel_width,
    )
    key_width_end = (
        tl.minimum(
            tl.maximum(last_query - kernel_width // 2, 0),
            width - kernel_width,
        )
        + kernel_width
    )

    running_max = tl.full((query_block,), _NEG_INF, dtype=tl.float32)
    running_sum = tl.zeros((query_block,), dtype=tl.float32)
    accumulator = tl.zeros((query_block, padded_head_dim), dtype=tl.float32)

    for key_frame in range(frame_start, frame_end):
        for key_height in range(height_start, height_end):
            plane = base + key_frame * stride_frames + key_height * stride_height
            for key_width_base in range(key_width_start, key_width_end, key_block):
                key_width = key_width_base + tl.arange(0, key_block)
                valid_key = key_width < key_width_end
                key_value_offsets = plane + key_width[:, None] * stride_width + head_offset[None, :]
                valid_key_value = valid_key[:, None] & valid_head[None, :]
                key = tl.load(key_ptr + key_value_offsets, mask=valid_key_value, other=0.0)
                if is_fp32:
                    scores = tl.dot(query, tl.trans(key), input_precision="ieee") * scale
                else:
                    scores = tl.dot(query, tl.trans(key)) * scale
                visible = (
                    (key_width[None, :] >= width_start[:, None])
                    & (key_width[None, :] < width_end[:, None])
                    & valid_key[None, :]
                )
                scores = tl.where(visible, scores, _NEG_INF)
                new_max = tl.maximum(running_max, tl.max(scores, 1))
                previous_scale = tl.exp(running_max - new_max)
                probabilities = tl.exp(scores - new_max[:, None])
                running_sum = running_sum * previous_scale + tl.sum(probabilities, 1)
                value = tl.load(value_ptr + key_value_offsets, mask=valid_key_value, other=0.0)
                if is_fp32:
                    accumulator = accumulator * previous_scale[:, None] + tl.dot(
                        probabilities,
                        value,
                        input_precision="ieee",
                    )
                else:
                    accumulator = accumulator * previous_scale[:, None] + tl.dot(
                        probabilities.to(value.dtype),
                        value,
                    )
                running_max = new_max

    output = accumulator / tl.maximum(running_sum, 1e-30)[:, None]
    output_offsets = (
        base
        + query_frame * stride_frames
        + query_height * stride_height
        + query_width[:, None] * stride_width
        + head_offset[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=valid_query[:, None] & valid_head[None, :],
    )


def na3d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply inward-shifted 3D neighborhood attention to BTHWNHD tensors."""
    if not HAS_TRITON:
        raise RuntimeError("LTX-2.5 Triton neighborhood attention requires a working Triton installation.")
    if query.device.type != "cuda":
        raise ValueError(f"LTX-2.5 Triton neighborhood attention requires a CUDA/ROCm device, got {query.device}.")
    if query.ndim != 6:
        raise ValueError(f"Expected query shape (B,T,H,W,NH,HD), got {tuple(query.shape)}.")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "LTX-2.5 Triton neighborhood attention requires matching query, key, and value shapes; "
            f"got query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}."
        )
    if key.device != query.device or value.device != query.device:
        raise ValueError("LTX-2.5 Triton neighborhood attention requires query, key, and value on one device.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("LTX-2.5 Triton neighborhood attention requires one query/key/value dtype.")
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported LTX-2.5 neighborhood-attention dtype: {query.dtype}.")
    if len(kernel_size) != 3 or any(size <= 0 for size in kernel_size):
        raise ValueError(f"Invalid LTX-2.5 neighborhood kernel: {kernel_size}.")

    batch, frames, height, width, heads, head_dim = query.shape
    if any(size > dimension for size, dimension in zip(kernel_size, (frames, height, width), strict=True)):
        raise ValueError(f"LTX-2.5 neighborhood kernel {kernel_size} exceeds token grid {(frames, height, width)}.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)
    padded_head_dim = max(16, triton.next_power_of_2(head_dim))
    query_block = 16
    key_block = max(16, min(32, triton.next_power_of_2(min(width, query_block + kernel_size[2]))))
    grid = (
        triton.cdiv(width, query_block),
        frames * height,
        batch * heads,
    )
    device_index = query.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    with torch.accelerator.device_index(device_index):
        _na3d_kernel[grid](
            query,
            key,
            value,
            output,
            frames,
            height,
            width,
            heads,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            query.stride(4),
            scale,
            kernel_frames=kernel_size[0],
            kernel_height=kernel_size[1],
            kernel_width=kernel_size[2],
            head_dim=head_dim,
            padded_head_dim=padded_head_dim,
            query_block=query_block,
            key_block=key_block,
            is_fp32=query.dtype == torch.float32,
            num_warps=4,
        )
    return output


__all__ = ["na3d"]
