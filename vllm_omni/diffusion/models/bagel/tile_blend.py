# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Blend a BAGEL VAE tile edge with one CUDA kernel."""

import math

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _blend_kernel(
    source,
    current,
    shape: tl.constexpr,
    source_strides: tl.constexpr,
    current_strides: tl.constexpr,
    axis: tl.constexpr,
    extent: tl.constexpr,
    source_start: tl.constexpr,
    count: tl.constexpr,
    block_size: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    remaining = index
    source_offset = tl.full((block_size,), 0, tl.int64)
    current_offset = tl.full((block_size,), 0, tl.int64)
    coordinate = tl.full((block_size,), 0, tl.int64)
    for dim in tl.static_range(3, -1, -1):
        position = remaining % shape[dim]
        remaining = remaining // shape[dim]
        current_offset += position * current_strides[dim]
        if dim == axis:
            coordinate = position
            position += source_start
        source_offset += position * source_strides[dim]
    alpha = tl.div_rn(coordinate.to(tl.float32), float(extent))
    # Do not subtract the rounded alpha from 1. Python first subtracts in FP64.
    complement = tl.div_rn((extent - coordinate).to(tl.float32), float(extent))
    a = tl.load(source + source_offset, index < count, other=0).to(tl.float32)
    b = tl.load(current + current_offset, index < count, other=0).to(tl.float32)
    dtype: tl.constexpr = current.dtype.element_ty
    # Keep the two product rounding steps from the original Torch expression.
    a = (a * complement).to(dtype).to(tl.float32)
    b = (b * alpha).to(dtype).to(tl.float32)
    tl.store(current + current_offset, a + b, index < count)


def try_blend(source: torch.Tensor, current: torch.Tensor, extent: int, axis: int) -> bool:
    """Return False without writing when the original loop is required."""
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or not current.is_cuda
        or source.device != current.device
        or source.dtype != current.dtype
        or current.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or torch.is_grad_enabled()
        or source.ndim != 4
        or current.ndim != 4
        or axis not in (2, 3)
        # For these extents, both divisions round like the Python coefficients.
        or not 0 < extent <= 1024
        or extent > min(source.shape[axis], current.shape[axis])
        or not current.numel()
        or any(source.shape[d] != current.shape[d] for d in range(4) if d != axis)
    ):
        return False
    # Sliced tiles need not be contiguous, but each destination must be unique.
    span = 1
    for size, stride in sorted(zip(current.shape, current.stride()), key=lambda item: item[1]):
        if size > 1:
            if stride < span:
                return False
            span += (size - 1) * stride
    source_span = 1 + sum((size - 1) * stride for size, stride in zip(source.shape, source.stride()))
    source_begin = source.data_ptr()
    current_begin = current.data_ptr()
    # Gathered tiles can share storage. Compare byte ranges, not storage owners.
    if (
        source_begin < current_begin + span * current.element_size()
        and current_begin < source_begin + source_span * source.element_size()
    ):
        return False
    shape = list(current.shape)
    shape[axis] = extent
    count = math.prod(shape)
    _blend_kernel[(triton.cdiv(count, 256),)](
        source,
        current,
        tuple(shape),
        source.stride(),
        current.stride(),
        axis,
        extent,
        source.shape[axis] - extent,
        count,
        256,
        enable_fp_fusion=False,
    )
    return True
