# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Copy both spatial halos in one launch, preserving channels-last ordering."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _copy_halo_pair_kernel(
        source_top,
        source_bottom,
        target_top,
        target_bottom,
        count: tl.constexpr,
        shape: tl.constexpr,
        source_top_strides: tl.constexpr,
        source_bottom_strides: tl.constexpr,
        target_top_strides: tl.constexpr,
        target_bottom_strides: tl.constexpr,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        remaining = offsets
        src_top = tl.full((BLOCK,), 0, tl.int64)
        src_bottom = tl.full((BLOCK,), 0, tl.int64)
        dst_top = tl.full((BLOCK,), 0, tl.int64)
        dst_bottom = tl.full((BLOCK,), 0, tl.int64)
        for dim in tl.static_range(len(shape)):
            coordinate = remaining % shape[dim]
            remaining = remaining // shape[dim]
            src_top += coordinate * source_top_strides[dim]
            src_bottom += coordinate * source_bottom_strides[dim]
            dst_top += coordinate * target_top_strides[dim]
            dst_bottom += coordinate * target_bottom_strides[dim]
        top = tl.load(source_top + src_top, offsets < count, 0)
        bottom = tl.load(source_bottom + src_bottom, offsets < count, 0)
        tl.store(target_top + dst_top, top, offsets < count)
        tl.store(target_bottom + dst_bottom, bottom, offsets < count)


def launch_halo_pair(
    source_top: torch.Tensor, source_bottom: torch.Tensor, target_top: torch.Tensor, target_bottom: torch.Tensor
) -> bool:
    """Copy disjoint CUDA halo views whose shape/layout the SP wrapper owns.

    The caller supplies four equally shaped 4D/5D tensors of the same dtype on
    one CUDA device. Targets must not overlap each other or the sources. Explicit
    strides allow packing and unpacking with the same kernel.
    """
    if not HAS_TRITON:
        return False
    tensors = (source_top, source_bottom, target_top, target_bottom)
    # Enumerate channels first, followed by spatial coordinates, time and batch.
    axes = (1, 4, 3, 2, 0) if source_top.ndim == 5 else (1, 3, 2, 0)
    original_shape = source_top.shape
    shape = tuple(original_shape[axis] for axis in axes)
    strides = [tuple(stride[axis] for axis in axes) for stride in (tensor.stride() for tensor in tensors)]
    count = source_top.numel()
    kernel = _copy_halo_pair_kernel[(triton.cdiv(count, 1024),)]
    if torch.accelerator.current_device_index() == source_top.device.index:
        kernel(*tensors, count, shape, *strides, 1024)
    else:
        with torch.get_device_module().device(source_top.device):
            kernel(*tensors, count, shape, *strides, 1024)

    return True
