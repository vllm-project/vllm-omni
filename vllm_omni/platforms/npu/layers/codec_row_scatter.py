# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from InferMatrixLab/qwen3-tts-rsi-npu-engine (Apache-2.0).
"""Pure selected-row stores for disjoint concurrent Codec graph state updates."""

import torch
from vllm.triton_utils import tl, triton


# Tail slices have different strides and pointer alignment at each chunk length.
# Keep them runtime values so a live tail cannot trigger a new compilation.
@triton.jit(
    do_not_specialize=["s0", "s1", "s2", "s3"],
    do_not_specialize_on_alignment=["source"],
)
def _scatter(
    source,
    indices,
    target,
    n: tl.constexpr,
    row_size: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    s0,
    s1,
    s2,
    s3,
    segment: tl.constexpr,
    element_stride: tl.constexpr,
    block: tl.constexpr,
):
    tiles_per_segment = tl.cdiv(segment, block)
    segments_per_row = row_size // segment
    for tile in range(tl.program_id(0), (n // segment) * tiles_per_segment, tl.num_programs(0)):
        subrow = tile // tiles_per_segment
        row = subrow // segments_per_row
        start = (subrow % segments_per_row) * segment
        inner = (tile % tiles_per_segment) * block + tl.arange(0, block)
        source_base = row * s0 + (start // (d2 * d3)) * s1 + (start // d3 % d2) * s2 + (start % d3) * s3
        destination = tl.load(indices + row)
        value = tl.load(source + source_base + inner * element_stride, inner < segment, other=0)
        tl.store(target + destination * row_size + start + inner, value, inner < segment)


def scatter_rows(target: torch.Tensor, indices: torch.Tensor, source: torch.Tensor) -> None:
    """Copy rows only; caller validates unique, in-range indices before capture."""
    if (
        target.device.type != "npu"
        or source.device != target.device
        or indices.device != target.device
        or indices.dtype != torch.long
        or indices.ndim != 1
        or not indices.is_contiguous()
        or not target.is_contiguous()
        or not 1 <= target.ndim <= 4
        or source.ndim != target.ndim
        or source.dtype != target.dtype
        or source.shape[1:] != target.shape[1:]
        or source.shape[0] != indices.numel()
        or source.numel() == 0
        or source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()
    ):
        raise ValueError("Expected independent source and contiguous selected-row target")
    shape = tuple(source.shape) + (1,) * (4 - source.ndim)
    strides = tuple(source.stride()) + (0,) * (4 - source.ndim)
    row = source.numel() // source.shape[0]
    segment = 1
    element_stride = source.stride(-1)
    for dimension in range(source.ndim - 1, 0, -1):
        if source.shape[dimension] == 1:
            continue
        if source.stride(dimension) != segment * element_stride:
            break
        segment *= source.shape[dimension]
    block = min(1024, max(32, triton.next_power_of_2(segment)))
    cores = torch.npu.get_device_properties(target.device).vector_core_num
    _scatter[(min(cores, source.numel() // segment * triton.cdiv(segment, block)),)](
        source,
        indices,
        target,
        source.numel(),
        row,
        *shape[1:],
        *strides,
        segment,
        element_stride,
        block,
    )
