# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framewise GroupNorm + SiLU without the NCTHW↔NTCHW transpose copies."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _frame_offset(
    group, index, spatial: tl.constexpr, frames: tl.constexpr, channels: tl.constexpr, groups: tl.constexpr
):
    group = group.to(tl.int64)
    index = index.to(tl.int64)
    batch = group // (frames * groups)
    frame = (group // groups) % frames
    channel = (group % groups) * (channels // groups) + index // spatial
    return batch * channels * frames * spatial + channel * frames * spatial + frame * spatial + index % spatial


@triton.jit
def _moments(
    input_ptr,
    partial_ptr,
    spatial: tl.constexpr,
    frames: tl.constexpr,
    channels_total: tl.constexpr,
    groups: tl.constexpr,
    group_size: tl.constexpr,
    tiles: tl.constexpr,
    block: tl.constexpr,
):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    i = tile * block + tl.arange(0, block)
    offset = _frame_offset(group, i, spatial, frames, channels_total, groups)
    x = tl.load(input_ptr + offset, i < group_size, 0).to(tl.float32)
    count = tl.minimum(block, group_size - tile * block)
    mean = tl.sum(x, 0) / count
    delta = tl.where(i < group_size, x - mean, 0)
    squares = tl.sum(delta * delta, 0)
    tl.store(partial_ptr + (group * tiles + tile) * 2, mean)
    tl.store(partial_ptr + (group * tiles + tile) * 2 + 1, squares)


@triton.jit
def _statistics(
    partial_ptr,
    stats_ptr,
    group_size: tl.constexpr,
    tiles: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    reduce_size: tl.constexpr,
):
    group = tl.program_id(0)
    # Merge centered partial moments; raw E[x²] - E[x]² loses precision
    # for nearly constant frames.
    parts = tl.arange(0, reduce_size)
    count = tl.maximum(0, tl.minimum(block, group_size - parts * block))
    means = tl.load(partial_ptr + (group * tiles + parts) * 2, parts < tiles, 0)
    mean = tl.sum(means * count, 0) / group_size
    centered = tl.load(partial_ptr + (group * tiles + parts) * 2 + 1, parts < tiles, 0)
    variance = tl.sum(centered + (means - mean) * (means - mean) * count, 0) / group_size
    inv = tl.rsqrt(tl.maximum(variance, 0) + eps)
    tl.store(stats_ptr + group * 2, mean)
    tl.store(stats_ptr + group * 2 + 1, inv)


@triton.jit
def _normalize(
    input_ptr,
    weight_ptr,
    bias_ptr,
    stats_ptr,
    output_ptr,
    spatial: tl.constexpr,
    frames: tl.constexpr,
    channels_total: tl.constexpr,
    groups: tl.constexpr,
    group_size: tl.constexpr,
    block: tl.constexpr,
):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    mean = tl.load(stats_ptr + group * 2)
    inv = tl.load(stats_ptr + group * 2 + 1)
    i = tile * block + tl.arange(0, block)
    channels = (group % groups) * (channels_total // groups) + i // spatial
    offset = _frame_offset(group, i, spatial, frames, channels_total, groups)
    x = tl.load(input_ptr + offset, i < group_size, 0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, i < group_size, 0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, i < group_size, 0).to(tl.float32)
    scale = inv * weight
    shift = bias - mean * scale
    # Preserve the FP16 GroupNorm output boundary before SiLU.
    normalized = (x * scale + shift).to(output_ptr.dtype.element_ty).to(tl.float32)
    y = normalized * tl.sigmoid(normalized)
    tl.store(output_ptr + offset, y, i < group_size)


def frame_norm_silu(norm: torch.nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """Normalize each frame/group in FP32 and return contiguous FP16 NCTHW."""
    x = x.contiguous()
    b, c, t, h, w = x.shape
    g = norm.num_groups
    n = c // g * h * w
    k = triton.cdiv(n, 1024)
    partial = torch.empty((b * t * g, k, 2), device=x.device, dtype=torch.float32)
    stats = torch.empty((b * t * g, 2), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    _moments[(b * t * g, k)](x, partial, h * w, t, c, g, n, k, 1024)
    _statistics[(b * t * g,)](partial, stats, n, k, norm.eps, 1024, triton.next_power_of_2(k), enable_fp_fusion=False)
    _normalize[(b * t * g, k)](
        x,
        norm.weight,
        norm.bias,
        stats,
        y,
        h * w,
        t,
        c,
        g,
        n,
        1024,
        enable_fp_fusion=False,
    )
    return y
