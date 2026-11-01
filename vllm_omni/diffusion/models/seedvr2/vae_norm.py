# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact FP32 affine application after unchanged distributed VAE statistics."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _apply_norm(
    grouped,
    mean,
    inverse,
    weight,
    bias,
    output,
    n: tl.constexpr,
    channels: tl.constexpr,
    groups: tl.constexpr,
    frames: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    mask = offsets < n
    batch = offsets // (channels * frames * pixels)
    channel = offsets // (frames * pixels) % channels
    frame = offsets // pixels % frames
    statistic = (batch * groups + channel // (channels // groups)) * frames + frame
    value = tl.load(grouped + offsets, mask, other=0).to(tl.float32)
    center = tl.load(mean + statistic, mask, other=0).to(tl.float32)
    scale = tl.load(inverse + statistic, mask, other=0).to(tl.float32)
    affine_weight = tl.load(weight + channel, mask, other=0).to(tl.float32)
    affine_bias = tl.load(bias + channel, mask, other=0).to(tl.float32)
    # Preserve each eager FP32 rounding boundary; launch disables FMA fusion.
    value = value - center
    value = value * scale
    value = value * affine_weight
    value = value + affine_bias
    tl.store(output + offsets, value, mask)


def apply_spatial_norm(
    grouped: torch.Tensor,
    mean: torch.Tensor,
    inverse: torch.Tensor,
    norm: torch.nn.GroupNorm,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """No new statistics, reassociation, reduced precision, or activation reuse."""
    b, c, t, h, w = shape
    assert grouped.device.type == "cuda" and grouped.dtype == torch.float32 and grouped.is_contiguous()
    assert c % norm.num_groups == 0 and mean.numel() == inverse.numel() == b * norm.num_groups * t
    assert mean.is_contiguous() and inverse.is_contiguous()
    assert norm.weight.is_contiguous() and norm.bias.is_contiguous()
    output = torch.empty(shape, device=grouped.device, dtype=dtype)
    count = grouped.numel()
    _apply_norm[(triton.cdiv(count, 256),)](
        grouped,
        mean,
        inverse,
        norm.weight,
        norm.bias,
        output,
        count,
        c,
        norm.num_groups,
        t,
        h * w,
        256,
        enable_fp_fusion=False,
    )
    return output
