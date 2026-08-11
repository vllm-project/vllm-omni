# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Channels-last-3D Wan VAE RMSNorm(+SiLU) Triton kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_CHANNELS = 1024


@triton.jit
def _wan_rmsnorm_silu_kernel(
    x_ptr,
    gamma_ptr,
    bias_ptr,
    output_ptr,
    channels: tl.constexpr,
    time_size,
    height_size,
    width_size,
    x_stride_batch,
    x_stride_channel,
    x_stride_time,
    x_stride_height,
    x_stride_width,
    output_stride_batch,
    output_stride_channel,
    output_stride_time,
    output_stride_height,
    output_stride_width,
    rms_scale,
    eps,
    has_bias: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    channel_offsets = tl.arange(0, BLOCK_C)
    channel_mask = channel_offsets < channels

    width = row % width_size
    remainder = row // width_size
    height = remainder % height_size
    remainder = remainder // height_size
    time = remainder % time_size
    batch = remainder // time_size

    x_base = batch * x_stride_batch + time * x_stride_time + height * x_stride_height + width * x_stride_width
    output_base = (
        batch * output_stride_batch
        + time * output_stride_time
        + height * output_stride_height
        + width * output_stride_width
    )

    x = tl.load(
        x_ptr + x_base + channel_offsets * x_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    inverse_norm = 1.0 / tl.maximum(norm, eps)

    # Match the eager dtype boundaries: normalize and scale in activation
    # dtype, affine in the promoted dtype, then SiLU in FP32.
    y = (x * inverse_norm).to(x_ptr.dtype.element_ty)
    gamma = tl.load(gamma_ptr + channel_offsets, mask=channel_mask, other=1.0)
    y = (y * rms_scale).to(x_ptr.dtype.element_ty)
    y = (y.to(tl.float32) * gamma.to(tl.float32)).to(output_ptr.dtype.element_ty)
    if has_bias:
        bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0)
        y = (y.to(tl.float32) + bias.to(tl.float32)).to(output_ptr.dtype.element_ty)
    y = y.to(tl.float32)
    y = y * tl.sigmoid(y)
    tl.store(
        output_ptr + output_base + channel_offsets * output_stride_channel,
        y,
        mask=channel_mask,
    )


def _affine_supported(x: torch.Tensor, affine: torch.Tensor) -> bool:
    return (
        affine.is_cuda
        and affine.device == x.device
        and (affine.dtype == x.dtype or affine.dtype == torch.float32)
        and affine.numel() == x.shape[1]
    )


def can_use_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    """Return whether ``wan_rmsnorm_silu`` supports this exact input."""

    return (
        x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_DTYPES
        and x.ndim == 5
        and x.numel() > 0
        and 0 < x.shape[1] <= _MAX_CHANNELS
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and _affine_supported(x, gamma)
        and (bias is None or _affine_supported(x, bias))
    )


def wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    rms_scale: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor | None:
    """Fuse ``SiLU(F.normalize(x, dim=1) * scale * gamma + bias)``."""

    if not can_use_wan_rmsnorm_silu(x, gamma, bias):
        return None

    batch_size, channels, time_size, height_size, width_size = x.shape
    output_dtype = torch.promote_types(x.dtype, gamma.dtype)
    output = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=output_dtype)
    block_channels = triton.next_power_of_2(channels)
    num_warps = 1 if block_channels <= 64 else 4 if block_channels <= 512 else 8
    gamma = gamma.reshape(channels).contiguous()
    has_bias = bias is not None
    bias = gamma if bias is None else bias.reshape(channels).contiguous()
    if rms_scale is None:
        rms_scale = channels**0.5

    with torch.cuda.device(x.device):
        _wan_rmsnorm_silu_kernel[(batch_size * time_size * height_size * width_size,)](
            x,
            gamma,
            bias,
            output,
            channels,
            time_size,
            height_size,
            width_size,
            *x.stride(),
            *output.stride(),
            float(rms_scale),
            eps,
            has_bias,
            BLOCK_C=block_channels,
            num_warps=num_warps,
        )
    return output


__all__ = ["can_use_wan_rmsnorm_silu", "wan_rmsnorm_silu"]
