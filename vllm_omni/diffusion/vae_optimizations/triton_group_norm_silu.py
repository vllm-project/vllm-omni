# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Channels-last two-pass GroupNorm(+SiLU) Triton kernels.

The kernels keep the channel dimension innermost throughout a 2D VAE
decoder. Statistics are accumulated in FP32, then a separate apply pass
performs the affine transform and optional SiLU epilogue. Unsupported inputs
return ``None`` so callers can fail closed to the reference implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_CHANNELS = 2048


@triton.jit
def _gn_partial_rows_kernel(
    x_ptr,
    psum_ptr,
    psq_ptr,
    rows,
    rows_per_prog,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    num_chunks = tl.num_programs(0)
    row0 = chunk * rows_per_prog
    cols = tl.arange(0, C)
    acc_sum = tl.zeros((C,), tl.float32)
    acc_sq = tl.zeros((C,), tl.float32)
    x_base = x_ptr + batch * rows * C
    for row_offset in range(0, rows_per_prog, BLOCK_R):
        row_indices = row0 + row_offset + tl.arange(0, BLOCK_R)
        mask = row_indices < rows
        x = tl.load(
            x_base + row_indices[:, None] * C + cols[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc_sum += tl.sum(x, 0)
        acc_sq += tl.sum(x * x, 0)
    output_offsets = (batch * num_chunks + chunk) * C + cols
    tl.store(psum_ptr + output_offsets, acc_sum)
    tl.store(psq_ptr + output_offsets, acc_sq)


@triton.jit
def _gn_finalize_kernel(
    psum_ptr,
    psq_ptr,
    weight_ptr,
    bias_ptr,
    scale_shift_ptr,
    num_chunks,
    group_numel,
    eps,
    channels,
    CPG: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    cols = group * CPG + tl.arange(0, CPG)
    acc_sum = tl.zeros((), tl.float32)
    acc_sq = tl.zeros((), tl.float32)
    for chunk0 in range(0, num_chunks, BLOCK_K):
        chunks = chunk0 + tl.arange(0, BLOCK_K)
        mask = chunks < num_chunks
        offsets = (batch * num_chunks + chunks)[:, None] * channels + cols[None, :]
        acc_sum += tl.sum(tl.load(psum_ptr + offsets, mask=mask[:, None], other=0.0))
        acc_sq += tl.sum(tl.load(psq_ptr + offsets, mask=mask[:, None], other=0.0))
    mean = acc_sum / group_numel
    variance = acc_sq / group_numel - mean * mean
    variance = tl.maximum(variance, 0.0)
    reciprocal_std = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    scale = weight * reciprocal_std
    shift = bias - mean * scale
    tl.store(scale_shift_ptr + batch * 2 * channels + cols, scale)
    tl.store(scale_shift_ptr + (batch * 2 + 1) * channels + cols, shift)


@triton.jit
def _gn_apply_rows_kernel(
    x_ptr,
    scale_shift_ptr,
    y_ptr,
    rows,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SILU: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, C)
    scale = tl.load(scale_shift_ptr + batch * 2 * C + cols)
    shift = tl.load(scale_shift_ptr + (batch * 2 + 1) * C + cols)
    row_indices = program * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = row_indices < rows
    offsets = batch * rows * C + row_indices[:, None] * C + cols[None, :]
    x = tl.load(x_ptr + offsets, mask=mask[:, None], other=0.0).to(tl.float32)
    y = x * scale[None, :] + shift[None, :]
    if SILU:
        y = y * tl.sigmoid(y)
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask[:, None])


def _group_norm_silu_rows(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    batch_size, rows, channels = x.shape
    channels_per_group = channels // num_groups
    block_rows = max(1, 8192 // channels)
    rows_per_program = block_rows * 32
    num_chunks = triton.cdiv(rows, rows_per_program)
    partial_sum = torch.empty((batch_size, num_chunks, channels), device=x.device, dtype=torch.float32)
    partial_sq = torch.empty_like(partial_sum)
    with torch.cuda.device(x.device):
        _gn_partial_rows_kernel[(num_chunks, batch_size)](
            x,
            partial_sum,
            partial_sq,
            rows,
            rows_per_program,
            C=channels,
            BLOCK_R=block_rows,
            num_warps=4,
        )
        scale_shift = torch.empty((batch_size, 2, channels), device=x.device, dtype=torch.float32)
        block_chunks = max(1, min(4096 // channels_per_group, triton.next_power_of_2(num_chunks)))
        _gn_finalize_kernel[(num_groups, batch_size)](
            partial_sum,
            partial_sq,
            weight,
            bias,
            scale_shift,
            num_chunks,
            rows * channels_per_group,
            eps,
            channels,
            CPG=channels_per_group,
            BLOCK_K=block_chunks,
            num_warps=4,
        )
        output = torch.empty_like(x)
        _gn_apply_rows_kernel[(triton.cdiv(rows, block_rows), batch_size)](
            x,
            scale_shift,
            output,
            rows,
            C=channels,
            BLOCK_R=block_rows,
            SILU=apply_silu,
            num_warps=4,
        )
    return output


def _is_supported(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
) -> bool:
    if not (x.is_cuda and x.numel() > 0 and not torch.is_grad_enabled()):
        return False
    if x.requires_grad or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if weight is None or bias is None:
        return False
    channels = x.shape[1] if x.dim() == 4 else x.shape[-1]
    if weight.shape != (channels,) or bias.shape != (channels,):
        return False
    if not (
        weight.device == bias.device == x.device
        and weight.dtype == bias.dtype == x.dtype
        and weight.is_contiguous()
        and bias.is_contiguous()
    ):
        return False
    if num_groups < 1 or channels % num_groups != 0:
        return False
    return triton.next_power_of_2(channels) == channels and channels <= _MAX_CHANNELS


def group_norm_silu_4d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool = True,
) -> torch.Tensor | None:
    """Run GroupNorm(+SiLU) on a channels-last ``(N, C, H, W)`` tensor."""

    if x.dim() != 4 or not _is_supported(x, weight, bias, num_groups):
        return None
    batch_size, channels, height, width = x.shape
    if not (channels > 1 and (height > 1 or width > 1) and x.is_contiguous(memory_format=torch.channels_last)):
        return None
    rows = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
    output = _group_norm_silu_rows(rows, weight, bias, num_groups, eps, apply_silu)
    return output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)


def group_norm_silu_rows(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool = True,
) -> torch.Tensor | None:
    """Run GroupNorm(+SiLU) over contiguous ``(N, rows, C)`` input."""

    if x.dim() != 3 or not x.is_contiguous():
        return None
    if not _is_supported(x, weight, bias, num_groups):
        return None
    return _group_norm_silu_rows(x, weight, bias, num_groups, eps, apply_silu)


__all__ = ["group_norm_silu_4d", "group_norm_silu_rows"]
