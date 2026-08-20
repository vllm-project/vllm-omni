# SPDX-License-Identifier: Apache-2.0
"""Fused MiniMax H3 modulation with FP32 accumulation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_scale_shift_kernel(
    output_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        output_ptr + row * stride_x_row + columns,
        x * (1.0 + scale) + shift,
        mask=mask,
    )


@triton.jit
def _indexed_gate_kernel(
    output_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_output_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + row * stride_other_row + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        output_ptr + row * stride_output_row + columns,
        x + gate * other,
        mask=mask,
    )


@triton.jit
def _rms_norm_indexed_scale_shift_kernel(
    output_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    normalized = x * tl.rsqrt(variance + eps) * weight

    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * hidden_size + columns,
        normalized * (1.0 + scale) + shift,
        mask=mask,
    )


@triton.jit
def _indexed_gate_rms_norm_scale_shift_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    stride_residual_row,
    stride_gate_row,
    stride_branch_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    residual = tl.load(residual_ptr + row * stride_residual_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + row * stride_branch_row + columns, mask=mask, other=0.0).to(tl.float32)
    updated = residual + gate * branch
    tl.store(residual_out_ptr + row * hidden_size + columns, updated, mask=mask)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(updated * updated, axis=0) / hidden_size
    normalized = updated * tl.rsqrt(variance + eps) * weight
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        modulated_out_ptr + row * hidden_size + columns,
        normalized * (1.0 + scale) + shift,
        mask=mask,
    )


def indexed_scale_shift_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply indexed scale/shift in-place to a disposable contiguous input."""
    if x.is_cpu:
        x.copy_((x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype))
        return x
    rows, hidden_size = x.shape
    if rows == 0:
        return x
    _indexed_scale_shift_kernel[(rows,)](
        x,
        x,
        shift,
        scale,
        indices,
        hidden_size,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return x


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return ``x + gate[indices] * other`` without indexed temporaries."""
    if x.is_cpu:
        return (x + gate.index_select(0, indices) * other).to(x.dtype)
    output = torch.empty_like(x)
    rows, hidden_size = x.shape
    if rows == 0:
        return output
    _indexed_gate_kernel[(rows,)](
        output,
        x,
        gate,
        other,
        indices,
        hidden_size,
        output.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return output


def rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse H3 RMSNorm with its indexed AdaLN affine transform."""
    if x.is_cpu:
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        return (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(input_dtype)
    output = torch.empty_like(x)
    rows, hidden_size = x.shape
    if rows:
        _rms_norm_indexed_scale_shift_kernel[(rows,)](
            output,
            x,
            weight,
            shift,
            scale,
            indices,
            hidden_size,
            eps,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            block_n=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return output


def indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse gated residual, the following RMSNorm, and AdaLN affine."""
    if residual.is_cpu:
        input_dtype = residual.dtype
        residual_out = (residual + gate.index_select(0, indices) * branch).to(input_dtype)
        normalized = residual_out.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        modulated_out = (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(
            input_dtype
        )
        return residual_out, modulated_out
    residual_out = torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    rows, hidden_size = residual.shape
    if rows:
        _indexed_gate_rms_norm_scale_shift_kernel[(rows,)](
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            hidden_size,
            eps,
            residual.stride(0),
            gate.stride(0),
            branch.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            block_n=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return residual_out, modulated_out


__all__ = [
    "indexed_gate",
    "indexed_gate_rms_norm_scale_shift",
    "indexed_scale_shift_",
    "rms_norm_indexed_scale_shift",
]
