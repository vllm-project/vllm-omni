# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused dots.tts LayerNorm modulation kernels with FP32 accumulation."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _layer_norm_indexed_scale_shift_kernel(
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
    mean = tl.sum(x, axis=0) / hidden_size
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    normalized = centered * tl.rsqrt(variance + eps) * weight

    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * hidden_size + columns,
        normalized * (1.0 + scale) + shift,
        mask=mask,
    )


@triton.jit
def _indexed_gate_layer_norm_scale_shift_kernel(
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
    # Match the BF16 residual value consumed by the unfused RMSNorm path.
    updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
    tl.store(residual_out_ptr + row * hidden_size + columns, updated, mask=mask)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(updated, axis=0) / hidden_size
    centered = updated - mean
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    normalized = centered * tl.rsqrt(variance + eps) * weight
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        modulated_out_ptr + row * hidden_size + columns,
        normalized * (1.0 + scale) + shift,
        mask=mask,
    )


def layer_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse LayerNorm with its indexed AdaLN affine transform."""
    if x.is_cpu:
        input_dtype = x.dtype
        normalized = torch.nn.functional.layer_norm(
            x.float(),
            (x.size(-1),),
            weight.float(),
            None,
            eps,
        ).to(input_dtype)
        return (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(input_dtype)
    output = torch.empty_like(x)
    rows, hidden_size = x.shape
    if rows:
        _layer_norm_indexed_scale_shift_kernel[(rows,)](
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


def indexed_gate_layer_norm_scale_shift(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse gated residual, the following LayerNorm, and AdaLN affine."""
    if residual.is_cpu:
        input_dtype = residual.dtype
        residual_out = (residual + gate.index_select(0, indices) * branch).to(input_dtype)
        normalized = torch.nn.functional.layer_norm(
            residual_out.float(),
            (residual_out.size(-1),),
            weight.float(),
            None,
            eps,
        ).to(input_dtype)
        modulated_out = (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(
            input_dtype
        )
        return residual_out, modulated_out
    residual_out = torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    rows, hidden_size = residual.shape
    if rows:
        _indexed_gate_layer_norm_scale_shift_kernel[(rows,)](
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
    "indexed_gate_layer_norm_scale_shift",
    "layer_norm_indexed_scale_shift",
]
