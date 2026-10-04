# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused tensor-indexed modulation with FP32 accumulation."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

# Ascend CANN limits the launch coreDim to uint16.
_MAX_1D_GRID_SIZE = 65535


def _iter_row_chunks(rows: int):
    for row_offset in range(0, rows, _MAX_1D_GRID_SIZE):
        yield row_offset, min(rows - row_offset, _MAX_1D_GRID_SIZE)


def _launch_row_chunks(kernel, rows: int, device_type: str, *args, **kwargs) -> None:
    if device_type != "npu":
        kernel[(rows,)](*args, 0, **kwargs)
        return
    for row_offset, chunk_rows in _iter_row_chunks(rows):
        kernel[(chunk_rows,)](*args, row_offset, **kwargs)


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
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
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
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
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
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
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
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
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
    _launch_row_chunks(
        _indexed_scale_shift_kernel,
        rows,
        x.device.type,
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
    _launch_row_chunks(
        _indexed_gate_kernel,
        rows,
        x.device.type,
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
    """Fuse RMSNorm with a row-indexed affine transform."""
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
        _launch_row_chunks(
            _rms_norm_indexed_scale_shift_kernel,
            rows,
            x.device.type,
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
        _launch_row_chunks(
            _indexed_gate_rms_norm_scale_shift_kernel,
            rows,
            residual.device.type,
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


@triton.jit
def _bf16_indexed_gate_add_kernel(
    bundle_ptr,
    gate_ptr,
    local_row_to_tail_ptr,
    bundle_stride_batch,
    bundle_stride_seq,
    bundle_stride_head,
    bundle_stride_dim,
    gate_stride_batch,
    gate_stride_seq,
    gate_stride_head,
    gate_stride_dim,
    local_rows: tl.constexpr,
    head_dim: tl.constexpr,
    row_width: tl.constexpr,
    feature_block: tl.constexpr,
):
    flat_local_row = tl.program_id(0)
    feature_block_index = tl.program_id(1)
    batch_index = flat_local_row // local_rows
    local_row = flat_local_row - batch_index * local_rows
    tail_slot = tl.load(local_row_to_tail_ptr + local_row)
    valid_row = tail_slot >= 0

    features = feature_block_index * feature_block + tl.arange(0, feature_block)
    feature_mask = features < row_width
    head_index = features // head_dim
    dim_index = features - head_index * head_dim
    fine_offsets = (
        batch_index * bundle_stride_batch
        + local_row * bundle_stride_seq
        + head_index * bundle_stride_head
        + dim_index * bundle_stride_dim
    )
    coarse_offsets = (
        batch_index * bundle_stride_batch
        + (local_rows + tail_slot) * bundle_stride_seq
        + head_index * bundle_stride_head
        + dim_index * bundle_stride_dim
    )
    gate_offsets = (
        batch_index * gate_stride_batch
        + local_row * gate_stride_seq
        + head_index * gate_stride_head
        + dim_index * gate_stride_dim
    )
    mask = valid_row & feature_mask
    fine_values = tl.load(bundle_ptr + fine_offsets, mask=mask, other=0.0)
    coarse_values = tl.load(bundle_ptr + coarse_offsets, mask=mask, other=0.0)
    gate_values = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0)
    values = tl.inline_asm_elementwise(
        """
        {
            .reg .b32 gated;
            mul.rn.bf16x2 gated, $2, $3;
            add.rn.bf16x2 $0, $1, gated;
        }
        """,
        constraints="=r,r,r,r",
        args=[fine_values, coarse_values, gate_values],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )
    tl.store(bundle_ptr + fine_offsets, values, mask=mask)


def bf16_indexed_gate_add_(bundle: torch.Tensor, gate: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    """Add indexed coarse rows to a fine prefix with separate BF16 mul/add rounding.

    ``bundle`` is [B, fine_rows + coarse_rows, H, D]; ``gate`` covers the fine
    prefix. Negative indices leave a fine row unchanged. The caller validates
    nonnegative indices against the coarse row count before device execution.
    The output aliases the fine prefix and the coarse tail remains unchanged.
    """
    if bundle.ndim != 4 or gate.ndim != 4 or bundle.shape[::2] != gate.shape[::2] or bundle.shape[-1] != gate.shape[-1]:
        raise ValueError("bundle and gate must be BSHD tensors with matching batch/head dimensions")
    rows = gate.shape[1]
    if rows < 1 or bundle.shape[1] <= rows or bundle.shape[-1] % 2:
        raise ValueError("expected a nonempty fine prefix, coarse tail and even head dimension")
    if bundle.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise TypeError("indexed gate addition requires BF16 tensors")
    if row_indices.shape != (rows,) or row_indices.dtype != torch.int32:
        raise ValueError("row_indices must be INT32 with one entry per fine row")
    if not all(
        t.is_cuda and t.device == bundle.device and t.is_contiguous() and not t.requires_grad
        for t in (bundle, gate, row_indices)
    ):
        raise ValueError("indexed gate addition requires contiguous inference tensors on one CUDA device")
    if bundle.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr():
        raise ValueError("gate must not alias the mutable output bundle")
    width = bundle.shape[2] * bundle.shape[3]
    _bf16_indexed_gate_add_kernel[(bundle.shape[0] * rows, triton.cdiv(width, 1024))](
        bundle,
        gate,
        row_indices,
        *bundle.stride(),
        *gate.stride(),
        local_rows=rows,
        head_dim=bundle.shape[3],
        row_width=width,
        feature_block=1024,
        num_warps=8,
    )
    return bundle[:, :rows]
