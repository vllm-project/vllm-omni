# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared gated-residual operator for diffusion transformer blocks."""

from __future__ import annotations

import math

import torch
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_GLOBAL_GATE = 0
_BATCH_GATE = 1
_ROW_GATE = 2
_EAGER_GATE = -1
_MAX_FUSED_HIDDEN_SIZE = 16384


if HAS_TRITON:

    @triton.jit
    def _gated_residual_kernel(
        residual_ptr,
        branch_ptr,
        gate_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        rows_per_batch: tl.constexpr,
        gate_row_stride: tl.constexpr,
        gate_mode: tl.constexpr,
        product_dtype: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_size)
        offsets = row * hidden_size + columns
        mask = columns < hidden_size

        if gate_mode == 0:
            gate_row = 0
        elif gate_mode == 1:
            gate_row = row // rows_per_batch
        else:
            gate_row = row

        gate_offsets = gate_row * gate_row_stride + columns
        residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
        branch = tl.load(branch_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + gate_offsets, mask=mask).to(tl.float32)

        # Match ``branch * gate`` followed by the residual add. The explicit
        # cast preserves the intermediate rounding of eager fp16/bf16 math.
        gated_branch = (branch * gate).to(product_dtype).to(tl.float32)
        tl.store(output_ptr + offsets, residual + gated_branch, mask=mask)


def _eager_gated_residual(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    gate_mode: int,
    rows_per_batch: int,
    gate_row_stride: int,
) -> torch.Tensor:
    del gate_mode, rows_per_batch, gate_row_stride
    return residual + branch * gate


def _fused_cuda_supported(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    gate_mode: int,
) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and residual.is_cuda
        and branch.is_cuda
        and gate.is_cuda
        and residual.dtype in (torch.float16, torch.bfloat16)
        and branch.dtype == residual.dtype
        and gate.dtype == residual.dtype
        and residual.device == branch.device == gate.device
        and residual.is_contiguous()
        and branch.is_contiguous()
        and gate.ndim > 0
        and gate.stride(-1) == 1
        and gate_mode != _EAGER_GATE
        and gate.shape[-1] == residual.shape[-1]
        and 0 < residual.shape[-1] <= _MAX_FUSED_HIDDEN_SIZE
    )


def _gated_residual_impl(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    gate_mode: int,
    rows_per_batch: int,
    gate_row_stride: int,
) -> torch.Tensor:
    if not _fused_cuda_supported(residual, branch, gate, gate_mode):
        return _eager_gated_residual(residual, branch, gate, gate_mode, rows_per_batch, gate_row_stride)

    output = torch.empty_like(residual)
    rows = residual.numel() // residual.shape[-1]
    if rows == 0:
        return output

    product_dtype = tl.float16 if residual.dtype == torch.float16 else tl.bfloat16
    block_size = triton.next_power_of_2(residual.shape[-1])
    _gated_residual_kernel[(rows,)](
        residual,
        branch,
        gate,
        output,
        hidden_size=residual.shape[-1],
        rows_per_batch=rows_per_batch,
        gate_row_stride=gate_row_stride,
        gate_mode=gate_mode,
        product_dtype=product_dtype,
        block_size=block_size,
        num_warps=8,
    )
    return output


def _gated_residual_fake(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    gate_mode: int,
    rows_per_batch: int,
    gate_row_stride: int,
) -> torch.Tensor:
    del branch, gate, gate_mode, rows_per_batch, gate_row_stride
    return torch.empty_like(residual)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "gated_residual"):
    direct_register_custom_op(
        op_name="gated_residual",
        op_func=_gated_residual_impl,
        fake_impl=_gated_residual_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def _row_stride(gate: torch.Tensor) -> int | None:
    if gate.ndim == 1:
        return gate.shape[-1]
    row_stride = gate.stride(-2)
    rows = 1
    for dimension in range(gate.ndim - 2, -1, -1):
        if gate.shape[dimension] != 1 and gate.stride(dimension) != rows * row_stride:
            return None
        rows *= gate.shape[dimension]
    return row_stride


def _gate_layout(residual: torch.Tensor, gate: torch.Tensor) -> tuple[int, int, int]:
    if gate.ndim > residual.ndim:
        return _EAGER_GATE, 1, 0

    padded_shape = (1,) * (residual.ndim - gate.ndim) + tuple(gate.shape)
    leading_shape = padded_shape[:-1]
    residual_leading_shape = tuple(residual.shape[:-1])
    if all(size == 1 for size in leading_shape):
        return _GLOBAL_GATE, 1, 0
    if residual.ndim >= 3 and leading_shape[0] == residual.shape[0] and all(size == 1 for size in leading_shape[1:]):
        return _BATCH_GATE, math.prod(residual.shape[1:-1]), gate.stride(0)
    if leading_shape == residual_leading_shape:
        row_stride = _row_stride(gate)
        if row_stride is not None:
            return _ROW_GATE, 1, row_stride
    return _EAGER_GATE, 1, 0


def gated_residual(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Return ``residual + branch * gate`` with a fused CUDA fast path.

    The fast path supports fp16 and bf16 activations with a gate shared
    globally, per batch item, or per token. Other broadcastable layouts use
    the eager reference implementation.
    """
    if residual.shape != branch.shape:
        raise ValueError(f"residual and branch must have the same shape, got {residual.shape} and {branch.shape}")
    if residual.ndim == 0:
        raise ValueError("residual and branch must have at least one dimension")
    if not residual.is_floating_point() or not branch.is_floating_point() or not gate.is_floating_point():
        raise TypeError("gated_residual requires floating-point tensors")

    try:
        broadcast_shape = torch.broadcast_shapes(tuple(residual.shape), tuple(gate.shape))
    except RuntimeError as error:
        raise ValueError(f"gate shape {gate.shape} is not broadcastable to residual shape {residual.shape}") from error
    if broadcast_shape != tuple(residual.shape):
        raise ValueError(f"gate shape {gate.shape} broadcasts beyond residual shape {residual.shape}")

    gate_mode, rows_per_batch, gate_row_stride = _gate_layout(residual, gate)
    if not _fused_cuda_supported(residual, branch, gate, gate_mode):
        return _gated_residual_impl(residual, branch, gate, gate_mode, rows_per_batch, gate_row_stride)
    return torch.ops.vllm_omni.gated_residual(
        residual,
        branch,
        gate,
        gate_mode,
        rows_per_batch,
        gate_row_stride,
    )


__all__ = ["gated_residual"]
