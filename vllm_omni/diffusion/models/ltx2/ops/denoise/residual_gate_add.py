# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact Triton residual gate additions for LTX-2."""

from __future__ import annotations

import logging

import torch
from vllm.triton_utils import tl, triton

from ..numerics import round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_ALIGNMENT = 16
_VECTOR_ELEMENTS = _ALIGNMENT // 2
_MIN_PROFITABLE_NUMEL = 1 << 20
_BLOCK_SIZE = 1024
_FAILED_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _residual_gate_add_kernel(
    output_ptr,
    residual_ptr,
    update_ptr,
    gate_ptr,
    gate_table_ptr,
    perturbation_mask_ptr,
    numel,
    rows_per_batch,
    hidden_size: tl.constexpr,
    gate_batch_stride,
    gate_row_stride,
    has_table: tl.constexpr,
    has_mask: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    valid = offsets < numel
    columns = offsets % hidden_size
    rows = offsets // hidden_size
    batches = rows // rows_per_batch
    tokens = rows % rows_per_batch

    residual = tl.load(residual_ptr + offsets, mask=valid).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=valid).to(tl.float32)
    gate_offsets = batches * gate_batch_stride + tokens * gate_row_stride + columns
    gate = tl.load(gate_ptr + gate_offsets, mask=valid).to(tl.float32)
    if has_table:
        gate_table = tl.load(gate_table_ptr + columns, mask=valid).to(tl.float32)
        gate = round_bf16_to_fp32(gate_table + gate)
    if has_mask:
        perturbation_mask = tl.load(perturbation_mask_ptr + batches, mask=valid).to(tl.float32)
        update = round_bf16_to_fp32(update * perturbation_mask)
    product = round_bf16_to_fp32(update * gate)
    tl.store(output_ptr + offsets, residual + product, mask=valid)


def _supported_inputs(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_table: torch.Tensor | None = None,
    perturbation_mask: torch.Tensor | None = None,
) -> bool:
    if residual.dim() != 3:
        return False
    gate_ok = (
        gate.dim() == 3
        and gate.shape[0] == residual.shape[0]
        and gate.shape[1] in (1, residual.shape[1])
        and gate.shape[2] == residual.shape[2]
        and gate.stride(-1) == 1
        and gate.stride(0) % _VECTOR_ELEMENTS == 0
        and gate.stride(1) % _VECTOR_ELEMENTS == 0
    )
    table_ok = gate_table is None or (
        gate_table.is_cuda
        and gate_table.device == residual.device
        and gate_table.dtype is residual.dtype
        and gate_table.shape == (residual.shape[-1],)
        and gate_table.is_contiguous()
        and gate_table.data_ptr() % _ALIGNMENT == 0
    )
    mask_ok = perturbation_mask is None or (
        perturbation_mask.is_cuda
        and perturbation_mask.device == residual.device
        and perturbation_mask.dtype is residual.dtype
        and perturbation_mask.shape == (residual.shape[0], 1, 1)
        and perturbation_mask.is_contiguous()
    )
    return (
        is_ltx2_ops_eligible(residual)
        and residual.device.index not in _FAILED_DEVICES
        and residual.dtype is torch.bfloat16
        and update.dtype is residual.dtype
        and gate.dtype is residual.dtype
        and residual.is_cuda
        and update.is_cuda
        and gate.is_cuda
        and residual.device == update.device == gate.device
        and residual.shape == update.shape
        and residual.numel() > 0
        and residual.shape[-1] % _VECTOR_ELEMENTS == 0
        and residual.is_contiguous()
        and update.is_contiguous()
        and gate_ok
        and table_ok
        and mask_ok
        and residual.data_ptr() % _ALIGNMENT == 0
        and update.data_ptr() % _ALIGNMENT == 0
        and gate.data_ptr() % _ALIGNMENT == 0
    )


def _run_residual_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_table: torch.Tensor | None = None,
    perturbation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _supported_inputs(residual, update, gate, gate_table, perturbation_mask):
        raise ValueError("unsupported tensors for LTX-2 Triton residual gate add")
    output = torch.empty_like(residual)
    table_arg = gate if gate_table is None else gate_table
    mask_arg = gate if perturbation_mask is None else perturbation_mask
    with torch.accelerator.device_index(residual.device.index):
        _residual_gate_add_kernel[(triton.cdiv(residual.numel(), _BLOCK_SIZE),)](
            output,
            residual,
            update,
            gate,
            table_arg,
            mask_arg,
            residual.numel(),
            residual.shape[1],
            hidden_size=residual.shape[2],
            gate_batch_stride=gate.stride(0),
            gate_row_stride=0 if gate.shape[1] == 1 else gate.stride(1),
            has_table=gate_table is not None,
            has_mask=perturbation_mask is not None,
            block_size=_BLOCK_SIZE,
            num_warps=8,
        )
    return output


def _disable_after_failure(residual: torch.Tensor, exc: Exception) -> None:
    _FAILED_DEVICES.add(residual.device.index)
    logger.warning(
        "Disabling LTX-2 Triton residual gate add on %s after failure: %s",
        residual.device,
        exc,
    )


def try_residual_gate_add_exact(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_table: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return the exact eager result, or ``None`` outside its contract."""

    if residual.numel() < _MIN_PROFITABLE_NUMEL or not _supported_inputs(
        residual,
        update,
        gate,
        gate_table,
    ):
        return None
    try:
        return _run_residual_gate_add(residual, update, gate, gate_table)
    except Exception as exc:  # noqa: BLE001 - fail closed after Triton failure
        _disable_after_failure(residual, exc)
        return None


def try_masked_residual_gate_add_exact(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    perturbation_mask: torch.Tensor,
    gate_table: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return the exact masked eager result, or ``None`` outside its contract."""

    if residual.numel() < _MIN_PROFITABLE_NUMEL or not _supported_inputs(
        residual,
        update,
        gate,
        gate_table,
        perturbation_mask=perturbation_mask,
    ):
        return None
    try:
        return _run_residual_gate_add(
            residual,
            update,
            gate,
            gate_table,
            perturbation_mask=perturbation_mask,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed after Triton failure
        _disable_after_failure(residual, exc)
        return None


__all__ = [
    "try_masked_residual_gate_add_exact",
    "try_residual_gate_add_exact",
]
