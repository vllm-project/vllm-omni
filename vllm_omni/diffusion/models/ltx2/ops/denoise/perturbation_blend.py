# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact Triton perturbation blending plus attention gating for LTX-2."""

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
def _perturbation_blend_attention_gate_kernel(
    output_ptr,
    primary_ptr,
    fallback_ptr,
    mask_ptr,
    gate_logits_ptr,
    numel,
    rows_per_batch,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    primary_batch_stride,
    primary_row_stride,
    fallback_batch_stride,
    fallback_row_stride,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    valid = offsets < numel
    columns = offsets % hidden_size
    rows = offsets // hidden_size
    batches = rows // rows_per_batch
    tokens = rows % rows_per_batch

    primary_offsets = batches * primary_batch_stride + tokens * primary_row_stride + columns
    fallback_offsets = batches * fallback_batch_stride + tokens * fallback_row_stride + columns
    primary = tl.load(primary_ptr + primary_offsets, mask=valid).to(tl.float32)
    fallback = tl.load(fallback_ptr + fallback_offsets, mask=valid).to(tl.float32)
    blend_mask = tl.load(mask_ptr + batches, mask=valid).to(tl.float32)
    inverse_mask = round_bf16_to_fp32(1.0 - blend_mask)
    primary_product = round_bf16_to_fp32(primary * blend_mask)
    fallback_product = round_bf16_to_fp32(fallback * inverse_mask)
    blended = round_bf16_to_fp32(primary_product + fallback_product)

    heads = columns // head_dim
    gate_offsets = rows * (hidden_size // head_dim) + heads
    logits = tl.load(gate_logits_ptr + gate_offsets, mask=valid).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-logits))
    gates = round_bf16_to_fp32(2.0 * round_bf16_to_fp32(sigmoid))
    tl.store(output_ptr + offsets, blended * gates, mask=valid)


def _supported_blend_inputs(
    primary: torch.Tensor,
    fallback: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    return (
        is_ltx2_ops_eligible(primary)
        and primary.device.index not in _FAILED_DEVICES
        and primary.is_cuda
        and fallback.is_cuda
        and mask.is_cuda
        and primary.device == fallback.device == mask.device
        and primary.dtype is torch.bfloat16
        and fallback.dtype is primary.dtype
        and mask.dtype is primary.dtype
        and primary.dim() == 3
        and fallback.shape == primary.shape
        and mask.shape == (primary.shape[0], 1, 1)
        and primary.numel() >= _MIN_PROFITABLE_NUMEL
        and primary.shape[-1] % _VECTOR_ELEMENTS == 0
        and primary.stride(-1) == 1
        and fallback.stride(-1) == 1
        and primary.stride(0) % _VECTOR_ELEMENTS == 0
        and primary.stride(1) % _VECTOR_ELEMENTS == 0
        and fallback.stride(0) % _VECTOR_ELEMENTS == 0
        and fallback.stride(1) % _VECTOR_ELEMENTS == 0
        and mask.is_contiguous()
        and primary.data_ptr() % _ALIGNMENT == 0
        and fallback.data_ptr() % _ALIGNMENT == 0
    )


def _supported_composite_inputs(
    primary: torch.Tensor,
    fallback: torch.Tensor,
    mask: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> bool:
    return (
        _supported_blend_inputs(primary, fallback, mask)
        and gate_logits.is_cuda
        and gate_logits.device == primary.device
        and gate_logits.dtype is primary.dtype
        and gate_logits.is_contiguous()
        and gate_logits.shape[:2] == primary.shape[:2]
        and gate_logits.dim() == 3
        and 0 < gate_logits.shape[2] <= 64
        and head_dim > 0
        and head_dim % _VECTOR_ELEMENTS == 0
        and primary.shape[2] == gate_logits.shape[2] * head_dim
    )


def _run_perturbation_blend_attention_gate(
    primary: torch.Tensor,
    fallback: torch.Tensor,
    mask: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if not _supported_composite_inputs(primary, fallback, mask, gate_logits, head_dim):
        raise ValueError("unsupported tensors for fused LTX-2 perturbation blend/gate")
    output = torch.empty_like(primary, memory_format=torch.contiguous_format)
    with torch.accelerator.device_index(primary.device.index):
        _perturbation_blend_attention_gate_kernel[(triton.cdiv(primary.numel(), _BLOCK_SIZE),)](
            output,
            primary,
            fallback,
            mask,
            gate_logits,
            primary.numel(),
            primary.shape[1],
            hidden_size=primary.shape[2],
            head_dim=head_dim,
            primary_batch_stride=primary.stride(0),
            primary_row_stride=primary.stride(1),
            fallback_batch_stride=fallback.stride(0),
            fallback_row_stride=fallback.stride(1),
            block_size=_BLOCK_SIZE,
            num_warps=8,
        )
    return output


def try_perturbation_blend_attention_gate_exact(
    primary: torch.Tensor,
    fallback: torch.Tensor,
    mask: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> torch.Tensor | None:
    """Return the exact eager composite, or ``None`` outside its contract."""

    if not _supported_composite_inputs(primary, fallback, mask, gate_logits, head_dim):
        return None
    try:
        return _run_perturbation_blend_attention_gate(
            primary,
            fallback,
            mask,
            gate_logits,
            head_dim,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed after Triton failure
        _FAILED_DEVICES.add(primary.device.index)
        logger.warning(
            "Disabling LTX-2 Triton perturbation blend/gate on %s after failure: %s",
            primary.device,
            exc,
        )
        return None


__all__ = ["try_perturbation_blend_attention_gate_exact"]
