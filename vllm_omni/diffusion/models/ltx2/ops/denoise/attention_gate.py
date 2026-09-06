# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact Triton fusion for the LTX-2 per-head attention gate."""

from __future__ import annotations

import logging

import torch
from vllm.triton_utils import tl, triton

from ..numerics import round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_ALIGNMENT = 16
_BLOCK_SIZE = 1024
_FAILED_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _attention_gate_kernel(
    output_ptr,
    hidden_states_ptr,
    gate_logits_ptr,
    numel,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    hidden_states = tl.load(hidden_states_ptr + offsets, mask=mask).to(tl.float32)
    gate_offsets = offsets // head_dim
    logits = tl.load(gate_logits_ptr + gate_offsets, mask=mask).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-logits))
    gates = round_bf16_to_fp32(2.0 * round_bf16_to_fp32(sigmoid))
    tl.store(output_ptr + offsets, hidden_states * gates, mask=mask)


def can_use_triton_attention_gate(
    hidden_states: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> bool:
    return (
        is_ltx2_ops_eligible(hidden_states)
        and hidden_states.device.index not in _FAILED_DEVICES
        and hidden_states.dtype is torch.bfloat16
        and gate_logits.dtype is hidden_states.dtype
        and hidden_states.is_cuda
        and gate_logits.is_cuda
        and hidden_states.device == gate_logits.device
        and hidden_states.dim() == 3
        and gate_logits.dim() == 3
        and hidden_states.shape[:2] == gate_logits.shape[:2]
        and head_dim > 0
        and head_dim % (_ALIGNMENT // hidden_states.element_size()) == 0
        and hidden_states.shape[2] == gate_logits.shape[2] * head_dim
        and hidden_states.numel() > 0
        and hidden_states.is_contiguous()
        and gate_logits.is_contiguous()
        and hidden_states.data_ptr() % _ALIGNMENT == 0
        and gate_logits.data_ptr() % _ALIGNMENT == 0
    )


def _run_attention_gate(
    hidden_states: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if not can_use_triton_attention_gate(hidden_states, gate_logits, head_dim):
        raise ValueError("unsupported tensors for LTX-2 Triton attention gate")
    output = torch.empty_like(hidden_states)
    with torch.accelerator.device_index(hidden_states.device.index):
        _attention_gate_kernel[(triton.cdiv(hidden_states.numel(), _BLOCK_SIZE),)](
            output,
            hidden_states,
            gate_logits,
            hidden_states.numel(),
            head_dim=head_dim,
            block_size=_BLOCK_SIZE,
            num_warps=8,
        )
    return output


def try_attention_gate_exact(
    hidden_states: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> torch.Tensor | None:
    """Return the exact verified-CUDA result, or ``None`` outside its contract."""

    if not can_use_triton_attention_gate(hidden_states, gate_logits, head_dim):
        return None
    try:
        return _run_attention_gate(hidden_states, gate_logits, head_dim)
    except Exception as exc:  # noqa: BLE001 - preserve inference after JIT failure
        _FAILED_DEVICES.add(hidden_states.device.index)
        logger.warning(
            "Disabling LTX-2 Triton attention gate on %s/%s after failure: %s",
            hidden_states.device,
            hidden_states.dtype,
            exc,
        )
        return None


__all__ = ["try_attention_gate_exact"]
