# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bit-exact workspace-reusing SwiGLU for the LTX diffusion VAE decoder."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from ..numerics import mul_rn_f32, round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_POINTWISE_BLOCK = 1024
_TILE_SIZE = 16384
_FAILED_KEYS: set[tuple[int | None, int, int, int, int]] = set()
_VERIFIED_KEYS: set[tuple[int | None, int, int, int, int]] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    elements,
    block: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=valid).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=valid).to(tl.float32)
    # Eager rounds SiLU to BF16 before the separate BF16 multiply.
    silu = round_bf16_to_fp32(mul_rn_f32(gate, tl.sigmoid(gate)))
    tl.store(gate_ptr + offsets, mul_rn_f32(silu, up), mask=valid)


def _weight_matches(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    shape: tuple[int, int],
) -> bool:
    return (
        weight.is_cuda
        and weight.device == hidden_states.device
        and weight.dtype is torch.bfloat16
        and weight.shape == shape
        and weight.is_contiguous()
    )


def _supported_inputs(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> bool:
    if hidden_states.ndim < 3:
        return False
    dim = hidden_states.shape[-1]
    hidden = gate_weight.shape[0] if gate_weight.ndim == 2 else 0
    return (
        is_ltx2_ops_eligible(hidden_states)
        and hidden_states.dtype is torch.bfloat16
        and hidden_states.shape[0] == 1
        and hidden_states.numel() > 0
        and hidden_states.is_contiguous()
        and 256 <= dim <= 2048
        and hidden == 4 * dim
        and _weight_matches(hidden_states, gate_weight, (hidden, dim))
        and _weight_matches(hidden_states, up_weight, (hidden, dim))
        and _weight_matches(hidden_states, down_weight, (dim, hidden))
    )


def _silu_mul_inplace(gate: torch.Tensor, up: torch.Tensor) -> None:
    with torch.accelerator.device_index(gate.device.index):
        _silu_mul_kernel[(triton.cdiv(gate.numel(), _POINTWISE_BLOCK),)](
            gate,
            up,
            gate.numel(),
            block=_POINTWISE_BLOCK,
            num_warps=4,
        )


def _reference_tile(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    return F.linear(
        F.silu(F.linear(hidden_states, gate_weight)) * F.linear(hidden_states, up_weight),
        down_weight,
    )


def _launch(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    dim = hidden_states.shape[-1]
    hidden = gate_weight.shape[0]
    flat = hidden_states.reshape(-1, dim)
    output = torch.empty_like(flat)
    workspace_gate = torch.empty((_TILE_SIZE, hidden), device=flat.device, dtype=flat.dtype)
    for start in range(0, flat.shape[0], _TILE_SIZE):
        end = min(start + _TILE_SIZE, flat.shape[0])
        tokens = end - start
        tile = flat[start:end]
        gate = workspace_gate[:tokens]
        torch.mm(tile, gate_weight.t(), out=gate)
        up = torch.mm(tile, up_weight.t())
        _silu_mul_inplace(gate, up)
        torch.mm(gate, down_weight.t(), out=output[start:end])
    return output.reshape(hidden_states.shape)


def _verify_edge_tiles(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> bool:
    dim = hidden_states.shape[-1]
    flat = hidden_states.reshape(-1, dim)
    actual = output.reshape(-1, dim)
    starts = {0, ((flat.shape[0] - 1) // _TILE_SIZE) * _TILE_SIZE}
    for start in starts:
        end = min(start + _TILE_SIZE, flat.shape[0])
        if not torch.equal(
            actual[start:end],
            _reference_tile(flat[start:end], gate_weight, up_weight, down_weight),
        ):
            return False
    return True


def try_swiglu_tiled_exact(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor | None:
    """Run workspace-reusing tiled SwiGLU, or expose the original module."""

    if not _supported_inputs(hidden_states, gate_weight, up_weight, down_weight):
        return None
    rows = hidden_states.numel() // hidden_states.shape[-1]
    runtime_key = (
        hidden_states.device.index,
        hidden_states.shape[-1],
        gate_weight.shape[0],
        min(rows, _TILE_SIZE),
        (rows - 1) % _TILE_SIZE + 1,
    )
    if runtime_key in _FAILED_KEYS:
        return None
    try:
        output = _launch(hidden_states, gate_weight, up_weight, down_weight)
        if runtime_key not in _VERIFIED_KEYS:
            if not _verify_edge_tiles(output, hidden_states, gate_weight, up_weight, down_weight):
                _FAILED_KEYS.add(runtime_key)
                logger.warning(
                    "Disabling LTX DiffVAE tiled SwiGLU fusion on %s after a bit-exactness mismatch",
                    hidden_states.device,
                )
                return None
            _VERIFIED_KEYS.add(runtime_key)
    except Exception as exc:  # noqa: BLE001 - fail closed after optimized-path failure
        _FAILED_KEYS.add(runtime_key)
        logger.warning(
            "Disabling LTX DiffVAE tiled SwiGLU fusion on %s after failure: %s",
            hidden_states.device,
            exc,
        )
        return None
    return output


__all__ = ["try_swiglu_tiled_exact"]
