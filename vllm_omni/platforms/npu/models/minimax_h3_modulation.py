# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from collections.abc import Callable
from ctypes import c_float
from functools import partial

import torch
from torch import nn

from vllm_omni.diffusion.layers.indexed_modulation import indexed_gate, rms_norm_indexed_scale_shift
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTBlock


def _supports_indexed_inputs(
    x: torch.Tensor, table: torch.Tensor, indices: torch.Tensor, *values: torch.Tensor
) -> bool:
    return (
        x.device.type == "npu"
        and x.ndim == 2
        and x.numel() > 0
        and table.ndim == 2
        and table.shape[0] > 0
        and table.shape[1] == x.shape[1]
        and indices.shape == (x.shape[0],)
        and indices.dtype in (torch.int32, torch.int64)
        and indices.device == x.device
        and all(t.device == x.device and t.dtype == torch.bfloat16 and t.stride(-1) == 1 for t in (x, table, *values))
    )


def _unit_inner_stride(t: torch.Tensor) -> torch.Tensor:
    # Omni's Triton kernels also address columns with unit stride.
    return t if t.stride(-1) == 1 else t.contiguous()


def _indexed_gate(
    sdk_gate: Callable[..., torch.Tensor],
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if other.shape == x.shape and _supports_indexed_inputs(x, gate, indices, other):
        return sdk_gate(torch.empty_like(x, memory_format=torch.contiguous_format), x, gate, other, indices)
    return indexed_gate(_unit_inner_stride(x), _unit_inner_stride(gate), _unit_inner_stride(other), indices)


def _rms_norm_indexed_scale_shift(
    sdk_rms: Callable[..., torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if (
        _supports_indexed_inputs(x, shift, indices, weight, scale)
        and x.shape[1] <= 8192
        and weight.shape == (x.shape[1],)
        and scale.shape == shift.shape
        and 0 < c_float(eps).value < math.inf
    ):
        return sdk_rms(x, weight, shift, scale, indices, eps)
    return rms_norm_indexed_scale_shift(
        _unit_inner_stride(x),
        _unit_inner_stride(weight),
        _unit_inner_stride(shift),
        _unit_inner_stride(scale),
        indices,
        eps,
    )


def apply_minimax_h3_modulation(model: nn.Module) -> None:
    """Install MindIE-SD modulation on the loaded H3 blocks on Ascend 910 and 950."""
    if not torch.npu.get_device_name().lower().startswith(("ascend910", "ascend950")):
        return
    try:
        from mindiesd import rms_norm_indexed_scale_shift as sdk_rms
        from mindiesd.layers.index_modulation import indexed_gate as sdk_gate
    except ImportError:
        return

    gate = partial(_indexed_gate, sdk_gate)
    rms = partial(_rms_norm_indexed_scale_shift, sdk_rms)
    for block in model.modules():
        if isinstance(block, MiniMaxH3DiTBlock):
            block._indexed_gate = gate  # type: ignore[method-assign]
            block._rms_norm_indexed_scale_shift = rms  # type: ignore[method-assign]
