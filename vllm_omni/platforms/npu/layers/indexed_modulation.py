# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from ctypes import c_float
from functools import cache
from math import inf
from types import ModuleType

import torch


@cache
def _supports_mindiesd_device(device: torch.device) -> bool:
    # The gate kernel only defines UB sizes for these chip families.
    return torch.npu.get_device_name(device).lower().startswith(("ascend910", "ascend950"))


@cache
def _load_mindiesd() -> ModuleType | None:
    try:
        from mindiesd.layers import index_modulation
    except ImportError:
        return None
    return index_modulation


def _supports_mindiesd_inputs(
    x: torch.Tensor, table: torch.Tensor, indices: torch.Tensor, *values: torch.Tensor
) -> bool:
    return (
        x.numel() > 0
        and table.shape[1:] == x.shape[1:]
        and table.numel() > 0
        and indices.shape == x.shape[:1]
        and indices.dtype in (torch.int32, torch.int64)
        and indices.device == x.device
        and all(t.device == x.device and t.dtype == torch.bfloat16 and t.stride(-1) == 1 for t in (x, table, *values))
        and _supports_mindiesd_device(x.device)
    )


def try_indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    if x.ndim != 2 or other.shape != x.shape or not _supports_mindiesd_inputs(x, gate, indices, other):
        return None
    sdk = _load_mindiesd()
    if sdk is None:
        return None
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    return sdk.indexed_gate(output, x, gate, other, indices)


def try_rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if (
        x.ndim != 2
        or x.shape[1] > 8192
        or weight.shape != x.shape[1:]
        or scale.shape != shift.shape
        or not 0 < c_float(eps).value < inf
        or not _supports_mindiesd_inputs(x, shift, indices, weight, scale)
    ):
        return None
    sdk = _load_mindiesd()
    if sdk is None:
        return None
    return sdk.rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
