# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend rotary embedding implementation for MAGI-2."""

from __future__ import annotations

import os

import torch

from ..native.rotary_embedding import apply_rotary_emb as native_apply_rotary_emb
from ..native.rotary_embedding import rotate_half

_USE_NPU_ROTARY_MUL = os.environ.get("MAGI2_USE_NPU_ROTARY_MUL", "1") != "0"


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply RoPE with ``torch_npu.npu_rotary_mul`` for the supported layout."""

    if not _USE_NPU_ROTARY_MUL or x.device.type != "npu" or x.ndim != 4 or interleaved:
        return native_apply_rotary_emb(x, cos, sin, interleaved=interleaved)

    import torch_npu

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    rotary_input = x[..., :rotary_dim]
    cos = cos.to(rotary_input.dtype)
    sin = sin.to(rotary_input.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rotated = torch_npu.npu_rotary_mul(
        rotary_input.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        rotary_mode="half",
    )
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


__all__ = ["apply_rotary_emb", "rotate_half"]
