# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAGI-2 layout adapter for the shared NPU rotary implementation."""

from __future__ import annotations

from importlib.util import find_spec

import torch

from vllm_omni.diffusion.layers.rope import apply_rotary_emb_mindiesd

from ..native.rope import apply_rotary_emb as native_apply_rotary_emb

_HAS_MINDIESD = find_spec("mindiesd") is not None


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Use shared MindIE-SD RoPE when its batch-angle contract is satisfied."""

    shared_angles = cos.ndim in (2, 3) and sin.ndim == cos.ndim
    if shared_angles and cos.ndim == 3:
        shared_angles = cos.shape[0] == sin.shape[0] == 1
    if x.device.type != "npu" or x.ndim != 4 or interleaved or not shared_angles or not _HAS_MINDIESD:
        return native_apply_rotary_emb(x, cos, sin, interleaved=interleaved)

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    rotated = apply_rotary_emb_mindiesd(
        x[..., :rotary_dim].contiguous(),
        cos.to(x.dtype),
        sin.to(x.dtype),
        interleaved=False,
        half_head_dim=True,
    )
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
