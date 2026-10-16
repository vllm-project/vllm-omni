# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU half-layout rotary operator for MAGI-2."""

from __future__ import annotations

import torch

from ..native.rope import apply_rotary_emb as native_apply_rotary_emb


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Use the NPU half-layout kernel for supported four-dimensional inputs."""

    if x.device.type != "npu" or x.ndim != 4 or interleaved:
        return native_apply_rotary_emb(x, cos, sin, interleaved=interleaved)

    import torch_npu

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2).to(x.dtype)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2).to(x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rotated = torch_npu.npu_rotary_mul(
        x[..., :rotary_dim].contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        rotary_mode="half",
    )
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
