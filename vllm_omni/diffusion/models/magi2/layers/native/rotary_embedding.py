# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native element-wise rotary embedding used by MAGI-2."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply the released element-wise RoPE layout to ``[...,H,D]``."""

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    if interleaved:
        cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)
        sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
    else:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim], interleaved) * sin
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


__all__ = ["apply_rotary_emb", "rotate_half"]
