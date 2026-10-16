# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""MAGI-2 rotary validation over the shared portable implementation."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch
from vllm_omni.diffusion.layers.rope import rotate_half as rotate_half


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply shared RoPE while preserving MAGI-2's invalid-dimension error."""

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    return apply_rotary_emb_torch(x, cos, sin, interleaved=interleaved)
