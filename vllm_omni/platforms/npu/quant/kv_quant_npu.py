# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Seeded rotation matrices for MindIE-SD quantized attention."""

from __future__ import annotations

import functools
import math

import torch


@functools.lru_cache(maxsize=128)
def get_quant_attention_rotation(device: torch.device, dtype: torch.dtype, head_dim: int) -> torch.Tensor:
    """Preserve Omni's fixed Walsh-Hadamard rotation without changing global RNG."""
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("Generated attention rotations require a power-of-two head dimension.")
    generator = torch.Generator(device="cpu").manual_seed(425500)
    signs = torch.randint(0, 2, (head_dim,), generator=generator, dtype=torch.int64)
    signs = signs.to(dtype=torch.float32).mul_(2).sub_(1)
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < head_dim:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), dim=1), torch.cat((hadamard, -hadamard), dim=1)), dim=0)
    rotation = signs[:, None] * hadamard / math.sqrt(head_dim)
    return rotation.to(device=device, dtype=dtype).contiguous()
