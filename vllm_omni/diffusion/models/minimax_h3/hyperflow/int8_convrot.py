# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Hadamard helper adapted from Video-Rebirth/hyperflow-h3-flash int8.py (Apache-2.0).
"""Hadamard rotation used to densify HyperFlow INT8 convrot weights."""

from __future__ import annotations

import functools

import torch
from torch import Tensor


@functools.cache
def regular_hadamard(group: int) -> Tensor:
    """``kron(H4 / 2, ...)`` of size ``group`` (power of 4): orthonormal, symmetric."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64) / 2
    h, n = torch.ones(1, 1, dtype=torch.float64), 1
    while n < group:
        h, n = torch.kron(h, h4), n * 4
    if n != group:
        raise ValueError(f"Hadamard group {group} is not a power of 4")
    return h.float()


__all__ = ["regular_hadamard"]
