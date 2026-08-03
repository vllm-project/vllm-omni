# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared tensor utilities for distributed layerwise offload.

These helpers are used by both DistributedLayerwiseOffloadHook and
DistributedLayerwiseOffloadBackend, and can be reused by other
offload backends.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor


def dtype_size(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch.dtype."""
    return torch.empty(1, dtype=dtype).element_size()


def is_dtensor(t: torch.Tensor) -> bool:
    """Check if tensor is a DTensor."""
    return isinstance(t, DTensor)


def set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
    """Replace target's underlying storage with value (zero-copy)."""
    if is_dtensor(target):
        target._local_tensor = value
    else:
        target.data = value


def make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
    """Create a zero-element placeholder to free GPU memory."""
    if is_dtensor(tensor):
        local_shape = tuple(tensor.to_local().shape)
        return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
    return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)


def is_materialized_tensor(t: torch.Tensor) -> bool:
    """Check if tensor holds real data (not meta or empty placeholder)."""
    if is_dtensor(t):
        local_t = t.to_local()
        return not local_t.is_meta
    return not t.is_meta and t.data.numel() > 0
