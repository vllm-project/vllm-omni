# SPDX-License-Identifier: Apache-2.0
"""Static, per-worker-GPU HBM budgets. No GPU imports or allocator mutations.

A budget is a profiled operating envelope, not a CUDA allocation hard limit.
The reserve covers graphs, transfers and slack not represented in profiling.
"""
from __future__ import annotations

import math

GIB = 1024**3


def budget_bytes(limit_gb: float | None, reserved_gb: float = 2.0) -> int | None:
    for name, value in (("hbm_limit_gb", limit_gb), ("hbm_reserved_gb", reserved_gb)):
        if value is None and name == "hbm_limit_gb":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (name == "hbm_limit_gb" and value == 0):
            raise ValueError(f"{name} has an invalid value: {value}")
    if limit_gb is None:
        return None
    limit = int(limit_gb * GIB)
    if limit <= int(reserved_gb * GIB):
        raise ValueError("hbm_limit_gb must exceed hbm_reserved_gb")
    return limit


def derive_kv_budget(
    limit_gb: float,
    reserved_gb: float,
    non_kv_bytes: int,
    explicit_kv_bytes: int | None = None,
) -> int:
    limit = budget_bytes(limit_gb, reserved_gb)
    assert limit is not None
    if non_kv_bytes < 0:
        raise ValueError("Profiled non-KV memory cannot be negative")
    available = limit - int(reserved_gb * GIB) - non_kv_bytes
    if available <= 0:
        raise ValueError(
            f"Static HBM budget exhausted: total={limit}, non_kv={non_kv_bytes}, "
            f"reserve={int(reserved_gb * GIB)} bytes; reduce execution limits or increase budget"
        )
    if explicit_kv_bytes is not None:
        if isinstance(explicit_kv_bytes, bool) or explicit_kv_bytes <= 0:
            raise ValueError("Explicit KV budget must be positive in static HBM mode")
        available = min(available, explicit_kv_bytes)
    return available


def initial_budget(limit_gb: float, reserved_gb: float, free_bytes: int) -> int:
    limit = budget_bytes(limit_gb, reserved_gb)
    assert limit is not None
    if limit > free_bytes:
        raise ValueError(f"Static HBM budget {limit} exceeds initial free memory {free_bytes} bytes")
    return limit
