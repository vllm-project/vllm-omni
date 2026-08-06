# SPDX-License-Identifier: Apache-2.0
"""Exact schedule identity and policy for MiniMax-H3 AdaLN reuse."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

MINIMAX_H3_ADALN_SCHEDULE_CACHE_ENV = "VLLM_OMNI_H3_ADALN_SCHEDULE_CACHE"
MINIMAX_H3_ADALN_OFFLOAD_WEIGHTS_ENV = "VLLM_OMNI_H3_ADALN_OFFLOAD_WEIGHTS"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

MiniMaxH3AdalnScheduleKey = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MiniMaxH3AdalnScheduleCache:
    """One model-local exact modulation schedule."""

    key: MiniMaxH3AdalnScheduleKey
    t_embs: tuple[torch.Tensor, ...]
    step_cursor: torch.Tensor
    steps: int
    max_unique_timesteps: int
    table_bytes: int
    projection_bytes: int = 0
    offloaded_projection_bytes: int = 0
    net_memory_saved_bytes: int = 0


def minimax_h3_adaln_schedule_cache_enabled() -> bool:
    """Return whether persistent exact-schedule AdaLN reuse is enabled."""
    return os.getenv(MINIMAX_H3_ADALN_SCHEDULE_CACHE_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def minimax_h3_adaln_weight_offload_enabled() -> bool:
    """Return whether exact tables should take ownership of AdaLN weights."""
    return minimax_h3_adaln_schedule_cache_enabled() and (
        os.getenv(MINIMAX_H3_ADALN_OFFLOAD_WEIGHTS_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES
    )


def minimax_h3_float32_bits(values: Any) -> tuple[int, ...]:
    """Return the signed int32 bit patterns of a CPU float32 tensor."""
    if not isinstance(values, torch.Tensor) or values.device.type != "cpu" or values.dtype != torch.float32:
        raise ValueError("schedule key values must be a CPU float32 tensor")
    return tuple(int(value) for value in values.contiguous().view(torch.int32).tolist())


def minimax_h3_adaln_schedule_key(
    metadata: Iterable[Any],
) -> MiniMaxH3AdalnScheduleKey:
    """Build an immutable schedule key without reading a device tensor."""
    key: list[tuple[int, ...]] = []
    for step in metadata:
        bits = getattr(step, "timestep_bits", None)
        if bits is None:
            raise ValueError("timestep metadata is missing host timestep_bits")
        key.append(tuple(int(value) for value in bits))
    if not key:
        raise ValueError("AdaLN schedule must contain at least one denoise step")
    return tuple(key)


def minimax_h3_build_adaln_table(
    projection: Any,
    t_embs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Project each step at its original shape and pad only unused rows."""
    rows_by_step = tuple(projection.compute_flat(t_emb) for t_emb in t_embs)
    max_rows = max(int(rows.shape[0]) for rows in rows_by_step)
    padded: list[torch.Tensor] = []
    for rows in rows_by_step:
        missing = max_rows - int(rows.shape[0])
        if missing:
            rows = torch.cat(
                [rows, rows.new_zeros((missing, rows.shape[1]))],
                dim=0,
            )
        padded.append(rows)
    return torch.stack(padded, dim=0)


__all__ = [
    "MINIMAX_H3_ADALN_OFFLOAD_WEIGHTS_ENV",
    "MINIMAX_H3_ADALN_SCHEDULE_CACHE_ENV",
    "MiniMaxH3AdalnScheduleCache",
    "MiniMaxH3AdalnScheduleKey",
    "minimax_h3_build_adaln_table",
    "minimax_h3_adaln_schedule_cache_enabled",
    "minimax_h3_adaln_weight_offload_enabled",
    "minimax_h3_adaln_schedule_key",
    "minimax_h3_float32_bits",
]
