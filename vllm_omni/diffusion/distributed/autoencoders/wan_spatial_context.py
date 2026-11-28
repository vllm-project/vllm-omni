# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-decode context shared by Wan spatial operators and kernels."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialShardContext:
    input_extent: int
    local_input_extent: int
    split_dim: str
    rank: int
    world_size: int


_SPATIAL_SHARD_CONTEXT: ContextVar[SpatialShardContext | None] = ContextVar(
    "wan_vae_spatial_shard_context",
    default=None,
)


def in_multi_gpu_spatial_decode() -> bool:
    context = _SPATIAL_SHARD_CONTEXT.get()
    return context is not None and context.world_size > 1
