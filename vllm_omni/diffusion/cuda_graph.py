# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared configuration for diffusion CUDA graph replay.

vLLM's core ``CUDAGraphWrapper`` is tied to the LLM forward context and expects
static input buffers to be managed by the caller. Diffusion pipelines that need
CUDA graphs provide model-specific runners, while this module keeps the CLI /
config surface shared across those runners.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass
class DiffusionCUDAGraphConfig:
    """Configuration for diffusion CUDA graph capture/replay."""

    enabled: bool = False
    max_graphs: int = 4
    warmup_steps: int = 1
    clone_outputs: bool = True
    use_global_graph_pool: bool = True
    clear_cuda_cache_on_capture: bool = False
    name: str = "diffusion"

    def __post_init__(self) -> None:
        if self.max_graphs < 1:
            raise ValueError("diffusion cuda_graph_config.max_graphs must be >= 1")

    @classmethod
    def from_value(
        cls,
        value: DiffusionCUDAGraphConfig | Mapping[str, Any] | None,
        *,
        enabled: bool | None = None,
    ) -> DiffusionCUDAGraphConfig:
        if value is None:
            config = cls()
        elif isinstance(value, cls):
            config = value
        elif isinstance(value, Mapping):
            valid_fields = {field.name for field in fields(cls)}
            unknown_fields = set(value) - valid_fields
            if unknown_fields:
                raise ValueError(
                    "Unknown diffusion cuda_graph_config field(s): "
                    f"{sorted(str(field) for field in unknown_fields)}. "
                    f"Valid fields are: {sorted(valid_fields)}."
                )
            config = cls(**dict(value))
        else:
            raise TypeError(
                f"cuda_graph_config must be a DiffusionCUDAGraphConfig, mapping, or None, got {type(value)!r}"
            )
        if enabled:
            config = replace(config, enabled=True)
        return config


__all__ = ["DiffusionCUDAGraphConfig"]
