# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validated configuration for central FIFO admission.

This module intentionally uses only the standard library. Configuration is
resolved once for head-side admission and routing; diffusion workers retain
their native execution configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class TailAwareSchedulingConfig:
    enabled: bool = False
    max_pending_requests: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        value = self.max_pending_requests
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_pending_requests must be an integer >= 1")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None = None) -> TailAwareSchedulingConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("tail_aware_scheduling_config must be an object")
        unknown = set(raw) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown tail-aware scheduling option(s): {', '.join(sorted(map(str, unknown)))}")
        return cls(**dict(raw))
