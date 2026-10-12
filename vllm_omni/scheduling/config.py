# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validated configuration for the unified tail-aware scheduling policy.

This module intentionally uses only the standard library. Configuration is
resolved once for head-side admission and routing; diffusion workers retain
their native execution configuration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class TailAwareSchedulingConfig:
    enabled: bool = False
    hardware_profile: str | None = None
    max_pending_requests: int = 1024
    risk_beta: float = 0.85
    band_risk_beta: float = 0.625
    band_min_pending: int = 10
    band_max_pending: int = 27

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if self.hardware_profile is not None and (
            not isinstance(self.hardware_profile, str) or self.hardware_profile not in {"910B2", "910B3"}
        ):
            raise ValueError("hardware_profile must be '910B2' or '910B3'")
        if self.enabled and self.hardware_profile is None:
            raise ValueError("tail-aware scheduling requires an explicit hardware_profile ('910B2' or '910B3')")
        for name in (
            "max_pending_requests",
            "band_min_pending",
            "band_max_pending",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("risk_beta", "band_risk_beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.band_max_pending < self.band_min_pending:
            raise ValueError("band_max_pending cannot be less than band_min_pending")

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
