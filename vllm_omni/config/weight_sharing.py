# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deployment policy for one colocated AR/DiT weight-sharing group.

The ratio covers both stages together on each GPU. Weight sizes and private
memory peaks are measured at startup; KV sizes remain stage-owned settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightSharingConfig:
    provider_stage: int
    consumer_stage: int
    gpu_memory_utilization: float = 0.9
    timeout: float = 120.0

    def __post_init__(self) -> None:
        for name in ("provider_stage", "consumer_stage"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"weight_sharing.{name} must be a nonnegative integer")
        if self.provider_stage == self.consumer_stage:
            raise ValueError("weight sharing requires distinct stages")
        ratio = self.gpu_memory_utilization
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("weight_sharing.gpu_memory_utilization must be finite and in (0, 1]")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("weight_sharing.timeout must be positive and finite")


def parse_weight_sharing(value: object) -> WeightSharingConfig | None:
    """Absence disables sharing; unknown keys and malformed policies fail early."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("weight_sharing must be a mapping or null")
    return WeightSharingConfig(**value)
