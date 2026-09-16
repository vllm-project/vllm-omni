# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SeaCacheConfig:
    """Configuration for SeaCache.

    Defaults are tuned for Cosmos3 and may require adjustment for other models.
    """

    threshold: float = 0.25
    residual_order: int = 1
    max_consecutive_cached: int = 2
    power_exp: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError(f"threshold must be finite and non-negative, got {self.threshold}")
        if isinstance(self.residual_order, bool) or not isinstance(self.residual_order, int) or self.residual_order < 0:
            raise ValueError(f"residual_order must be a non-negative integer, got {self.residual_order!r}")
        if (
            isinstance(self.max_consecutive_cached, bool)
            or not isinstance(self.max_consecutive_cached, int)
            or self.max_consecutive_cached < 0
        ):
            raise ValueError(
                f"max_consecutive_cached must be a non-negative integer, got {self.max_consecutive_cached!r}"
            )
        if not math.isfinite(self.power_exp) or self.power_exp <= 0:
            raise ValueError(f"power_exp must be finite and positive, got {self.power_exp}")
