# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class SeaCacheState:
    """Per-cache-context trajectory state."""

    last_step: int | None = None
    accumulated_distance: float = 0.0
    previous_indicator: list[torch.Tensor] | None = None
    history: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    consecutive_cached: int = 0

    def reset(self) -> None:
        self.last_step = None
        self.accumulated_distance = 0.0
        self.previous_indicator = None
        self.history.clear()
        self.consecutive_cached = 0
