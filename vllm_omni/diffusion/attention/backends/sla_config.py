# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from dataclasses import dataclass


@dataclass
class SLASpec:
    """MindIE-SD sparse-linear attention options for MiniMax-H3.

    ``topk`` is the retained block fraction. ``zero_init`` permits missing
    projection weights for wiring checks, without claiming checkpoint quality.
    """

    topk: float = 0.05
    blkq: int = 64
    blkk: int = 64
    feature_map: str = "softmax"
    zero_init: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.topk, bool) or not isinstance(self.topk, (int, float)):
            raise ValueError("sla.topk must be a number in (0, 1].")
        if not math.isfinite(self.topk) or not 0 < self.topk <= 1:
            raise ValueError("sla.topk must be finite and in (0, 1].")
        for name in ("blkq", "blkk"):
            value = getattr(self, name)
            if type(value) is not int or value not in (64, 128):
                raise ValueError(f"sla.{name} must be 64 or 128.")
        if self.feature_map not in ("softmax", "elu", "relu"):
            raise ValueError("sla.feature_map must be softmax, elu or relu.")
        if type(self.zero_init) is not bool:
            raise ValueError("sla.zero_init must be a boolean.")
