# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""K/V demand routing and manifest controls for sparse Ring attention."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SparSPSpec:
    """Use the same sparse selector, K/V dtype, and NCCL transport in every mode."""

    routing: Literal["full_ring", "selective_direct", "selective_pair_relay"] = "full_ring"
    cdf: float = 0.90
    sim_threshold: float = 0.1
    pv_threshold: float = 50.0
    demand_planner: Literal["torch", "manifest"] = "torch"
    placement: Literal["none", "interval_periodic", "auto"] = "none"

    def __post_init__(self) -> None:
        if self.routing not in ("full_ring", "selective_direct", "selective_pair_relay"):
            raise ValueError("invalid SparSP routing")
        if self.demand_planner not in ("torch", "manifest"):
            raise ValueError("SparSP demand_planner must be torch or manifest")
        if self.placement not in ("none", "interval_periodic", "auto"):
            raise ValueError("invalid SparSP placement")
        if self.demand_planner == "manifest" and self.routing == "full_ring":
            raise ValueError("SparSP manifest demand planner requires selective routing")
        if self.routing == "selective_pair_relay" and self.demand_planner != "manifest":
            raise ValueError("SparSP pair relay requires the manifest demand planner")
        for name in ("cdf", "sim_threshold", "pv_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"SparSP {name} must be a finite number")
        if not 0 < self.cdf <= 1 or not -1 <= self.sim_threshold <= 1 or self.pv_threshold <= 0:
            raise ValueError("SparSP requires CDF in (0, 1], similarity in [-1, 1] and positive PV threshold")
