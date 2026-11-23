# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""K/V demand routing and planner controls for sparse Ring attention."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SparSPSpec:
    """Use the same sparse selector, K/V dtype, and NCCL transport in every mode."""

    routing: Literal["full_ring", "selective_direct"] = "full_ring"
    cdf: float = 0.90
    sim_threshold: float = 0.1
    pv_threshold: float = 50.0
    demand_planner: Literal["torch", "manifest", "per_hop"] = "torch"
    placement: Literal["none", "interval_periodic", "auto"] = "none"

    def __post_init__(self) -> None:
        if self.routing not in ("full_ring", "selective_direct"):
            raise ValueError("invalid SparSP routing")
        if self.demand_planner not in ("torch", "manifest", "per_hop"):
            raise ValueError("SparSP demand_planner must be torch, manifest or per_hop")
        if self.placement not in ("none", "interval_periodic", "auto"):
            raise ValueError("invalid SparSP placement")
        if self.demand_planner != "torch" and self.routing != "selective_direct":
            raise ValueError(f"SparSP {self.demand_planner} demand planner requires selective_direct routing")
        for name in ("cdf", "sim_threshold", "pv_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"SparSP {name} must be a finite number")
        if not 0 < self.cdf <= 1 or not -1 <= self.sim_threshold <= 1 or self.pv_threshold <= 0:
            raise ValueError("SparSP requires CDF in (0, 1], similarity in [-1, 1] and positive PV threshold")
