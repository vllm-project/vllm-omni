# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic startup handoff between model loaders and offload backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan

_STARTUP_STATE_ATTR = "_omni_offload_startup_state"


@dataclass
class OffloadStartupState:
    """Loader-owned state consumed by the offloader startup boundary."""

    host_weight_plan: HostWeightPlan | None = None
    fresh_model_loader: Callable[[], nn.Module] | None = None
    allow_fresh_retry: bool = False

    def __post_init__(self) -> None:
        if self.allow_fresh_retry != (self.fresh_model_loader is not None):
            raise ValueError("fresh-model retry policy requires exactly one retry callback")

    def close_loader_ownership(self) -> None:
        """Release a plan that never reached a backend."""
        if self.host_weight_plan is None:
            return
        carrier = self.host_weight_plan.lease_carrier
        if carrier is not None:
            carrier.close()


def attach_offload_startup_state(model: nn.Module, state: OffloadStartupState) -> None:
    """Attach one process-local startup handoff to a loaded pipeline."""
    previous = getattr(model, _STARTUP_STATE_ATTR, None)
    if previous is not None:
        previous.close_loader_ownership()
    setattr(model, _STARTUP_STATE_ATTR, state)


def take_offload_startup_state(model: nn.Module) -> OffloadStartupState | None:
    """Take and remove the loader handoff from a pipeline exactly once."""
    state = getattr(model, _STARTUP_STATE_ATTR, None)
    if state is None:
        return None
    delattr(model, _STARTUP_STATE_ATTR)
    return state


__all__ = [
    "OffloadStartupState",
    "attach_offload_startup_state",
    "take_offload_startup_state",
]
