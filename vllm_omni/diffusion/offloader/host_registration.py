# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Platform-neutral lifecycle for registering existing host mappings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from vllm_omni.host_weight_runtime import MappedHostRegion


class HostRegistrationError(RuntimeError):
    """Raised when existing host ranges cannot be registered safely."""


class HostRegistrationBudgetError(HostRegistrationError):
    """Raised before mutation when the complete mapping exceeds its budget."""


class HostRegistrationCleanupError(HostRegistrationError):
    """Raised when registered ranges cannot be released safely."""

    def __init__(self, message: str, *, active_registration: HostRegistration | None = None) -> None:
        super().__init__(message)
        self.active_registration = active_registration


class HostRegistration(Protocol):
    """Transport-owned registration for one process-local host mapping."""

    @property
    def total_bytes(self) -> int: ...

    @property
    def region_count(self) -> int: ...

    def close(self) -> tuple[str, ...]: ...


def register_host_mappings(
    regions: Sequence[MappedHostRegion],
    *,
    device: torch.device,
    max_bytes: int | None,
) -> HostRegistration:
    """Register immutable ranges with the active platform implementation."""
    if device.type == "cuda":
        from .cuda_host_registration import CudaHostRegistration

        return CudaHostRegistration.create(regions, max_bytes=max_bytes)
    raise HostRegistrationError(f"host-mapping registration is not supported on {device.type}")


__all__ = [
    "HostRegistration",
    "HostRegistrationBudgetError",
    "HostRegistrationCleanupError",
    "HostRegistrationError",
    "register_host_mappings",
]
