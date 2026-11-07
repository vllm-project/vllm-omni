# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unified Loader / Offloader / Host Weight Manager weight stack.

Sub-RFC 01-1 establishes the package: the shared data types and protocols
(:mod:`vllm_omni.diffusion.weight_stack.interfaces`), the error hierarchy
(:mod:`vllm_omni.diffusion.weight_stack.errors`) and the resource ledger
(:mod:`vllm_omni.diffusion.weight_stack.startup`).  Later sub-RFCs add the
weight-stack switch (01-2), the offload execution facts (01-3), the replica-wide
confirmation protocol (01-4), the startup driver (01-5) and the worker dispatch
(01-6).

This package root only exports the startup state enum and the error types, so
importing it never pulls in a concrete Loader, host weight manager, Offloader or
configuration implementation.
"""

from __future__ import annotations

from vllm_omni.diffusion.weight_stack.errors import (
    AllocationError,
    CommitError,
    ConfigError,
    ConfirmationTimeoutError,
    ConstructionError,
    DeviceCapacityError,
    FinalizeError,
    GroupConfirmationError,
    HostCapacityError,
    LifecycleError,
    PeerFailureError,
    PlanningError,
    PreflightError,
    PrepareError,
    RuntimeFault,
    StackUnavailableError,
    WeightStackError,
    WriteError,
)
from vllm_omni.diffusion.weight_stack.interfaces import LoadState

__all__ = [
    "AllocationError",
    "CommitError",
    "ConfigError",
    "ConfirmationTimeoutError",
    "ConstructionError",
    "DeviceCapacityError",
    "FinalizeError",
    "GroupConfirmationError",
    "HostCapacityError",
    "LifecycleError",
    "LoadState",
    "PeerFailureError",
    "PlanningError",
    "PrepareError",
    "PreflightError",
    "RuntimeFault",
    "StackUnavailableError",
    "WeightStackError",
    "WriteError",
]
