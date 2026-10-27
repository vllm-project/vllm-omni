# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Diffusion dispatch policies and backend execution schedulers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler, SchedulerInterface
    from vllm_omni.diffusion.sched.interface import (
        CachedRequestData,
        DiffusionRequestStatus,
        DiffusionSchedulerOutput,
        KVPrefetchJob,
        NewRequestData,
        SchedulerRequestState,
        StepBatchSamplingParamsKey,
    )
    from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
    from vllm_omni.diffusion.sched.sigma_schedule import BASE_SCHEDULE_KEY, DMD2SigmaSchedule
    from vllm_omni.diffusion.sched.step_scheduler import StepScheduler

    Scheduler = RequestScheduler

# Dispatch configuration must not load worker or model execution dependencies.
_LAZY_MODULES = {
    "DiffusionRequestStatus": "interface",
    "CachedRequestData": "interface",
    "DiffusionSchedulerOutput": "interface",
    "KVPrefetchJob": "interface",
    "NewRequestData": "interface",
    "SchedulerRequestState": "interface",
    "BaseScheduler": "base_scheduler",
    "SchedulerInterface": "base_scheduler",
    "StepBatchSamplingParamsKey": "interface",
    "BASE_SCHEDULE_KEY": "sigma_schedule",
    "DMD2SigmaSchedule": "sigma_schedule",
    "RequestScheduler": "request_scheduler",
    "StepScheduler": "step_scheduler",
    "Scheduler": "request_scheduler",
}

__all__ = [
    "DiffusionRequestStatus",
    "CachedRequestData",
    "DiffusionSchedulerOutput",
    "KVPrefetchJob",
    "NewRequestData",
    "SchedulerRequestState",
    "BaseScheduler",
    "SchedulerInterface",
    "StepBatchSamplingParamsKey",
    "BASE_SCHEDULE_KEY",
    "DMD2SigmaSchedule",
    "RequestScheduler",
    "StepScheduler",
    "Scheduler",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr_name = "RequestScheduler" if name == "Scheduler" else name
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
