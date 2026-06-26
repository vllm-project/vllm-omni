# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput


class DiffusionRequestStatus(enum.IntEnum):
    """Request status tracked by diffusion scheduler."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()

    # if any status is after or equal to FINISHED_COMPLETED, it is considered finished
    FINISHED_COMPLETED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    @staticmethod
    def is_finished(status: DiffusionRequestStatus) -> bool:
        return status >= DiffusionRequestStatus.FINISHED_COMPLETED


@dataclass(frozen=True, eq=True)
class SamplingParamsKey:
    """Batch-compatibility key derived from ``OmniDiffusionSamplingParams``.

    Only requests with the same key can be batched together.
    Fields not included here are treated as request-local and do not
    participate in the current homogeneous batching policy.
    """

    # Spatial / temporal shape.
    height: int | None = None
    width: int | None = None
    num_frames: int = 1
    resolution: int | str | None = None
    fps: int | None = None
    frame_rate: float | None = None
    boundary_ratio: float | None = None

    # CFG / guidance.
    do_classifier_free_guidance: bool = False
    guidance_scale: float = 0.0
    guidance_scale_provided: bool = False
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    true_cfg_scale: float | None = None
    cfg_normalize: bool = False

    # LoRA identity. Requests with different adapters or scales must run in
    # separate batches so the worker can activate exactly one adapter per step.
    lora_int_id: int | None = None
    lora_scale: float = 1.0


@dataclass
class DiffusionRequestState:
    """Scheduler-owned state for one queued OmniDiffusionRequest."""

    request_id: str
    req: OmniDiffusionRequest
    sampling_params_key: SamplingParamsKey | None = None
    status: DiffusionRequestStatus = DiffusionRequestStatus.WAITING
    error: str | None = None

    def is_finished(self) -> bool:
        return DiffusionRequestStatus.is_finished(self.status)


@dataclass
class NewRequestData:
    """Full request payload for a newly scheduled diffusion request."""

    request_id: str
    req: OmniDiffusionRequest

    @classmethod
    def from_state(cls, state: DiffusionRequestState) -> NewRequestData:
        return cls(request_id=state.request_id, req=state.req)


@dataclass
class CachedRequestData:
    """Cached diffusion requests that only need their request ids resent."""

    request_ids: list[str]

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(request_ids=[])


@dataclass
class DiffusionSchedulerOutput:
    """Output of a single scheduling cycle."""

    step_id: int  # global step index
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int
    # next request to background-prefetch KV
    kv_prefetch_jobs: dict | None = None

    @cached_property
    def scheduled_request_ids(self) -> list[str]:
        """
        All scheduled request ids in this cycle, including both new and cached ones.
        """
        return [
            *(req.request_id for req in self.scheduled_new_reqs),
            *self.scheduled_cached_reqs.request_ids,
        ]

    @property
    def num_scheduled_reqs(self) -> int:
        return len(self.scheduled_request_ids)

    @property
    def is_empty(self) -> bool:
        return self.num_scheduled_reqs == 0


class SchedulerInterface(ABC):
    """Abstract lifecycle contract for diffusion schedulers."""

    @abstractmethod
    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize or reset scheduler state."""

    @abstractmethod
    def add_request(self, request: OmniDiffusionRequest) -> str:
        """Add a request and return the scheduler-owned request id."""

    @abstractmethod
    def schedule(self) -> DiffusionSchedulerOutput:
        """Run one scheduling cycle."""

    @abstractmethod
    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        """Update scheduler state from executor output."""

    @abstractmethod
    def get_request_state(self, request_id: str) -> DiffusionRequestState | None:
        """Return request state if present."""

    @abstractmethod
    def has_requests(self) -> bool:
        """Return whether the scheduler still owns runnable requests."""

    @abstractmethod
    def pop_request_state(self, request_id: str) -> DiffusionRequestState | None:
        """Remove and return request state if present."""

    @abstractmethod
    def preempt_request(self, request_id: str) -> bool:
        """Preempt a running request back to waiting."""

    @abstractmethod
    def finish_requests(self, request_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        """Mark one or more requests finished."""

    @abstractmethod
    def close(self) -> None:
        """Release scheduler-owned state."""
