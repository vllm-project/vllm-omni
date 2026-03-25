# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


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


@dataclass
class DiffusionRequestState:
    """Scheduler-owned state for one queued OmniDiffusionRequest."""

    # Unique scheduler-owned request ID.
    # NOTE: This identifies one OmniDiffusionRequest, which may contain multiple request_ids.
    # TODO: Align this with OmniDiffusionRequest.request_ids once scheduler batching is supported.
    sched_req_id: str
    req: OmniDiffusionRequest
    status: DiffusionRequestStatus = DiffusionRequestStatus.WAITING
    error: str | None = None

    def is_finished(self) -> bool:
        return DiffusionRequestStatus.is_finished(self.status)


@dataclass
class NewRequestData:
    """Full request payload for a newly scheduled diffusion request."""

    sched_req_id: str
    req: OmniDiffusionRequest

    @classmethod
    def from_state(cls, state: DiffusionRequestState) -> NewRequestData:
        return cls(sched_req_id=state.sched_req_id, req=state.req)


@dataclass
class CachedRequestData:
    """Cached diffusion requests that only need their scheduler ids resent."""

    sched_req_ids: list[str]

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(sched_req_ids=[])


@dataclass
class DiffusionSchedulerOutput:
    """Output of a single scheduling cycle."""

    step_id: int
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int

    @cached_property
    def scheduled_req_ids(self) -> list[str]:
        """
        All scheduled request ids in this cycle, including both new and cached ones.
        NOTE:
            This id is generated and owned by the scheduler,
            and may be different from the OmniDiffusionRequest.request_ids.
        """
        return [
            *(req.sched_req_id for req in self.scheduled_new_reqs),
            *self.scheduled_cached_reqs.sched_req_ids,
        ]

    @property
    def num_scheduled_reqs(self) -> int:
        return len(self.scheduled_req_ids)

    @property
    def is_empty(self) -> bool:
        return self.num_scheduled_reqs == 0


class SchedulerInterface(ABC):
    """Abstract lifecycle contract for diffusion schedulers."""

    def _make_sched_req_id(self, request: OmniDiffusionRequest) -> str:
        """
        Generate a unique scheduler request ID for the given request.
            The default implementation uses the first request_id from the request if available,
            otherwise generates a random one.
        """
        if request.request_ids:
            base = request.request_ids[0]
        else:
            logger.warning("Request has no request_ids, generating a random one. Request: %s", request)
            base = f"req_{uuid.uuid4().hex[:8]}"

        sched_req_id = base
        suffix = 1
        while self.get_request_state(sched_req_id) is not None:
            sched_req_id = f"{base}#{suffix}"
            suffix += 1
        return sched_req_id

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
    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: DiffusionOutput) -> set[str]:
        """Update scheduler state from executor output."""

    @abstractmethod
    def get_request_state(self, sched_req_id: str) -> DiffusionRequestState | None:
        """Return request state if present."""

    @abstractmethod
    def has_requests(self) -> bool:
        """Return whether the scheduler still owns runnable requests."""

    @abstractmethod
    def get_sched_req_id(self, request_id: str) -> str | None:
        """Resolve a public request_id to the active scheduler request id."""

    @abstractmethod
    def pop_request_state(self, sched_req_id: str) -> DiffusionRequestState | None:
        """Remove and return request state if present."""

    @abstractmethod
    def preempt_request(self, sched_req_id: str) -> bool:
        """Preempt a running request back to waiting."""

    @abstractmethod
    def finish_requests(self, sched_req_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        """Mark one or more requests finished."""

    @abstractmethod
    def close(self) -> None:
        """Release scheduler-owned state."""
