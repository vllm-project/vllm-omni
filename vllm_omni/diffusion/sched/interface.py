# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from dataclasses import dataclass

from vllm_omni.diffusion.request import OmniDiffusionRequest


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

    sched_req_id: str
    req: OmniDiffusionRequest
    status: DiffusionRequestStatus = DiffusionRequestStatus.WAITING
    error: str | None = None

    def is_finished(self) -> bool:
        return DiffusionRequestStatus.is_finished(self.status)


@dataclass
class DiffusionSchedulerOutput:
    """Output of a single scheduling cycle.

    Kept intentionally small so step-execution components can share a stable
    transport shape while scheduler policy continues to evolve.
    """

    step_id: int
    req_states: list[DiffusionRequestState]
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int
