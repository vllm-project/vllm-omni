# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput


class RequestScheduler(_BaseScheduler):
    """Diffusion scheduler with vLLM-style waiting/running queues."""

    def add_request(self, request: OmniDiffusionRequest) -> str:
        return super().add_request(request)

    def schedule(self) -> DiffusionSchedulerOutput:
        return super().schedule()

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        result = output.result
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            if result is None:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "No output result"
            elif result.error:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = result.error
            else:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)
