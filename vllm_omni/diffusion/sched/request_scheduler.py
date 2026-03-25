# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
)

logger = init_logger(__name__)


class RequestScheduler(_BaseScheduler):
    """Diffusion scheduler with vLLM-style waiting/running queues."""

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        state = DiffusionRequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._register_request_ids(request.request_ids, sched_req_id)
        self._waiting.append(sched_req_id)
        logger.debug("Scheduler add_request: %s (waiting=%d)", sched_req_id, len(self._waiting))
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        # First, schedule the RUNNING request(s)
        for sched_req_id in self._running:
            state = self._request_states.get(sched_req_id)
            if state is not None:
                scheduled_cached_req_ids.append(sched_req_id)

        # Second, schedule WAITING requests while capacity remains.
        while self._waiting and len(self._running) < self._max_batch_size:
            sched_req_id = self._waiting.popleft()
            state = self._request_states.get(sched_req_id)
            if state is None:
                continue
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(sched_req_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: DiffusionOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        finished_req_ids = {
            sched_req_id for sched_req_id in scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        # NOTE: request-mode currently assumes one executor call produces one
        # DiffusionOutput for the single scheduled request in this cycle.
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            if output.error:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = output.error
            else:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None

        finished_req_ids |= self._finish_requests(terminal_statuses, terminal_errors)
        return finished_req_ids

    def abort_request(self, sched_req_id: str) -> bool:
        if self.get_request_state(sched_req_id) is None:
            return False
        self.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        return True
