# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import time
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    SchedulerRequestState,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)


@dataclass
class _StepProgress:
    current_step: int
    total_steps: int
    interaction_generation: int = 0
    scheduled_interaction_generation: int = 0
    wait_deadline: float | None = None  # None: runnable; inf: indefinite wait.
    resume: Literal["interaction", "timeout"] | None = None


class StepScheduler(BaseScheduler):
    """Scheduler that advances each request by one denoise step per update."""

    def __init__(self) -> None:
        super().__init__()
        self._request_progress: dict[str, _StepProgress] = {}

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()

    def _can_schedule_running(self, state: SchedulerRequestState) -> bool:
        return self._request_progress[state.request_id].wait_deadline is None

    def get_execution_wait_timeout(self, now: float) -> float | None:
        deadlines = [p.wait_deadline for p in self._request_progress.values() if p.wait_deadline is not None]
        if not deadlines or any(self._request_progress[rid].wait_deadline is None for rid in self._running):
            return 0.0
        if len(self._running) < self.max_num_running_reqs and any(
            self._can_schedule_waiting(self._request_states[request_id]) for request_id in self._waiting
        ):
            return 0.0
        if self._finished_req_ids or self._kv_transfer_request_ids or self._kv_draining_requests:
            return 0.0
        deadline = min(deadlines)
        return None if deadline == inf else max(deadline - now, 0.0)

    def notify_interaction(self, request_id: str) -> None:
        state = self._request_states.get(request_id)
        if state is None or state.is_finished():
            return
        progress = self._request_progress[request_id]
        progress.interaction_generation += 1
        if progress.wait_deadline is not None:
            progress.wait_deadline = None
            progress.resume = "interaction"

    def schedule(self) -> DiffusionSchedulerOutput:
        now = time.monotonic()
        for progress in self._request_progress.values():
            if progress.wait_deadline is not None and now >= progress.wait_deadline:
                progress.wait_deadline = None
                progress.resume = "timeout"
        output = super().schedule()
        for request_id in output.scheduled_request_ids:
            progress = self._request_progress[request_id]
            progress.scheduled_interaction_generation = progress.interaction_generation
            if progress.resume is not None:
                output.step_resumes[request_id] = progress.resume
                progress.resume = None
        return output

    def add_request(self, request: OmniDiffusionRequest) -> str:
        request_id = request.request_id
        total_steps = self._get_total_steps(request)
        if total_steps <= 0:
            raise ValueError(f"Diffusion request {request_id} must have positive total_steps, got {total_steps}")

        current_step = request.sampling_params.step_index or 0
        if current_step < 0 or current_step >= total_steps:
            raise ValueError(
                f"Diffusion request {request_id} has invalid initial step_index {current_step} "
                f"for total_steps={total_steps}"
            )

        request.sampling_params.step_index = current_step
        request_id = self._add_request_with_request_id(request_id, request)
        self._request_progress[request_id] = _StepProgress(current_step=current_step, total_steps=total_steps)
        logger.debug(
            "StepScheduler add_request: %s (step=%d/%d, waiting=%d)",
            request_id,
            current_step,
            total_steps,
            len(self._waiting),
        )
        return request_id

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not super()._can_schedule_waiting(state):
            return False

        is_first_step = self._request_progress[state.request_id].current_step == 0
        for request_id in self._running:
            running = self._request_states[request_id]
            if state.req.allow_mixed_step_phases and running.req.allow_mixed_step_phases:
                continue
            if (self._request_progress[request_id].current_step == 0) != is_first_step:
                return False
        return True

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: BaseRunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids and not sched_output.finished_req_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in scheduled_request_ids:
            state = self._request_states.get(request_id)
            progress = self._request_progress.get(request_id)
            if state is None or progress is None or state.is_finished():
                continue
            req_output = output.get_request_output(request_id)
            if req_output is None:
                logger.warning(
                    "No RunnerOutput for request %s, treating as error",
                    request_id,
                )
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = "No output for request"
                continue

            req_result = req_output.result
            if req_result is not None and req_result.aborted:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ABORTED
                terminal_errors[request_id] = None
                continue
            output_error = req_result.error if req_result is not None else None
            if output_error is not None:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = output_error
                continue

            if req_output.step_index is None:
                logger.warning(
                    "Received RunnerOutput with no step_index for request %s, treating as error",
                    request_id,
                )
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = "Missing step_index in RunnerOutput"
                continue

            # We assume that the decoding stage is executed immediately after the denoising stage completes.
            progress.current_step = req_output.step_index
            state.req.sampling_params.step_index = req_output.step_index
            if req_output.finished:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None
            else:
                state.error = None
                wait = req_output.wait_for_interaction
                if wait is not None:
                    # RPCs are drained before this update. Latch successful
                    # interactions even when the wait was not installed yet.
                    if progress.interaction_generation != progress.scheduled_interaction_generation:
                        progress.resume = "interaction"
                    else:
                        progress.wait_deadline = (
                            inf if wait.timeout_seconds is None else time.monotonic() + wait.timeout_seconds
                        )

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def _pop_extra_request_state(self, request_id: str) -> None:
        self._request_progress.pop(request_id, None)

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        finished = super()._finish_requests(statuses, errors)
        for request_id in finished:
            progress = self._request_progress.get(request_id)
            if progress is not None:
                progress.wait_deadline = None
                progress.resume = None
        return finished

    def _get_total_steps(self, request: OmniDiffusionRequest) -> int:
        sampling = request.sampling_params

        if sampling.timesteps is not None:
            return self._sequence_length(sampling.timesteps)
        if sampling.sigmas is not None:
            return len(sampling.sigmas)
        assert sampling.num_inference_steps is not None
        return int(sampling.num_inference_steps)

    @staticmethod
    def _sequence_length(values: Any) -> int:
        ndim = getattr(values, "ndim", None)
        if ndim == 0:
            return 1

        shape = getattr(values, "shape", None)
        if shape is not None:
            return int(shape[0])

        return len(values)
