# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from dataclasses import fields

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
    SamplingParamsKey,
    SchedulerInterface,
)

logger = init_logger(__name__)

_KEY_FIELD_NAMES = frozenset(f.name for f in fields(SamplingParamsKey))


def get_sampling_params_key(request: OmniDiffusionRequest) -> SamplingParamsKey | None:
    """Build a batch-compatibility key from the request's sampling params."""
    if len(request.prompts) != 1:
        return None

    sampling = request.sampling_params
    return SamplingParamsKey(**{name: getattr(sampling, name) for name in _KEY_FIELD_NAMES})


class _BaseScheduler(SchedulerInterface):
    """Shared queue/state bookkeeping for diffusion schedulers."""

    def __init__(self) -> None:
        self.od_config: OmniDiffusionConfig | None = None
        self._request_states: dict[str, DiffusionRequestState] = {}
        self._request_id_to_sched_req_id: dict[str, str] = {}
        self._step_id: int = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._running_sampling_params_key: SamplingParamsKey | None = None
        self._finished_req_ids: set[str] = set()
        self.max_num_running_reqs: int = 1

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._request_states.clear()
        self._request_id_to_sched_req_id.clear()
        self._step_id = 0
        self._waiting.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        max_num_seqs = getattr(od_config, "max_num_seqs", 1)
        try:
            self.max_num_running_reqs = max(1, int(max_num_seqs))
        except (TypeError, ValueError):
            self.max_num_running_reqs = 1
        self._reset_scheduler_state()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        return self._add_request_with_sched_req_id(sched_req_id, request)

    def _add_request_with_sched_req_id(self, sched_req_id: str, request: OmniDiffusionRequest) -> str:
        state = self._make_request_state(sched_req_id, request)
        self._request_states[sched_req_id] = state
        self._register_request_ids(request.request_ids, sched_req_id)
        self._waiting.append(sched_req_id)
        logger.debug("%s add_request: %s (waiting=%d)", self.__class__.__name__, sched_req_id, len(self._waiting))
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
        while self._waiting and len(self._running) < self.max_num_running_reqs:
            sched_req_id = self._waiting[0]
            state = self._request_states.get(sched_req_id)
            if state is None:
                self._waiting.popleft()
                continue
            if not self._can_schedule_waiting(state):
                break

            self._waiting.popleft()
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            if not self._running:
                self._running_sampling_params_key = state.sampling_params_key
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

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def get_request_state(self, sched_req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(sched_req_id)

    def get_sched_req_id(self, request_id: str) -> str | None:
        return self._request_id_to_sched_req_id.get(request_id)

    def pop_request_state(self, sched_req_id: str) -> DiffusionRequestState | None:
        self._pop_extra_request_state(sched_req_id)
        state = self._request_states.pop(sched_req_id, None)
        if state is not None:
            self._unregister_request_ids(state.req.request_ids, sched_req_id)
        return state

    def preempt_request(self, sched_req_id: str) -> bool:
        if sched_req_id not in self._request_states:
            return False
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
            if not self._running:
                self._running_sampling_params_key = None
            self._waiting.appendleft(sched_req_id)
            self._request_states[sched_req_id].status = DiffusionRequestStatus.PREEMPTED
            return True
        return False

    def finish_requests(self, sched_req_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        if isinstance(sched_req_ids, str):
            sched_req_ids = [sched_req_ids]
        self._finish_requests({sched_req_id: status for sched_req_id in sched_req_ids})

    def close(self) -> None:
        self._request_states.clear()
        self._request_id_to_sched_req_id.clear()
        self._waiting.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        self._reset_scheduler_state()

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()

        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()

        for sched_req_id, status in statuses.items():
            assert DiffusionRequestStatus.is_finished(status)
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue

            finished_req_ids.add(sched_req_id)
            if sched_req_id in self._running:
                running_to_remove.add(sched_req_id)
            if sched_req_id in self._waiting:
                waiting_to_remove.add(sched_req_id)

        if running_to_remove:
            self._running = [sched_req_id for sched_req_id in self._running if sched_req_id not in running_to_remove]
            if not self._running:
                self._running_sampling_params_key = None
        if waiting_to_remove:
            self._waiting = deque(
                sched_req_id for sched_req_id in self._waiting if sched_req_id not in waiting_to_remove
            )

        for sched_req_id in finished_req_ids:
            state = self._request_states[sched_req_id]
            status = statuses[sched_req_id]
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(sched_req_id)
            else:
                state.error = None

        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _finalize_update_from_output(
        self,
        sched_output: DiffusionSchedulerOutput,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        finished_req_ids = {
            sched_req_id for sched_req_id in sched_output.scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""

    def _make_request_state(self, sched_req_id: str, request: OmniDiffusionRequest) -> DiffusionRequestState:
        return DiffusionRequestState(
            sched_req_id=sched_req_id,
            req=request,
            sampling_params_key=get_sampling_params_key(request),
        )

    def _can_schedule_waiting(self, state: DiffusionRequestState) -> bool:
        if not self._running:
            return True

        current_key = self._current_sampling_params_key()
        return current_key is not None and current_key == state.sampling_params_key

    def _current_sampling_params_key(self) -> SamplingParamsKey | None:
        if self._running_sampling_params_key is not None or not self._running:
            return self._running_sampling_params_key
        state = self._request_states.get(self._running[0])
        self._running_sampling_params_key = None if state is None else state.sampling_params_key
        return self._running_sampling_params_key

    def _register_request_ids(self, request_ids: list[str], sched_req_id: str) -> None:
        for request_id in request_ids:
            existing = self._request_id_to_sched_req_id.get(request_id)
            if existing is not None and existing != sched_req_id:
                raise ValueError(f"request_id {request_id!r} is already mapped to active sched_req_id {existing!r}.")
            self._request_id_to_sched_req_id[request_id] = sched_req_id

    def _unregister_request_ids(self, request_ids: list[str], sched_req_id: str) -> None:
        for request_id in request_ids:
            if self._request_id_to_sched_req_id.get(request_id) == sched_req_id:
                self._request_id_to_sched_req_id.pop(request_id, None)
