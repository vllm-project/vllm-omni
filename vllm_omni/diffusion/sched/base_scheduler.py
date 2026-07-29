# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import fields

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    KVPrefetchJob,
    NewRequestData,
    RequestBatchSamplingParamsKey,
    SchedulerRequestState,
    StepBatchSamplingParamsKey,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)

BatchSamplingParamsKey = StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey

# LoRA identity is derived from `sampling.lora_request`, not a same-named field
# on sampling params, so it must be resolved separately from the bulk lookup.
_STEP_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(field.name for field in fields(StepBatchSamplingParamsKey)) - {
    "lora_int_id"
}


class BaseScheduler(ABC):
    """Shared queue/state bookkeeping for diffusion schedulers."""

    def __init__(self) -> None:
        self.od_config: OmniDiffusionConfig | None = None
        self._request_states: dict[str, SchedulerRequestState] = {}
        self._step_id: int = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._running_sampling_params_key: BatchSamplingParamsKey | None = None
        self._finished_req_ids: set[str] = set()
        self.max_num_running_reqs: int = 1
        self._prefetch_enabled: bool = False

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._request_states.clear()
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
        omni_kv = getattr(od_config, "omni_kv_config", None) or {}
        self._prefetch_enabled = bool(omni_kv.get("enable_kv_async_prefetch", False))
        self._reset_scheduler_state()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        return self._add_request_with_request_id(request.request_id, request)

    def _add_request_with_request_id(self, request_id: str, request: OmniDiffusionRequest) -> str:
        if request_id in self._request_states:
            raise ValueError(f"request_id {request_id!r} is already active.")
        state = self._make_request_state(request_id, request)
        self._request_states[request_id] = state
        self._waiting.append(request_id)
        logger.debug("%s add_request: %s (waiting=%d)", self.__class__.__name__, request_id, len(self._waiting))
        return request_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_request_ids: list[str] = []

        # First, schedule the RUNNING request(s)
        for request_id in self._running:
            state = self._request_states.get(request_id)
            if state is not None:
                scheduled_cached_request_ids.append(request_id)

        # Second, schedule WAITING requests while capacity remains.
        while self._waiting and len(self._running) < self.max_num_running_reqs:
            request_id = self._waiting[0]
            state = self._request_states.get(request_id)
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
            self._running.append(request_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_request_ids.append(request_id)

        # Expose the next waiting request (serial mode) so the runner can
        # prefetch its KV during this forward.  Skip a request without
        # kv_sender_info (would target the wrong sender under multi-replica) or
        # one already finished/aborted (would consume its sender buffer for
        # nothing).
        kv_prefetch_job: KVPrefetchJob | None = None
        if self._prefetch_enabled and self._waiting:
            nxt = self._request_states.get(self._waiting[0])
            if nxt is not None and not nxt.is_finished():
                sender_info = getattr(nxt.req, "kv_sender_info", None)
                if sender_info:
                    kv_prefetch_job = {
                        "request_id": nxt.request_id,
                        "kv_sender_info": sender_info,
                    }

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(request_ids=scheduled_cached_request_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
            kv_prefetch_job=kv_prefetch_job,
        )

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    @abstractmethod
    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        pass

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def num_waiting_requests(self) -> int:
        return len(self._waiting)

    def num_running_requests(self) -> int:
        return len(self._running)

    def get_request_state(self, request_id: str) -> SchedulerRequestState | None:
        return self._request_states.get(request_id)

    def pop_request_state(self, request_id: str) -> SchedulerRequestState | None:
        self._pop_extra_request_state(request_id)
        return self._request_states.pop(request_id, None)

    def preempt_request(self, request_id: str) -> bool:
        if request_id not in self._request_states:
            return False
        if request_id in self._running:
            self._running.remove(request_id)
            if not self._running:
                self._running_sampling_params_key = None
            self._waiting.appendleft(request_id)
            self._request_states[request_id].status = DiffusionRequestStatus.PREEMPTED
            return True
        return False

    def finish_requests(self, request_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        self._finish_requests({request_id: status for request_id in request_ids})

    def close(self) -> None:
        self._request_states.clear()
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

        for request_id, status in statuses.items():
            assert DiffusionRequestStatus.is_finished(status)
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue

            finished_req_ids.add(request_id)
            if request_id in self._running:
                running_to_remove.add(request_id)
            if request_id in self._waiting:
                waiting_to_remove.add(request_id)

        if running_to_remove:
            self._running = [request_id for request_id in self._running if request_id not in running_to_remove]
            if not self._running:
                self._running_sampling_params_key = None
        if waiting_to_remove:
            self._waiting = deque(request_id for request_id in self._waiting if request_id not in waiting_to_remove)

        for request_id in finished_req_ids:
            state = self._request_states[request_id]
            status = statuses[request_id]
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(request_id)
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
            request_id for request_id in sched_output.scheduled_request_ids if request_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, request_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""

    def _make_request_state(self, request_id: str, request: OmniDiffusionRequest) -> SchedulerRequestState:
        return SchedulerRequestState(
            request_id=request_id,
            req=request,
            sampling_params_key=self._build_sampling_params_key(request),
        )

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not self._running:
            return True

        current_key = self._current_sampling_params_key()
        return current_key is not None and current_key == state.sampling_params_key

    def _current_sampling_params_key(self) -> BatchSamplingParamsKey | None:
        if self._running_sampling_params_key is not None or not self._running:
            return self._running_sampling_params_key
        state = self._request_states.get(self._running[0])
        self._running_sampling_params_key = None if state is None else state.sampling_params_key
        return self._running_sampling_params_key

    def _build_sampling_params_key(
        self, request: OmniDiffusionRequest
    ) -> StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey:  # return type loosened for subclassing
        """Build a step-batch compatibility key from sampling parameters."""
        sampling = request.sampling_params
        # LoRA identity is optional on sampling params (and on test stubs).
        lora_request = getattr(sampling, "lora_request", None)
        return StepBatchSamplingParamsKey(
            lora_int_id=lora_request.lora_int_id if lora_request is not None else None,
            **{name: getattr(sampling, name) for name in _STEP_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES},
        )


class SchedulerInterface(BaseScheduler):
    """Deprecated compatibility base for custom scheduler injection.

    Prefer subclassing :class:`BaseScheduler` directly. Subclassing this name
    still works but emits a :class:`DeprecationWarning`.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        import warnings

        warnings.warn(
            "SchedulerInterface is deprecated; subclass BaseScheduler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
