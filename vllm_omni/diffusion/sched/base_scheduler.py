# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import fields

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    KVTransferRegistrationError,
    commit_kv_load,
    native_prefetch_enabled,
    prepare_kv_requests,
    validate_kv_transfer_boundaries,
)
from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
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
    _AdmissionWaitDecision,
)
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)

BatchSamplingParamsKey = StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey

# LoRA identity and execution mode are request-owned rather than same-named
# sampling-param fields, so they must be resolved separately from the bulk lookup.
_STEP_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(field.name for field in fields(StepBatchSamplingParamsKey)) - {
    "condition_key",
    "lora_int_id",
    "use_step_execution",
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
        self._diffusion_kv_manager: DiffusionKVCacheManager | None = None
        self._kv_connector = None
        self._kv_transfer_request_ids: set[str] = set()
        self._kv_finished_request_ids: set[str] = set()
        self._kv_loading_request_ids: set[str] = set()
        self._kv_request_generations: dict[str, int] = {}
        self._kv_draining_requests: dict[str, SchedulerRequestState] = {}
        self._kv_received_request_ids: set[str] = set()
        self._native_prefetch_enabled = False
        self._native_prefetch_request_id: str | None = None

    def initialize(
        self,
        od_config: OmniDiffusionConfig,
        *,
        kv_cache_config: KVCacheConfig | None = None,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        kv_vllm_config: VllmConfig | None = None,
    ) -> None:
        self.od_config = od_config
        self._native_prefetch_enabled = native_prefetch_enabled(od_config)
        self._native_prefetch_request_id = None
        self._request_states.clear()
        self._step_id = 0
        self._waiting.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        self._reset_kv_transfer_state()
        max_num_seqs = getattr(od_config, "max_num_seqs", 1)
        try:
            self.max_num_running_reqs = max(1, int(max_num_seqs))
        except (TypeError, ValueError):
            self.max_num_running_reqs = 1
        omni_kv = getattr(od_config, "omni_kv_config", None) or {}
        self._prefetch_enabled = bool(omni_kv.get("enable_kv_async_prefetch", False))
        diffusion_kv_enabled = (
            getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
        )
        if diffusion_kv_enabled:
            if kv_cache_config is None:
                raise ValueError("paged_scheduler Diffusion KV requires a native Scheduler KVCacheConfig")
            if scheduler_block_size is None or hash_block_size is None:
                raise ValueError("paged_scheduler Diffusion KV requires native scheduler/hash block sizes")
            if kv_vllm_config is None:
                raise ValueError("paged_scheduler Diffusion KV requires the native VllmConfig used for cache sizing")
            max_rows_per_request = od_config.diffusion_kv_max_rows_per_request
            if max_rows_per_request is None:
                raise ValueError("paged_scheduler Diffusion KV requires diffusion_kv_max_rows_per_request")
            self._diffusion_kv_manager = DiffusionKVCacheManager(
                kv_cache_config,
                max_model_len=kv_vllm_config.model_config.max_model_len,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
                max_rows_per_request=max_rows_per_request,
                max_in_flight_tokens=kv_vllm_config.max_in_flight_tokens,
                enable_prefix_caching=bool(getattr(kv_vllm_config.cache_config, "enable_prefix_caching", False)),
                prefix_caching_hash_algo=getattr(
                    kv_vllm_config.cache_config,
                    "prefix_caching_hash_algo",
                    "sha256",
                ),
            )
        else:
            if any(
                value is not None
                for value in (
                    kv_cache_config,
                    scheduler_block_size,
                    hash_block_size,
                    kv_vllm_config,
                )
            ):
                raise ValueError("dense_legacy Scheduler received unexpected Diffusion KV cache initialization state")
            self._diffusion_kv_manager = None
        from vllm_omni.diffusion.diffusion_kv.kv_connector import create_scheduler_kv_connector

        self._kv_connector = create_scheduler_kv_connector(od_config, kv_cache_config, kv_vllm_config)
        self._reset_scheduler_state()

    @property
    def kv_connector(self):
        """Upstream vLLM Scheduler-role connector, when configured."""

        return self._kv_connector

    def add_request(self, request: OmniDiffusionRequest) -> str:
        return self._add_request_with_request_id(request.request_id, request)

    def _add_request_with_request_id(self, request_id: str, request: OmniDiffusionRequest) -> str:
        if request_id in self._request_states or request_id in self._kv_draining_requests:
            raise ValueError(f"request_id {request_id!r} is already active.")
        state = self._make_request_state(request_id, request)
        state.queued_at = time.perf_counter()
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
        deferred: list[str] = []
        waiting_budget = len(self._waiting)
        while self._waiting and waiting_budget and len(self._running) < self.max_num_running_reqs:
            waiting_budget -= 1
            request_id = self._waiting[0]
            state = self._request_states.get(request_id)
            if state is None:
                self._waiting.popleft()
                continue
            if not self._can_schedule_waiting(state):
                break

            diffusion_kv_metadata: DiffusionKVMetadata | None = None
            if self._diffusion_kv_manager is not None:
                already_reserved = self._diffusion_kv_manager.has_request(request_id)
                if already_reserved:
                    diffusion_kv_metadata = self._diffusion_kv_manager.get_metadata(request_id)
                else:
                    matched_tokens: list[int] = []
                    if self._kv_connector is not None:
                        for request in state.diffusion_kv_requests:
                            num_tokens, _ = self._kv_connector.get_num_new_matched_tokens(request, 0)
                            if num_tokens is None:
                                break
                            matched_tokens.append(num_tokens)
                        if len(matched_tokens) != len(state.diffusion_kv_requests):
                            deferred.append(self._waiting.popleft())
                            continue
                    try:
                        allocation = self._diffusion_kv_manager.reserve_request(
                            request_id,
                            state.diffusion_kv_requests,
                        )
                    except Exception as exc:
                        # Reservation rolls back on failure and has not yet
                        # handed any destination pages to the connector.
                        self._finish_requests(
                            {request_id: DiffusionRequestStatus.FINISHED_ERROR},
                            {request_id: str(exc)},
                        )
                        continue
                    if allocation is None:
                        break
                    # Check the lookup result before registering destination
                    # pages with the connector: deferred pages can be freed.
                    if not self._can_schedule_waiting(state):
                        self._diffusion_kv_manager.free_request(request_id)
                        break
                    diffusion_kv_metadata = allocation
                    self._kv_request_generations[request_id] = allocation.allocation_generation
                    if self._kv_connector is not None:
                        try:
                            transfer_ids = commit_kv_load(
                                self._kv_connector,
                                self._diffusion_kv_manager.native_manager,
                                state.diffusion_kv_requests,
                                matched_tokens,
                            )
                        except KVTransferRegistrationError as exc:
                            self._finish_requests(
                                {request_id: DiffusionRequestStatus.FINISHED_ERROR}, {request_id: str(exc)}
                            )
                            continue
                        # Other exceptions have no proven rollback. Keep pages
                        # allocated until the Engine shuts down the Workers.
                        self._kv_transfer_request_ids.update(transfer_ids)
                        if transfer_ids:
                            self._kv_loading_request_ids.add(request_id)
                        for sequence, request in zip(allocation.sequences, state.diffusion_kv_requests, strict=True):
                            sequence.num_computed_tokens = request.num_computed_tokens

            self._waiting.popleft()
            if request_id == self._native_prefetch_request_id:
                self._native_prefetch_request_id = None
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            if not self._running:
                self._running_sampling_params_key = state.sampling_params_key
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(request_id)
            if was_new_request:
                state.req.scheduler_queue_wait_ms = max((time.perf_counter() - state.queued_at) * 1000.0, 0.0)
                scheduled_new_reqs.append(
                    NewRequestData.from_state(
                        state,
                        diffusion_kv_metadata=diffusion_kv_metadata,
                    )
                )
            else:
                scheduled_cached_request_ids.append(request_id)

        self._waiting.extendleft(reversed(deferred))

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
            kv_finished_request_ids=set(self._kv_finished_request_ids),
        )
        if self._kv_connector is not None and (
            self._kv_transfer_request_ids or self._kv_finished_request_ids or self._kv_draining_requests
        ):
            scheduler_output.kv_connector_metadata = self._kv_connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_transfer_request_ids = self._kv_transfer_request_ids
            self._kv_transfer_request_ids = set()
        if self._native_prefetch_enabled:
            scheduler_output.kv_required_request_ids = self._loading_sequence_ids(self._running)
            # Flush current-request metadata before staging B. Mooncake
            # batches ready requests from the same producer into one write;
            # combining A+B would make A's completion depend on B's bytes.
            self._try_native_prefetch()
            prefetch_finished_ids = self._kv_finished_request_ids - scheduler_output.kv_finished_request_ids
            if self._kv_transfer_request_ids or prefetch_finished_ids:
                scheduler_output.kv_prefetch_connector_metadata = self._kv_connector.build_connector_meta(
                    scheduler_output
                )
                scheduler_output.kv_prefetch_request_ids = self._kv_transfer_request_ids
                scheduler_output.kv_transfer_request_ids |= self._kv_transfer_request_ids
                self._kv_transfer_request_ids = set()
            scheduler_output.finished_req_ids.update(self._finished_req_ids)
            scheduler_output.kv_finished_request_ids.update(prefetch_finished_ids)
            scheduler_output.num_waiting_reqs = len(self._waiting)

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        self._kv_finished_request_ids.clear()
        return scheduler_output

    def _try_native_prefetch(self) -> None:
        if not self._running or not self._waiting or self._native_prefetch_request_id is not None:
            return
        request_id = self._waiting[0]
        state = self._request_states.get(request_id)
        if state is None or state.is_finished() or not state.diffusion_kv_requests:
            return
        manager = self._diffusion_kv_manager
        if manager.has_request(request_id):
            return
        matched_tokens = []
        for request in state.diffusion_kv_requests:
            params = request.kv_transfer_params or {}
            if (
                not params.get("do_remote_prefill")
                or not all(params.get(key) for key in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id"))
                or type(params.get("num_transfer_tokens")) is not int
                or params["num_transfer_tokens"] <= 0
            ):
                return
            num_tokens, _ = self._kv_connector.get_num_new_matched_tokens(request, 0)
            if num_tokens is None:
                return
            matched_tokens.append(num_tokens)
        try:
            validate_kv_transfer_boundaries(state.diffusion_kv_requests, matched_tokens)
        except KVTransferRegistrationError as exc:
            # Do not reserve B's pages for a transfer that cannot be registered.
            # Normal admission will report the request-scoped error later.
            logger.debug("Native KV prefetch skipped for %s: invalid boundary (%s)", request_id, exc)
            return
        # The manager atomically reserves all CFG rows, including capacity
        # needed when B eventually executes. B remains in WAITING.
        try:
            allocation = manager.reserve_request(request_id, state.diffusion_kv_requests)
        except Exception as exc:
            # reserve_request rolls back atomically. Normal admission will
            # surface B's error later; speculative allocation must not fail A.
            logger.debug("Native KV prefetch skipped for %s: reservation failed (%s)", request_id, exc)
            return
        if allocation is None:
            logger.debug("Native KV prefetch skipped for %s: insufficient blocks", request_id)
            return
        self._kv_request_generations[request_id] = allocation.allocation_generation
        try:
            transfer_ids = commit_kv_load(
                self._kv_connector, manager.native_manager, state.diffusion_kv_requests, matched_tokens
            )
        except KVTransferRegistrationError as exc:
            # Rollback may already notify the producer to release its pages.
            # B cannot retry that transfer ticket, but A can still proceed.
            self._finish_requests({request_id: DiffusionRequestStatus.FINISHED_ERROR}, {request_id: str(exc)})
            return
        self._kv_transfer_request_ids.update(transfer_ids)
        if transfer_ids:
            self._kv_loading_request_ids.add(request_id)
        for sequence, request in zip(allocation.sequences, state.diffusion_kv_requests, strict=True):
            sequence.num_computed_tokens = request.num_computed_tokens
        self._native_prefetch_request_id = request_id
        logger.debug("Native KV prefetch submitted for %s: %s", request_id, sorted(transfer_ids))

    def _loading_sequence_ids(self, request_ids: Iterable[str]) -> set[str]:
        sequence_ids: set[str] = set()
        for request_id in request_ids:
            if request_id not in self._kv_loading_request_ids:
                continue
            state = self._request_states.get(request_id) or self._kv_draining_requests.get(request_id)
            if state is not None:
                sequence_ids.update(request.request_id for request in state.diffusion_kv_requests)
        return sequence_ids

    def native_kv_poll_output(self, *, drain_request_ids: list[str] | None = None) -> DiffusionSchedulerOutput | None:
        """Poll after compute; cancellation/close may wait for selected loads."""
        if not self._native_prefetch_enabled or not self._kv_loading_request_ids:
            return None
        return DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
            kv_required_request_ids=self._loading_sequence_ids(drain_request_ids or []),
            kv_poll_only=True,
        )

    @abstractmethod
    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: BaseRunnerOutput) -> set[str]:
        pass

    def has_requests(self) -> bool:
        return bool(
            self._waiting
            or self._running
            or self._kv_transfer_request_ids
            or self._kv_finished_request_ids
            or self._kv_draining_requests
        )

    def update_kv_connector_output(self, output: KVConnectorOutput | None) -> None:
        if output is None:
            return
        assert self._kv_connector is not None
        self._kv_connector.update_connector_output(output)
        finished_ids = output.finished_recving or set()
        if not finished_ids:
            return
        for request_id in tuple(self._kv_loading_request_ids):
            state = self._request_states.get(request_id) or self._kv_draining_requests.get(request_id)
            if state is None:
                continue
            internal_ids = {request.request_id for request in state.diffusion_kv_requests}
            self._kv_received_request_ids.update(internal_ids.intersection(finished_ids))
            if internal_ids and internal_ids.issubset(self._kv_received_request_ids):
                self._kv_loading_request_ids.discard(request_id)
                self._kv_received_request_ids.difference_update(internal_ids)

    def fail_incomplete_kv_loads(self, transfer_ids: set[str]) -> set[str]:
        statuses = {}
        for request_id in self._kv_loading_request_ids:
            state = self._request_states.get(request_id)
            if state is not None and any(req.request_id in transfer_ids for req in state.diffusion_kv_requests):
                statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
        return self._finish_requests(statuses, {rid: "Timed out receiving diffusion KV" for rid in statuses})

    def completed_kv_drains(self) -> set[str]:
        return self._kv_draining_requests.keys() - self._kv_loading_request_ids

    def release_kv_drains(self, request_ids: set[str]) -> None:
        """Called only after all ranks completed and Worker row cleanup succeeded."""
        for request_id in request_ids:
            if request_id in self._kv_loading_request_ids:
                raise RuntimeError("Cannot release diffusion pages before KV receive completes")
            state = self._kv_draining_requests[request_id]
            self._release_request_kv(state, state.status)
            del self._kv_draining_requests[request_id]
            if request_id not in self._request_states:
                self._kv_request_generations.pop(request_id, None)

    def num_waiting_requests(self) -> int:
        return len(self._waiting)

    def num_running_requests(self) -> int:
        return len(self._running)

    def pending_finished_request_ids(self) -> set[str]:
        """Finished requests whose state the engine has not consumed yet."""
        return {request_id for request_id in self._finished_req_ids if request_id in self._request_states}

    def get_admission_wait_decision(
        self,
        *,
        now: float,
        dp_concurrent: bool = False,
    ) -> _AdmissionWaitDecision:
        """Return the admission-delay policy for the next scheduling wave."""
        del now, dp_concurrent
        return _AdmissionWaitDecision(should_wait=False)

    def should_end_admission_wait(
        self,
        decision: _AdmissionWaitDecision,
        *,
        now: float,
        stable_since: float,
    ) -> bool:
        """Return whether an active admission delay should end."""
        del decision, now, stable_since
        return True

    def get_request_state(self, request_id: str) -> SchedulerRequestState | None:
        return self._request_states.get(request_id)

    def get_diffusion_kv_cleanup_targets(self, request_ids: list[str]) -> list[str | tuple[str, int]]:
        targets: list[str | tuple[str, int]] = []
        for request_id in request_ids:
            if request_id in self._kv_loading_request_ids:
                continue
            generation = self._kv_request_generations.get(request_id)
            if generation is not None:
                targets.append((request_id, generation))
            else:
                targets.append(request_id)
        return targets

    def pop_request_state(self, request_id: str) -> SchedulerRequestState | None:
        self._pop_extra_request_state(request_id)
        if request_id not in self._kv_draining_requests:
            self._kv_request_generations.pop(request_id, None)
        return self._request_states.pop(request_id, None)

    def preempt_request(self, request_id: str) -> bool:
        if request_id in self._kv_loading_request_ids:
            return False
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
        self._shutdown_diffusion_kv()
        self._request_states.clear()
        self._waiting.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        self._reset_scheduler_state()

    def _reset_kv_transfer_state(self) -> None:
        """Clear request bookkeeping for native KV transfers."""
        self._kv_transfer_request_ids.clear()
        self._kv_finished_request_ids.clear()
        self._kv_loading_request_ids.clear()
        self._kv_draining_requests.clear()
        self._kv_received_request_ids.clear()
        self._kv_request_generations.clear()
        self._native_prefetch_request_id = None

    def _shutdown_diffusion_kv(self) -> None:
        """Stop transfers before releasing their cache and request state."""
        from vllm_omni.diffusion.diffusion_kv.kv_connector import shutdown_kv_connector

        shutdown_kv_connector(scheduler_connector=self._kv_connector)
        self._kv_connector = None
        if self._diffusion_kv_manager is not None:
            self._diffusion_kv_manager.close()
            self._diffusion_kv_manager = None
        self._reset_kv_transfer_state()

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
            if request_id == self._native_prefetch_request_id:
                self._native_prefetch_request_id = None
            if request_id in self._kv_loading_request_ids:
                self._kv_draining_requests[request_id] = state
            else:
                self._release_request_kv(state, status)
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(request_id)
            else:
                state.error = None

        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _release_request_kv(self, state: SchedulerRequestState, status: DiffusionRequestStatus) -> None:
        request_id = state.request_id
        if self._kv_connector is not None:
            for request in state.diffusion_kv_requests:
                request.status = (
                    RequestStatus.FINISHED_ABORTED
                    if status == DiffusionRequestStatus.FINISHED_ABORTED
                    else RequestStatus.FINISHED_STOPPED
                )
                if request.kv_transfer_params:
                    self._kv_finished_request_ids.add(request.request_id)
                block_ids = []
                if self._diffusion_kv_manager.has_request(request_id):
                    blocks = self._diffusion_kv_manager.native_manager.get_blocks(request.request_id)
                    block_ids = blocks.get_block_ids()[0]
                delay_free, _ = self._kv_connector.request_finished(request, block_ids)
                if delay_free:
                    raise RuntimeError("Diffusion consumer cannot release pages while KV transfer is active")
        if self._diffusion_kv_manager is not None:
            self._diffusion_kv_manager.free_request(request_id)

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
        # Also surface admission failures recorded while schedule() built this
        # output. Older finished ids retained only for Worker cleanup have
        # already been popped by the Engine and are deliberately ignored.
        if self._diffusion_kv_manager is not None:
            for request_id, status in statuses.items():
                if status == DiffusionRequestStatus.FINISHED_COMPLETED:
                    self._diffusion_kv_manager.publish_request(request_id)

        finished_req_ids = {
            request_id for request_id in sched_output.finished_req_ids if request_id in self._request_states
        }
        finished_req_ids |= {
            request_id for request_id in sched_output.scheduled_request_ids if request_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, request_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""

    def _make_request_state(self, request_id: str, request: OmniDiffusionRequest) -> SchedulerRequestState:
        kv_requests = request.diffusion_kv_requests or ()
        if self._diffusion_kv_manager is not None:
            self._reject_legacy_dense_kv(request)
            if not kv_requests:
                raise ValueError("paged_scheduler request preprocessing did not produce DiffusionKVRequest state")
        elif kv_requests:
            raise ValueError("dense_legacy request unexpectedly contains Scheduler Diffusion KV requests")

        kv_transfer_params = getattr(request, "kv_transfer_params", None)
        if kv_transfer_params is not None:
            prepare_kv_requests(kv_requests, kv_transfer_params)

        # DiffusionKVRequest objects are mutable Scheduler/native-KVCacheManager
        # state and must never ride the normal request payload to a Worker.
        request.diffusion_kv_requests = None
        return SchedulerRequestState(
            request_id=request_id,
            req=request,
            sampling_params_key=self._build_sampling_params_key(request),
            diffusion_kv_requests=kv_requests,
        )

    @staticmethod
    def _reject_legacy_dense_kv(request: OmniDiffusionRequest) -> None:
        """Keep dense injected KV out of the Scheduler-owned paged path."""

        populated_fields: list[str] = []
        for owner_name, owner in (
            ("request", request),
            ("sampling_params", request.sampling_params),
        ):
            for field_name, value in vars(owner).items():
                if value is not None and (field_name == "past_key_values" or field_name.endswith("_past_key_values")):
                    populated_fields.append(f"{owner_name}.{field_name}")
        if populated_fields:
            fields_text = ", ".join(sorted(populated_fields))
            raise ValueError(
                "paged_scheduler Diffusion KV does not accept legacy dense KV payloads; "
                f"clear these fields before admission: {fields_text}"
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
            condition_key=getattr(request, "batch_compatibility_key", None),
            lora_int_id=lora_request.lora_int_id if lora_request is not None else None,
            use_step_execution=getattr(request, "use_step_execution", True),
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
