# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduling-side coordination for chunk and full-payload input waiting.

Manages WAITING_FOR_CHUNK and WAITING_FOR_INPUT state transitions based on
readiness signals from OmniConnectorOutput, without calling connector I/O.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.output import OmniChunkRecvHandle
from vllm_omni.outputs import SchedulingMetadataUpdate

logger = init_logger(__name__)


def uses_native_mrv2_data_plane(
    model_config: Any,
    *,
    use_v2_model_runner: bool,
) -> bool:
    return bool(use_v2_model_runner and getattr(model_config, "supports_native_mrv2_data_plane", False))


def uses_full_payload_input_coordinator(model_config: Any) -> bool:
    """Returns True if this stage parks pending requests in
    WAITING_FOR_INPUT awaiting a full_payload delivery on the worker connector.

    Gated by the topology-declared ``requires_full_payload_input`` capability on
    downstream (stage_id > 0) stages, and only on the non-async-chunk path
    (async-chunk stages are fed through the streamed connector instead).
    """
    if getattr(model_config, "stage_id", 0) <= 0:
        return False
    if getattr(model_config, "async_chunk", False):
        return False
    return bool(getattr(model_config, "requires_full_payload_input", False))


class OmniSchedulingCoordinator:
    """Pure-scheduling coordinator for chunk and full-payload input waiting.

    The Scheduler owns an instance of this class.  It consumes readiness
    signals produced by the Model Runner's ``OmniConnectorModelRunnerMixin``
    (via ``OmniConnectorOutput``) and manages ``WAITING_FOR_INPUT`` state
    transitions accordingly.
    """

    def __init__(
        self,
        scheduler_max_num_seqs: int = 0,
        stage_id: int = 0,
        async_chunk: bool = False,
    ) -> None:
        self._stage_id = stage_id
        self._scheduler_max_num_seqs = scheduler_max_num_seqs
        self._async_chunk = async_chunk

        self.finished_requests: set[str] = set()
        self.requests_with_ready_chunks: set[str] = set()
        self.input_terminal_req_ids: set[str] = set()
        self._full_payload_input_received: set[str] = set()

        self._waiting_for_chunk_waiting: deque[Any] = deque()
        self._waiting_for_chunk_running: deque[Any] = deque()

        # Request IDs that were newly registered for chunk recv this cycle.
        # The engine/Model Runner should call register_chunk_recv() for these
        # so the bg thread starts polling.
        self.pending_chunk_registrations: list[OmniChunkRecvHandle] = []
        # Requests waiting for full_payload stage input (WAITING_FOR_INPUT).
        self._waiting_for_input: deque[Any] = deque()
        # Per-cycle list of minimal handles to ship to the model runner so it
        # can call register_chunk_recv().  Typed concretely (not list[Any]) so
        # the surrounding OmniSchedulerOutput stays msgspec-friendly across
        # default, PD-disagg, and multi-node executor IPC paths.
        self.pending_input_registrations: list[OmniChunkRecvHandle] = []

        # Monotonic timestamp recording when each request first entered
        # WAITING_FOR_INPUT.  Used by collect_timed_out_request_ids() to
        # detect orphaned waits.
        self._waiting_since: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  Core scheduling methods
    # ------------------------------------------------------------------ #

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        chunk_ready_req_ids: set[str],
        chunk_finished_req_ids: set[str],
    ) -> None:
        """Transition requests whose chunks have arrived.

        Args:
            waiting_queue: Scheduler's waiting request queue.
            running_queue: Scheduler's running request list.
            chunk_ready_req_ids: IDs with a newly arrived chunk this cycle.
            chunk_finished_req_ids: IDs whose final chunk has arrived.
        """
        if self._stage_id == 0 or not self._async_chunk:
            return

        self.finished_requests.update(chunk_finished_req_ids)
        self.pending_chunk_registrations = []

        self._process_chunk_queue(
            waiting_queue,
            self._waiting_for_chunk_waiting,
            RequestStatus.WAITING,
            chunk_ready_req_ids,
        )
        self._process_chunk_queue(
            running_queue,
            self._waiting_for_chunk_running,
            RequestStatus.RUNNING,
            chunk_ready_req_ids,
        )
        while len(running_queue) > self._scheduler_max_num_seqs:
            request = running_queue.pop()
            # Must reset status to WAITING so the scheduler treats it as
            # schedulable work.  KV blocks are NOT freed here (unlike a
            # real preemption), so PREEMPTED would be incorrect.
            request.status = RequestStatus.WAITING
            waiting_queue.prepend_requests([request])

    def process_pending_full_payload_inputs(
        self,
        waiting_queue: Any,
        stage_recv_req_ids: set[str],
    ) -> None:
        """Manage WAITING_FOR_INPUT lifecycle for full_payload_mode.

        For non-Stage-0 stages in full_payload mode:
        1. Fresh WAITING requests are transitioned to WAITING_FOR_INPUT
           and registered for bg-thread polling.
        2. WAITING_FOR_INPUT requests whose data has arrived (in
           ``stage_recv_req_ids``) are transitioned back to WAITING.
        """
        if self._stage_id == 0:
            return

        self._full_payload_input_received.update(stage_recv_req_ids)
        if stage_recv_req_ids:
            self.finished_requests.update(stage_recv_req_ids)
            logger.debug(
                "[Coordinator stage-%s] full_payload recv -> finished_requests: %s",
                self._stage_id,
                stage_recv_req_ids,
            )
        self.pending_input_registrations = []

        remaining: deque[Any] = deque()
        for request in self._waiting_for_input:
            if request.request_id in stage_recv_req_ids:
                request.status = RequestStatus.WAITING
                self._waiting_since.pop(request.request_id, None)
                waiting_queue.add_request(request)
            else:
                remaining.append(request)
        self._waiting_for_input = remaining

        to_remove: list[Any] = []
        queue_snapshot = list(waiting_queue)
        for request in queue_snapshot:
            if request.status == RequestStatus.WAITING:
                if request.request_id in self._full_payload_input_received:
                    continue
                if request.request_id in self.finished_requests:
                    continue
                request.status = RequestStatus.WAITING_FOR_INPUT
                self._waiting_since.setdefault(request.request_id, time.monotonic())
                to_remove.append(request)
                self._waiting_for_input.append(request)
                self.pending_input_registrations.append(
                    OmniChunkRecvHandle(
                        request_id=request.request_id,
                        external_req_id=getattr(request, "external_req_id", None),
                        payload_sender_info=getattr(request, "payload_sender_info", None),
                    )
                )
            elif request.status == RequestStatus.WAITING_FOR_INPUT:
                if request.request_id in stage_recv_req_ids:
                    request.status = RequestStatus.WAITING
                    self._waiting_since.pop(request.request_id, None)
                else:
                    to_remove.append(request)
                    self._waiting_for_input.append(request)
                    self.pending_input_registrations.append(
                        OmniChunkRecvHandle(
                            request_id=request.request_id,
                            external_req_id=getattr(request, "external_req_id", None),
                            payload_sender_info=getattr(request, "payload_sender_info", None),
                        )
                    )
        if to_remove:
            # Use the bulk-remove helper: one O(N) sweep instead of N
            # repeated O(N) removes from a list-backed queue.
            waiting_queue.remove_requests(to_remove)

    def free_finished_request(self, request_id: str) -> None:
        """Prune all coordinator state owned by a freed request."""
        self._full_payload_input_received.discard(request_id)
        self.finished_requests.discard(request_id)
        self.requests_with_ready_chunks.discard(request_id)
        self.input_terminal_req_ids.discard(request_id)
        self._waiting_since.pop(request_id, None)
        for queue_attr in (
            "_waiting_for_chunk_waiting",
            "_waiting_for_chunk_running",
            "_waiting_for_input",
        ):
            queue = getattr(self, queue_attr)
            setattr(
                self,
                queue_attr,
                deque(request for request in queue if request.request_id != request_id),
            )
        self.pending_chunk_registrations = [
            handle for handle in self.pending_chunk_registrations if handle.request_id != request_id
        ]
        self.pending_input_registrations = [
            handle for handle in self.pending_input_registrations if handle.request_id != request_id
        ]

    def collect_timed_out_request_ids(
        self,
        timeout_s: float,
    ) -> set[str]:
        """Return IDs of requests that have been waiting longer than *timeout_s*.

        Uses ``_waiting_since`` timestamps (always up-to-date) to detect
        timed-out requests.  This method is safe to call at any point in
        the scheduling cycle — it does **not** rely on coordinator internal
        queues (which are empty after ``restore_queues()``).

        Clears ``_waiting_since`` for timed-out IDs and defensively removes
        them from coordinator internal queues if present.  The caller
        (scheduler) should then remove the requests from its queues,
        set ``FINISHED_ERROR``, and call ``_free_request()`` so that
        ``cleanup_finished_request()`` fires in the model runner mixin.
        """
        if timeout_s <= 0:
            return set()
        now = time.monotonic()
        timed_out_ids: set[str] = set()
        for req_id, start_time in self._waiting_since.items():
            if now - start_time > timeout_s:
                timed_out_ids.add(req_id)
        if not timed_out_ids:
            return set()

        # Defensively remove from coordinator internal queues (may already
        # be empty if restore_queues() has run).
        for queue_attr in (
            "_waiting_for_chunk_waiting",
            "_waiting_for_chunk_running",
            "_waiting_for_input",
        ):
            queue = getattr(self, queue_attr)
            remaining = deque(request for request in queue if request.request_id not in timed_out_ids)
            setattr(self, queue_attr, remaining)

        for req_id in timed_out_ids:
            self._waiting_since.pop(req_id, None)
            logger.warning(
                "[Coordinator stage-%s] Request %s timed out waiting for input (waited > %.0fs)",
                self._stage_id,
                req_id,
                timeout_s,
            )

        return timed_out_ids

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request] | None = None,
    ) -> None:
        """Return waiting-for-chunk/input requests to scheduling queues."""
        for request in self._waiting_for_chunk_waiting:
            waiting_queue.add_request(request)
        self._waiting_for_chunk_waiting = deque()

        if running_queue is not None and self._waiting_for_chunk_running:
            running_queue.extend(self._waiting_for_chunk_running)
        self._waiting_for_chunk_running = deque()

        for request in self._waiting_for_input:
            waiting_queue.add_request(request)
        self._waiting_for_input = deque()

    def update_request_metadata(
        self,
        requests: dict[str, Request],
        request_metadata: dict[str, SchedulingMetadataUpdate],
    ) -> None:
        """Apply typed runner updates without interpreting payload metadata."""
        for req_id, update in request_metadata.items():
            request = requests.get(req_id)
            if request is None:
                continue
            if update.input_terminal:
                self.input_terminal_req_ids.add(req_id)
            self._apply_scheduling_update(request, update)

    def _apply_scheduling_update(self, request: Request, update: SchedulingMetadataUpdate) -> None:
        if update.resize_prompt_to is not None:
            output_token_ids = getattr(request, "_output_token_ids", None)
            if output_token_ids is None or not output_token_ids:
                next_len = update.resize_prompt_to
                current_prompt_ids = getattr(request, "prompt_token_ids", ()) or ()
                if len(current_prompt_ids) != next_len or getattr(request, "num_prompt_tokens", None) != next_len:
                    new_prompt = [0] * next_len
                    request.prompt_token_ids = new_prompt
                    request.num_prompt_tokens = next_len
                    request._all_token_ids.clear()
                    request._all_token_ids.extend(new_prompt)
                    request._output_token_ids.clear()
                    request.num_computed_tokens = 0

        if update.prompt_token_ids is not None:
            prompt_token_ids = list(update.prompt_token_ids)
            request.prompt_token_ids = prompt_token_ids
            request.num_prompt_tokens = len(prompt_token_ids)
            request._all_token_ids.clear()
            request._all_token_ids.extend(prompt_token_ids)
            request._output_token_ids.clear()
            request.num_computed_tokens = 0

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """Clear per-cycle ready state after scheduler output is materialized."""
        self._clear_chunk_ready(scheduler_output)
        self.input_terminal_req_ids.difference_update(
            self._scheduled_request_ids(scheduler_output),
        )

    def get_scheduled_input_terminal_req_ids(
        self,
        scheduler_output: Any,
    ) -> set[str]:
        """Return terminal inputs executed by this immutable scheduler step."""
        return self.input_terminal_req_ids.intersection(
            self._scheduled_request_ids(scheduler_output),
        )

    @staticmethod
    def _scheduled_request_ids(scheduler_output: Any) -> set[str]:
        num_scheduled_tokens = getattr(
            scheduler_output,
            "num_scheduled_tokens",
            None,
        )
        if num_scheduled_tokens is not None:
            return set(num_scheduled_tokens)

        request_ids = {req.req_id for req in getattr(scheduler_output, "scheduled_new_reqs", ())}
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        request_ids.update(getattr(cached_reqs, "req_ids", ()))
        return request_ids

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        chunk_ready_req_ids: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            request_can_run = self._request_can_run(request, chunk_ready_req_ids)
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request_can_run:
                    if request.request_id in chunk_ready_req_ids:
                        self.requests_with_ready_chunks.add(request.request_id)
                    continue
                if request.status == RequestStatus.WAITING_FOR_INPUT:
                    continue
                self.pending_chunk_registrations.append(
                    OmniChunkRecvHandle(
                        request_id=request.request_id,
                        external_req_id=getattr(request, "external_req_id", None),
                    )
                )
                request.status = RequestStatus.WAITING_FOR_CHUNK
                self._waiting_since.setdefault(request.request_id, time.monotonic())
            else:
                if request_can_run:
                    request.status = target_status
                    if request.request_id in chunk_ready_req_ids:
                        self.requests_with_ready_chunks.add(request.request_id)
                    self._waiting_since.pop(request.request_id, None)
                    continue
            if isinstance(queue, list):
                queue.remove(request)
            else:
                queue.remove_request(request)
            waiting_for_chunk_list.append(request)

    def _request_can_run(
        self,
        request: Request,
        chunk_ready_req_ids: set[str],
    ) -> bool:
        req_id = request.request_id
        if req_id in self.finished_requests:
            # Admit the final payload or terminal marker for cleanup.
            return True

        return req_id in chunk_ready_req_ids or req_id in self.requests_with_ready_chunks

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                req_id = getattr(req_data, "req_id", None)
                if req_id is not None:
                    self.requests_with_ready_chunks.discard(req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self.requests_with_ready_chunks.discard(req_id)
