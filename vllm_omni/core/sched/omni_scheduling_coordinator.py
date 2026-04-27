# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling-side coordination for chunk and full_payload input waiting.

Manages WAITING_FOR_CHUNK and WAITING_FOR_INPUT state transitions
based on readiness signals from OmniConnectorOutput, without ever
calling connector.put()/get().

This replaces the scheduling half of OmniChunkTransferAdapter; the
transport half lives in OmniConnectorModelRunnerMixin.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class OmniSchedulingCoordinator:
    """Pure-scheduling coordinator for chunk and full_payload input waiting.

    The Scheduler owns an instance of this class.  It consumes readiness
    signals produced by the Model Runner's ``OmniConnectorModelRunnerMixin``
    (via ``OmniConnectorOutput``) and manages ``WAITING_FOR_CHUNK`` and
    ``WAITING_FOR_INPUT`` state transitions accordingly.
    """

    def __init__(self, scheduler_max_num_seqs: int, stage_id: int = 0, async_chunk: bool = False):
        self._stage_id = stage_id
        self._scheduler_max_num_seqs = scheduler_max_num_seqs
        self._async_chunk = async_chunk

        self.finished_requests: set[str] = set()
        self.requests_with_ready_chunks: set[str] = set()
        self._full_payload_input_received: set[str] = set()

        self._waiting_for_chunk_waiting: deque[Any] = deque()
        self._waiting_for_chunk_running: deque[Any] = deque()

        # Request IDs that were newly registered for chunk recv this cycle.
        # The engine/Model Runner should call register_chunk_recv() for these
        # so the bg thread starts polling.
        self.pending_chunk_registrations: list[Any] = []

        # Requests waiting for full_payload stage input (WAITING_FOR_INPUT).
        self._waiting_for_input: deque[Any] = deque()
        self.pending_input_registrations: list[Any] = []

        # Monotonic timestamp recording when each request first entered
        # WAITING_FOR_CHUNK or WAITING_FOR_INPUT.  Used by
        # collect_timed_out_request_ids() to detect orphaned waits.
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

        terminal_ready_req_ids = chunk_ready_req_ids.intersection(chunk_finished_req_ids)
        self.finished_requests.update(chunk_finished_req_ids - terminal_ready_req_ids)
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
        self.finished_requests.update(terminal_ready_req_ids)

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
        running_queue: list[Request],
        stage_recv_req_ids: set[str],
    ) -> None:
        """Manage WAITING_FOR_INPUT lifecycle for full_payload_mode.

        For non-Stage-0 stages in full_payload_mode (``async_chunk=False``):
        1. Fresh WAITING requests are transitioned to WAITING_FOR_INPUT
           and registered for bg-thread polling.
        2. WAITING_FOR_INPUT requests whose data has arrived (in
           ``stage_recv_req_ids``) are transitioned back to WAITING.
        """
        if self._stage_id == 0:
            return

        self._full_payload_input_received.update(stage_recv_req_ids)
        if not self._async_chunk and stage_recv_req_ids:
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

        if not self._async_chunk:
            to_remove: list[Any] = []
            queue_snapshot = list(waiting_queue)
            for request in queue_snapshot:
                if request.status == RequestStatus.WAITING:
                    if request.request_id in self._full_payload_input_received:
                        continue
                    if request.request_id in self.requests_with_ready_chunks:
                        continue
                    if request.request_id in self.finished_requests:
                        continue
                    request.status = RequestStatus.WAITING_FOR_INPUT
                    self._waiting_since.setdefault(request.request_id, time.monotonic())
                    to_remove.append(request)
                    self._waiting_for_input.append(request)
                    self.pending_input_registrations.append(request)
                elif request.status == RequestStatus.WAITING_FOR_INPUT:
                    if request.request_id in stage_recv_req_ids:
                        request.status = RequestStatus.WAITING
                        self._waiting_since.pop(request.request_id, None)
                    else:
                        to_remove.append(request)
                        self._waiting_for_input.append(request)
                        self.pending_input_registrations.append(request)
            for request in to_remove:
                waiting_queue.remove(request)

    def process_pending_full_payload_inputs_legacy(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        stage_recv_req_ids: set[str],
    ) -> None:
        """Compatibility wrapper for ``process_pending_full_payload_inputs``."""
        self.process_pending_full_payload_inputs(waiting_queue, running_queue, stage_recv_req_ids)

    def free_finished_request(self, request_id: str) -> None:
        """Prune internal tracking sets for a freed request to prevent unbounded growth."""
        self._full_payload_input_received.discard(request_id)
        self.finished_requests.discard(request_id)
        self.requests_with_ready_chunks.discard(request_id)
        self._waiting_since.pop(request_id, None)

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
            remaining: deque[Any] = deque()
            for request in queue:
                if request.request_id not in timed_out_ids:
                    remaining.append(request)
            setattr(self, queue_attr, remaining)

        for req_id in timed_out_ids:
            self._waiting_since.pop(req_id, None)
            logger.warning(
                "[Coordinator stage-%s] Request %s timed out waiting for chunk/input (waited > %.0fs)",
                self._stage_id,
                req_id,
                timeout_s,
            )

        return timed_out_ids

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """Return waiting-for-chunk/input requests to scheduling queues."""
        for request in self._waiting_for_chunk_waiting:
            waiting_queue.add_request(request)
        self._waiting_for_chunk_waiting = deque()

        if self._waiting_for_chunk_running:
            running_queue.extend(self._waiting_for_chunk_running)
        self._waiting_for_chunk_running = deque()

        for request in self._waiting_for_input:
            waiting_queue.add_request(request)
        self._waiting_for_input = deque()

    def update_request_metadata(
        self,
        requests: dict[str, Request],
        request_metadata: dict[str, dict[str, Any]],
        model_mode: str = "ar",
    ) -> None:
        """Apply received scheduling metadata to request objects.

        For AR mode: only scheduler-visible metadata is applied locally.
        For Generation mode: updates ``request.prompt_token_ids``.

        Additionally, if the payload contains ``next_stage_prompt_len``,
        updates the request's ``prompt_token_ids`` to the correct length.
        """
        for req_id, metadata in request_metadata.items():
            request = requests.get(req_id)
            if request is None:
                continue

            # Handle next_stage_prompt_len if present (for models like Qwen3-Omni).
            # Only apply when the request has not started decoding yet
            # (no output tokens). Resetting a mid-decode request would
            # destroy generated tokens and desync KV cache state.
            if "next_stage_prompt_len" in metadata:
                next_len = metadata["next_stage_prompt_len"]
                if isinstance(next_len, int) and next_len > 0:
                    output_token_ids = getattr(request, "_output_token_ids", None)
                    has_decode_output = output_token_ids is not None and len(output_token_ids) > 0
                    if has_decode_output:
                        logger.debug(
                            "[Coordinator stage-%s] Skipping prompt resize for req %s: "
                            "request already has %s output tokens",
                            self._stage_id,
                            req_id,
                            len(output_token_ids),
                        )
                    else:
                        current_prompt_ids = getattr(request, "prompt_token_ids", []) or []
                        current_prompt_len = len(current_prompt_ids)
                        if current_prompt_len != next_len or getattr(request, "num_prompt_tokens", None) != next_len:
                            new_prompt = [0] * next_len
                            request.prompt_token_ids = new_prompt
                            request.num_prompt_tokens = next_len
                            request._all_token_ids.clear()
                            request._all_token_ids.extend(new_prompt)
                            request._output_token_ids.clear()
                            request.num_computed_tokens = 0
                            logger.debug(
                                "[Coordinator stage-%s] Updated prompt_token_ids length to %s for req %s",
                                self._stage_id,
                                next_len,
                                req_id,
                            )

            if model_mode != "ar":
                new_ids = metadata.get("code_predictor_codes", [])
                runtime_seed = None
                if "left_context_size" in metadata:
                    runtime_seed = {
                        "meta": {"left_context_size": metadata["left_context_size"]},
                    }
                request._omni_initial_model_buffer = runtime_seed
                if new_ids:
                    request.prompt_token_ids = new_ids
                    request.num_computed_tokens = 0

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """Clear per-cycle ready state after scheduler output is materialized."""
        self._clear_chunk_ready(scheduler_output)

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
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    continue
                if request.request_id in self.finished_requests:
                    continue
                if request.status == RequestStatus.WAITING_FOR_INPUT:
                    continue
                if request.request_id in chunk_ready_req_ids:
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
                self.pending_chunk_registrations.append(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
                self._waiting_since.setdefault(request.request_id, time.monotonic())
            else:
                if request.request_id in chunk_ready_req_ids:
                    request.status = target_status
                    self.requests_with_ready_chunks.add(request.request_id)
                    self._waiting_since.pop(request.request_id, None)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                self.requests_with_ready_chunks.discard(
                    getattr(req_data, "req_id", None),
                )

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self.requests_with_ready_chunks.discard(req_id)


# Backward-compatible alias
ChunkSchedulingCoordinator = OmniSchedulingCoordinator
