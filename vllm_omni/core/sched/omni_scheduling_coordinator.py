# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling-side coordination for full_payload input waiting.

Manages WAITING_FOR_INPUT state transitions based on readiness signals
from OmniConnectorOutput, without ever calling connector.put()/get().

Chunk waiting (WAITING_FOR_CHUNK) lives on OmniChunkTransferAdapter.
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
    """Pure-scheduling coordinator for full_payload input waiting.

    The Scheduler owns an instance of this class.  It consumes readiness
    signals produced by the Model Runner's ``OmniConnectorModelRunnerMixin``
    (via ``OmniConnectorOutput``) and manages ``WAITING_FOR_INPUT`` state
    transitions accordingly.
    """

    def __init__(self, stage_id: int = 0):
        self._stage_id = stage_id

        self.finished_requests: set[str] = set()
        self._full_payload_input_received: set[str] = set()

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
                        )
                    )
        if to_remove:
            # Use the bulk-remove helper: one O(N) sweep instead of N
            # repeated O(N) removes from a list-backed queue.
            waiting_queue.remove_requests(to_remove)

    def free_finished_request(self, request_id: str) -> None:
        """Prune internal tracking sets for a freed request to prevent unbounded growth."""
        self._full_payload_input_received.discard(request_id)
        self.finished_requests.discard(request_id)
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
        remaining: deque[Any] = deque()
        for request in self._waiting_for_input:
            if request.request_id not in timed_out_ids:
                remaining.append(request)
        self._waiting_for_input = remaining

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
    ) -> None:
        """Return waiting-for-input requests to the waiting queue."""
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
