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

    @staticmethod
    def _flatten_prompt_token_ids(value: Any) -> list[int]:
        """Normalize connector metadata into flat prompt token ids."""
        if value is None:
            return []
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            flattened: list[int] = []
            for item in value:
                if hasattr(item, "detach") and hasattr(item, "cpu") and hasattr(item, "tolist"):
                    item = item.detach().cpu().tolist()
                elif hasattr(item, "tolist") and not isinstance(item, (list, tuple)):
                    item = item.tolist()
                if isinstance(item, (list, tuple)):
                    flattened.extend(int(token_id) for token_id in item)
                else:
                    flattened.append(int(item))
            return flattened
        return [int(value)]

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
                new_ids = self._flatten_prompt_token_ids(metadata.get("code_predictor_codes"))
                if new_ids:
                    request.prompt_token_ids = new_ids
                    request.num_prompt_tokens = len(new_ids)
                    request._all_token_ids.clear()
                    request._all_token_ids.extend(new_ids)
                    request._output_token_ids.clear()
                    request.num_computed_tokens = 0
