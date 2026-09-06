# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

logger = init_logger(__name__)


def uses_native_mrv2_data_plane(
    model_config: Any,
    *,
    use_v2_model_runner: bool,
) -> bool:
    return bool(
        use_v2_model_runner
        and getattr(model_config, "async_chunk", False)
        and getattr(model_config, "supports_native_mrv2_data_plane", False)
    )


# (arch, model_stage) pairs that route their full_payload stage input via
# the worker connector and therefore need the scheduler-side coordinator to
# park requests in WAITING_FOR_INPUT until the recv side delivers.  This set
# must stay aligned with the arch scope of `init_omni_connectors` in
# gpu_ar_model_runner.py and gpu_generation_model_runner.py.  Adding a stage
# here without also wiring its worker connector init produces a permanent
# Stage 1 hang (gate parks the request, no transport ever releases it).
#
_FULL_PAYLOAD_INPUT_STAGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("Qwen3OmniMoeForConditionalGeneration", "talker"),
        ("Qwen3OmniMoeForConditionalGeneration", "code2wav"),
        # qwen2_5_omni thinker->talker uses the real full-payload
        # producer builder (text_hidden_states routed via
        # pooler_output["hidden"] -> accumulator -> connector).  Both
        # stages of qwen2_5_omni are enabled.
        ("Qwen2_5OmniForConditionalGeneration", "talker"),
        ("Qwen2_5OmniForConditionalGeneration", "code2wav"),
        # covo_audio: fused_thinker_talker (Stage 0) -> code2wav (Stage 1).
        ("CovoAudioForConditionalGeneration", "code2wav"),
        # mimo_audio: fused_thinker_talker (Stage 0) -> code2wav (Stage 1).
        ("MiMoAudioModel", "code2wav"),
        # qwen3_tts: Qwen3TTSTalkerForConditionalGeneration (Stage 0)
        # -> Qwen3TTSCode2Wav (Stage 1).  Stage 1 is the consumer.
        ("Qwen3TTSCode2Wav", "code2wav"),
        # minicpmo_4_5: Talker (Stage 1) -> Code2Wav (Stage 2).
        ("MiniCPMO45Code2Wav", "code2wav"),
        # cosyvoice3: cosyvoice3_talker (Stage 0) -> cosyvoice3_code2wav (Stage 1).
        ("CosyVoice3Model", "cosyvoice3_code2wav"),
        # nemotron_voicechat: talker (Stage 1) -> code2wav (Stage 2). Stage 2
        # waits for the talker's full-payload code stacks; the thinker (Stage 0)
        # -> talker (Stage 1) hop is token-path only and must NOT be listed.
        ("NemotronVoiceChatCode2Wav", "code2wav"),
        # audex TTS sync path: thinker (Stage 0) -> streaming decoder (Stage 1).
        # The default deploy is async_chunk; this covers async_chunk: false.
        ("AudexCode2Wav", "audex_code2wav"),
        # audex TTA: tta thinker (Stage 0) -> XCodec1 (Stage 1, always sync
        # full-payload — CNN codec decoded over the full sequence).
        ("AudexXCodec1", "audex_xcodec"),
        # indextts2 / indextts2_5: talker (Stage 0) -> s2mel decoder
        # (Stage 1). Stage 1 consumes the complete mel/optional-latent payload.
        ("IndexTTS2S2MelDecoder", "indextts2_s2mel_decoder"),
        ("IndexTTS25S2MelDecoder", "indextts2_5_s2mel_decoder"),
        # dynin: token2text (Stage 0) -> token2image (Stage 1) ->
        # token2audio (Stage 2).  Producer wires via
        # custom_process_next_stage_input_func: *_full_payload in deploy yaml.
        ("DyninOmniForConditionalGeneration", "token2image"),
        ("DyninOmniForConditionalGeneration", "token2audio"),
    }
)


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
        # Absolute Thinker decode horizon visible to each Stage-1 Talker.
        # A Talker may consume row i iff i < decode_token_end.  This is
        # deliberately monotonic and independent of connector chunk count:
        # one chunk can expose multiple conditioning rows.
        self.decode_token_horizons: dict[str, int] = {}
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
        """Prune all coordinator state owned by a freed request."""
        self._full_payload_input_received.discard(request_id)
        self.finished_requests.discard(request_id)
        self.requests_with_ready_chunks.discard(request_id)
        self.input_terminal_req_ids.discard(request_id)
        self.decode_token_horizons.pop(request_id, None)
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

            if metadata.get("input_terminal") is True:
                self.input_terminal_req_ids.add(req_id)

            decode_token_end = metadata.get("decode_token_end")
            if decode_token_end is not None:
                decode_token_end = int(decode_token_end)
                previous_end = self.decode_token_horizons.get(req_id)
                self.decode_token_horizons[req_id] = max(
                    previous_end if previous_end is not None else decode_token_end,
                    decode_token_end,
                )

            # Handle the downstream sampler history if present (for models like
            # Qwen3-Omni). Exact IDs are part of sampling semantics; a same-sized
            # zero placeholder is sufficient for KV allocation but changes
            # repetition-penalty behavior and therefore the seeded Talker path.
            # Only apply when the request has not started decoding yet
            # (no output tokens). Resetting a mid-decode request would
            # destroy generated tokens and desync KV cache state.
            prompt_ids_value = metadata.get("next_stage_prompt_ids")
            next_prompt_ids = None
            if isinstance(prompt_ids_value, (list, tuple)) and prompt_ids_value:
                next_prompt_ids = [int(token_id) for token_id in prompt_ids_value]
            next_len = metadata.get("next_stage_prompt_len")
            if next_prompt_ids is not None or (isinstance(next_len, int) and next_len > 0):
                if next_prompt_ids is None:
                    next_prompt_ids = [0] * int(next_len)
                else:
                    next_len = len(next_prompt_ids)
                if next_len > 0:
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
                        if (
                            current_prompt_ids != next_prompt_ids
                            or current_prompt_len != next_len
                            or getattr(request, "num_prompt_tokens", None) != next_len
                        ):
                            request.prompt_token_ids = next_prompt_ids
                            request.num_prompt_tokens = next_len
                            request._all_token_ids.clear()
                            request._all_token_ids.extend(next_prompt_ids)
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
                runtime_seed = None
                if "left_context_size" in metadata:
                    runtime_seed = {
                        "meta": {"left_context_size": metadata["left_context_size"]},
                    }
                request._omni_initial_model_buffer = runtime_seed
                if new_ids:
                    request.prompt_token_ids = new_ids
                    request.num_prompt_tokens = len(new_ids)
                    request._all_token_ids.clear()
                    request._all_token_ids.extend(new_ids)
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
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _request_can_run(
        self,
        request: Request,
        chunk_ready_req_ids: set[str],
    ) -> bool:
        req_id = request.request_id
        if req_id in self.finished_requests:
            # Once the producer is terminal, the Talker must be allowed to
            # inject TTS EOS and subsequent pad frames until its own sampler
            # reaches codec EOS.
            return True

        decode_token_end = self.decode_token_horizons.get(req_id)
        if self._stage_id == 1 and decode_token_end is not None:
            output_token_ids = getattr(request, "_output_token_ids", ())
            # Async scheduling reserves future decode positions with output
            # placeholders before sampled tokens are committed to
            # _output_token_ids. Count both or the scheduler can overbook one
            # conditioning row while the previous Talker step is in flight.
            in_flight_outputs = int(getattr(request, "num_output_placeholders", 0) or 0)
            next_decode_token = len(output_token_ids) + in_flight_outputs
            if next_decode_token == 0:
                # The initial prefill payload has no decode span.  Its
                # one-shot connector readiness admits Talker prefill.
                return req_id in chunk_ready_req_ids or req_id in self.requests_with_ready_chunks
            return next_decode_token < decode_token_end

        return req_id in chunk_ready_req_ids or req_id in self.requests_with_ready_chunks

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                self.requests_with_ready_chunks.discard(
                    getattr(req_data, "req_id", None),
                )

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self.requests_with_ready_chunks.discard(req_id)
