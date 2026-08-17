from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable
from typing import Any

import torch
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)
from vllm_omni.core.sched.output import (
    OmniChunkRecvHandle,
    OmniNewRequestData,
    OmniSchedulerOutput,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine import OmniEngineCoreOutput

logger = init_logger(__name__)

_STATS_INTERVAL_S = 1.0

# Upper bound on how long a request may wait for stage input before the
# scheduler force-fails it.  Defends against stuck consumer-side requests when
# the producer drops a payload, send fails, or recv never arrives.  Override
# per-deployment via VLLM_OMNI_INPUT_WAIT_TIMEOUT_S; set exactly 0 to disable
# the safety net. Negative and non-finite values are rejected at startup.
#
# Scope: both transports.  The full-payload path measures time parked in
# WAITING_FOR_INPUT (``OmniSchedulingCoordinator._waiting_since``); the
# async-chunk path measures time stalled in WAITING_FOR_CHUNK
# (``OmniChunkTransferAdapter._waiting_since``, reset on each chunk arrival).
# One knob, because the question it answers -- "how long may a request wait for
# stage input?" -- does not depend on which transport carries it.
_INPUT_WAIT_TIMEOUT_DEFAULT = 600.0
_INPUT_WAIT_TIMEOUT_RAW = os.environ.get("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", "600")
try:
    DEFAULT_INPUT_WAIT_TIMEOUT_S: float = float(_INPUT_WAIT_TIMEOUT_RAW)
except ValueError:
    logger.warning(
        "Invalid VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=%r; falling back to %s seconds.",
        _INPUT_WAIT_TIMEOUT_RAW,
        _INPUT_WAIT_TIMEOUT_DEFAULT,
    )
    DEFAULT_INPUT_WAIT_TIMEOUT_S = _INPUT_WAIT_TIMEOUT_DEFAULT

if not math.isfinite(DEFAULT_INPUT_WAIT_TIMEOUT_S):
    # nan compares false against everything, so every deadline check silently
    # passes; inf makes the deadline unreachable. Both disable the net while
    # looking like a number, which is the failure this item exists to stop.
    raise ValueError(
        f"VLLM_OMNI_INPUT_WAIT_TIMEOUT_S={_INPUT_WAIT_TIMEOUT_RAW!r} is not finite. "
        "Use a positive number of seconds, or exactly 0 to disable the deadline."
    )
if DEFAULT_INPUT_WAIT_TIMEOUT_S < 0:
    # Fail at startup rather than substituting a default. `-1` is a common idiom
    # for "no limit", so silently turning it into 600 would give an operator who
    # meant "never time out" mysterious failures ten minutes in.
    raise ValueError(
        f"VLLM_OMNI_INPUT_WAIT_TIMEOUT_S={_INPUT_WAIT_TIMEOUT_RAW!r} is negative. "
        "Use a positive number of seconds, or exactly 0 to disable the deadline."
    )
elif DEFAULT_INPUT_WAIT_TIMEOUT_S == 0:
    # Zero stays a supported opt-out, but it is no longer silent: it disables
    # the deadline on BOTH transports, so a request whose stage input never
    # arrives waits forever. Say which nets just went away.
    logger.warning(
        "VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0: stage-input deadlines are DISABLED for "
        "both the full-payload path (WAITING_FOR_INPUT) and the async-chunk path "
        "(WAITING_FOR_CHUNK). A request whose input never arrives will wait "
        "indefinitely instead of failing."
    )


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    # ------------------------------------------------------------------ #
    #  Shared scheduler/output helpers (lift the AR / generation duplicates)
    # ------------------------------------------------------------------ #

    def _init_omni_io_scheduling_state(self) -> None:
        """Initialize scheduler state shared by AR and generation stages."""
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = (
            OmniChunkTransferAdapter(self.vllm_config) if getattr(model_config, "async_chunk", False) else None
        )
        self.input_coordinator = (
            OmniSchedulingCoordinator(stage_id=getattr(model_config, "stage_id", 0))
            if uses_full_payload_input_coordinator(model_config)
            else None
        )
        self._latest_omni_connector_output = None
        # Optional per-stage pooling-output decoder hook (dotted path in
        # model_config); applied worker-side before IPC.
        self._pooling_output_decoder = None
        _decoder_path = getattr(model_config, "pooling_output_decoder", None)
        if _decoder_path:
            self._pooling_output_decoder = resolve_obj_by_qualname(str(_decoder_path))

    def _maybe_decode_pooling_output(self, request: Request, pooler_output: Any) -> Any:
        """Apply the stage's pooling-output decoder hook to the pooler tensor
        before IPC, or pass it through unchanged when none is configured.
        Decoder exceptions propagate; callers fail the request with
        FinishReason.ERROR rather than emitting an empty success."""
        if self._pooling_output_decoder is None:
            return pooler_output
        if pooler_output is None or getattr(request, "pooling_params", None) is None:
            return pooler_output
        if not isinstance(pooler_output, torch.Tensor):
            return pooler_output
        return self._pooling_output_decoder(
            pooler_output,
            request,
            self.vllm_config.model_config.hf_config,
        )

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

    def _replace_streaming_session(self, session: Request, update: StreamingUpdate) -> None:
        """Replace a downstream stage's placeholder with its next payload."""
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is not None:
            adapter.segment_finished_requests.discard(session.request_id)
        session._output_token_ids.clear()
        session._all_token_ids.clear()
        # In-flight outputs from the previous segment were optimistically
        # scheduled (async lookahead). Mark them stale so update_from_output
        # drops them instead of underflowing num_output_placeholders
        # (vLLM 0.27 a0c092ee72 removed async_tokens_to_discard). Seed in
        # SCHEDULED-token units — num_in_flight_tokens matches what each
        # pre-replacement frame will drain, so the counter reaches exactly
        # zero; a placeholder-based seed swallowed valid new-segment frames
        # whenever placeholder counts diverged from scheduled counts.
        session.num_stale_output_tokens += int(getattr(session, "num_in_flight_tokens", 0) or 0)
        session.num_output_placeholders = 0
        session.spec_token_ids = []
        new_prompt = update.prompt_token_ids or ()
        session._all_token_ids.extend(new_prompt)
        session.num_computed_tokens = 0
        session.prompt_token_ids = new_prompt
        session.additional_information = update.additional_information or None
        session.model_intermediate_buffer = getattr(
            update,
            "model_intermediate_buffer",
            None,
        )
        session.update_block_hashes()
        session.num_prompt_tokens = len(new_prompt)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING
        if session in self.skipped_waiting:
            self.skipped_waiting.remove_requests((session,))
            self._enqueue_waiting_request(session)
        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _consume_pending_connector_output(self, model_mode: str) -> None:
        """Drain ``self._latest_omni_connector_output`` into the coordinator.

        Called at the top of every ``schedule()`` cycle.  Identical between
        AR and generation schedulers except for the ``model_mode`` argument
        forwarded to ``update_request_metadata``.
        """
        connector_output = getattr(self, "_latest_omni_connector_output", None)
        self._latest_omni_connector_output = None
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        if connector_output and connector_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests, connector_output.request_metadata, model_mode=model_mode
            )
        input_coordinator.process_pending_full_payload_inputs(
            self.waiting,
            connector_output.stage_recv_req_ids if connector_output else set(),
        )

    def _process_pending_omni_inputs(self, model_mode: str) -> None:
        """Apply pending connector inputs, timeouts, and async chunks."""
        self._consume_pending_connector_output(model_mode)
        self._process_pending_input_timeouts()
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting,
                self.running,
                scheduler_requests=self.requests,
            )
            self._process_pending_chunk_timeouts()
            self._log_failed_chunk_sends()

    def _restore_omni_wait_queues(self) -> None:
        """Restore requests temporarily parked by Omni input gates."""
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.restore_queues(
                self.waiting,
                self.running,
                scheduler_requests=self.requests,
            )
        if self.input_coordinator:
            self.input_coordinator.restore_queues(self.waiting)

    def _process_pending_input_timeouts(self) -> None:
        """Force-fail requests waiting on the full-payload coordinator too long.

        Called at the top of every ``schedule()`` cycle, right after
        ``_consume_pending_connector_output``.  Without this hook, a request
        whose producer dropped a payload would sit in the
        full-payload-input wait state indefinitely (the runner mixin
        protects ``_pending_load_reqs`` from prune sweeps).

        Reads ``_waiting_since`` timestamps maintained by the input
        coordinator and delegates to the base scheduler's
        ``finish_requests`` to mark expired requests FINISHED_ERROR.
        Disabled when ``DEFAULT_INPUT_WAIT_TIMEOUT_S`` is <= 0.

        Scope: only covers ``input_coordinator`` (full-payload path).
        Async-chunk requests park in ``chunk_transfer_adapter`` instead and
        are handled by ``_process_pending_chunk_timeouts`` below, which shares
        this timeout.
        """
        if DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0:
            return
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        timed_out_ids = input_coordinator.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)
        if not timed_out_ids:
            return
        present_ids = {req_id for req_id in timed_out_ids if req_id in self.requests}
        if not present_ids:
            return
        logger.warning(
            "Marking %d request(s) as FINISHED_ERROR after waiting > %.0fs for connector input: %s",
            len(present_ids),
            DEFAULT_INPUT_WAIT_TIMEOUT_S,
            sorted(present_ids),
        )
        self.finish_requests(present_ids, RequestStatus.FINISHED_ERROR)

    def _log_failed_chunk_sends(self) -> None:
        """Surface chunks the sender gave up on (R1.2 of #4855).

        ``save_loop`` pops a task with ``popleft()`` and never re-queues it, and
        ``connector.put`` can report failure without raising, so a dropped chunk
        is a give-up rather than a retry. Until this change the only trace was a
        warning that printed ``None`` for the request id, because the task dict
        has no ``request_id`` key.

        This only reports. Failing the request from here does not work: the
        producer is not the ``final_output`` stage, so an abort synthesized by
        ``finish_requests`` is not turned into a client-visible termination by
        ``Orchestrator._route_output`` (see the ``final_output`` guards there).
        Making a producer-side drop client-visible needs an orchestrator-side
        path like ``_fail_request_dead_stage``; the consumer's R1.1 deadline is
        what actually ends the request today.
        """
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None:
            return
        failures = adapter.collect_failed_send_request_ids()
        for request_id, reason in failures.items():
            if request_id in self.requests:
                logger.error(
                    "[OmniScheduler] req=%s: chunk send gave up (%s); the consumer will "
                    "fail on the stage-input deadline",
                    request_id,
                    reason,
                )
            else:
                # Drained but no longer scheduled here: the request finished or
                # was aborted between the failed send and this sweep. Still say
                # so -- collect_* empties the map, so staying quiet would make
                # this the one drop path in a change whose point is visibility.
                logger.warning(
                    "[OmniScheduler] req=%s: chunk send gave up (%s) after the request "
                    "left this scheduler (finished or aborted)",
                    request_id,
                    reason,
                )

    def _process_pending_chunk_timeouts(self) -> None:
        """Force-fail requests stalled in ``WAITING_FOR_CHUNK`` too long.

        The async-chunk counterpart of ``_process_pending_input_timeouts``,
        called right after ``chunk_transfer_adapter.process_pending_chunks``.
        Until now the chunk path had no safety net at all: the scheduler-side
        comment on the full-payload net said a similar guard "belongs in the
        chunk adapter", while the worker mixin assumed the full-payload net
        already covered it.  Only the first was true, so a dropped terminal
        chunk or a producer that exhausted its send retries parked the request
        forever (vllm-project/vllm-omni#3833).  Since ``async_chunk: true`` is
        the default for the flagship TTS pipelines, that was the default
        streaming path.

        Shares ``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S`` with the full-payload net:
        one knob for "how long may a request wait for stage input", regardless
        of which transport carries it.  Disabled when the value is <= 0.
        """
        if DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0:
            return
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None or not getattr(adapter, "receives_chunks", False):
            return
        timed_out_ids = adapter.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)
        if not timed_out_ids:
            return
        present_ids = {req_id for req_id in timed_out_ids if req_id in self.requests}
        if not present_ids:
            return
        logger.warning(
            "Marking %d request(s) as FINISHED_ERROR after stalling > %.0fs waiting for a chunk: %s",
            len(present_ids),
            DEFAULT_INPUT_WAIT_TIMEOUT_S,
            sorted(present_ids),
        )
        self.finish_requests(present_ids, RequestStatus.FINISHED_ERROR)

    def _capture_omni_connector_output(self, model_runner_output: Any) -> None:
        """Stash the model runner's omni_connector_output for next schedule().

        Called at the tail of every ``update_from_output()`` -- identical
        between AR and generation schedulers.  Only stashes the output;
        applying the metadata is the responsibility of
        ``_consume_pending_connector_output()`` at the start of the next
        ``schedule()`` cycle.  Applying it twice (once here, once on
        consume) is unsafe under ``update_request_metadata`` in
        generation mode, which resets ``prompt_token_ids`` /
        ``_output_token_ids`` / ``num_computed_tokens`` and would
        clobber any progress between the two calls.
        """
        omni_output = getattr(model_runner_output, "omni_connector_output", None)
        if omni_output is None:
            return
        self._latest_omni_connector_output = omni_output

    def _wrap_omni_scheduler_output(
        self,
        base: SchedulerOutput,
        *,
        finished_requests_needing_kv_transfer: dict | None = None,
        pending_input_registrations: list[OmniChunkRecvHandle] | None = None,
    ) -> OmniSchedulerOutput:
        """Wrap a base ``SchedulerOutput`` in ``OmniSchedulerOutput``.

        Pulls each base ``SchedulerOutput`` dataclass field via ``getattr``
        and forwards optional omni-specific fields.  Lifted from 4 separate
        copy-pastes between AR (1) and generation (3) schedulers.
        """
        base_data = {name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__}
        input_coordinator = getattr(self, "input_coordinator", None)
        if pending_input_registrations is None:
            pending_input_registrations = input_coordinator.pending_input_registrations if input_coordinator else []
        return OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_requests_needing_kv_transfer or {},
            pending_input_registrations=pending_input_registrations,
        )

    def _rewrap_scheduled_new_reqs(self, scheduler_output: SchedulerOutput) -> None:
        """Attach Omni payloads without reconstructing existing Omni entries."""
        scheduler_output.scheduled_new_reqs = [  # type: ignore[assignment]
            data
            if isinstance(data, OmniNewRequestData)
            else OmniNewRequestData.from_base(data, self.requests.get(data.req_id))
            for data in scheduler_output.scheduled_new_reqs
        ]

    def _postprocess_omni_schedule_output(
        self,
        scheduler_output: SchedulerOutput,
        *,
        include_cached_payloads: bool = False,
    ) -> None:
        """Enrich new requests and apply async-chunk output bookkeeping."""
        self._rewrap_scheduled_new_reqs(scheduler_output)
        if not self.chunk_transfer_adapter:
            return
        if include_cached_payloads:
            self.chunk_transfer_adapter.postprocess_scheduler_output(
                scheduler_output,
                self.requests,
            )
        else:
            self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output)

    def _make_omni_engine_output(
        self,
        request: Request,
        *,
        new_token_ids: list[int],
        finish_reason: Any = None,
        new_logprobs: Any = None,
        new_prompt_logprobs_tensors: Any = None,
        pooling_output: Any = None,
        multimodal_output: Any = None,
        stop_reason: Any = None,
        prefill_stats: Any = None,
        kv_transfer_params: Any = None,
        routed_experts: Any = None,
        num_nans_in_logits: int = 0,
        is_segment_finished: bool | None = False,
        new_prompt_len_snapshot: int | None = None,
    ) -> OmniEngineCoreOutput:
        """Build the common request-output envelope used by LLM schedulers."""
        return OmniEngineCoreOutput(
            request_id=request.request_id,
            new_token_ids=new_token_ids,
            finish_reason=finish_reason,
            new_logprobs=new_logprobs,
            new_prompt_logprobs_tensors=new_prompt_logprobs_tensors,
            pooling_output=pooling_output,
            multimodal_output=multimodal_output,
            stop_reason=stop_reason,
            events=request.take_events(),
            prefill_stats=prefill_stats,
            kv_transfer_params=kv_transfer_params,
            trace_headers=request.trace_headers,
            routed_experts=routed_experts,
            num_nans_in_logits=num_nans_in_logits,
            is_segment_finished=is_segment_finished,
            new_prompt_len_snapshot=new_prompt_len_snapshot,
        )

    def _append_request_output(
        self,
        outputs: dict[int, list[EngineCoreOutput]],
        request: Request,
        **output_fields: Any,
    ) -> None:
        outputs[request.client_index].append(
            OmniSchedulerMixin._make_omni_engine_output(
                self,
                request,
                **output_fields,
            )
        )

    def _handle_failed_kv_load_outputs(
        self,
        failed_request_ids: set[str] | None,
        outputs: dict[int, list[EngineCoreOutput]],
    ) -> list[Request]:
        """Finish unrecoverable KV loads and emit their terminal outputs."""
        if not failed_request_ids or self.recompute_kv_load_failures:
            return []
        requests = [self.requests[req_id] for req_id in failed_request_ids]
        self.finish_requests(failed_request_ids, RequestStatus.FINISHED_ERROR)
        for request in requests:
            OmniSchedulerMixin._append_request_output(
                self,
                outputs,
                request,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
            )
        return requests

    def _attach_finished_request_sets(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        *,
        synthesize_abort_outputs: bool,
    ) -> None:
        """Attach finished IDs while keeping AR's synthetic-abort policy explicit."""
        finished_req_ids = self.finished_req_ids_dict
        if not finished_req_ids:
            return
        for client_index, finished_set in finished_req_ids.items():
            output = engine_core_outputs.get(client_index)
            if output is None:
                output = EngineCoreOutputs()
                engine_core_outputs[client_index] = output
            if synthesize_abort_outputs:
                emitted = {item.request_id for item in output.outputs}
                output.outputs.extend(
                    EngineCoreOutput(req_id, [], finish_reason=FinishReason.ABORT)
                    for req_id in finished_set
                    if req_id not in emitted
                )
            output.finished_requests = finished_set
        finished_req_ids.clear()

    def _remove_stopped_requests_from_queues(
        self,
        stopped_running_reqs: set[Request],
        stopped_preempted_reqs: set[Request],
    ) -> None:
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

    def _aggregate_kv_connector_stats(
        self,
        kv_connector_output: Any,
    ) -> KVConnectorStats | None:
        stats = kv_connector_output.kv_connector_stats if kv_connector_output else None
        if self.connector:
            scheduler_stats = self.connector.get_kv_connector_stats()
            if scheduler_stats is not None and not scheduler_stats.is_empty():
                stats = stats.aggregate(scheduler_stats) if stats is not None else scheduler_stats
        return stats

    def _publish_kv_cache_events(self) -> None:
        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)
        if events:
            self.kv_event_publisher.publish(KVEventBatch(ts=time.time(), events=events))

    def _attach_scheduler_stats(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        spec_decoding_stats: SpecDecodingStats | None,
        kv_connector_stats: KVConnectorStats | None,
        cudagraph_stats: CUDAGraphStat | None,
        perf_stats: PerfStats | None,
    ) -> None:
        stats = self.make_stats(
            spec_decoding_stats,
            kv_connector_stats,
            cudagraph_stats,
            perf_stats,
        )
        if stats is None:
            return
        if (output := next(iter(engine_core_outputs.values()), None)) is None:
            engine_core_outputs[0] = output = EngineCoreOutputs()
        output.scheduler_stats = stats

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[Request]:
        """Finish requests and clean all Omni-owned queue/coordinator state."""
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        self._realign_request_status_to_queues(request_ids)
        finished = super().finish_requests(request_ids, finished_status)
        self._purge_finished_from_running()

        for request in finished:
            self._free_input_coordinator_request(request.request_id)
        return finished

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        now = time.monotonic()
        if now - getattr(self, "_last_stats_time", 0.0) < _STATS_INTERVAL_S:
            return None
        self._last_stats_time = now
        return super().make_stats(*args, **kwargs)

    def _realign_request_status_to_queues(
        self,
        request_ids: str | Iterable[str] | None,
    ) -> None:
        """Realign ``request.status`` to actual queue membership.

        ``OmniChunkTransferAdapter._process_chunk_queue`` stamps
        ``requests_origin_status[req.id] = WAITING`` (or ``RUNNING``) when
        first parking a request in a chunk-transfer deque. On the next
        tick, when the chunk arrives, ``_process_chunk_queue`` sets
        ``request.status = target_status`` and continues, but
        ``requests_origin_status`` is left at its first-park value -- no
        hook updates it on the ``waiting → running`` admit transition
        that ``super().schedule()`` later performs. The table stays
        stale until the request makes another deque round-trip.

        If an abort lands in the gap between admit and the next deque
        round-trip, ``chunk_transfer_adapter.finish_requests`` reads the
        stale ``WAITING`` from ``requests_origin_status``, stomps it
        onto ``request.status``, and the upstream
        ``Scheduler.finish_requests`` else branch silently fails to
        remove from ``self.running`` -- the request stays alive in
        ``self.running`` and the worker's ``input_batch`` slot leaks.
        After ``max_num_seqs`` such aborts every new request hangs at
        ``chunks=0`` until the client times out.

        Realign here: if a request lives in ``self.running`` but its
        status is not ``RUNNING``, set it to ``RUNNING``; symmetrically
        flip ``RUNNING → WAITING`` when the request is actually in
        ``self.waiting``. This is a localized safety net for
        ``requests_origin_status`` staleness on the admit transition;
        it does not touch the adapter's invariants and is complementary
        to the chunk-transfer-adapter deque purge that already runs
        inside ``process_pending_chunks`` / ``restore_queues``.

        Note on scope: only the ``async_chunk`` path actually triggers
        the ``requests_origin_status`` staleness this helper repairs.
        When ``async_chunk`` is disabled, no chunk-transfer round-trip
        occurs between admit and finish, so the realignment walk is a
        cheap O(n) no-op over an already-aligned set. The call is kept
        unconditional in ``finish_requests`` to (a) keep the abort path
        uniform and (b) defend any future configuration that re-enables
        chunk transfer from rediscovering the same regression.

        See https://github.com/vllm-project/vllm-omni/pull/3774 and the
        residual-hang reproduction discussed in that PR.
        """
        # Mirror the upstream Scheduler.finish_requests resolution of
        # ``request_ids`` so realignment touches exactly the set that
        # ``super().finish_requests`` will then walk.
        if isinstance(request_ids, str):
            ids_to_align: Iterable[str] = (request_ids,)
        elif request_ids is None:
            ids_to_align = list(self.requests.keys())
        else:
            ids_to_align = list(request_ids)

        if not ids_to_align:
            return

        running_ids = {r.request_id for r in self.running}
        waiting_ids = {r.request_id for r in self.waiting}

        for rid in ids_to_align:
            req = self.requests.get(rid)
            if req is None or req.is_finished():
                continue
            if rid in running_ids and req.status != RequestStatus.RUNNING:
                req.status = RequestStatus.RUNNING
            elif rid in waiting_ids and req.status == RequestStatus.RUNNING:
                req.status = RequestStatus.WAITING

    def _purge_finished_from_running(self) -> None:
        """Defensive post-finish sweep of ``self.running``.

        Belt-and-suspenders to ``_realign_request_status_to_queues``:
        even after status realignment lets upstream
        ``Scheduler.finish_requests`` pick the right removal branch,
        a future regression or an unexpected ``status`` mid-transition
        could still leave already-finished entries in ``self.running``.
        Sweeping here guarantees the worker's ``input_batch`` slot is
        not pinned by a freed request.

        Complementary to ``_realign_request_status_to_queues``: realign
        is preventive (fix ``status`` before ``super().finish_requests``
        so the right branch fires); this purge is defensive (sweep the
        residue after ``super().finish_requests`` so any stale entries
        are reclaimed).

        Scope of the predicate. ``is_finished()`` covers entries the
        upstream ``finish_requests`` already drained from ``self.requests``
        but failed to remove from ``self.running``; the
        ``request_id not in self.requests`` arm catches the same surface
        from a different angle and is the post-cleanup mirror of the
        deque purge ``_purge_untracked_chunk_requests`` already runs at
        the chunk-transfer-adapter layer. It does **not** by itself make
        arbitrary direct deletions of ``self.requests`` safe -- callers
        that pop ``self.requests`` outside the standard finish path
        still have to go through ``_free_request`` (or equivalent) for
        block / connector / coordinator cleanup. This sweep only
        reclaims the ``self.running`` slot reference.

        In-place via ``self.running[:] = ...`` for minor consistency
        with idiomatic vLLM scheduler mutation; upstream
        ``Scheduler.finish_requests`` itself rebinds ``self.running``,
        so list identity across the whole call is not preserved -- the
        slice form is just to avoid an extra rebind inside this helper.

        Assumes the upstream V1 invariant that scheduler ticks are
        serialized on a single thread; in-place mutation here is no more
        racy than the rest of the scheduler under that assumption.

        See https://github.com/vllm-project/vllm-omni/pull/3774
        discussion.
        """
        if not self.running:
            return
        self.running[:] = [req for req in self.running if not req.is_finished() and req.request_id in self.requests]
