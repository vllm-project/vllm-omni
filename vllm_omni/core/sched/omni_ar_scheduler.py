from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as AsyncVLLMScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.utils import omni_routed_experts_for_request
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)


class SampledLogprobContractError(RuntimeError):
    """The model runner returned unusable sampled-token logprobs."""


def _slice_sampled_logprobs(logprobs: Any, req_index: int, sampled_token_ids: list[int]) -> Any:
    """Slice and validate the sampled-token logprobs for one AR request."""
    if logprobs is None:
        raise SampledLogprobContractError("AR logprobs were requested, but the model runner returned none")

    sliced = logprobs.slice_request(req_index, len(sampled_token_ids))
    token_rows = np.asarray(sliced.logprob_token_ids)
    value_rows = np.asarray(sliced.logprobs)
    expected_rows = len(sampled_token_ids)

    if token_rows.ndim != 2 or value_rows.ndim != 2:
        raise SampledLogprobContractError(
            "AR sampled-token logprobs must be rank-2 arrays, "
            f"got token_ids={token_rows.shape} logprobs={value_rows.shape}"
        )
    if token_rows.shape[0] != expected_rows or value_rows.shape[0] != expected_rows:
        raise SampledLogprobContractError(
            "AR sampled-token logprob row count does not match generated tokens: "
            f"tokens={expected_rows} token_id_rows={token_rows.shape[0]} "
            f"logprob_rows={value_rows.shape[0]}"
        )
    if expected_rows == 0:
        return sliced
    if token_rows.shape[1] == 0 or value_rows.shape[1] == 0:
        raise SampledLogprobContractError("AR sampled-token logprob rows are empty")

    sampled = np.asarray(sampled_token_ids)
    if not np.array_equal(token_rows[:, 0], sampled):
        mismatch = np.flatnonzero(token_rows[:, 0] != sampled)
        first = int(mismatch[0])
        raise SampledLogprobContractError(
            "AR sampled-token logprobs are misaligned: "
            f"row={first} generated_token={int(sampled[first])} "
            f"logprob_token={int(token_rows[first, 0])}"
        )
    if not np.isfinite(value_rows[:, 0]).all():
        bad_rows = np.flatnonzero(~np.isfinite(value_rows[:, 0])).tolist()
        raise SampledLogprobContractError(f"AR sampled-token logprobs contain non-finite values at rows {bad_rows}")
    return sliced


class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    """Synchronous AutoRegressive scheduler for vLLM-Omni. This class is also
    used as a base class for the OmniARAsyncScheduler and holds most of the
    core scheduling logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # Requests marked for deferred stop: keep running until KV extraction
        # completes so that kv_ready can be emitted while the request is still
        # alive.  Stopped on the first scheduler step after extraction ack.
        self.pending_stop_after_extraction: set[str] = set()

        self.finished_req_ids_dict = defaultdict(set)

        # [Omni] Pre-parse KV transfer criteria
        self._omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()

        # Cache per-request flag to avoid repeated deserialization of additional_information
        self._omits_kv_transfer_cache: dict[str, bool] = {}
        self._init_omni_io_scheduling_state()
        # Snapshot prompt length for each streaming input update
        self._new_prompt_len_snapshot: dict[str, int] = {}

    def _get_confirmed_num_computed_tokens(self, request: Request) -> int:
        """num_computed_tokens minus async placeholders (KV actually on GPU)."""
        # Output placeholders are zero when async scheduling isn't used
        return request.num_computed_tokens - request.num_output_placeholders

    def _get_kv_transfer_criteria(self) -> dict | None:
        return self._get_omni_kv_config_value("kv_transfer_criteria")

    def _get_omni_kv_config_value(self, key: str, default: Any = None) -> Any:
        config = getattr(self, "_omni_kv_config", None)
        if config is None and hasattr(self, "vllm_config"):
            config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default) if config is not None else default

    def _request_omits_kv_transfer_to_next_stage(self, request: Request) -> bool:
        """True when this stage-zero-final request does not need downstream KV.

        The result is cached per request to avoid repeated deserialization of
        additional_information on every scheduler tick.
        """
        rid = request.request_id
        cached = self._omits_kv_transfer_cache.get(rid)
        if cached is not None:
            return cached

        payload = getattr(request, "additional_information", None)
        if payload is None:
            result = False
        else:
            info = deserialize_additional_information(payload)
            result = info.get("omni_final_stage_id") == 0 and not bool(info.get("omni_force_kv_transfer", False))

        self._omits_kv_transfer_cache[rid] = result
        return result

    def _should_defer_waiting_admission(self) -> bool:
        return False

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        # Text-only requests finalize at stage 0; do not prefill-stop for DiT KV.
        if self._request_omits_kv_transfer_to_next_stage(request):
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")
        stop_decode_on_trigger = self.kv_transfer_criteria.get("stop_after_transfer", True)

        if request.request_id in self.transfer_triggered_requests:
            # Deferred stop: once KV extraction is complete (no longer in
            # active_kv_transfers), stop the request.  This guarantees the
            # kv_ready signal was emitted while the request was still alive.
            if (
                request.request_id in self.pending_stop_after_extraction
                and request.request_id not in self.active_kv_transfers
            ):
                self.pending_stop_after_extraction.discard(request.request_id)
                request.status = RequestStatus.FINISHED_STOPPED
                return True
            return False

        # seq_len for KV transfer must exclude async placeholders.
        confirmed_computed = self._get_confirmed_num_computed_tokens(request)

        if criteria_type == "prefill_finished":
            if confirmed_computed >= request.num_prompt_tokens:
                self._commit_kv_transfer_trigger(
                    request.request_id,
                    confirmed_computed,
                    stop_decode_on_trigger,
                )
                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                idx = new_token_ids.index(target_token_id)
                tokens_to_exclude = len(new_token_ids) - (idx + 1)
                snapshot_len = confirmed_computed - tokens_to_exclude
                self._commit_kv_transfer_trigger(
                    request.request_id,
                    snapshot_len,
                    stop_decode_on_trigger,
                )
                return False

        return False

    def _commit_kv_transfer_trigger(
        self,
        req_id: str,
        seq_len: int,
        stop_after_transfer: bool,
    ) -> None:
        self.transfer_triggered_requests.add(req_id)
        self._mark_request_for_kv_transfer(req_id, seq_len)
        if stop_after_transfer and req_id in self.requests_needing_kv_transfer:
            self.pending_stop_after_extraction.add(req_id)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # Remove FINISHED_ABORTED requests before the upstream scheduler sees
        # them. Upstream vllm raises RuntimeError on this status; omni allows
        # async abort (e.g. client disconnect during TTS streaming) to leave
        # requests in the waiting/running queues temporarily.
        for queue in (self.waiting, self.running):
            for req in list(queue):
                if getattr(req, "status", None) == RequestStatus.FINISHED_ABORTED:
                    queue.remove(req)
        self._process_pending_omni_inputs(model_mode="ar")

        original_waiting = None
        if self._should_defer_waiting_admission():
            original_waiting = self.waiting
            self.waiting = create_request_queue(self.policy)

        try:
            scheduler_output = super().schedule(throttle_prefills)
        finally:
            if original_waiting is not None:
                deferred_waiting = list(self.waiting)
                if deferred_waiting:
                    original_waiting.prepend_requests(deferred_waiting)
                self.waiting = original_waiting
            self._restore_omni_wait_queues()

        self._postprocess_omni_schedule_output(
            scheduler_output,
            include_cached_payloads=True,
        )
        finished_reqs = self.get_finished_requests_needing_kv_transfer()

        # Wrap in omni scheduler output to carry transfer metadata.
        return self._wrap_omni_scheduler_output(
            scheduler_output,
            finished_requests_needing_kv_transfer=finished_reqs,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        mm_outputs = getattr(model_runner_output, "multimodal_outputs", None)
        inter_stage_outputs = getattr(model_runner_output, "inter_stage_outputs", None)
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Pre-process KV extraction acks so that the per-request loop below
        # can see up-to-date active_kv_transfers state and emit kv_ready
        # signals while requests are still alive (before any deferred stop).
        kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    self.active_kv_transfers.discard(req_id)
                    req = self.requests.get(req_id)
                    if req is not None and not req.is_finished():
                        outputs[req.client_index].append(
                            OmniEngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=[],
                                kv_transfer_params={"kv_ready": True},
                            )
                        )
                except Exception:
                    init_logger(__name__).exception("Failed to pre-process KV extraction for %s", req_id)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is not None:
                # vLLM 0.26: settle the in-flight tokens counted in schedule().
                # Must happen before the skips below — failed-KV-load and
                # already-finished requests were incremented too, and the two
                # readers (allocate_slots, _connector_finished) clamp with
                # max(0, computed - in_flight), so a leaked counter silently
                # freezes sliding-window block freeing.
                request.num_in_flight_tokens -= num_tokens_scheduled
            # vLLM 0.27 (a0c092ee72) removed the async_tokens_to_discard
            # handling from the upstream scheduler and replaced it with the
            # num_stale_output_tokens/is_stale mechanism. Omni's discard
            # sites (segment stop, streaming-session replacement) record the
            # in-flight share here; the delayed outputs are dropped below
            # instead of decrementing num_output_placeholders (which the
            # discard zeroed) and underflowing the upstream assert.
            output_is_stale = False
            if request is not None and request.num_stale_output_tokens > 0:
                output_is_stale = True
                request.num_stale_output_tokens -= num_tokens_scheduled
                assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or async scheduling).
                continue
            if output_is_stale:
                # Output of a step scheduled before the request's in-flight
                # tokens were discarded (segment stop / session replacement).
                # num_computed_tokens was rolled back at the discard site, so
                # this output must not be appended or emitted.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
            status_before_stop = request.status
            new_logprobs = None
            logprob_validation_failed = False

            # Validate before mutating request token state. A bad runner output
            # is request-local: terminate only this request and keep processing
            # the rest of the batch.
            if (
                generated_token_ids
                and request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
            ):
                try:
                    new_logprobs = _slice_sampled_logprobs(logprobs, req_index, generated_token_ids)
                except SampledLogprobContractError as exc:
                    logger.error("Invalid AR sampled-token logprobs for request %s: %s", req_id, exc)
                    request.status = RequestStatus.FINISHED_ERROR
                    request.stop_reason = str(exc)
                    request.resumable = False
                    generated_token_ids = []
                    logprob_validation_failed = True

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = logprob_validation_failed
            is_segment_finished = False
            finished = False
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            mm_output = mm_outputs[req_index] if mm_outputs else None
            inter_stage_output = inter_stage_outputs[req_index] if inter_stage_outputs else None
            kv_transfer_params = None
            finish_reason = None
            routed_experts = None

            # Check for stop and update request status.
            if new_token_ids:
                num_sampled_tokens = len(new_token_ids)
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
                if new_logprobs is not None and len(new_token_ids) < num_sampled_tokens:
                    # A mid-step stop (e.g. spec-decode tokens sampled past
                    # EOS) trims new_token_ids after the validation slice
                    # above; re-slice so the emitted logprob rows stay 1:1
                    # with the emitted tokens, as upstream vLLM does by
                    # slicing after the trim.
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # If criteria returns True, it means we must STOP the request.
            # If criteria returns False, it might have triggered a background
            # transfer (e.g. prefill finished / special token) but continues decoding.
            if not stopped and self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            if stopped:
                if model_runner_output.routed_experts is not None:
                    routed_experts = omni_routed_experts_for_request(model_runner_output.routed_experts, request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                is_segment_finished = not finished
                if finished:
                    request.resumable = False
                if not finished:
                    # for streaming input request only
                    if self.chunk_transfer_adapter:
                        if self.vllm_config.model_config.stage_id != 0:
                            # Downstream async-chunk stages receive real payloads from the
                            # connector. This update only resumes polling for the next segment.
                            self.chunk_transfer_adapter.segment_finished_requests.discard(request.request_id)
                    outstanding_async_tokens = request.num_output_placeholders
                    # Always record the discard signal (0 when nothing is in
                    # flight). Upstream a0c092ee72 removed the
                    # `async_tokens_to_discard` default from `Request`; it
                    # remains an omni-only signal set here on every segment
                    # stop, so a stop with no outstanding placeholders
                    # explicitly records 0.
                    request.async_tokens_to_discard = outstanding_async_tokens
                    # Seed the stale share in SCHEDULED-token units:
                    # num_in_flight_tokens is exactly the unreported steps'
                    # num_tokens_scheduled sum (settled per frame at the top
                    # of this loop), and the drain subtracts each arriving
                    # frame's num_tokens_scheduled — commensurable by
                    # construction, so pre-discard frames drain to exactly
                    # zero. Seeding from num_output_placeholders swallowed
                    # valid new-segment frames or underflowed the drain
                    # assert whenever placeholder counts diverged from
                    # scheduled counts (spec drafts, in-flight prefill).
                    if request.num_in_flight_tokens > 0:
                        request.num_stale_output_tokens += request.num_in_flight_tokens
                    if outstanding_async_tokens > 0:
                        # Discard only outputs that are already in flight and
                        # roll back their optimistic computed-token accounting.
                        request.num_computed_tokens -= outstanding_async_tokens
                        request.num_output_placeholders = 0
                    request.spec_token_ids = []
                    request._output_token_ids.clear()
                if finished:
                    kv_transfer_params, _ = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or mm_output is not None or pooler_output is not None or kv_transfer_params or stopped:
                OmniSchedulerMixin._append_request_output(
                    self,
                    outputs,
                    request,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=new_logprobs,
                    new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                    pooling_output=pooler_output,
                    multimodal_output=mm_output,
                    stop_reason=request.stop_reason,
                    prefill_stats=request.take_prefill_stats(),
                    kv_transfer_params=kv_transfer_params,
                    routed_experts=routed_experts,
                    num_nans_in_logits=request.num_nans_in_logits,
                    is_segment_finished=is_segment_finished,
                    new_prompt_len_snapshot=self._new_prompt_len_snapshot.get(req_id),
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if self.chunk_transfer_adapter is not None and (
                inter_stage_output is not None or is_segment_finished or finished
            ):
                self.chunk_transfer_adapter.save_async(
                    inter_stage_output,
                    request,
                    is_segment_finished,
                )

        self._remove_stopped_requests_from_queues(
            stopped_running_reqs,
            stopped_preempted_reqs,
        )

        failed_requests = self._handle_failed_kv_load_outputs(
            failed_kv_load_req_ids,
            outputs,
        )
        if self.chunk_transfer_adapter is not None:
            for request in failed_requests:
                self.chunk_transfer_adapter.cleanup_receiver(request.request_id)

        self._cleanup_kv_tracking(req.request_id for req in stopped_running_reqs | stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        kv_connector_stats = self._aggregate_kv_connector_stats(kv_connector_output)
        self._publish_kv_cache_events()

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        self._attach_finished_request_sets(
            engine_core_outputs,
            synthesize_abort_outputs=True,
        )

        self._attach_scheduler_stats(
            engine_core_outputs,
            spec_decoding_stats,
            kv_connector_stats,
            cudagraph_stats,
            perf_stats,
        )

        self._capture_omni_connector_output(model_runner_output)

        # Free blocks that were held for transfer (kv_ready and
        # active_kv_transfers updates already done before the per-request loop).
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    if req_id in self.waiting_for_transfer_free:
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            if req_id in self.transfer_triggered_requests:
                                self.transfer_triggered_requests.remove(req_id)
                            self.active_kv_transfers.discard(req_id)
                            self.pending_stop_after_extraction.discard(req_id)
                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
                except Exception:
                    init_logger(__name__).exception("Failed to free blocks for %s after transfer", req_id)

        return engine_core_outputs

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Only extend prompt at stage 0, and replace
        the existing session with the next streaming update at other stages.

        Discards the last sampled output token from the prior input chunk at stage 0.
        """
        req_id = session.request_id
        self._new_prompt_len_snapshot[req_id] = len(update.prompt_token_ids)
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        # Seed the stale share in SCHEDULED-token units (see the segment-stop
        # site in update_from_output): num_in_flight_tokens matches what each
        # pre-replacement frame will drain, so the counter reaches exactly
        # zero and the new segment's frames are never swallowed. This also
        # covers an in-flight prefill chunk, which carries no placeholders
        # but must still have its late output dropped.
        in_flight_tokens = int(getattr(session, "num_in_flight_tokens", 0) or 0)
        if in_flight_tokens > 0:
            session.num_stale_output_tokens += in_flight_tokens
        if outstanding_async_tokens > 0:
            # Async scheduling may already have sampled the previous
            # segment's next token. Drop that late token instead of
            # appending it to the new streaming segment.
            session.async_tokens_to_discard = 1
            session.num_computed_tokens -= session.num_output_placeholders
            session.num_output_placeholders = 0
            session.spec_token_ids = []
        stage_id = self.vllm_config.model_config.stage_id
        if self.chunk_transfer_adapter and self.chunk_transfer_adapter.receives_chunks:
            self.chunk_transfer_adapter.requests_num_chunks_sent.pop(session.external_req_id, None)
            if stage_id != 0:
                # Downstream async-chunk stages receive real payloads from the
                # connector. This update only resumes polling for the next segment.
                self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)
                # Do not replace prompt/additional_information here; the next
                # upstream chunk will populate them in chunk transfer adapter.
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
                return
        update_infos = (
            getattr(update, "model_intermediate_buffer", None),
            getattr(update, "additional_information", None),
        )
        replace_streaming_prompt = any(
            isinstance(info, dict)
            and isinstance(info.get("meta"), dict)
            and info["meta"].get("replace_streaming_prompt") is True
            for info in update_infos
        )
        if replace_streaming_prompt:
            self._replace_streaming_session(session, update)
            return
        super()._update_request_as_session(session, update)
        if hasattr(update, "model_intermediate_buffer"):
            session.model_intermediate_buffer = update.model_intermediate_buffer

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()

        self._omits_kv_transfer_cache.pop(request.request_id, None)

        # [Upstream compat] Discard request from in-flight prefills set added
        # upstream for routed-experts in-flight reservation tracking.
        # Use getattr for safety with test __new__ code paths.
        getattr(self, "_inflight_prefills", set()).discard(request)

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self._new_prompt_len_snapshot.pop(request_id, None)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # Mirror the generation scheduler's try/finally pattern so the
        # input_coordinator entry is always pruned along every return path,
        # including the early returns for in-flight / waiting KV transfers
        # below. _free_input_coordinator_request is a no-op when the
        # coordinator is None, so the unconditional finally is safe.
        try:
            # 2. Omni Specific: Check if we need to transfer KV
            if self._should_transfer_kv_for_request(request_id):
                already_triggered = request_id in self.transfer_triggered_requests
                is_active = request_id in self.active_kv_transfers

                if already_triggered:
                    if is_active:
                        # It triggered but hasn't finished yet. We MUST wait.
                        logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                        self.waiting_for_transfer_free.add(request_id)
                        kv_xfer_params = None
                        return kv_xfer_params, None
                    elif request_id in self.waiting_for_transfer_free:
                        # Blocks held until KV extraction completes in a future step.
                        return None, None
                    else:
                        logger.debug(
                            f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                            "Freeing immediately."
                        )
                else:
                    self.waiting_for_transfer_free.add(request_id)
                    confirmed_computed = self._get_confirmed_num_computed_tokens(request)
                    self._mark_request_for_kv_transfer(request_id, confirmed_computed)
                    # Return KV transfer metadata so it propagates to RequestOutput
                    if request_id in self.requests_needing_kv_transfer:
                        transfer_data = self.requests_needing_kv_transfer[request_id]
                        kv_xfer_params = {
                            "past_key_values": transfer_data["block_ids"],
                            "kv_metadata": {
                                "seq_len": transfer_data["seq_len"],
                                "block_ids": transfer_data["block_ids"],
                            },
                        }
                        # Also update request.additional_information for good measure
                        add_info = getattr(request, "additional_information", None)
                        # If additional_information is an AdditionalInformationPayload-like object,
                        # unpack it into a plain dict.
                        if (
                            add_info is not None
                            and hasattr(add_info, "entries")
                            and isinstance(getattr(add_info, "entries"), dict)
                        ):
                            request.additional_information = deserialize_additional_information(add_info)
                            add_info = request.additional_information
                        if add_info is None:
                            request.additional_information = {}
                            add_info = request.additional_information
                        if isinstance(add_info, dict):
                            add_info.update(kv_xfer_params)

                    return kv_xfer_params, None

            # 3. Standard Freeing
            delay_free_blocks |= connector_delay_free_blocks
            if not delay_free_blocks:
                self._free_blocks(request)

            return kv_xfer_params, None
        finally:
            self._free_input_coordinator_request(request_id)
            # Normal completion runs through here, not finish_requests()
            # (the abort path) -- see vllm-project/vllm-omni#5349.
            if self.chunk_transfer_adapter is not None:
                self.chunk_transfer_adapter.cleanup_receiver(request_id)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        if not self._get_omni_kv_config_value("need_send_cache", False):
            return False
        request = self.requests.get(req_id)
        if request is not None and self._request_omits_kv_transfer_to_next_stage(request):
            return False
        return True

    def _cleanup_kv_tracking(self, request_ids: Iterable[str]) -> None:
        for req_id in request_ids:
            if req_id in self.waiting_for_transfer_free:
                continue
            self.transfer_triggered_requests.discard(req_id)
            self.active_kv_transfers.discard(req_id)
            self.pending_stop_after_extraction.discard(req_id)

    def _has_pending_kv_work(self) -> bool:
        return bool(self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free)

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        return self._has_pending_kv_work() or super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        return self._has_pending_kv_work() or super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        return self._has_pending_kv_work() or super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests


class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    """Asynchronous AutoRegressive scheduler."""
