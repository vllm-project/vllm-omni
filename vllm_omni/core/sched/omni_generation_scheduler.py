from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
)
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData
from vllm_omni.core.sched.utils import omni_routed_experts_for_request
from vllm_omni.outputs import OmniModelRunnerOutput

logger = init_logger(__name__)


class OmniGenerationScheduler(OmniSchedulerMixin, VLLMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.vllm_config.model_config
        self._init_omni_io_scheduling_state()
        self._retains_state_across_chunks = bool(getattr(model_config, "retains_state_across_chunks", False))
        self._pending_finish_reqs: list[Request] = []

    @staticmethod
    def _record_prefill_stats(request: Request) -> None:
        """Mirror upstream first-prefill prompt accounting for generation stages.

        Async-chunk generation stages can grow ``prompt_token_ids`` after the
        first schedule; until this path tracks final chunk lengths, this records
        the first scheduled prompt snapshot. Keep this in sync with vLLM
        Scheduler.schedule() if this fast path gains prefix-cache or external-KV
        accounting.
        """
        if request.prefill_stats is None:
            return
        request.prefill_stats.set(
            num_prompt_tokens=request.num_prompt_tokens,
            num_local_cached_tokens=0,
            num_external_cached_tokens=0,
        )

    def _handle_stopped_request(self, request: Request) -> bool:
        if (
            request.resumable
            and not request.streaming_queue
            and self.chunk_transfer_adapter is not None
            and self.chunk_transfer_adapter.receives_chunks
        ):
            # Downstream async-chunk stages receive the next segment from the
            # connector, not from an API StreamingUpdate. Enqueue them as
            # schedulable before the base class can park them in skipped_waiting.
            request.status = RequestStatus.WAITING
            self._enqueue_waiting_request(request)
            return False
        return super()._handle_stopped_request(request)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """One-shot generation fast path:
        - Feed all input tokens of the request at once
          (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the
          default vLLM scheduling.
        """

        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running_reqs: list[Request] = []
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}
        cached_additional_information: dict[str, dict | None] = {}

        # Temporary queue: preserve waiting order while requests await input.
        skipped_waiting_requests = create_request_queue(self.policy)
        req_index = 0
        self._process_pending_omni_inputs(model_mode="generation")

        # OMNI: Track requests that are already finished (e.g., marked by connector)
        # These should be removed from running and not scheduled
        already_finished_reqs: set[Request] = set()
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                already_finished_reqs.add(request)
                req_index += 1
                continue

            num_computed_tokens = request.num_computed_tokens
            required_tokens = len(request.prompt_token_ids) - num_computed_tokens
            if not self.scheduler_config.enable_chunked_prefill and required_tokens > token_budget:
                # If chunked_prefill is disabled,
                # we can stop the scheduling here.
                break
            # async_chunk: don't schedule placeholder tokens when no new chunk is available.
            if required_tokens <= 0:
                if self.chunk_transfer_adapter is not None and self.chunk_transfer_adapter.is_done_receiving_chunks(
                    request.request_id
                ):
                    self._pending_finish_reqs.append(request)
                req_index += 1
                continue
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if num_computed_tokens == 0:
                self._record_prefill_stats(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            cached_additional_information[request.request_id] = getattr(request, "additional_information", None)
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_index += 1

        # OMNI: Remove already finished requests from running queue
        if already_finished_reqs:
            self.running = remove_all(self.running, already_finished_reqs)

        # Fast path selection and scheduling for one-shot generation requests,
        # independent of pooling_params.
        while self.waiting and token_budget > 0 and self._pause_state == PauseState.UNPAUSED:
            # Requests waiting for their next chunk are temporarily absent
            # from `running`, but stateful models still retain their model
            # runner slot. Mirror vLLM's treatment of
            # `num_waiting_for_streaming_input` when enforcing max_num_seqs.
            num_running = len(self.running)
            if self._retains_state_across_chunks and self.chunk_transfer_adapter is not None:
                num_running += self.chunk_transfer_adapter.num_running_waiting_for_chunk
            if num_running >= self.max_num_running_reqs:
                break

            request = self.waiting.peek_request()
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                # Pop the finished request from waiting queue and don't schedule it
                self.waiting.pop_request()
                continue

            # async_chunk: wait for the first upstream chunk (don't start with placeholders).
            if self.chunk_transfer_adapter is not None and len(request.prompt_token_ids) == 0:
                if self.chunk_transfer_adapter.is_done_receiving_chunks(request.request_id):
                    self.waiting.pop_request()
                    self._pending_finish_reqs.append(request)
                    continue
                else:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

            # Allocate all input tokens for the request in one shot
            # (allocate 1 placeholder if zero)
            required_tokens = max(len(request.prompt_token_ids), 1)
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break

            # Officially schedule this request
            request = self.waiting.pop_request()
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if request.num_computed_tokens == 0:
                self._record_prefill_stats(request)

            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            scheduled_new_reqs.append(request)

        # Return skipped waiting requests
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # If fast path scheduled none, fall back to the original scheduling
        if not num_scheduled_tokens:
            if self.chunk_transfer_adapter:
                # Don't fall back: base scheduler doesn't handle async_chunk
                # requests with empty prompt_token_ids.
                self._restore_omni_wait_queues()
            else:
                res = super().schedule(throttle_prefills)
                self._restore_omni_wait_queues()
                self._postprocess_omni_schedule_output(res)
                return self._wrap_omni_scheduler_output(res)

        # Compute common prefix blocks (aligned with v1)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request.request_id)

        # Assemble SchedulerOutput (align with v0.14.0)
        if self.use_v2_model_runner:
            # No resumed reqs in fast path; pass prefill_token_ids for new reqs.
            new_reqs_data = [
                OmniNewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    getattr(req, "_all_token_ids", None),
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                OmniNewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]
        # No running/resumed reqs scheduled in our fast path
        cached_reqs_data = self._make_cached_request_data(
            running_reqs=scheduled_running_reqs,
            resumed_reqs=[],
            num_scheduled_tokens=num_scheduled_tokens,
            spec_decode_tokens=scheduled_spec_decode_tokens,
            req_to_new_blocks=req_to_new_blocks,
        )

        cached_reqs_data = OmniCachedRequestData(
            req_ids=cached_reqs_data.req_ids,
            resumed_req_ids=cached_reqs_data.resumed_req_ids,
            new_token_ids=cached_reqs_data.new_token_ids,
            all_token_ids=cached_reqs_data.all_token_ids,
            new_block_ids=cached_reqs_data.new_block_ids,
            num_computed_tokens=cached_reqs_data.num_computed_tokens,
            num_output_tokens=cached_reqs_data.num_output_tokens,
            prompt_token_ids=cached_prompt_token_ids,
            additional_information=cached_additional_information,
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        # Record the request ids scheduled in this step (v0.14.0 behavior).
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # KVTransfer: package metadata
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        # EC Connector: package metadata
        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Update internal state (advance num_computed_tokens, free encoder inputs,
        # etc.)
        self._update_after_schedule(scheduler_output)

        try:
            self._postprocess_omni_schedule_output(scheduler_output)
        finally:
            self._restore_omni_wait_queues()

        return self._wrap_omni_scheduler_output(scheduler_output)

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.input_coordinator is None:
            return super()._free_request(request, delay_free_blocks)

        try:
            return super()._free_request(request, delay_free_blocks)
        finally:
            self._free_input_coordinator_request(request.request_id)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update scheduler state and finish completed one-shot work."""
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        mm_outputs = getattr(model_runner_output, "multimodal_outputs", None)
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and getattr(kv_connector_output, "invalid_block_ids", None):
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

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
                # Request may already be finished (e.g., aborted during
                # execution / pipeline parallelism / async scheduling).
                continue
            if output_is_stale:
                # Output of a step scheduled before the request's in-flight
                # tokens were discarded (segment stop / session replacement).
                # num_computed_tokens was rolled back at the discard site, so
                # this output must not be appended or emitted.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

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

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            mm_output = mm_outputs[req_index] if mm_outputs else None
            status_before_stop = request.status
            finish_reason = None
            is_segment_finished = False
            routed_experts = None

            # Decode the pooling output before stop handling so a decoder
            # failure finishes the request with FinishReason.ERROR (500).
            try:
                pooling_output_payload = self._maybe_decode_pooling_output(request, pooler_output)
            except Exception as exc:
                logger.exception("[pooling] decoder hook failed for request %s", req_id)
                pooling_output_payload = None
                request.status = RequestStatus.FINISHED_ERROR
                request.stop_reason = f"pooling output decode failed: {exc}"
                request.resumable = False

            # One-shot generation request: finish after its current input unit
            # has been fully processed.
            if (
                request.status == RequestStatus.FINISHED_ERROR
                or request.status == RequestStatus.FINISHED_STOPPED
                or (self.chunk_transfer_adapter is None and request.num_computed_tokens >= request.num_prompt_tokens)
                or (
                    self.chunk_transfer_adapter is not None
                    and self.chunk_transfer_adapter.is_done_receiving_chunks(request.request_id)
                    and request.num_computed_tokens >= len(request.prompt_token_ids)
                )
            ):
                if request.status != RequestStatus.FINISHED_ERROR:
                    request.status = RequestStatus.FINISHED_STOPPED
                # Optional: set a stop_reason for front-end clarity
                # (does not affect protocol)
                stopped = True

            if stopped:
                if model_runner_output.routed_experts is not None:
                    routed_experts = omni_routed_experts_for_request(model_runner_output.routed_experts, request)
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                is_segment_finished = not finished
                if not finished:
                    # for streaming input request only
                    if self.chunk_transfer_adapter:
                        self.chunk_transfer_adapter.segment_finished_requests.discard(req_id)
                if finished:
                    kv_transfer_params, _ = self._free_request(request)
                    if self.chunk_transfer_adapter is not None:
                        self.chunk_transfer_adapter.cleanup(
                            request.request_id,
                            getattr(request, "external_req_id", None),
                        )
                if status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_running_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.num_logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                # NOTE: structured_output_request should not be None if
                # use_structured_output, we have check above, so safe to ignore
                # type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]  # noqa: E501
                    req_id, new_token_ids
                )

            # spec_token_ids comes from the model runner output
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
                    pooling_output=pooling_output_payload,
                    multimodal_output=mm_output,
                    stop_reason=request.stop_reason,
                    prefill_stats=request.take_prefill_stats(),
                    kv_transfer_params=kv_transfer_params,
                    routed_experts=routed_experts,
                    num_nans_in_logits=request.num_nans_in_logits,
                    is_segment_finished=is_segment_finished,
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Finish async_chunk requests that schedule() collected because their
        # upstream completed with no remaining codec tokens.
        for request in self._pending_finish_reqs:
            if request in stopped_running_reqs or request.is_finished():
                continue
            request.status = RequestStatus.FINISHED_STOPPED
            finish_reason = request.get_finished_reason()
            finished = self._handle_stopped_request(request)
            is_segment_finished = not finished
            kv_transfer_params = None
            if finished:
                kv_transfer_params, _ = self._free_request(request)
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup(
                        request.request_id,
                        getattr(request, "external_req_id", None),
                    )
            OmniSchedulerMixin._append_request_output(
                self,
                outputs,
                request,
                new_token_ids=[],
                finish_reason=finish_reason,
                stop_reason=request.stop_reason,
                kv_transfer_params=kv_transfer_params,
                is_segment_finished=is_segment_finished,
            )
            stopped_running_reqs.add(request)
        self._pending_finish_reqs.clear()

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
                self.chunk_transfer_adapter.cleanup(
                    request.request_id,
                    getattr(request, "external_req_id", None),
                )

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
            synthesize_abort_outputs=False,
        )

        self._attach_scheduler_stats(
            engine_core_outputs,
            spec_decoding_stats,
            kv_connector_stats,
            cudagraph_stats,
            perf_stats,
        )

        self._capture_omni_connector_output(model_runner_output)

        return engine_core_outputs

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Just replace the existing session with the next streaming update.

        Do not expend prompt id using update.
        """
        self._replace_streaming_session(session, update)
