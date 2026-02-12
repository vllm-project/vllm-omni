import time
from collections import defaultdict

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.outputs import OmniModelRunnerOutput


class OmniGenerationScheduler(VLLMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)

    def schedule(self) -> SchedulerOutput:
        """Diffusion fast path:
        - Feed all input tokens of the request at once
          (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the
          default vLLM scheduling.
        """

        token_budget = self.max_num_scheduled_tokens
        scheduled_timestamp = time.monotonic()

        scheduled_new_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running_reqs: list[Request] = []
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}

        # Temporary queue: preserve waiting order, do not disturb non-diffusion requests
        skipped_waiting_requests = create_request_queue(self.policy)
        req_index = 0
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)

        # OMNI: Track requests that are already finished (e.g., marked by connector)
        # These should be removed from running and not scheduled
        already_finished_reqs: set[Request] = set()
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # OMNI: Skip requests that are not in self.requests
            # This can happen when connector marks request as finished and it's removed from requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                already_finished_reqs.add(request)
                req_index += 1
                continue

            num_computed_tokens = request.num_computed_tokens
            required_tokens = max(len(request.prompt_token_ids) - num_computed_tokens, 1)
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
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_index += 1

        # OMNI: Remove already finished requests from running queue
        if already_finished_reqs:
            self.running = remove_all(self.running, already_finished_reqs)

        # Fast path selection and scheduling (treat all as diffusion requests,
        # independent of pooling_params)
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_running_reqs:
            request = self.waiting.peek_request()
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                # Pop the finished request from waiting queue and don't schedule it
                self.waiting.pop_request()
                continue

            # Uniformly treat as diffusion. A feature flag can be added later
            # via config or request tag.

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

            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            scheduled_new_reqs.append(request)

        # Return skipped waiting requests
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # If fast path scheduled none, fall back to the original scheduling
        if not num_scheduled_tokens:
            res = super().schedule()
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
                self.chunk_transfer_adapter.postprocess_scheduler_output(res)
            return res

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
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
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
        )

        # Record the request ids scheduled in this step (v0.14.0 behavior).
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # KVTransfer: package metadata
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        # EC Connector: package metadata
        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Update internal state (advance num_computed_tokens, free encoder inputs,
        # etc.)
        self._update_after_schedule(scheduler_output)

        try:
            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]

            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output)

        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")

        return scheduler_output

    """
    Scheduler for the diffusion model.
    This scheduler is modified to stop the request immediately for the diffusion model.
    This is because the diffusion model can generate the final image/audio in one step.
    Note: This is just a minimal modification to the original scheduler,
    and there should be some further efforts to optimize the scheduler.
    The original scheduler is still used for the AR model.
    """

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update the scheduler state based on the model runner output.

        This method is modified to stop the request immediately for the diffusion model.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        # Merge connector-side stats (align with v0.14.0)
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and getattr(kv_connector_output, "invalid_block_ids", None):
            failed_kv_load_req_ids = self._handle_invalid_blocks(kv_connector_output.invalid_block_ids)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
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
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            status_before_stop = request.status
            finish_reason = None
            routed_experts = None

            # Diffusion request: completes in one step; mark finished and free resources
            if request.status == RequestStatus.FINISHED_STOPPED or (
                self.chunk_transfer_adapter is None and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                request.status = RequestStatus.FINISHED_STOPPED
                # Optional: set a stop_reason for front-end clarity
                # (does not affect protocol)
                request.stop_reason = request.stop_reason  # or "generation_done"
                stopped = True

            if stopped:
                routed_experts = self._get_routed_experts(request)
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
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
            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Collect and publish KV cache events (align with v0.14.0)
        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs
