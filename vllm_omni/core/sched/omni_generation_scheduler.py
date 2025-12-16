import time
from collections import defaultdict

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import (
    Request,
    RequestStatus,
    SchedulerOutput,
    SpecDecodingStats,
)
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.outputs import OmniModelRunnerOutput


class OmniGenerationScheduler(VLLMScheduler):
    def schedule(self) -> SchedulerOutput:
        """Diffusion fast path:
        - Feed all input tokens of the request at once
          (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the
          default vLLM scheduling.
        """

        # Select requests with zero prompt and using pooling
        # (diffusion results returned via pooler_output)
        token_budget = self.max_num_scheduled_tokens
        capacity = self.max_num_running_reqs - len(self.running)
        scheduled_timestamp = time.monotonic()

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}

        # Temporary queue: preserve waiting order, do not disturb non-diffusion requests
        skipped_waiting_requests = create_request_queue(self.policy)

        # Fast path selection and scheduling (treat all as diffusion requests,
        # independent of pooling_params)
        while self.waiting and token_budget > 0 and capacity > 0:
            request = self.waiting.peek_request()
            # Uniformly treat as diffusion. A feature flag can be added later
            # via config or request tag.

            # Allocate all input tokens for the request in one shot
            # (allocate 1 placeholder if zero)
            required_tokens = max(getattr(request, "num_prompt_tokens", 0), 1)
            if required_tokens > token_budget:
                # Insufficient budget to process all inputs at once;
                # stop fast path attempt
                break
            num_new_tokens = required_tokens
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
            request.status = RequestStatus.RUNNING
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)

            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            capacity -= 1
            scheduled_new_reqs.append(request)

        # Return skipped waiting requests
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # If fast path scheduled none, fall back to the original scheduling
        if not num_scheduled_tokens:
            return super().schedule()

        # Compute common prefix blocks (aligned with v1)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(
                any_request, len(self.running)
            )

        # Assemble SchedulerOutput
        new_reqs_data = [
            OmniNewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = scheduled_new_reqs + scheduled_running_reqs + scheduled_resumed_reqs
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(
            scheduled_requests, scheduled_spec_decode_tokens
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
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # KVTransfer: package metadata
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Update internal state (advance num_computed_tokens, free encoder inputs,
        # etc.)
        self._update_after_schedule(scheduler_output)
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

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats = kv_connector_output.kv_connector_stats if kv_connector_output else None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
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
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]

            # Diffusion request: completes in one step; mark finished and free resources
            request.status = RequestStatus.FINISHED_STOPPED
            # Optional: set a stop_reason for front-end clarity
            # (does not affect protocol)
            request.stop_reason = request.stop_reason  # or "generation_done"
            kv_transfer_params = self._free_request(request)
            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

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
            # Convert pooler_output tensor to dict format expected by OmniEngineCoreOutput
            pooling_output_dict = None
            if pooler_output is not None:
                # Wrap tensor in dict to match OmniEngineCoreOutput.pooling_output type
                # which expects Optional[dict[str, torch.Tensor]]
                pooling_output_dict = {"model_outputs": pooler_output}
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooling_output_dict,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
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
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(model_runner_output.kv_connector_output)

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

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs
