from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from time import time
from typing import Any

from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

logger = init_logger(__name__)


@dataclass
class KVCacheTransferData:
    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OmniARScheduler(VLLMScheduler):
    """
    OmniARScheduler: Scheduler for vLLM-Omni multimodal processing.

    This scheduler extends vLLM's scheduler to support multimodal and
    non-autoregressive processing with additional fields and methods
    specific to vLLM-Omni.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        self.requests_needing_kv_transfer: dict[str, int] = {}
        # Track requests that have already been processed for KV transfer
        self.processed_kv_transfers: set[str] = set()
        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

    # Ensure scheduled_new_reqs carry omni-specific payloads
    # (e.g., additional_information)
    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        scheduler_output = super().schedule()
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
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

            # Add information about requests needing KV cache transfer
            scheduler_output.finished_requests_needing_kv_transfer = self.get_finished_requests_needing_kv_transfer()
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            scheduler_output.finished_requests_needing_kv_transfer = {}

        return scheduler_output

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
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
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
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                if request.output_token_ids:
                    stopped = check_stop(request, self.max_model_len, pooler_output)

            if stopped:
                # If we need transfer, we do NOT call _free_request immediately (or effectively, we delay block freeing)
                # But _free_request does other things too (metrics, etc).
                # We need a way to perform the non-block-freeing parts of _free_request.

                # Let's use our modified _free_request which handles this check internally
                kv_transfer_params = self._free_request(request)

                # If the request was marked for transfer inside _free_request,
                # self.requests_needing_kv_transfer will have it.

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
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

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
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

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        # [Omni] Handle finished transfers reported by Model Runner
        # This is where we free blocks that were held for transfer
        try:
            kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
            if kv_extracted_ids:
                for req_id in kv_extracted_ids:
                    if req_id in self.waiting_for_transfer_free:
                        # Now it's safe to free blocks
                        req = self.requests.get(req_id)
                        if req:
                            # Use _free_blocks from parent logic
                            # Since we don't have direct access to it without calling super()._free_blocks
                            # (which doesn't exist separately in all versions,
                            # usually it's just in _free_request or self.kv_cache_manager.free).
                            # We can call self.kv_cache_manager.free(req) directly and remove from requests.
                            self.kv_cache_manager.free(req)
                            # Remove from requests dict - effectively what _free_blocks does
                            if req_id in self.requests:
                                del self.requests[req_id]

                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
        except Exception:
            init_logger(__name__).exception("Failed to process finished transfer requests")

        return engine_core_outputs

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        # This overrides VLLMScheduler._free_request completely to add delayed freeing logic.

        # 1. Standard cleanup parts from base _free_request
        # We need to replicate base behavior but add our hook

        # From base: invoke connector finished
        delay_free_blocks = False
        kv_xfer_params = None
        if self.connector is not None:
            # This assumes base class has _connector_finished or we need to copy that too if it's private
            # Since we inherit, we should have access if it's protected/public.
            # VLLMScheduler has _connector_finished
            delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # 2. Omni Specific: Check if we need to transfer KV
        if self._should_transfer_kv_for_request(request_id):
            self.waiting_for_transfer_free.add(request_id)
            # Store req_id and seq_len
            self._mark_request_for_kv_transfer(request_id, request.num_computed_tokens)
            # FORCE delay freeing blocks regardless of connector status
            # We return early so _free_blocks is not called
            return kv_xfer_params

        # 3. Standard Freeing
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        # Helper to match base class structure if not directly available
        # VLLMScheduler has _free_blocks
        super()._free_blocks(request)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking
        if req_id in self.requests_needing_kv_transfer or req_id in self.processed_kv_transfers:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]
                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        need_send = False
        import os

        # 1. Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_config attribute
        omni_config = getattr(self.vllm_config, "omni_config", None)
        if omni_config:
            # omni_config could be an object or a dict
            if isinstance(omni_config, dict):
                need_send = omni_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_config, "need_send_cache", False)

        # 2. Environment variable as override/fallback
        if os.getenv("OMNI_NEED_SEND_CACHE"):
            need_send = os.getenv("OMNI_NEED_SEND_CACHE", "false").lower() == "true"

        # 3. Compatibility check
        kv_role = os.getenv("VLLM_KV_ROLE", "")
        if kv_role == "kv_producer":
            need_send = True

        return need_send

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        if self.requests_needing_kv_transfer:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()
        # Mark these requests as processed to avoid duplicate processing
        self.processed_kv_transfers.update(requests.keys())
        self.requests_needing_kv_transfer.clear()
        return requests
