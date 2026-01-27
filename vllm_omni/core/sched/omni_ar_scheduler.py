from __future__ import annotations

import importlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from time import time
from typing import Any

import numpy as np
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.distributed.omni_connectors.adapter import get_chunk, put_chunk
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

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
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()
        model_config = self.vllm_config.model_config
        self.omni_connector = None
        if model_config.async_chunk:
            connector_config = model_config.stage_connector_config
            connector_specs = ConnectorSpec(
                name=connector_config.get("name", "SharedMemoryConnector"),
                extra=connector_config.get("extra", {}),
            )
            self.omni_connector = OmniConnectorFactory.create_connector(connector_specs)

            custom_process_next_stage_input_func = getattr(
                self.vllm_config.model_config, "custom_process_next_stage_input_func", None
            )
            if custom_process_next_stage_input_func:
                module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self.custom_process_next_stage_input_func = getattr(module, func_name)

        self.stage_id = getattr(self.vllm_config.model_config, "stage_id", None)

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                return omni_kv_config.get("kv_transfer_criteria", None)
            else:
                return getattr(omni_kv_config, "kv_transfer_criteria", None)
        return None

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")

        # Universal duplicate check for once semantics
        if request.request_id in self.transfer_triggered_requests:
            return False

        if criteria_type == "prefill_finished":
            if request.num_computed_tokens >= request.num_prompt_tokens:
                logger.debug(f"[Omni] Request {request.request_id} triggered prefill_finished transfer (Non-Stop)")
                self.transfer_triggered_requests.add(request.request_id)
                self._mark_request_for_kv_transfer(request.request_id, request.num_computed_tokens)

                # Return False means "Do NOT stop the request" -> Continue Decoding
                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                logger.debug(f"[Omni] Request {request.request_id} triggered special_token criteria (Non-Stop)")

                self.transfer_triggered_requests.add(request.request_id)

                # Calculate precise snapshot length (trim to sentinel)
                # Find the FIRST occurrence of the sentinel
                try:
                    idx = new_token_ids.index(target_token_id)
                    # seq_len = tokens_before_this_step + idx + 1 (include sentinel)
                    # request.num_computed_tokens already includes ALL new_token_ids
                    # so we subtract (len(new_token_ids) - (idx + 1))
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = request.num_computed_tokens - tokens_to_exclude
                except ValueError:
                    snapshot_len = request.num_computed_tokens

                # Trigger Transfer
                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)

                # Do NOT stop request
                return False

        return False

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
            if self.omni_connector is not None:
                get_chunk(self.omni_connector, scheduler_output)

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
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

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
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

            # If criteria returns True, it means we must STOP the request (e.g. legacy behavior, not used now)
            # If criteria returns False, it might have triggered a background transfer
            # (e.g. prefill finished / special token) but continues decoding
            if not stopped and self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                if request.output_token_ids:
                    stopped = check_stop(request, self.max_model_len)
            routed_experts = None
            if stopped:
                # [Omni] Handle routed experts if enabled
                if self.vllm_config.model_config.enable_return_routed_experts:
                    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
                    block_ids = kv_blocks.get_block_ids()[0]
                    num_tokens = request.num_tokens - 1

                    # compute slot mapping
                    block_ids_array = np.array(block_ids, dtype=np.int32)
                    num_blocks = len(block_ids)
                    block_size = self.block_size

                    # generate block offsets
                    block_offsets = np.arange(0, block_size)

                    # compute slot mapping: slot = block_id * block_size + offset
                    slot_mapping = (
                        block_offsets.reshape((1, block_size)) + block_ids_array.reshape((num_blocks, 1)) * block_size
                    ).flatten()[:num_tokens]

                    routed_experts = self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

                kv_transfer_params = self._free_request(request)
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
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

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
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
                if self.omni_connector is not None:
                    custom_process_next_stage_input_func = self.custom_process_next_stage_input_func
                    put_chunk(self.omni_connector, pooler_output, request, custom_process_next_stage_input_func)
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # [Main] Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # [Omni] Cleanup state for finished requests
        for req in stopped_running_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)

        # Same for preempted
        for req in stopped_preempted_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
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

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        # This is where we free blocks that were held for transfer
        try:
            kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
            if kv_extracted_ids:
                for req_id in kv_extracted_ids:
                    # Mark transfer as finished
                    if req_id in self.active_kv_transfers:
                        self.active_kv_transfers.remove(req_id)
                        logger.debug(f"[Omni] KV Transfer finished for {req_id}")

                    if req_id in self.waiting_for_transfer_free:
                        # Now it's safe to free blocks
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            if req_id in self.transfer_triggered_requests:
                                self.transfer_triggered_requests.remove(req_id)
                            if req_id in self.active_kv_transfers:
                                self.active_kv_transfers.remove(req_id)

                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
        except Exception:
            init_logger(__name__).exception("Failed to process finished transfer requests")

        return engine_core_outputs

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""

        # 1. Standard cleanup parts from base _free_request
        delay_free_blocks = False
        kv_xfer_params = None
        if self.connector is not None:
            delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # 2. Omni Specific: Check if we need to transfer KV
        if self._should_transfer_kv_for_request(request_id):
            already_triggered = request_id in self.transfer_triggered_requests
            is_active = request_id in self.active_kv_transfers

            if already_triggered:
                if is_active:
                    # It triggered but hasn't finished yet. We MUST wait.
                    logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                    self.waiting_for_transfer_free.add(request_id)
                    # We do NOT mark for transfer again, just wait.
                    kv_xfer_params = None  # No new transfer params
                    return kv_xfer_params
                else:
                    logger.debug(
                        f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                        "Freeing immediately."
                    )
            else:
                self.waiting_for_transfer_free.add(request_id)
                self._mark_request_for_kv_transfer(request_id, request.num_computed_tokens)
                # Return KV transfer metadata so it propagates to RequestOutput
                if request_id in self.requests_needing_kv_transfer:
                    transfer_data = self.requests_needing_kv_transfer[request_id]
                    kv_xfer_params = {
                        "past_key_values": transfer_data["block_ids"],
                        "kv_metadata": {"seq_len": transfer_data["seq_len"], "block_ids": transfer_data["block_ids"]},
                    }
                    # Also update request.additional_information for good measure
                    add_info = getattr(request, "additional_information", None)
                    # If additional_information is an AdditionalInformationPayload-like object,
                    # unpack list_data into a plain dict.
                    if (
                        add_info is not None
                        and hasattr(add_info, "entries")
                        and isinstance(getattr(add_info, "entries"), dict)
                    ):
                        request.additional_information = {
                            k: getattr(v, "list_data")
                            for k, v in getattr(add_info, "entries").items()
                            if getattr(v, "list_data", None) is not None
                        }
                        add_info = request.additional_information
                    if add_info is None:
                        request.additional_information = {}
                        add_info = request.additional_information
                    if isinstance(add_info, dict):
                        add_info.update(kv_xfer_params)

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
        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        return need_send

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests
