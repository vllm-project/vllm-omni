# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve codec cache positions across atomic, asynchronous audio chunks."""

import copy
import threading

import torch
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.stage_input_processors.easymagpie import _cleanup_codec_request


class EasyMagpieTalkerTransferAdapter(OmniChunkTransferAdapter):
    @staticmethod
    def _snapshot_processor_request(request):
        # This processor consumes ids, terminal flags and two immutable input
        # settings. Token histories and text conditioning never cross this edge.
        snapshot = copy.copy(request)
        info = deserialize_additional_information(request.additional_information)
        snapshot.additional_information = {
            key: info[key] for key in ("text_prefill_num", "reset_codec_on_segment") if key in info
        }
        return snapshot

    def _clear_sender_state_locked(self, external_req_id):
        super()._clear_sender_state_locked(external_req_id)
        _cleanup_codec_request(self, external_req_id)


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Keep prefills separate from graph-replayed decode batches.

    Allow two decode steps after admission so the startup audio chunk can
    finish while waiting prompts continue to make progress under load.
    """

    chunk_transfer_adapter_cls = EasyMagpieTalkerTransferAdapter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._separate_prefills = getattr(self.vllm_config.model_config.hf_config, "easymagpie_optimized", False)
        self._decode_steps_remaining = 0
        self._throttle_prefills = False
        self._prefill_step = False
        self._deferred_decode_steps = []

    @staticmethod
    def _is_prefill(request):
        pending = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
        return request.is_prefill_chunk or pending > 1

    def _should_defer_waiting_admission(self):
        defer = super()._should_defer_waiting_admission()
        if (
            not self._separate_prefills
            or self._throttle_prefills
            or self.num_spec_tokens
            or self.parallel_config.pipeline_parallel_size != 1
        ):
            return defer
        ready = [
            request
            for request in self.running
            if request.status == RequestStatus.RUNNING
            and request.next_decode_eligible_step <= self.current_step + 1
            and request.num_tokens_with_spec + request.num_output_placeholders > request.num_computed_tokens
        ]
        has_decode = any(not self._is_prefill(request) for request in ready)
        can_admit = len(self.running) + self.num_waiting_for_streaming_input < self.max_num_running_reqs
        has_prefill = any(self._is_prefill(request) for request in ready) or (
            can_admit and bool(self.waiting) and not defer
        )
        self._prefill_step = has_prefill and not (has_decode and self._decode_steps_remaining)
        for request in ready:
            if self._is_prefill(request) != self._prefill_step:
                self._deferred_decode_steps.append((request, request.next_decode_eligible_step))
                request.next_decode_eligible_step = self.current_step + 2
        return defer or not self._prefill_step

    def schedule(self, throttle_prefills=False):
        self._throttle_prefills = throttle_prefills
        try:
            output = super().schedule(throttle_prefills)
        finally:
            for request, previous_step in self._deferred_decode_steps:
                request.next_decode_eligible_step = previous_step
            self._deferred_decode_steps.clear()
        if output.num_scheduled_tokens:
            self._decode_steps_remaining = 2 if self._prefill_step else max(0, self._decode_steps_remaining - 1)
        return output


class EasyMagpieCodecTransferAdapter(OmniChunkTransferAdapter):
    def __init__(self, vllm_config):
        self.codec_lock = threading.RLock()
        super().__init__(vllm_config)

    def _poll_single_request(self, entry):
        with self.codec_lock:
            if entry.request.num_in_flight_tokens:
                return False
            return super()._poll_single_request(entry)

    def _commit_received_chunk(self, request, result, **kwargs):
        # The generic generation adapter replaces the prompt for each chunk.
        # A causal codec instead appends one placeholder per frame, preserving
        # the position used by its vLLM-managed recurrent state pages.
        computed = request.num_computed_tokens
        prompt = list(request.prompt_token_ids or ())[:computed]
        token_ids = list(request._all_token_ids)[:computed]
        ready = super()._commit_received_chunk(request, result, **kwargs)
        codes = result[0].get("codes", {}).get("audio")
        frames = codes.shape[0] if isinstance(codes, torch.Tensor) and codes.ndim == 2 else 0
        placeholders = [0] * frames
        request.prompt_token_ids = prompt + placeholders
        request._all_token_ids[:] = token_ids + placeholders
        request.num_prompt_tokens = len(request.prompt_token_ids)
        request.num_computed_tokens = computed
        request.update_block_hashes()
        return ready

    def process_pending_chunks(self, waiting_queue, running_queue, scheduler_requests=None):
        super().process_pending_chunks(waiting_queue, running_queue, scheduler_requests=scheduler_requests)
        # A terminal chunk may already be materialized while its PCM copy is
        # in flight. Keep its state and terminal flags until output settles.
        for request in list(running_queue):
            if request.num_in_flight_tokens:
                running_queue.remove(request)
                self.waiting_for_chunk_running_requests.append(request)


class EasyMagpieCodecScheduler(OmniGenerationScheduler):
    chunk_transfer_adapter_cls = EasyMagpieCodecTransferAdapter

    def schedule(self, *args, **kwargs):
        with self.chunk_transfer_adapter.codec_lock:
            return super().schedule(*args, **kwargs)

    def update_from_output(self, *args, **kwargs):
        with self.chunk_transfer_adapter.codec_lock:
            return super().update_from_output(*args, **kwargs)
