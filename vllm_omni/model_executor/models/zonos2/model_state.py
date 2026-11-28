# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2's eager MRv2 lifecycle, nine-codebook sampler and owned output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import GrammarOutput, NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.data_entry_keys import (
    LAST_OUTPUT_TOKEN_KEY,
    OUTPUT_TOKEN_COUNT_KEY,
    REQUEST_ID_KEY,
    REQUEST_SAMPLING_PARAMS_KEY,
)
from vllm_omni.model_executor.models.zonos2.zonos2_keys import FRAMES, STATE
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample
from vllm_omni.worker_v2.warmup import is_warmup_request

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.zonos2.zonos2_talker import Zonos2TalkerForConditionalGeneration


class Zonos2ModelState(OmniModelState):
    model: Zonos2TalkerForConditionalGeneration

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: Zonos2TalkerForConditionalGeneration,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        # Slot removal also occurs on preemption. Retain this scalar and the
        # model's codec history by request ID until the runner reports finish.
        self.last_tokens: dict[str, int | torch.Tensor | None] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        prompt_len = len(new_req_data.prompt_token_ids or [])
        tokens = new_req_data.prefill_token_ids
        self.last_tokens[new_req_data.req_id] = tokens[-1] if len(tokens) > prompt_len else None

    def on_requests_finished(self, req_ids: set[str]) -> None:
        super().on_requests_finished(req_ids)
        for req_id in req_ids:
            self.last_tokens.pop(req_id, None)

    def custom_sampler(self, sampler: Sampler) -> tuple[Zonos2Sampler, None]:
        return Zonos2Sampler(sampler, self), None

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Any = None,
    ) -> None:
        if all(is_warmup_request(key) for key in input_batch.req_ids):
            self.model._sampling_plan = []
            return
        if req_states is None:
            raise ValueError("ZONOS2 MRv2 preprocessing requires request state")
        eligible = self._sample_keep_mask(input_batch).tolist()
        snapshots: list[dict[str, Any]] = []
        input_ids = model_inputs["input_ids"]
        embeds = model_inputs["inputs_embeds"]
        for row, key in enumerate(input_batch.req_ids):
            slot = int(input_batch.idx_mapping_np[row])
            buffer = self.intermediate_buffer.buffers[slot]
            if buffer.get("req_id") != key or key not in self.last_tokens:
                raise ValueError("ZONOS2 MRv2 request identity does not match its slot")
            info = dict(buffer)
            frames = info.get(FRAMES)
            if not isinstance(frames, torch.Tensor):
                raise ValueError("ZONOS2 MRv2 request is missing its prompt frames")
            prompt_len = len(frames)
            if int(req_states.prompt_len.np[slot]) != prompt_len:
                raise ValueError("ZONOS2 MRv2 prompt length does not match its frames")
            offset = int(input_batch.num_computed_tokens_np[row])
            span = int(input_batch.num_scheduled_tokens[row])
            # With synchronous, non-speculative scheduling these CPU positions
            # are exact. Re-prefill includes the already committed lifecycle
            # tokens, even while its current chunk is still inside the prompt.
            committed = max(0, int(input_batch.prefill_len_np[row]) - prompt_len, offset + span - prompt_len)
            params = info["sampling_params"]
            info.update(
                {
                    REQUEST_ID_KEY: key,
                    OUTPUT_TOKEN_COUNT_KEY: committed,
                    LAST_OUTPUT_TOKEN_KEY: self.last_tokens[key],
                    REQUEST_SAMPLING_PARAMS_KEY: {
                        name: getattr(params, name)
                        for name in (
                            "temperature",
                            "top_k",
                            "top_p",
                            "min_p",
                            "seed",
                            "max_tokens",
                            "ignore_eos",
                            "repetition_penalty",
                            "extra_args",
                        )
                    },
                    "_omni_num_computed_tokens": offset,
                }
            )
            start = int(input_batch.query_start_loc_np[row])
            ids, hidden, updates = self.model.preprocess(input_ids[start : start + span], None, **info)
            input_ids[start : start + span].copy_(ids)
            embeds[start : start + span].copy_(hidden)
            # Only model-owned state persists. Its tensors are replaced, never
            # modified, so MRv2 can retain their storage without a second copy.
            self.intermediate_buffer.update(
                slot, {STATE: updates[STATE]}, self.model.gpu_resident_buffer_keys, owned=True
            )
            snapshots.append({**info, **updates})
        model_inputs["model_intermediate_buffer"] = snapshots
        model_inputs["runtime_additional_information"] = snapshots
        model_inputs["request_sample_eligible"] = eligible

    def postprocess_model_output(
        self, model_output: torch.Tensor, input_batch: InputBatch, req_states: RequestState
    ) -> tuple[torch.Tensor, dict]:
        # Codec output belongs to the current sampling step, after forward.
        return model_output, {}


class Zonos2Sampler(OmniSampler):
    def __init__(self, sampler: Sampler, state: Zonos2ModelState) -> None:
        super().__init__(sampler)
        self.state = state

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        if all(is_warmup_request(key) for key in input_batch.req_ids):
            return super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        if grammar_output is not None or input_batch.num_draft_tokens:
            raise ValueError("ZONOS2 MRv2 does not support grammar or speculative sampling")
        model = self.state.model
        if [key for key, _, _ in model._sampling_plan] != input_batch.req_ids:
            raise ValueError("ZONOS2 MRv2 sampling plan does not match the current batch")
        if input_batch.logits_indices.numel() != input_batch.num_reqs:
            raise ValueError("ZONOS2 MRv2 requires one logits row per request")
        hidden = hidden_states[input_batch.logits_indices]
        logits = model.compute_logits(hidden)
        payload = model.make_omni_output(hidden).multimodal_outputs
        output = model.sample(logits, None)
        if output is None:
            raise RuntimeError("ZONOS2 MRv2 sampler did not produce lifecycle tokens")
        for row, (key, eligible, _) in enumerate(model._sampling_plan):
            if not eligible:
                continue
            self.state.last_tokens[key] = output.sampled_token_ids[row, 0]
            slot = int(input_batch.idx_mapping_np[row])
            self.state.intermediate_buffer.update(
                slot, {STATE: model._request_states[key].payload()}, model.gpu_resident_buffer_keys, owned=True
            )
        return OmniSamplingOutput(
            output,
            output.num_sampled,
            output.num_rejected,
            payload,
            include_hidden_states=False,
            owns_multimodal_outputs=True,
        )
