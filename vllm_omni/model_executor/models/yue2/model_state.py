# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind native MRv2 batches to YuE2's request-owned sampling state."""

from __future__ import annotations

import weakref
from typing import Any

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_sampler import OmniSampler
from vllm_omni.worker_v2.warmup import is_warmup_request


class Yue2ModelState(OmniModelState):
    _sampling_batch: InputBatch | None = None

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        self._sampling_batch = None
        # Native dummy batches have fresh IDs even when their slots contain live requests.
        if not any(rid in req_states.req_id_to_index for rid in input_batch.req_ids) or all(
            is_warmup_request(rid) for rid in input_batch.req_ids
        ):
            return inputs
        if input_batch.num_draft_tokens or input_batch.has_structured_output_reqs:
            raise ValueError("YuE2 sampling does not support speculative decoding or structured output")
        buffers = inputs["model_intermediate_buffer"]
        args = []
        for rid, buffer in zip(input_batch.req_ids, buffers, strict=True):
            if buffer.get("req_id") != rid:
                raise ValueError(f"YuE2 sampling slot does not belong to request {rid}")
            args.append(buffer["sampling_params"].extra_args)
        self.model.prepare_runner_inputs(
            req_ids=input_batch.req_ids,
            num_computed_tokens=input_batch.num_computed_tokens_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            input_ids=input_batch.input_ids,
            positions=input_batch.positions,
            sampling_extra_args=args,
            discard_mask=~self._sample_keep_mask(input_batch),
        )
        self._sampling_batch = input_batch
        return inputs

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._sampling_batch is None:
            return self.model.compute_prompt_logits(hidden_states)
        return self.model.compute_logits(hidden_states)

    def custom_sampler(self, sampler: Sampler) -> tuple[Yue2Sampler, None]:
        return Yue2Sampler(sampler, self), None


class Yue2Sampler(OmniSampler):
    def __init__(self, sampler: Sampler, state: Yue2ModelState):
        super().__init__(sampler)
        # The native runner retains its sampler after dropping model/model_state on shutdown.
        self.state = weakref.proxy(state)

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        if self.state._sampling_batch is not input_batch:
            return self.base_sampler(logits, input_batch)
        token_ids = self.state.model.sample(logits)
        counts, rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=counts,
            num_rejected=rejected,
        )
