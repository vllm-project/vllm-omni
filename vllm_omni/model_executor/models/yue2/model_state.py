# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind YuE2's request-owned sampler to Model Runner V2 batches."""

from typing import Any

import torch
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample
from vllm_omni.worker_v2.warmup import is_warmup_request


class Yue2ModelState(OmniModelState):
    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        if all(is_warmup_request(rid) for rid in input_batch.req_ids):
            return inputs
        # Profiling/capture uses InputBatch.make_dummy with random IDs and no
        # admitted requests. Its slot numbers may overlap live buffers.
        if not any(rid in req_states.req_id_to_index for rid in input_batch.req_ids):
            return inputs
        # Slot indices and batch rows differ after admission/preemption. Gather
        # parameters in forward order, including both members of a CFG pair.
        buffers = self.intermediate_buffer.gather(input_batch)
        self.model.prepare_runner_inputs(
            req_ids=input_batch.req_ids,
            num_computed_tokens=input_batch.num_computed_tokens_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            input_ids=input_batch.input_ids,
            positions=input_batch.positions,
            sampling_extra_args=[buf["sampling_params"].extra_args or {} for buf in buffers],
            discard_mask=~self._sample_keep_mask(input_batch),
        )
        return inputs

    def postprocess_model_output(
        self, model_output: torch.Tensor, input_batch: InputBatch, req_states: RequestState
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Acoustic jobs finish after the AR slot is released. The YuE2 worker
        # publishes their request-keyed audio separately, including idle steps.
        return model_output, {}

    def custom_sampler(self, sampler: Sampler) -> tuple[OmniSampler, None]:
        return Yue2Sampler(sampler, self), None


class Yue2Sampler(OmniSampler):
    def __init__(self, base_sampler: Sampler, state: Yue2ModelState):
        super().__init__(base_sampler)
        self.state = state

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        if all(is_warmup_request(rid) for rid in input_batch.req_ids):
            return super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        if grammar_output is not None or input_batch.num_draft_tokens:
            raise ValueError("YuE2 does not support grammar or speculative sampling")
        model = self.state.model
        logits = model.compute_logits(hidden_states[input_batch.logits_indices])
        sampled = model.sample(logits, None)
        if sampled is None:
            raise RuntimeError("YuE2's model-owned sampler did not return tokens")
        count, rejected = get_num_sampled_and_rejected(
            torch.ones(input_batch.num_reqs, dtype=torch.int32, device=hidden_states.device),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            req_states.prefill_len.gpu,
        )
        output = SamplerOutput(
            sampled_token_ids=sampled.sampled_token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=count,
            num_rejected=rejected,
        )
        return OmniSamplingOutput(output, count, rejected, include_hidden_states=False)
