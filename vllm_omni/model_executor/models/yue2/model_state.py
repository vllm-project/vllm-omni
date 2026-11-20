# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind YuE2's request-owned sampler to Model Runner V2 batches."""

import time
from dataclasses import fields
from typing import Any

import torch
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState, RequestLifecycleUpdate
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample
from vllm_omni.worker_v2.warmup import is_warmup_request

from .runtime import Yue2ModelRunnerOutput, Yue2SchedulerOutput


class Yue2ModelState(OmniModelState):
    def get_request_lifecycle(self, scheduler_output: SchedulerOutput) -> RequestLifecycleUpdate | None:
        if not isinstance(scheduler_output, Yue2SchedulerOutput):
            return None
        return RequestLifecycleUpdate(
            reset_req_ids=scheduler_output.replaced_req_ids,
            release_req_ids=scheduler_output.acoustic_req_ids,
        )

    def finalize_runner_output(
        self, output: ModelRunnerOutput | AsyncModelRunnerOutput, *, idle: bool
    ) -> Yue2ModelRunnerOutput:
        # YuE2 requires synchronous scheduling. Resolve tokens before polling
        # acoustic jobs, and keep the existing request-keyed scheduler contract.
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()
        # V2's transient attributes are not dataclass constructor arguments.
        result = Yue2ModelRunnerOutput(**{item.name: getattr(output, item.name) for item in fields(output)})
        result.acoustic_ready = self.model.poll_audio()
        result.acoustic_started = list(self.model.acoustic_started)
        self.model.acoustic_started.clear()
        if idle and not result.acoustic_ready:
            # This callback runs on the worker thread, outside the API event
            # loop. Retain the acoustic poll cadence without busy-spinning.
            time.sleep(0.001)
        return result

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        if all(is_warmup_request(rid) for rid in input_batch.req_ids):
            return inputs
        # Profiling/capture uses InputBatch.make_dummy with random IDs and no
        # admitted requests. Its slot numbers may overlap live buffers.
        if not any(rid in req_states.req_id_to_index for rid in input_batch.req_ids):
            return inputs
        # The parent supplies buffers in forward order, including both CFG rows.
        buffers = inputs["model_intermediate_buffer"]
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
        # Acoustic jobs finish after the AR slot is released. The output hook
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
