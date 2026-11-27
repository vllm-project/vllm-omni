# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 batch lifecycle and in-step audio sampling for Haru."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.haru.runtime import HaruRuntime
from vllm_omni.worker_v2.model_states.omni_mamba_hybrid import OmniMambaHybridModelState
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.haru.haru import HaruForConditionalGeneration


class HaruModelState(OmniMambaHybridModelState):
    model: HaruForConditionalGeneration

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        self.model.runtime.begin_step()
        super().run_preprocess(input_batch, model_inputs, req_states, mtp_batch_descriptor_dispatcher)

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_sample_logits(hidden_states)
        if self.model.runtime.has_rows:
            return self.model.runtime.adjust_logits(hidden_states, logits)
        return logits

    def custom_sampler(self, sampler: Sampler) -> tuple[HaruSampler, None]:
        return HaruSampler(sampler, self.model.runtime), None


class HaruSampler(OmniSampler):
    def __init__(self, base_sampler: Sampler, runtime: HaruRuntime) -> None:
        super().__init__(base_sampler)
        self.runtime = runtime

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        result = super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        if not self.runtime.has_rows:
            return result

        # The stock sampler has one token column even for unfinished prompt chunks.
        # Mask dropped samples before copying the token IDs needed for the next frame.
        tokens = result.sampler_output.sampled_token_ids[:, 0].masked_fill(result.num_sampled == 0, -1).tolist()
        invalid_rows = [row for row, token in enumerate(tokens) if token == -1]
        self.runtime.record_sampled([[] if token == -1 else [token] for token in tokens])
        result.multimodal_outputs = self.runtime.finish_step(input_batch.req_ids, invalid_rows)
        result.include_hidden_states = False
        result.owns_multimodal_outputs = True
        return result
