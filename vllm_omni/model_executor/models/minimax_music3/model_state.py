# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 state for the MiniMax Music 3 talker.

The talker already owns everything that makes a step: the row-to-request
binding and frame feedback (``prepare_runner_inputs``), the guided frame draw
(``sample``) and the window emission (``postprocess`` / ``make_omni_output``).
This state only feeds those hooks from MRv2's batch instead of V1's, so both
runners execute the same model code and produce the same audio for a seed.

Where each V1 input comes from here:

* ``prepare_runner_inputs`` runs in ``prepare_inputs``, before the forward and
  before a FULL decode graph is replayed, with the batch's own computed and
  scheduled token counts.
* ``sampling_extra_args`` (CFG role and scale, seed) comes from each row's
  sampling params, which the intermediate buffer keeps per request.
* ``request_sample_eligible`` marks the rows whose step produces a token, the
  ones that are past their prompt once this step's tokens are computed.
* ``sample`` runs in the registered sampler's ``sample_step`` on the logits of
  the hidden states the forward just produced. Its draw is seeded by request
  and frame position, so nothing here depends on the stock sampler's RNG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample
from vllm_omni.worker_v2.warmup import is_warmup_request

if TYPE_CHECKING:
    from .talker import MiniMaxMusic3TalkerForConditionalGeneration

__all__ = ["MiniMaxMusic3TalkerModelState"]


class MiniMaxMusic3TalkerModelState(OmniModelState):
    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        num_reqs = input_batch.num_reqs
        computed = input_batch.num_computed_tokens_np[:num_reqs]
        scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        # Writes the frame feedback slab a replayed decode graph reads, and
        # rebinds rows to requests for the sampler.
        self.model.prepare_runner_inputs(
            req_ids=input_batch.req_ids[:num_reqs],
            num_computed_tokens=computed,
            num_scheduled_tokens=scheduled,
            input_ids=input_batch.input_ids,
            positions=input_batch.positions,
        )
        infos = inputs["model_intermediate_buffer"][:num_reqs]
        inputs["sampling_extra_args"] = [
            dict(getattr(info.get("sampling_params"), "extra_args", None) or {}) for info in infos
        ]
        inputs["request_sample_eligible"] = (computed + scheduled >= input_batch.prefill_len_np[:num_reqs]).tolist()
        return inputs

    def custom_sampler(self, sampler: Sampler) -> tuple[MiniMaxMusic3Sampler, None]:
        return MiniMaxMusic3Sampler(sampler, self.model), None


class MiniMaxMusic3Sampler(OmniSampler):
    """Hands the talker's guided frame draw to the runner as its sampler."""

    def __init__(self, base_sampler: Sampler, talker: MiniMaxMusic3TalkerForConditionalGeneration) -> None:
        super().__init__(base_sampler)
        self.talker = talker

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        # vLLM's warmup requests carry no CFG role. They are not songs, so the
        # stock sampler takes them, as it does V1's warm-up batches.
        if input_batch.num_reqs == 0 or all(is_warmup_request(r) for r in input_batch.req_ids):
            return super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        if grammar_output is not None or input_batch.num_draft_tokens:
            raise ValueError("MiniMax Music 3 on MRv2 does not support structured output or speculative decoding")
        logits = self.talker.compute_logits(hidden_states[input_batch.logits_indices])
        output = self.talker.sample(logits, None)
        if output is None:
            # No guided row resolved this step; the stock sampler handles it,
            # as it does on V1.
            return super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        num_reqs = input_batch.num_reqs
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            req_states.prefill_len.gpu,
        )
        sampler_output = SamplerOutput(output.sampled_token_ids[:num_reqs], None, None, num_sampled, num_rejected)
        # The talker sets omni_pooler_payload_include_hidden = False, which V1
        # honors; without this the backbone hidden state would also be
        # published, under the same "latent" key as the song.
        #
        # Without async_chunk the runner builds per-request payloads only for
        # hidden states or a finalizer, so keep the talker's payload as is.
        return OmniSamplingOutput(
            sampler_output,
            num_sampled,
            num_rejected,
            include_hidden_states=False,
            finalize_multimodal=None if self.talker.publishes_windows else _keep_payload,
        )


def _keep_payload(payload: dict[str, Any], num_sampled: list[int]) -> dict[str, Any]:
    return payload


def create_minimax_music3_talker_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> MiniMaxMusic3TalkerModelState:
    return MiniMaxMusic3TalkerModelState(vllm_config, model, encoder_cache, device)
