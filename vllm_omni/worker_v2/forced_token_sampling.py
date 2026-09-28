# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fast path for models whose text distribution has exactly one finite logit.

Keep the ordinary sampler for every feature that can alter or inspect that
distribution. Audio-channel sampling remains in the model's depth transformer.
"""

import numpy as np
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler


def try_sample_forced_tokens(runner, input_batch, grammar_output):
    hook = getattr(runner.model, "get_forced_token_ids_mrv2", None)
    sampler = runner.sampler
    if (
        not callable(hook)
        or type(sampler) is not Sampler
        or runner.batch_sharder is not None
        or grammar_output is not None
        or input_batch.num_draft_tokens
        or input_batch.num_reqs == 0
        or input_batch.logits_indices.numel() != input_batch.num_reqs
        or sampler.compute_nans
        or sampler.return_sampling_mask
        or sampler.trace_replay_state is not None
        or sampler.thinking_budget_state.enabled
    ):
        return None
    indices = input_batch.idx_mapping_np
    if (
        sampler.get_logprobs_dims(indices) is not None
        # The seeded upstream path is stateless. An unseeded sampler may
        # consume the global CUDA RNG also used by model-side sampling.
        or not np.all(sampler.sampling_states.seeds_set[indices])
        or np.any(sampler.logit_bias_state.use_logit_bias[indices])
        or np.any(sampler.penalties_state.use_penalty[indices])
        or np.any(sampler.bad_words_state.num_bad_words.np[indices])
    ):
        return None
    tokens = hook(input_batch.num_reqs)
    if tokens is None:
        return None
    # Preserve the upstream chunked-prefill suppression and request-slot mapping.
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        input_batch.seq_lens.new_ones(input_batch.num_reqs),
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        sampler.req_states.prefill_len.gpu,
    )
    output = SamplerOutput(
        sampled_token_ids=tokens.view(-1, 1),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
    return output, num_sampled, num_rejected
