# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.worker_v2.forced_token_sampling import try_sample_forced_tokens


def setup(params=None):
    device = torch.device("cuda")
    states = RequestState(4, 32, 16, 0, 32, device)
    sampler = Sampler(4, 32, device, states)
    params = params or SamplingParams(temperature=1.7, top_k=25, top_p=0.8, seed=42)
    for i in range(3):
        states.add_request(str(i), 5, [1, 2, 3, 4, 5], 0, 16)
        sampler.add_request(states.req_id_to_index[str(i)], 5, params)
    states.apply_staged_writes()
    sampler.apply_staged_writes()
    mapping = torch.tensor([1, 3, 2], device=device, dtype=torch.int32)
    indices = torch.arange(3, device=device)
    batch = SimpleNamespace(
        num_reqs=3,
        num_draft_tokens=0,
        logits_indices=indices,
        idx_mapping=mapping,
        expanded_idx_mapping=mapping,
        idx_mapping_np=np.array([1, 3, 2], dtype=np.int32),
        seq_lens=torch.tensor([3, 6, 5], device=device, dtype=torch.int32),
        cu_num_logits=torch.arange(4, device=device, dtype=torch.int32),
        cu_num_logits_np=np.arange(4, dtype=np.int32),
        positions=torch.tensor([2, 5, 4], device=device),
        input_ids=torch.tensor([3, 3, 3], device=device),
        expanded_local_pos=torch.zeros(3, device=device, dtype=torch.int32),
    )
    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    nn.Module.__init__(model)
    model.text_vocab_size = 32
    model.audio_assistant_slot_token_id = 7
    model.im_end_token_id = 9
    model._batch_state = []
    model._batch_should_continue = torch.tensor([True, False, True], device=device)
    return SimpleNamespace(model=model, sampler=sampler, batch_sharder=None), batch


@pytest.mark.parametrize(
    "temperature,top_k,top_p,min_p", [(1.7, 25, 0.8, 0.0), (0.0, -1, 1.0, 0.0), (1.0, 1, 0.01, 0.9)]
)
def test_matches_real_sampler_with_mixed_prefill_and_reordered_slots(temperature, top_k, top_p, min_p):
    runner, batch = setup(SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=42))
    logits = runner.model.compute_logits(torch.zeros(3, 8, device="cuda"))
    expected = runner.sampler(logits, batch)
    actual, counts, rejected = try_sample_forced_tokens(runner, batch, None)
    torch.testing.assert_close(actual.sampled_token_ids, expected.sampled_token_ids)
    torch.testing.assert_close(counts, expected.num_sampled)
    torch.testing.assert_close(rejected, expected.num_rejected)
    assert counts.tolist() == [0, 1, 1]
    # Changing the next step's graph-backed stop flags cannot overwrite outputs.
    runner.model._batch_should_continue.logical_not_()
    assert actual.sampled_token_ids.flatten().tolist() == [7, 9, 7]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logprobs": 1},
        {"logit_bias": {7: -100}},
        {"allowed_token_ids": [9]},
        {"min_tokens": 5, "stop_token_ids": [9]},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 1.0},
        {"frequency_penalty": 1.0},
        {"seed": None},
    ],
)
def test_constraints_fall_back_to_standard_sampler(kwargs):
    params = {"seed": 42, **kwargs}
    runner, batch = setup(SamplingParams(**params))
    assert try_sample_forced_tokens(runner, batch, None) is None


@pytest.mark.parametrize(
    "feature", ["grammar", "draft", "shard", "nans", "mask", "trace", "thinking", "bad_words", "stale_mask"]
)
def test_nonstandard_modes_fall_back(feature):
    runner, batch = setup()
    grammar = None
    if feature == "grammar":
        grammar = object()
    elif feature == "draft":
        batch.num_draft_tokens = 1
    elif feature == "shard":
        runner.batch_sharder = object()
    elif feature == "nans":
        runner.sampler.compute_nans = True
    elif feature == "mask":
        runner.sampler.return_sampling_mask = True
    elif feature == "trace":
        runner.sampler.trace_replay_state = object()
    elif feature == "thinking":
        runner.sampler.thinking_budget_state.enabled = True
    elif feature == "bad_words":
        runner.sampler.bad_words_state.num_bad_words.np[3] = 1
    elif feature == "stale_mask":
        runner.model._batch_should_continue = torch.ones(4, device="cuda", dtype=torch.bool)
    assert try_sample_forced_tokens(runner, batch, grammar) is None


pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
