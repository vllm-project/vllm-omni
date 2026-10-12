# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Determined text tokens retain the upstream sampling/counting contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from tests.model_executor.models.moss_tts.test_local_model_state import _batch, _state
from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = pytest.mark.core_model


@pytest.mark.cuda
@pytest.mark.parametrize("seed", [None, 17])
@pytest.mark.parametrize("temperature", [0.0, 1.7])
def test_matches_real_sampler_for_mixed_prefill_decode_stop_and_reordering(seed, temperature):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    reqs = RequestState(5, 32, 16, 0, 8, device)
    sampler = Sampler(VllmConfig(), 5, 8, device, reqs)
    for i, length in enumerate([4, 1, 3]):
        reqs.add_request(str(i), length, [2] * length, 0, 8)
        slot = reqs.req_id_to_index[str(i)]
        sampler.add_request(slot, SamplingParams(temperature=temperature, top_k=3, top_p=0.8, seed=seed))
    reqs.apply_staged_writes()
    sampler.apply_staged_writes()
    state = _state(MossLocalModelState, device)
    state._direct_tokens = True
    state.model._batch_state = None
    batch = _batch(device, [2, 4, 3], [1, 2, 1])
    batch.seq_lens = torch.tensor([5, 2, 2], device=device, dtype=torch.int32)
    batch.seq_lens_cpu_upper_bound = torch.tensor([5, 2, 2], dtype=torch.int32)
    batch.cu_num_logits_np = np.arange(4, dtype=np.int32)
    batch.cu_num_logits = torch.tensor(batch.cu_num_logits_np, device=device)
    batch.expanded_idx_mapping = batch.idx_mapping
    batch.expanded_local_pos = torch.zeros(3, dtype=torch.int32, device=device)
    batch.logits_indices = batch.query_start_loc[1:].long() - 1
    batch.positions = torch.arange(batch.num_tokens, device=device)
    state.model._batch_should_continue = torch.tensor([False, True, True], device=device)
    with torch.inference_mode():
        logits = state.model.compute_logits(torch.ones(3, 4, device=device))
        expected = sampler(logits, batch)
        actual = state.sample_determined_tokens(batch, sampler)
    assert actual is not None
    for field in ["sampled_token_ids", "num_sampled", "num_rejected"]:
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    assert actual.num_sampled.tolist() == [1, 0, 1]
    snapshot = actual.sampled_token_ids.clone()
    state.model._batch_should_continue.logical_not_()
    torch.testing.assert_close(actual.sampled_token_ids, snapshot, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "constraint",
    [
        {"min_tokens": 2, "stop_token_ids": [1]},
        {"allowed_token_ids": [1, 2]},
        {"logit_bias": {1: -10.0}},
    ],
)
def test_real_sampler_constraints_disable_determined_tokens(constraint):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    reqs = RequestState(5, 32, 16, 0, 8, device)
    sampler = Sampler(VllmConfig(), 5, 8, device, reqs)
    reqs.add_request("constrained", 1, [2], 0, 8)
    slot = reqs.req_id_to_index["constrained"]
    sampler.add_request(slot, SamplingParams(**constraint))
    state = _state(MossLocalModelState, device)
    state._direct_tokens = True
    # Use the actual upstream request registration, including its combined
    # min-token/allowed-ID/bias flag, rather than mocking that flag ourselves.
    assert state.sample_determined_tokens(SimpleNamespace(idx_mapping_np=np.array([slot])), sampler) is None


@pytest.mark.cpu
@pytest.mark.parametrize(
    "feature", ["logprobs", "bias", "penalty", "bad_words", "custom", "thinking", "trace", "mask", "nans"]
)
def test_distribution_features_fall_back_before_using_gpu(feature):
    sampler = object.__new__(Sampler)
    sampler.compute_nans = feature == "nans"
    sampler.return_sampling_mask = feature == "mask"
    sampler.trace_replay_state = object() if feature == "trace" else None
    sampler.get_logprobs_dims = lambda rows: (1, 0) if feature == "logprobs" else None
    bias = object.__new__(LogitBiasState)
    bias.use_logit_bias = np.array([feature == "bias"])
    penalties = object.__new__(PenaltiesState)
    penalties.use_penalty = np.array([feature == "penalty"])
    bad_words = object.__new__(BadWordsState)
    bad_words.num_bad_words = SimpleNamespace(np=np.array([feature == "bad_words"]))
    sampler.logits_processors = [bias, penalties, bad_words] + ([object()] if feature == "custom" else [])
    sampler.thinking_budget_state = SimpleNamespace(enabled=True, use_thinking_budget=np.array([feature == "thinking"]))
    state = object.__new__(MossLocalModelState)
    state._direct_tokens = True
    assert state.sample_determined_tokens(SimpleNamespace(idx_mapping_np=np.array([0])), sampler) is None


@pytest.mark.cpu
@pytest.mark.parametrize("feature", ["grammar", "sharding", "pipeline", "speculative", "empty", "declined"])
def test_runner_retains_upstream_path_for_unsupported_execution(mocker, feature):
    hook = mocker.Mock(return_value=None)

    class State:
        def sample_determined_tokens(self, batch, sampler):
            return hook(batch, sampler)

    runner = object.__new__(OmniARModelRunner)
    runner.model_state = State()
    runner.sampler = object()
    runner.batch_sharder = object() if feature == "sharding" else None
    runner.pp_handler = object() if feature == "pipeline" else None
    batch = SimpleNamespace(num_draft_tokens=int(feature == "speculative"), num_reqs=0 if feature == "empty" else 1)
    grammar = object() if feature == "grammar" else None
    fallback = mocker.patch.object(OmniGPUModelRunner, "sample", return_value="upstream")
    hidden = object()
    assert runner.sample(hidden, batch, grammar) == "upstream"
    fallback.assert_called_once_with(hidden, batch, grammar)
    assert hook.call_count == int(feature == "declined")
