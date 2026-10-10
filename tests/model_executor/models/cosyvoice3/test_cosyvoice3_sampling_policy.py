# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 request-owned host routing, independent of model weights or CUDA."""

import pytest
import torch
from vllm.sampling_params import SamplingParams

from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    _make_sampling_metadata,
    _make_talker_model,
)
from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _host_params(temperature=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    params = SamplingParams(
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    # Keep boundary values for the float32 routing comparison.
    params.temperature = temperature
    return params


@pytest.mark.parametrize("mixed", [False, True])
def test_host_policy_avoids_parameter_tensor_reads(monkeypatch, mixed):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[], []])
    metadata.all_random = not mixed
    temperatures = [0.0 if mixed else 1.0, 1.0]
    metadata.temperature = torch.tensor(temperatures)
    params = [_host_params(t) for t in temperatures]

    # No history => no rejection predicate read. The finite-logits check must
    # remain, but parameter reductions and mixed-temperature readback must not.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "any", lambda *a, **k: pytest.fail("device penalty probe"))
        patch.setattr(torch.Tensor, "tolist", lambda *a, **k: pytest.fail("greedy mask readback"))
        out = model.sample(torch.tensor([[float("-inf"), 2.0]] * 2), metadata, per_req_sampling_params=params)
    assert out.sampled_token_ids.tolist() == [[1], [1]]


@pytest.mark.parametrize("temperature", [0.0, 1e-6, 1e-5])
def test_host_policy_matches_float32_greedy_boundary(monkeypatch, temperature):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    metadata.all_random = False
    metadata.temperature = torch.tensor([temperature], dtype=torch.float32)
    assert float(metadata.temperature[0]) < model._sampling_eps
    monkeypatch.setattr(cosyvoice3, "random_sample", lambda *a, **k: pytest.fail("greedy must not advance RNG"))
    out = model.sample(torch.tensor([[0.0, 2.0]]), metadata, per_req_sampling_params=[_host_params(temperature)])
    assert out.sampled_token_ids.tolist() == [[1]]


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
@pytest.mark.parametrize("value", [0.0, 1e-50, -0.2, 0.5])
def test_host_policy_matches_tensor_penalty_routing(field, value):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[], []])
    metadata.frequency_penalties = torch.zeros(2)
    metadata.presence_penalties = torch.zeros(2)
    params = [_host_params(), _host_params()]
    setattr(params[1], field, value)
    tensor_field = "frequency_penalties" if field == "frequency_penalty" else "presence_penalties"
    getattr(metadata, tensor_field)[1] = value
    assert model._cosyvoice3_ras_enabled(metadata, params) == model._cosyvoice3_ras_enabled(metadata)


@pytest.mark.parametrize("reason", ["frequency", "presence", "logprobs", "bad_words", "all_greedy", "standard"])
def test_host_policy_preserves_generic_sampler_route(monkeypatch, reason):
    model = _make_talker_model()
    model.config.vocab_size = 5
    metadata = _make_sampling_metadata(output_token_ids=[[], []])
    params = [_host_params(), _host_params()]
    if reason == "frequency":
        params[1].frequency_penalty = 0.1
        metadata.frequency_penalties = torch.tensor([0.0, 0.1])
    elif reason == "presence":
        params[1].presence_penalty = -0.5
        metadata.presence_penalties = torch.tensor([0.0, -0.5])
    elif reason == "logprobs":
        metadata.max_num_logprobs = 1
    elif reason == "bad_words":
        metadata.bad_words_token_ids = {1: [[0]]}
    elif reason == "all_greedy":
        metadata.all_greedy = True
        metadata.all_random = False
        metadata.temperature = None
    else:
        monkeypatch.setattr(cosyvoice3, "cosyvoice3_standard_sampling", lambda config: True)

    def generic_sampler(*, logits, sampling_metadata):
        assert sampling_metadata is metadata
        if reason == "standard":
            assert logits.shape == (2, 2)
        else:
            assert logits.shape == (2, 5)
            assert torch.isneginf(logits[:, 2:]).all()
        return "generic"

    model._talker_sampler = generic_sampler
    monkeypatch.setattr(torch, "any", lambda *a, **k: pytest.fail("host route should not probe GPU penalties"))
    assert model.sample(torch.ones(2, 2), metadata, per_req_sampling_params=params) == "generic"


@pytest.mark.parametrize("host_params", [None, [], [None, None], [_host_params(), None], [_host_params()]])
def test_incomplete_host_policy_uses_tensor_routing(host_params):
    model = _make_talker_model()
    model.config.vocab_size = 2
    metadata = _make_sampling_metadata(output_token_ids=[[], []])
    # Missing host information must not hide a penalty on another row.
    metadata.presence_penalties = torch.tensor([0.0, 0.5])
    model._talker_sampler = lambda **kwargs: "generic"
    assert model.sample(torch.ones(2, 2), metadata, per_req_sampling_params=host_params) == "generic"


def test_host_policy_does_not_read_unused_penalty_buffers(monkeypatch):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]], repetition_penalty=1.0)
    metadata.no_penalties = True
    # InputBatch does not upload penalty buffers when no_penalties is true.
    metadata.frequency_penalties.fill_(float("nan"))
    metadata.presence_penalties.fill_(float("nan"))
    monkeypatch.setattr(torch, "any", lambda *a, **k: pytest.fail("unused penalty buffer"))
    out = model.sample(torch.tensor([[float("-inf"), 2.0]]), metadata, per_req_sampling_params=[_host_params()])
    assert out.sampled_token_ids.tolist() == [[1]]


def test_host_policy_keeps_invalid_logits_check_and_rng_state():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    generator = torch.Generator().manual_seed(12)
    metadata.generators = {0: generator}
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match="no finite logits"):
        model.sample(torch.full((1, 2), float("nan")), metadata, per_req_sampling_params=[_host_params()])
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize("repetition_penalty", [1.0, 1.0001])
def test_host_policy_real_input_batch_and_runner(monkeypatch, repetition_penalty):
    """Validate the host/GPU metadata contract through the actual V1 entry point."""
    from vllm.v1.worker import gpu_input_batch
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    monkeypatch.setattr(gpu_input_batch, "PIN_MEMORY", False)
    batch = InputBatch(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=4,
        block_sizes=[1],
        kernel_block_sizes=[1],
        max_num_blocks_per_req=[16],
        logitsprocs_need_output_token_ids=False,
    )
    runner = object.__new__(GPUARModelRunner)
    runner.model = _make_talker_model()
    runner.input_batch = batch
    runner.requests = {}
    runner.sampler = lambda **kwargs: pytest.fail("unexpected runner fallback")
    runner._apply_duplex_sampling = lambda *args: None

    def add(req_id, temperature, seed):
        request = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[0],
            mm_features=[],
            sampling_params=SamplingParams(
                temperature=temperature, seed=seed, top_k=1, top_p=0.8, repetition_penalty=repetition_penalty
            ),
            generator=torch.Generator().manual_seed(seed),
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[1],
        )
        runner.requests[req_id] = request
        batch.add_request(request)
        return request

    greedy = add("greedy", 0.0, 31)
    random = add("random", 1.0, 32)
    greedy_state = greedy.generator.get_state().clone()
    random_state = random.generator.get_state().clone()

    def run():
        batch.sampling_metadata = batch._make_sampling_metadata()
        # Metadata construction can use torch.any; only forbid parameter probes
        # during the actual sampler call, after InputBatch has normalized it.
        with monkeypatch.context() as patch:
            patch.setattr(torch, "any", lambda *a, **k: pytest.fail("device penalty probe"))
            out = runner._sample(torch.tensor([[float("-inf"), 2.0]] * batch.num_reqs), spec_decode_metadata=None)
        assert out.sampled_token_ids.tolist() == [[1]] * batch.num_reqs

    run()
    batch.swap_states(0, 1)
    run()
    assert torch.equal(greedy.generator.get_state(), greedy_state)
    assert not torch.equal(random.generator.get_state(), random_state)
    batch.remove_request("greedy")
    del runner.requests["greedy"]
    batch.condense()
    add("replacement", 0.7, 33)
    run()


def test_host_policy_benchmark_initializes_rng_in_fresh_process():
    """Pytest already imports Omni; a fresh CLI process must use the same RNG."""
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchmarks.tts.benchmark_cosyvoice3_b1 import load_runtime; "
            "r = load_runtime(); assert getattr(r.random_sample, '_omni_batched_seeded', False)",
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
    )
