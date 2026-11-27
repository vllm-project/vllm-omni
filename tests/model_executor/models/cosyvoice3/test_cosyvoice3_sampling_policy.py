# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 request-owned host routing, independent of model weights or CUDA."""

import pytest

pytest.importorskip(
    "vllm.v1.worker.gpu_input_batch", reason="CosyVoice V1-only sampling contract is retired; V2 sampling tests remain"
)
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


@pytest.mark.parametrize(
    "temperature", [0.0, 1e-5, torch.nextafter(torch.tensor(1e-5), torch.tensor(float("inf"))).item()]
)
def test_host_policy_matches_float32_greedy_boundary(monkeypatch, temperature):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    metadata.all_random = False
    metadata.temperature = torch.tensor([temperature], dtype=torch.float32)
    greedy = float(metadata.temperature[0]) < model._sampling_eps

    def nucleus(*args, **kwargs):
        assert not greedy, "greedy must not advance RNG"
        return torch.tensor([0]), None

    monkeypatch.setattr(cosyvoice3, "nucleus_sample", nucleus)
    out = model.sample(torch.tensor([[0.0, 2.0]]), metadata, per_req_sampling_params=[_host_params(temperature)])
    assert out.sampled_token_ids.tolist() == [[1 if greedy else 0]]


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
@pytest.mark.parametrize("value", [0.0, 1e-50, 0.5])
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


@pytest.mark.parametrize("host_params", [None, [_host_params()], [_host_params(), None]])
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
