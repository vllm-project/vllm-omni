# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, sample_with_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _output():
    counts, rejected = torch.tensor([1, 0]), torch.tensor([0, 0])
    return SamplerOutput(torch.tensor([[7], [0]]), None, None, counts, rejected), counts, rejected


@pytest.mark.parametrize("wrapped", [False, True])
def test_standard_sampling_keeps_counts_grammar_and_lifecycle(wrapped):
    base = Mock()
    base.req_states = object()
    sampler = OmniSampler(base) if wrapped else base
    expected = _output()
    standard = Mock(return_value=expected)
    hidden, batch, grammar = torch.zeros(2, 4), object(), object()
    result = sample_with_output(sampler, standard, hidden, batch, base.req_states, grammar)
    assert result.sampler_output is expected[0]
    assert result.num_sampled is expected[1] and result.num_rejected is expected[2]
    assert result.multimodal_outputs is None
    standard.assert_called_once_with(hidden, batch, grammar)
    sampler.add_request(3, "request")
    sampler.apply_staged_writes()
    base.add_request.assert_called_once_with(3, "request")
    base.apply_staged_writes.assert_called_once_with()
    assert sampler.req_states is base.req_states
    sampler(hidden, batch)
    base.assert_called_once_with(hidden, batch)


@pytest.mark.parametrize("payload", [{}, {"codes.audio": torch.tensor([[1, 2]])}])
def test_audio_sampler_uses_registered_adapter_without_standard_resampling(payload):
    expected = OmniSamplingOutput(*_output(), multimodal_outputs=payload)

    class AudioSampler(OmniSampler):
        def sample_step(self, hidden, batch, states, grammar, standard):
            assert states is self.req_states
            if grammar is not None:
                raise ValueError("unsupported grammar")
            return expected

    sampler = AudioSampler(SimpleNamespace(req_states=object()))
    standard = Mock(side_effect=AssertionError("must not resample"))
    result = sample_with_output(sampler, standard, torch.zeros(2, 4), object(), sampler.req_states, None)
    assert result is expected and result.multimodal_outputs is payload
    with pytest.raises(ValueError, match="unsupported grammar"):
        sample_with_output(sampler, standard, torch.zeros(2, 4), object(), sampler.req_states, object())
    standard.assert_not_called()


def test_adapter_preserves_stock_sampler_staged_write_fast_path():
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    assert OmniSampler(object.__new__(Sampler)).omni_static_staged_writes
    assert not OmniSampler(SimpleNamespace()).omni_static_staged_writes
    assert OmniSampler(SimpleNamespace(omni_static_staged_writes=True)).omni_static_staged_writes


@pytest.mark.parametrize("model_warmup", [False, True])
def test_ar_runner_uses_model_state_sampling_warmup_or_upstream(monkeypatch, model_warmup):
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    calls = []

    class State:
        def warmup_sampling(self, hidden, requests):
            calls.append((hidden, requests))

    runner = object.__new__(OmniARModelRunner)
    runner.model_state = State() if model_warmup else SimpleNamespace()
    runner.req_states = object()
    upstream = Mock()
    monkeypatch.setattr(OmniGPUModelRunner, "_dummy_sampler_run", upstream)
    hidden = torch.zeros(2, 8)
    runner._dummy_sampler_run(hidden)
    assert len(calls) == int(model_warmup)
    assert upstream.call_count == int(not model_warmup)
    if model_warmup:
        assert calls[0][0] is hidden and calls[0][1] is runner.req_states
