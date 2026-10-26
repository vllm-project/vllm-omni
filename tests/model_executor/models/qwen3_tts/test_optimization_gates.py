# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only optimizations leave unsupported platforms and samplers upstream."""

import pytest
import torch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _base_sampler():
    base = object.__new__(Sampler)
    base.req_states = object.__new__(RequestState)
    base.req_states.max_num_reqs = 8
    base.sampling_states = object.__new__(SamplingStates)
    base.sampling_states.vocab_size = 3072
    base.compute_nans = False
    base.use_fp64_gumbel = False
    base.trace_replay_state = None
    base.return_sampling_mask = False
    base.logits_processors = [object.__new__(c) for c in (LogitBiasState, PenaltiesState, BadWordsState)]
    return base


@pytest.mark.parametrize("reason", ["platform", "fp64", "plugin"])
def test_disabled_custom_sampler_preserves_stock_registration(monkeypatch, reason):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: reason != "platform")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    base = _base_sampler()
    if reason == "fp64":
        base.use_fp64_gumbel = True
    if reason == "plugin":
        base.logits_processors.append(object())
    talker = object.__new__(Qwen3TTSTalkerForConditionalGeneration)
    assert talker.mrv2_custom_sampler(base) is None


def test_cuda_sampler_registers_when_supported(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    talker = object.__new__(Qwen3TTSTalkerForConditionalGeneration)
    registered = talker.mrv2_custom_sampler(_base_sampler())
    assert registered is not None
    sampler, rejection = registered
    assert sampler.enabled and rejection is None


def test_rocm_does_not_probe_cuda_pdl_capability(monkeypatch):
    from vllm_omni.model_executor.models.qwen3_tts.fused_code_predictor import _pdl_supported
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: pytest.fail("NVIDIA-only capability probe"))
    assert not _pdl_supported(torch.device("cuda"))
