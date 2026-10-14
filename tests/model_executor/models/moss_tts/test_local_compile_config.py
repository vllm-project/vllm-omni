# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deployment sampler settings override the legacy environment switch."""

import pytest
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts import modeling_moss_tts_local_depth as depth

pytestmark = [pytest.mark.core_model, pytest.mark.tts, pytest.mark.cpu]


@pytest.mark.parametrize(
    "configured,env,expected", [(None, "0", False), (None, "1", True), (True, "0", True), (False, "1", False)]
)
def test_sampler_compile_setting_and_idempotent_setup(monkeypatch, configured, env, expected):
    monkeypatch.setenv("VLLM_OMNI_MOSS_LOCAL_COMPILE_AUDIO_SAMPLER", env)
    monkeypatch.setattr(depth.current_omni_platform, "supports_torch_inductor", lambda: True)
    calls = []

    def compile_fn(fn, **kwargs):
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(depth.torch, "compile", compile_fn)
    model = depth.MossTTSLocalDepthTransformer(
        GPT2Config(n_embd=80, n_head=1, n_inner=160), compile_audio_sampler=configured
    )
    model.setup_compile()
    model.setup_compile()
    assert len(calls) == 1 + expected
    assert (model._compiled_audio_sampler is depth._sample_token) is expected
    sampler_calls = [kwargs for fn, kwargs in calls if fn is depth._sample_token]
    assert len(sampler_calls) == int(expected)
    if expected:
        assert sampler_calls[0] == {"fullgraph": True, "dynamic": True, "options": {"fallback_random": True}}


def test_sampler_compile_setting_respects_platform_support(monkeypatch):
    monkeypatch.setattr(depth.current_omni_platform, "supports_torch_inductor", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("compile called on unsupported platform")

    monkeypatch.setattr(depth.torch, "compile", forbidden)
    model = depth.MossTTSLocalDepthTransformer(GPT2Config(n_embd=80, n_head=1, n_inner=160), compile_audio_sampler=True)
    model.setup_compile()
    assert model._compiled_audio_sampler is None
