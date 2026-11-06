# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Construction-time install contract for length bucketing.

Review feedback (gcanlin): the install belongs inside the model that owns the
encoder, not as a runner hook. These tests pin that contract:

- constructing a verified model installs the bucketed wrapper on the model's
  own ``encoder`` (only ``module.forward`` — ``embed_multimodal`` is never
  touched) and arms the feature when ``VLLM_OMNI_AUDIO_BUCKET_SECONDS`` is
  set;
- without the env var the constructor still installs the pass-through wrapper
  (the model's ``lengths`` call contract must stay valid) but the feature
  stays disarmed — the forward's behavior is unchanged;
- a stage without a bucketable encoder (Token2Wav) never sees the installer,
  whatever the env says — it has no install call by construction.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.covo_audio import covo_audio_llm
from vllm_omni.model_executor.models.step_audio2 import step_audio2_thinker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeVllmConfig:
    """Just enough of VllmConfig for a model constructor."""

    def __init__(self, model_type: str) -> None:
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            multimodal_config=SimpleNamespace(),
        )

    def with_hf_config(self, hf_config, architectures=None):
        return self


def _stub_lm():
    return SimpleNamespace(make_empty_intermediate_tensors=lambda: None)


def test_covo_llm_construction_installs_bucketed_encoder(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "0.4,1,2")
    monkeypatch.setattr(covo_audio_llm, "init_vllm_registered_model", lambda **kwargs: _stub_lm())
    monkeypatch.setattr(covo_audio_llm, "WhisperEncoder", lambda cfg: torch.nn.Linear(4, 4))
    monkeypatch.setattr(covo_audio_llm, "AudioAdapter", lambda *args, **kwargs: torch.nn.Linear(4, 4))

    vllm_config = _FakeVllmConfig("covo_audio")
    vllm_config.model_config.hf_config.encoder_config = None
    vllm_config.model_config.hf_config.whisper_feats_dim = 16
    vllm_config.model_config.hf_config.llm_config = SimpleNamespace(hidden_size=8)
    vllm_config.model_config.hf_config.adapter_downsample = 2

    model = covo_audio_llm.CovoAudioLLMForConditionalGeneration(vllm_config=vllm_config)

    assert getattr(model, "_bucketed_installed", False), "model constructor must install the bucketed encoder"
    wrapper = model.encoder.forward  # the installer assigns the _BucketedEncoder here
    assert type(wrapper).__name__ == "_BucketedEncoder", "the model's own encoder must be wrapped"
    assert wrapper._active is True, "verified model + env must arm the feature"
    # No wrapper state: per-item lengths ride the encoder call itself.
    assert not hasattr(wrapper, "_lengths")
    # embed_multimodal is the model's own bound method — never wrapped.
    assert getattr(model.embed_multimodal, "__self__", None) is model


def test_step_audio2_thinker_construction_installs_bucketed_encoder(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "0.4,1,2")
    monkeypatch.setattr(step_audio2_thinker, "init_vllm_registered_model", lambda **kwargs: _stub_lm())

    vllm_config = _FakeVllmConfig("step_audio_2")
    # Tiny real AudioEncoder/Adaptor: the contract is the install placement,
    # but the installer wraps whatever nn.Module is named "encoder".
    vllm_config.model_config.hf_config.audio_encoder_config = {
        "n_mels": 4,
        "n_audio_ctx": 8,
        "n_audio_state": 8,
        "n_audio_head": 2,
        "n_audio_layer": 1,
        "kernel_size": 3,
        "adapter_stride": 2,
        "llm_dim": 8,
    }

    model = step_audio2_thinker.StepAudio2ThinkerForConditionalGeneration(vllm_config=vllm_config)

    assert getattr(model, "_bucketed_installed", False), "model constructor must install the bucketed encoder"
    wrapper = model.encoder.forward
    assert type(wrapper).__name__ == "_BucketedEncoder", "the model's own encoder must be wrapped"
    assert wrapper._active is True, "verified model + env must arm the feature"
    assert not hasattr(wrapper, "_lengths")
    assert getattr(model.embed_multimodal, "__self__", None) is model


def test_construction_without_env_installs_pass_through_only(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", raising=False)
    monkeypatch.setattr(covo_audio_llm, "init_vllm_registered_model", lambda **kwargs: _stub_lm())
    monkeypatch.setattr(covo_audio_llm, "WhisperEncoder", lambda cfg: torch.nn.Linear(4, 4))
    monkeypatch.setattr(covo_audio_llm, "AudioAdapter", lambda *args, **kwargs: torch.nn.Linear(4, 4))

    vllm_config = _FakeVllmConfig("covo_audio")
    vllm_config.model_config.hf_config.encoder_config = None
    vllm_config.model_config.hf_config.whisper_feats_dim = 16
    vllm_config.model_config.hf_config.llm_config = SimpleNamespace(hidden_size=8)
    vllm_config.model_config.hf_config.adapter_downsample = 2

    model = covo_audio_llm.CovoAudioLLMForConditionalGeneration(vllm_config=vllm_config)

    # The wrapper is installed (the model's ``lengths`` call contract must
    # stay valid) but the feature is disarmed: the forward is a pass-through.
    assert getattr(model, "_bucketed_installed", False)
    wrapper = model.encoder.forward
    assert type(wrapper).__name__ == "_BucketedEncoder"
    assert wrapper._active is False, "feature must stay off without the env"
    assert getattr(model.embed_multimodal, "__self__", None) is model


def test_token2wav_stage_never_installs_bucketed_encoder(monkeypatch):
    """A stage without a bucketable encoder must never see the installer —
    the Token2Wav stage carries no install call, so no DAC/tts tokenizer
    encoder can be mis-wrapped regardless of the env."""
    try:
        from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
            StepAudio2Token2WavForConditionalGeneration,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"Step-Audio2 Token2Wav extras not installed ({exc.name})")

    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "0.4,1,2")

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(token2wav_path="/tmp/fake-token2wav", token2wav_float16=False),
            model="/tmp/fake-model",
        ),
        device_config=SimpleNamespace(device="cpu"),
    )
    model = StepAudio2Token2WavForConditionalGeneration(vllm_config=vllm_config)

    assert not getattr(model, "_bucketed_installed", False)
    assert not any(type(module.forward).__name__ == "_BucketedEncoder" for module in model.modules())
