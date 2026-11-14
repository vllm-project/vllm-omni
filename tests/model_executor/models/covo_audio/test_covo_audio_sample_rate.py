# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav emits a config-derived sampling rate under ``SAMPLE_RATE_KEY``."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.covo_audio.covo_audio import CovoAudioForConditionalGeneration
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# An unusual rate proves the emission reads the code2wav config
# (``CovoAudioCode2WavConfig.wav_input_sr``) instead of a literal.
UNUSUAL_RATE = 22050


def _bare_code2wav_model() -> CovoAudioForConditionalGeneration:
    model = object.__new__(CovoAudioForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"
    model.code2wav = SimpleNamespace(sample_rate=UNUSUAL_RATE)
    return model


def test_code2wav_emits_config_derived_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_code2wav_model()
    monkeypatch.setattr(model, "generate_audio", lambda code: torch.zeros(1, 40))

    out = model.forward(torch.tensor([5, 6, 7]), torch.arange(3))

    payload = out.multimodal_outputs
    assert SAMPLE_RATE_KEY in payload
    assert get_sampling_rate(payload) == UNUSUAL_RATE


def test_text_only_sentinel_emits_sample_rate() -> None:
    model = _bare_code2wav_model()

    out = model.forward(torch.tensor([-1]), torch.arange(1))

    payload = out.multimodal_outputs
    assert get_sampling_rate(payload) == UNUSUAL_RATE
    # 0.1s of silence at the emitted rate.
    assert payload["model_outputs"].shape == (1, UNUSUAL_RATE // 10)
