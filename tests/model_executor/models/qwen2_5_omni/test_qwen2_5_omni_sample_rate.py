# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav emits a config-derived sampling rate under ``SAMPLE_RATE_KEY``."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.metrics.definitions import DEFAULT_AUDIO_SAMPLE_RATE
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# An unusual rate proves the emission reads the config instead of a literal.
UNUSUAL_RATE = 22050


def _bare_code2wav_model(token2wav_config: object) -> Qwen2_5OmniForConditionalGeneration:
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"
    model.token2wav_config = token2wav_config
    return model


def test_code2wav_emits_config_derived_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_code2wav_model(SimpleNamespace(sample_rate=UNUSUAL_RATE))
    monkeypatch.setattr(model, "generate_audio", lambda code, speaker: torch.zeros(4))

    out = model.forward(torch.tensor([5, 6, 7]), torch.arange(3))

    payload = out.multimodal_outputs
    assert SAMPLE_RATE_KEY in payload
    assert get_sampling_rate(payload) == UNUSUAL_RATE


def test_code2wav_without_token2wav_config_uses_shared_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The transformers token2wav config chain carries no rate field; the
    # resolver's shared default applies (same behavior as qwen3_omni).
    model = _bare_code2wav_model(None)
    monkeypatch.setattr(model, "generate_audio", lambda code, speaker: torch.zeros(4))

    out = model.forward(torch.tensor([5, 6, 7]), torch.arange(3))

    assert get_sampling_rate(out.multimodal_outputs) == DEFAULT_AUDIO_SAMPLE_RATE
