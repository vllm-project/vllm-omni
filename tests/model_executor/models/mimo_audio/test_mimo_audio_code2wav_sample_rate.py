# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav emits the model's sampling rate under ``SAMPLE_RATE_KEY``."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import (
    MiMoAudioToken2WavForConditionalGenerationVLLM,
)
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# An unusual rate proves the emission reads the model attribute (itself
# derived from the audio tokenizer config) instead of a literal.
UNUSUAL_RATE = 22050


def _bare_model() -> MiMoAudioToken2WavForConditionalGenerationVLLM:
    model = object.__new__(MiMoAudioToken2WavForConditionalGenerationVLLM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.sample_rate = UNUSUAL_RATE
    return model


def test_empty_codes_emits_sample_rate() -> None:
    model = _bare_model()

    out = model.forward(codes=torch.zeros(0, dtype=torch.long))

    payload = out.multimodal_outputs
    assert SAMPLE_RATE_KEY in payload
    assert get_sampling_rate(payload) == UNUSUAL_RATE


def test_make_omni_output_wraps_raw_tensor_with_sample_rate() -> None:
    model = _bare_model()

    out = model.make_omni_output(torch.zeros(4))

    assert get_sampling_rate(out.multimodal_outputs) == UNUSUAL_RATE
