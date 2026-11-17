# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Nemotron VoiceChat code2wav emits the canonical sample-rate payload shape."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_code2wav import (
    NemotronVoiceChatCode2Wav,
)
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_model() -> NemotronVoiceChatCode2Wav:
    model = object.__new__(NemotronVoiceChatCode2Wav)
    nn.Module.__init__(model)
    # Unusual rate proves the emission follows the codec-config-derived
    # attribute instead of the 22050 fallback literal.
    model._sample_rate = UNUSUAL_RATE
    # forward() only needs a parameter-bearing module for the device.
    model.audio_codec = nn.Linear(1, 1)
    return model


def test_dummy_profile_step_emits_canonical_rate() -> None:
    model = _bare_model()

    # Dummy-profile requests decode nothing but keep their rate entry.
    output = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        runtime_additional_information=[{"meta": {"nemotron_voicechat_dummy_profile": True}}],
    )

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["model_outputs"]) == 1
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
