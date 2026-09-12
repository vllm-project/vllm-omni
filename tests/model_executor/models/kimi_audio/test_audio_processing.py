# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for Whisper preprocessing boundaries."""

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from vllm_omni.model_executor.models.kimi_audio.audio_processing import (
    prepare_whisper_inputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]


@pytest.fixture(scope="module")
def feature_extractor():
    return WhisperFeatureExtractor(feature_size=128)


@pytest.mark.parametrize("samples,expected", [(1280, (1,)), (1281, (2,)), (480000, (375,)), (480001, (375, 1))])
def test_real_feature_extractor_preserves_audio_boundaries(feature_extractor, samples, expected):
    waveform = np.zeros(samples, dtype=np.float32)
    inputs = prepare_whisper_inputs(waveform, feature_extractor, sampling_rate=16000)
    assert inputs.token_lengths == expected
    assert inputs.input_features.shape == (len(expected), 128, 3000)
    assert inputs.input_features.dtype == torch.float32
    assert np.count_nonzero(waveform) == 0


def test_recording_after_thirty_seconds_is_not_truncated(feature_extractor):
    waveform = np.zeros(480000 + 16000, dtype=np.float32)
    waveform[480000:] = np.sin(np.arange(16000) * (2 * np.pi * 440 / 16000))
    inputs = prepare_whisper_inputs(waveform, feature_extractor, sampling_rate=16000)
    assert inputs.token_lengths == (375, 13)
    assert not torch.equal(inputs.input_features[0], inputs.input_features[1])
