# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real duplex PCM format routing and shared encoder metadata."""

from __future__ import annotations

import base64
import wave
from io import BytesIO

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio
from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("response_format", ["pcm16", "PCM16", "pcm"])
def test_duplex_pcm_alias_uses_real_raw_encoder_without_container_frames(monkeypatch, response_format):
    observed = []
    original = AudioMixin.create_audio

    def capture(instance, request):
        response = original(instance, request)
        observed.append(response)
        return response

    monkeypatch.setattr(AudioMixin, "create_audio", capture)
    samples = np.array([-1.2, -1, -0.5, -0.00001, 0, 0.00001, 0.5, 0.999, 1, 1.2], dtype=np.float32)
    encoded = encode_audio(torch.from_numpy(samples), 24000, response_format, None)
    decoded = base64.b64decode(encoded)
    actual = np.frombuffer(decoded, dtype="<i2")
    expected = np.clip(np.floor(samples.astype(np.float64) * 32768), -32768, 32767).astype("<i2")
    assert len(decoded) == 2 * samples.size and actual.size == samples.size
    np.testing.assert_array_equal(actual, expected)
    assert observed[0].media_type == "audio/pcm"
    assert observed[0].audio_metadata.format == "pcm"
    assert observed[0].audio_metadata.frame_count == samples.size
    assert observed[0].audio_metadata.sample_rate_hz == 24000
    assert observed[0].audio_metadata.channels == 1


def test_duplex_other_format_still_uses_real_wav_encoder():
    encoded = encode_audio(np.zeros(10, dtype=np.float32), 24000, "wav", 1.0)
    with wave.open(BytesIO(base64.b64decode(encoded))) as audio:
        assert audio.getnframes() == 10 and audio.getnchannels() == 1
        assert audio.getframerate() == 24000 and audio.getsampwidth() == 2
