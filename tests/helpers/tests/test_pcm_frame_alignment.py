# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject incomplete PCM frames without losing failure diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers.assertions import _assert_pcm_int16_speech_hnr, assert_audio_speech_response

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _partial_frame_pcm(channels: int) -> bytes:
    if channels == 1:
        return bytes(3)
    # Otherwise valid stereo speech-like audio with one extra int16 sample.
    samples = 10_000 * np.sin(2 * np.pi * 220 * np.arange(24_000) / 48_000)
    return np.repeat(samples, 2).astype("<i2").tobytes() + bytes(2)


@pytest.mark.parametrize("channels", [1, 2])
def test_pcm_hnr_rejects_partial_frame(channels):
    with pytest.raises(AssertionError, match="aligned to int16"):
        _assert_pcm_int16_speech_hnr(_partial_frame_pcm(channels), sr=48_000, channels=channels)


def test_stereo_pcm_frame_failure_retains_original_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_OMNI_FAILED_SPEECH_AUDIO_DIR", str(tmp_path))
    pcm_bytes = _partial_frame_pcm(2)
    response = SimpleNamespace(success=True, audio_bytes=pcm_bytes, audio_format="audio/pcm")

    with pytest.raises(AssertionError, match="aligned to int16 frames"):
        assert_audio_speech_response(
            response,
            {"response_format": "pcm", "expected_sample_rate": 48_000, "transcript_pcm_channels": 2},
            "advanced_model",
        )

    (retained_audio,) = tmp_path.glob("*.pcm")
    assert retained_audio.read_bytes() == pcm_bytes
    metadata = json.loads(retained_audio.with_suffix(".json").read_text())
    assert metadata["sample_rate"] == 48_000
    assert metadata["audio_format"] == "pcm"
    assert "aligned to int16 frames" in metadata["assertion"]
