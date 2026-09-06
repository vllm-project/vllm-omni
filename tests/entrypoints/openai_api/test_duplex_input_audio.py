# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for audio normalization used by the turn-based duplex fallback."""

import base64
import io
import wave

import numpy as np

from vllm_omni.entrypoints.duplex.audio import pcm_f32le_payload_to_wav
from vllm_omni.entrypoints.duplex.chat_fallback import _audio_metadata


def test_pcm_f32le_payload_is_wrapped_as_wav():
    samples = np.array([0.0, 0.5, -0.5, 1.0], dtype="<f4")
    payload = base64.b64encode(samples.tobytes()).decode("ascii")

    encoded, audio_format, sample_rate_hz = pcm_f32le_payload_to_wav(payload, 16_000)

    assert audio_format == "wav"
    assert sample_rate_hz == 16_000
    with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == len(samples)


def test_wav_audio_duration_is_reported_in_milliseconds():
    wav = io.BytesIO()
    with wave.open(wav, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\0\0" * 16_000)

    encoded = base64.b64encode(wav.getvalue()).decode("ascii")
    assert _audio_metadata(encoded, fmt="wav") == (1000, 16_000)


def test_pcm_audio_without_sample_rate_does_not_guess_metadata():
    encoded = base64.b64encode(b"\0\0" * 24_000).decode("ascii")

    assert _audio_metadata(encoded, fmt="pcm") == (0, None)
