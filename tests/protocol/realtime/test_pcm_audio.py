# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression coverage for PCM on the live protocol conversion path."""

import io
import wave

import numpy as np
import pybase64 as base64
import pytest

from vllm_omni.protocol.duplex import convert_input_audio_with_rate, pcm_f32le_payload_to_wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("source_rate_hz", [16_000, 24_000, 48_000])
def test_float_pcm_is_normalized_to_the_engine_rate(source_rate_hz):
    samples = np.full(source_rate_hz // 10, 0.25, dtype="<f4")
    audio = base64.b64encode(samples.tobytes()).decode("ascii")

    encoded, fmt, rate = convert_input_audio_with_rate(audio, "pcm_f32le", sample_rate_hz=source_rate_hz)

    decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
    assert fmt == "pcm_f32le"
    assert rate == 16_000
    assert decoded.size == 1600
    assert np.isfinite(decoded).all()
    np.testing.assert_allclose(decoded[100:-100], 0.25, atol=0.01)


def test_float_pcm_wav_wrap_preserves_samples_and_clips_to_pcm16():
    samples = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype="<f4")
    audio = base64.b64encode(samples.tobytes()).decode("ascii")

    encoded, fmt, rate = pcm_f32le_payload_to_wav(audio, 16_000)

    assert (fmt, rate) == ("wav", 16_000)
    with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, rate)
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(decoded, [-32768, -16384, 0, 16384, 32767])


@pytest.mark.parametrize("raw", [b"", b"\0", np.array([float("nan")], dtype="<f4").tobytes()])
def test_float_pcm_wav_wrap_rejects_invalid_samples(raw):
    with pytest.raises(ValueError):
        pcm_f32le_payload_to_wav(base64.b64encode(raw).decode("ascii"), 16_000)


def test_float_pcm_wav_wrap_rejects_invalid_base64():
    with pytest.raises(ValueError, match="base64"):
        pcm_f32le_payload_to_wav("not!base64", 16_000)
