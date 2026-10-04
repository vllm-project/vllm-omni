# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Duplex output audio: numpy writes PCM and WAV with exactly soundfile's bytes, soundfile does the rest."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.entrypoints.duplex import audio_encoding
from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _fresh_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_encoding, "_pcm16_mode_by_key", {})


def _chunks() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    chunks = [(rng.standard_normal(size) * 0.4).astype(np.float32) for size in (1, 1920, 9600, 9601)]
    loud = rng.uniform(-3.0, 3.0, 4800).astype(np.float32)  # clips on both sides
    # Every half-step of the int16 range: where libsndfile 1.2's int32 shift
    # and the rounding quantizers of older builds disagree.
    ties = ((np.arange(-4000, 4000) + 0.5) / 32768.0).astype(np.float32)
    # One float32 ulp outside small whole steps: libsndfile 1.2 carries these
    # to the step, a plain floor(x * 32768) does not.
    steps = (np.arange(1, 128) / 32768.0).astype(np.float32)
    carries = np.concatenate([np.nextafter(steps, np.float32(0.0)), np.nextafter(-steps, np.float32(-1.0))])
    return [*chunks, loud, np.zeros(960, dtype=np.float32), ties, carries]


@pytest.mark.parametrize(("fmt", "rate"), [("pcm", 24000), ("WAV", 16000)])
def test_pcm_and_wav_bytes_equal_soundfile(fmt: str, rate: int) -> None:
    for chunk in _chunks():
        expected = audio_encoding._encode_with_soundfile(chunk, rate, fmt, 1.0)
        assert encode_audio(chunk, rate, fmt, None) == expected
    # The fast path is what produced them: the probe matched soundfile.
    assert audio_encoding._pcm16_mode(fmt.lower(), rate) is not None


def test_only_pcm_and_wav_at_normal_speed_skip_soundfile(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunks()[2]
    audio_encoding._pcm16_mode("pcm", 24000)  # calibrate before counting soundfile calls
    calls: list[tuple[str, float]] = []
    original = audio_encoding._encode_with_soundfile

    def counting(audio: np.ndarray, rate: int, fmt: str, speed: float) -> str:
        calls.append((fmt, speed))
        return original(audio, rate, fmt, speed)

    monkeypatch.setattr(audio_encoding, "_encode_with_soundfile", counting)
    encode_audio(chunk, 24000, "pcm", 1.0)
    assert calls == []
    encode_audio(chunk, 24000, "pcm", 1.5)
    assert calls == [("pcm", 1.5)]


def test_non_finite_samples_are_left_to_soundfile() -> None:
    chunk = _chunks()[1].copy()
    chunk[5] = np.nan
    chunk[9] = np.inf
    assert encode_audio(chunk, 24000, "pcm", 1.0) == audio_encoding._encode_with_soundfile(chunk, 24000, "pcm", 1.0)


def test_a_probe_mismatch_keeps_soundfile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_encoding, "_PCM16_QUANTIZERS", {"zeros": lambda samples: np.zeros(samples.shape, "<i2")})
    chunk = _chunks()[2]

    assert encode_audio(chunk, 24000, "wav", 1.0) == audio_encoding._encode_with_soundfile(chunk, 24000, "wav", 1.0)
    assert audio_encoding._pcm16_mode_by_key == {("wav", 24000): None}


def test_the_int32_mode_writes_libsndfile_12_bytes() -> None:
    """libsndfile 1.2 rounds x * 2**31 half to even into an int32 and keeps its high 16 bits."""
    for chunk in _chunks():
        wide = np.rint(chunk.astype(np.float64) * 2.0**31)
        expected = (np.clip(wide, -(2.0**31), 2.0**31 - 1).astype(np.int64) >> 16).astype("<i2")
        encoded = audio_encoding._encode_pcm16(chunk, 24000, "pcm", "int32_high")
        assert encoded == base64.b64encode(expected.tobytes()).decode("utf-8")
