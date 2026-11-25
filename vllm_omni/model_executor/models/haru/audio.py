# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/audio.py @ d3612b0.
"""Voice-reference audio helpers (mono float32 at 24 kHz inside Haru)."""

from __future__ import annotations

import io
import math

import numpy as np

SAMPLE_RATE = 24_000
_SILENCE_DBFS = -45.0
_SILENCE_MARGIN_SECONDS = 0.2


def decode_audio(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Audio file bytes (any format libsndfile reads) -> mono float32 at ``sample_rate``."""
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        waveform, source_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as error:
        raise ValueError(f"unsupported audio format: {error}") from error
    mono = waveform.mean(axis=1)
    if int(source_rate) != sample_rate:
        gcd = math.gcd(int(source_rate), sample_rate)
        mono = resample_poly(mono, sample_rate // gcd, int(source_rate) // gcd)
    return mono.astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cut leading and trailing audio quieter than -45 dBFS (20 ms RMS windows)."""
    window = sample_rate // 50
    count = audio.shape[0] // window
    if count == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: count * window].reshape(count, window) ** 2, axis=1) + 1e-12)
    voiced = np.flatnonzero(20.0 * np.log10(rms) > _SILENCE_DBFS)
    if voiced.size == 0:
        return audio[:0]
    margin = int(_SILENCE_MARGIN_SECONDS * sample_rate)
    start = max(0, int(voiced[0]) * window - margin)
    end = min(audio.shape[0], (int(voiced[-1]) + 1) * window + margin)
    return audio[start:end]


def prepare_reference(audio: np.ndarray, max_seconds: float = 180.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Voice reference clip: silence trimmed, at most the last ``max_seconds``."""
    audio = trim_silence(np.asarray(audio, dtype=np.float32), sample_rate)
    return audio[-int(max_seconds * sample_rate) :]
