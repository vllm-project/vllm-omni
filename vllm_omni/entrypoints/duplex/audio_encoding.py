# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audio encoding for duplex model output (entrypoint layer, injected into ``DuplexOmniEngine``)."""

from __future__ import annotations

import struct
from collections.abc import Callable

import numpy as np
import pybase64 as base64
from vllm.logger import init_logger

logger = init_logger(__name__)

# The largest float32 below 2**31: where x * 2**31 saturates the int32 range.
_FLOAT32_BELOW_2_31 = float(np.nextafter(np.float32(2.0**31), np.float32(0.0)))


def _pcm16_int32_high(samples: np.ndarray) -> np.ndarray:
    """libsndfile 1.2: round ``x * 2**31`` half to even into an int32, keep its high 16 bits."""
    # Scaling by a power of two is exact, so rint rounds the value libsndfile rounds.
    scaled = samples * np.float32(2.0**31)
    np.clip(scaled, -(2.0**31), _FLOAT32_BELOW_2_31, out=scaled)
    return (np.rint(scaled, out=scaled).astype(np.int32) >> 16).astype("<i2")


def _pcm16_rint(scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """Older libsndfile builds: clip ``x * scale`` to int16 and round it half to even (``lrintf``)."""

    def quantize(samples: np.ndarray) -> np.ndarray:
        scaled = samples * np.float32(scale)
        np.clip(scaled, -32768.0, 32767.0, out=scaled)
        return np.rint(scaled, out=scaled).astype("<i2")

    return quantize


def _pcm16_floor_f64(samples: np.ndarray) -> np.ndarray:
    """``AudioMixin``'s own RAW writer: ``floor(x * 32768)`` in float64, clipped to int16."""
    return np.clip(np.floor(samples.astype(np.float64) * 32768.0), -32768, 32767).astype("<i2")


# Float-to-PCM_16 quantizers that libsndfile builds and AudioMixin's RAW writer
# have used; calibration picks the one that reproduces the fallback's bytes.
_PCM16_QUANTIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "int32_high": _pcm16_int32_high,
    "rint_32768": _pcm16_rint(32768.0),
    "rint_32767": _pcm16_rint(32767.0),
    "floor_f64": _pcm16_floor_f64,
}
# (format, sample rate) -> the calibrated quantizer's name, or None.
_pcm16_mode_by_key: dict[tuple[str, int], str | None] = {}


def encode_audio(
    audio_data: object,
    sample_rate_hz: int,
    response_format: str,
    speed: float | None,
) -> str | None:
    """Encode a model audio tensor/array into base64 in ``response_format``.

    Moved from the serving runtime bridge; uses the shared ``AudioMixin``
    encoder directly instead of going through the chat service. PCM and WAV
    at normal speed are written with numpy instead, once a probe showed that
    this gives soundfile's exact bytes for the format and rate.
    """
    if audio_data is None:
        return None
    try:
        import torch

        if isinstance(audio_data, torch.Tensor):
            audio_tensor = audio_data.detach().cpu().float().numpy()
        else:
            audio_tensor = np.asarray(audio_data, dtype=np.float32)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.reshape(-1)
        speed_value = float(speed) if isinstance(speed, int | float) and speed > 0 else 1.0
        if speed_value == 1.0 and audio_tensor.ndim == 1 and audio_tensor.size > 0 and isinstance(sample_rate_hz, int):
            fmt = response_format.lower()
            mode = _pcm16_mode(fmt, sample_rate_hz) if fmt in ("pcm", "wav") else None
            if mode is not None:
                encoded = _encode_pcm16(audio_tensor, sample_rate_hz, fmt, mode)
                if encoded is not None:
                    return encoded
        return _encode_with_soundfile(audio_tensor, sample_rate_hz, response_format, speed_value)
    except Exception:
        logger.exception("Failed to encode duplex data-plane audio output")
        return None


def _encode_with_soundfile(audio_tensor: np.ndarray, sample_rate_hz: int, response_format: str, speed: float) -> str:
    from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
    from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

    audio_response = AudioMixin().create_audio(
        CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=sample_rate_hz,
            response_format=response_format,
            speed=speed,
            stream_format="audio",
            base64_encode=True,
        )
    )
    return str(audio_response.audio_data)


def _encode_pcm16(audio: np.ndarray, sample_rate_hz: int, fmt: str, mode: str) -> str | None:
    """Base64 of soundfile's PCM_16 bytes for *audio* under quantizer *mode*; ``None`` if a sample is not finite."""
    if not np.isfinite(audio.sum()):
        return None
    data = _PCM16_QUANTIZERS[mode](audio.astype(np.float32, copy=False)).tobytes()
    if fmt == "wav":
        data = _wav_pcm16_mono_header(sample_rate_hz, len(data)) + data
    return base64.b64encode(data).decode("utf-8")


def _wav_pcm16_mono_header(sample_rate_hz: int, data_bytes: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,  # WAVE_FORMAT_PCM
        1,  # mono
        sample_rate_hz,
        sample_rate_hz * 2,
        2,
        16,
        b"data",
        data_bytes,
    )


def _pcm16_mode(fmt: str, sample_rate_hz: int) -> str | None:
    key = (fmt, sample_rate_hz)
    if key not in _pcm16_mode_by_key:
        _pcm16_mode_by_key[key] = _calibrate_pcm16(fmt, sample_rate_hz)
    return _pcm16_mode_by_key[key]


def _calibration_probe() -> np.ndarray:
    """Samples around every edge of the conversion: clipping, rounding ties, tiny and random values."""
    edges = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 1e5, -1e5, 1e-6, -1e-6, 1e-40, -1e-40]
    edges += [np.nextafter(np.float32(1.0), np.float32(2.0)), np.nextafter(np.float32(-1.0), np.float32(-2.0))]
    edges += [np.nextafter(np.float32(1.0), np.float32(0.0)), np.nextafter(np.float32(-1.0), np.float32(0.0))]
    # Where x * 2**31 rounds to 0 or to +-1 in the int32, and normal-but-tiny
    # values a decay tail can leave.
    half = np.float32(2.0**-32)
    edges += [np.nextafter(half, np.float32(0.0)), half, np.nextafter(half, np.float32(1.0)), -half]
    edges += [np.nextafter(-half, np.float32(-1.0)), 1e-37, -1e-37, 1e-20, -1e-20]
    for scale in (32767.0, 32768.0):
        for steps in (0.5, 1.5, 2.5, 100.5, 32766.5, 32767.5):
            edges += [steps / scale, -steps / scale]
    # One float32 ulp outside whole int16 steps: rounding x * 2**31 into the
    # int32 carries these to the step, a plain floor(x * 32768) does not.
    for steps in (1, 2, 17, 63, 64, 82, 127):
        step = np.float32(steps / 32768.0)
        edges += [np.nextafter(step, np.float32(0.0)), np.nextafter(-step, np.float32(-1.0))]
    rng = np.random.default_rng(0)
    return np.concatenate(
        [
            np.asarray(edges, dtype=np.float32),
            (rng.standard_normal(1500) * 0.3).astype(np.float32),
            rng.uniform(-1.2, 1.2, 1500).astype(np.float32),
        ]
    )


def _calibrate_pcm16(fmt: str, sample_rate_hz: int) -> str | None:
    """The quantizer that makes ``_encode_pcm16`` equal soundfile's output for *fmt* at this rate, if any."""
    probe = _calibration_probe()
    # Three lengths, so the WAV header's size fields are checked too.
    probes = (probe, probe[:1001], probe[:1])
    try:
        expected = [_encode_with_soundfile(samples, sample_rate_hz, fmt, 1.0) for samples in probes]
    except Exception:
        logger.info(
            "Duplex %s audio at %s Hz: soundfile rejected the probe; encoding stays on soundfile", fmt, sample_rate_hz
        )
        return None
    for mode in _PCM16_QUANTIZERS:
        encoded = [_encode_pcm16(samples, sample_rate_hz, fmt, mode) for samples in probes]
        if encoded == expected:
            return mode
    logger.info(
        "Duplex %s audio at %s Hz: numpy PCM_16 differs from soundfile; encoding stays on soundfile",
        fmt,
        sample_rate_hz,
    )
    return None


__all__ = ["encode_audio"]
