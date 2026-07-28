# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii
import io
import math
import wave

import numpy as np

try:
    from audioop import alaw2lin, lin2alaw, lin2ulaw, ulaw2lin
except ImportError:  # pragma: no cover - audioop is removed in newer Python.
    alaw2lin = lin2alaw = lin2ulaw = ulaw2lin = None


MIN_INPUT_SAMPLE_RATE_HZ = 8_000
MAX_INPUT_SAMPLE_RATE_HZ = 192_000


def validate_input_sample_rate_hz(sample_rate_hz: int | float) -> int:
    if isinstance(sample_rate_hz, bool) or (isinstance(sample_rate_hz, float) and not math.isfinite(sample_rate_hz)):
        raise ValueError("sample_rate_hz must be a finite integer")
    rate = int(sample_rate_hz)
    if rate != sample_rate_hz:
        raise ValueError("sample_rate_hz must be an integer")
    if not MIN_INPUT_SAMPLE_RATE_HZ <= rate <= MAX_INPUT_SAMPLE_RATE_HZ:
        raise ValueError(f"sample_rate_hz must be between {MIN_INPUT_SAMPLE_RATE_HZ} and {MAX_INPUT_SAMPLE_RATE_HZ}")
    return rate


def resample_pcm16_mono(raw: bytes, *, source_rate_hz: int, target_rate_hz: int) -> bytes:
    if source_rate_hz <= 0 or target_rate_hz <= 0 or source_rate_hz == target_rate_hz:
        return raw
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if samples.size <= 1:
        return raw
    target_size = max(1, int(round(samples.size * target_rate_hz / source_rate_hz)))
    source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
    target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
    resampled = np.interp(target_x, source_x, samples)
    return np.clip(resampled, -32768, 32767).astype("<i2").tobytes()


def decode_g711_ulaw(raw: bytes) -> bytes:
    if ulaw2lin is not None:
        return ulaw2lin(raw, 2)
    data = np.frombuffer(raw, dtype=np.uint8)
    value = np.bitwise_not(data).astype(np.int16)
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    return np.where(sign != 0, -(sample - 0x84), sample - 0x84).astype("<i2").tobytes()


def decode_g711_alaw(raw: bytes) -> bytes:
    if alaw2lin is not None:
        return alaw2lin(raw, 2)
    data = np.bitwise_xor(np.frombuffer(raw, dtype=np.uint8), 0x55).astype(np.int16)
    sign = data & 0x80
    exponent = (data >> 4) & 0x07
    mantissa = data & 0x0F
    sample = np.where(exponent == 0, (mantissa << 4) + 8, ((mantissa << 4) + 0x108) << (exponent - 1))
    return np.where(sign != 0, sample, -sample).astype("<i2").tobytes()


def encode_g711_ulaw(raw: bytes) -> bytes:
    if lin2ulaw is not None:
        return lin2ulaw(raw, 2)
    pcm = np.clip(np.frombuffer(raw, dtype="<i2").astype(np.int32), -32635, 32635)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.abs(pcm) + 0x84
    exponent = np.zeros_like(magnitude)
    for exp in range(7):
        exponent = np.where(magnitude > (0xFF << exp), exp + 1, exponent)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (np.bitwise_not(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def encode_g711_alaw(raw: bytes) -> bytes:
    if lin2alaw is not None:
        return lin2alaw(raw, 2)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    sign = np.where(pcm >= 0, 0x80, 0x00)
    magnitude = np.abs(pcm)
    exponent = np.zeros_like(magnitude)
    for exp in range(1, 8):
        exponent = np.where(magnitude >= (1 << (exp + 7)), exp, exponent)
    mantissa = np.where(
        exponent == 0,
        (magnitude >> 4) & 0x0F,
        (magnitude >> (exponent + 3)) & 0x0F,
    )
    return ((sign | (exponent << 4) | mantissa) ^ 0x55).astype(np.uint8).tobytes()


def wav_payload_to_pcm16(raw: bytes) -> tuple[bytes | None, int | None]:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate_hz = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_width != 2:
            return None, sample_rate_hz
        if channels <= 1:
            return frames, sample_rate_hz
        pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
        mono = np.mean(pcm.astype(np.float32), axis=1)
        return np.clip(mono, -32768, 32767).astype("<i2").tobytes(), sample_rate_hz
    except (EOFError, ValueError, wave.Error):
        return None, None


def convert_input_audio_with_rate(
    audio: object,
    fmt: object,
    *,
    sample_rate_hz: int | float | None = None,
    target_sample_rate_hz: int = 16_000,
) -> tuple[object, object, int | float | None]:
    if not isinstance(audio, str) or not isinstance(fmt, str):
        return audio, fmt, sample_rate_hz
    normalized = fmt.lower()
    if normalized not in {"pcm16", "pcm_s16le", "s16le", "g711_ulaw", "g711_alaw"}:
        return audio, fmt, sample_rate_hz
    if isinstance(sample_rate_hz, int | float):
        sample_rate_hz = validate_input_sample_rate_hz(sample_rate_hz)
    try:
        raw = base64.b64decode(audio.strip(), validate=False)
    except (binascii.Error, ValueError):
        return audio, fmt, sample_rate_hz
    if normalized == "g711_ulaw":
        raw = decode_g711_ulaw(raw)
        sample_rate_hz = sample_rate_hz if isinstance(sample_rate_hz, int | float) else 8_000
    elif normalized == "g711_alaw":
        raw = decode_g711_alaw(raw)
        sample_rate_hz = sample_rate_hz if isinstance(sample_rate_hz, int | float) else 8_000
    elif len(raw) % 2:
        return audio, fmt, sample_rate_hz
    if isinstance(sample_rate_hz, int | float) and int(sample_rate_hz) != target_sample_rate_hz:
        raw = resample_pcm16_mono(
            raw,
            source_rate_hz=int(sample_rate_hz),
            target_rate_hz=target_sample_rate_hz,
        )
        sample_rate_hz = target_sample_rate_hz
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    encoded = base64.b64encode(np.ascontiguousarray(pcm, dtype="<f4").tobytes()).decode("ascii")
    return encoded, "pcm_f32le", sample_rate_hz


def convert_output_audio(
    audio: str,
    *,
    source_fmt: str,
    target_fmt: str,
    source_sample_rate_hz: int | None = None,
    target_sample_rate_hz: int | None = None,
) -> tuple[str, str, int | None]:
    target = target_fmt.lower()
    if target not in {"g711_ulaw", "g711_alaw"}:
        return audio, source_fmt, source_sample_rate_hz
    try:
        raw = base64.b64decode(audio, validate=False)
    except (binascii.Error, ValueError):
        return audio, source_fmt, source_sample_rate_hz
    source = source_fmt.lower()
    if source == "wav":
        pcm_raw, wav_rate = wav_payload_to_pcm16(raw)
        if pcm_raw is None:
            return audio, source_fmt, source_sample_rate_hz
        raw = pcm_raw
        source_sample_rate_hz = source_sample_rate_hz or wav_rate
    elif source not in {"pcm", "pcm16", "pcm_s16le", "s16le"} or len(raw) % 2:
        return audio, source_fmt, source_sample_rate_hz
    target_rate = target_sample_rate_hz or 8_000
    if source_sample_rate_hz is not None:
        raw = resample_pcm16_mono(raw, source_rate_hz=source_sample_rate_hz, target_rate_hz=target_rate)
    encoded = encode_g711_ulaw(raw) if target == "g711_ulaw" else encode_g711_alaw(raw)
    return base64.b64encode(encoded).decode("ascii"), target, target_rate
