# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live session audio: format negotiation and transcoding.

The session format (``audio/pcm`` at 16 or 24 kHz, ``audio/pcmu``,
``audio/pcma``) applies to both directions. Internally audio is mono float32
in [-1, 1] at the model's native input and output rates.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybase64 as base64

from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError, unsupported_session_config
from vllm_omni.protocol.realtime.audio import (
    decode_g711_alaw,
    decode_g711_ulaw,
    encode_g711_alaw,
    encode_g711_ulaw,
)
from vllm_omni.utils.audio_resample import StreamingAudioResampler

G711_RATE_HZ = 8_000
PCM_RATES_HZ = (16_000, 24_000)
DEFAULT_FORMAT: dict[str, Any] = {"type": "audio/pcm", "rate": 24_000}
# Hard cap on one append; 10 s at 24 kHz PCM16 is 480 KB.
MAX_APPEND_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class LiveAudioFormat:
    type: str
    rate: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "rate": self.rate}


def parse_audio_format(raw: Any) -> LiveAudioFormat:
    """Resolve ``session.audio.format``; ``None`` selects 24 kHz PCM."""
    if raw is None:
        raw = DEFAULT_FORMAT
    if not isinstance(raw, dict):
        raise unsupported_session_config("audio.format")
    fmt_type = raw.get("type")
    rate = raw.get("rate")
    if fmt_type == "audio/pcm":
        if rate not in PCM_RATES_HZ:
            raise unsupported_session_config("audio.format")
        return LiveAudioFormat(fmt_type, int(rate))
    if fmt_type in ("audio/pcmu", "audio/pcma"):
        if rate not in (None, G711_RATE_HZ):
            raise unsupported_session_config("audio.format")
        return LiveAudioFormat(fmt_type, G711_RATE_HZ)
    raise unsupported_session_config("audio.format")


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """One-shot resample of a complete clip."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or samples.size == 0:
        return samples
    return StreamingAudioResampler(source_rate, target_rate).process(samples, final=True)


class ResampleStream:
    """Stateful resampler for one continuous stream; identity when rates match."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self._resampler = None if source_rate == target_rate else StreamingAudioResampler(source_rate, target_rate)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._resampler is None:
            return samples
        return self._resampler.process(samples)


def pcm16_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) * np.float32(1.0 / 32768.0)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()


class InputAudioDecoder:
    """Decodes ``session.input_audio.append`` payloads for one session."""

    def __init__(self, fmt: LiveAudioFormat) -> None:
        self.format = fmt

    def decode(self, audio_b64: Any) -> tuple[np.ndarray, float]:
        """Return (float32 samples at the session rate, duration in ms).

        Raises ``LiveProtocolError(invalid_audio)`` for malformed input.
        """
        if not isinstance(audio_b64, str):
            raise LiveProtocolError("invalid_audio", "'audio' must be a base64 string.", param="audio")
        if len(audio_b64) > MAX_APPEND_BYTES * 4 // 3 + 4:
            raise LiveProtocolError("invalid_audio", "Audio append is too large.", param="audio")
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise LiveProtocolError("invalid_audio", "'audio' is not valid base64.", param="audio") from exc
        if self.format.type == "audio/pcm":
            if len(raw) % 2:
                raise LiveProtocolError(
                    "invalid_audio", "PCM16 audio must contain a whole number of samples.", param="audio"
                )
            samples = pcm16_to_float(raw)
        elif self.format.type == "audio/pcmu":
            samples = pcm16_to_float(decode_g711_ulaw(raw))
        else:
            samples = pcm16_to_float(decode_g711_alaw(raw))
        return samples, samples.size * 1000.0 / self.format.rate


class OutputAudioEncoder:
    """Encodes model audio into ``session.output_audio.delta`` payloads."""

    def __init__(self, fmt: LiveAudioFormat, model_rate: int) -> None:
        self.format = fmt
        self._stream = ResampleStream(model_rate, fmt.rate)

    def encode(self, samples: np.ndarray) -> str:
        pcm = float_to_pcm16(self._stream.process(np.asarray(samples, dtype=np.float32).reshape(-1)))
        if self.format.type == "audio/pcmu":
            pcm = encode_g711_ulaw(pcm)
        elif self.format.type == "audio/pcma":
            pcm = encode_g711_alaw(pcm)
        return base64.b64encode(pcm).decode("ascii")
