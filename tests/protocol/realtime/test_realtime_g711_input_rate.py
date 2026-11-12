# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A G.711 append with no rate is 8 kHz and must be delivered as 16 kHz audio.

G.711 is an 8 kHz codec and an OpenAI ``input_audio_buffer.append`` has no rate
field, so the decoder has to upsample it. Otherwise the 8 kHz bytes are tagged
as 16 kHz: half the duration and double the pitch, silently.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.protocol.realtime.audio import encode_g711_alaw, encode_g711_ulaw
from vllm_omni.protocol.realtime.audio_input import decode_audio_append
from vllm_omni.protocol.realtime.session import (
    RealtimeInputDefaults,
    apply_realtime_session_defaults,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SECONDS = 1.0
SOURCE_RATE_HZ = 8_000
TARGET_RATE_HZ = 16_000
TONE_HZ = 440.0

_CODECS = [
    ("g711_ulaw", "audio/g711_ulaw", encode_g711_ulaw),
    ("g711_alaw", "audio/g711_alaw", encode_g711_alaw),
]


def _g711_b64(encode) -> str:
    t = np.arange(int(SOURCE_RATE_HZ * SECONDS)) / SOURCE_RATE_HZ
    pcm16 = (0.5 * np.sin(2 * np.pi * TONE_HZ * t) * 32767).astype("<i2").tobytes()
    return base64.b64encode(encode(pcm16)).decode("ascii")


def _assert_16khz_tone(decoded) -> None:
    samples = np.frombuffer(decoded.audio, dtype="<f4")
    assert decoded.format == "pcm_f32le"
    assert decoded.sample_rate_hz == TARGET_RATE_HZ
    assert samples.size == int(TARGET_RATE_HZ * SECONDS)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    dominant = float(np.fft.rfftfreq(samples.size, 1 / TARGET_RATE_HZ)[int(np.argmax(spectrum))])
    assert dominant == pytest.approx(TONE_HZ, abs=5.0)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_append_without_rate_is_upsampled(event_format, _session_format, encode):
    decoded = decode_audio_append(
        {"audio": _g711_b64(encode), "format": event_format},
        defaults=RealtimeInputDefaults(),
    )

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("_event_format", "session_format", "encode"), _CODECS)
def test_g711_session_format_without_append_rate_is_upsampled(_event_format, session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(), {"audio": {"input": {"format": session_format}}}
    )

    decoded = decode_audio_append({"audio": _g711_b64(encode)}, defaults=defaults)

    _assert_16khz_tone(decoded)
