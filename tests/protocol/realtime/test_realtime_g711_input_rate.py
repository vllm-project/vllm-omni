# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A G.711 append decodes at the rate that was actually declared.

G.711 is an 8 kHz codec and an OpenAI ``input_audio_buffer.append`` has no
rate field. Only a genuinely unspecified rate may fall back to the codec's
native 8 kHz; a rate carried by the append, by its format object, by the
session, or inherited from an earlier ``session.update`` must win.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    realtime_session_payload,
)
from vllm_omni.entrypoints.duplex.realtime_input import RealtimeEnvelope
from vllm_omni.protocol.realtime.audio import encode_g711_alaw, encode_g711_ulaw
from vllm_omni.protocol.realtime.audio_input import decode_audio_append
from vllm_omni.protocol.realtime.session import (
    RealtimeInputDefaults,
    apply_realtime_session_defaults,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SECONDS = 1.0
NATIVE_RATE_HZ = 8_000
DECLARED_RATE_HZ = 16_000
TARGET_RATE_HZ = 16_000
TONE_HZ = 440.0

_CODECS = [
    ("g711_ulaw", "audio/g711_ulaw", encode_g711_ulaw),
    ("g711_alaw", "audio/g711_alaw", encode_g711_alaw),
]


def _tone_pcm16(source_rate_hz: int) -> bytes:
    t = np.arange(int(source_rate_hz * SECONDS)) / source_rate_hz
    return (0.5 * np.sin(2 * np.pi * TONE_HZ * t) * 32767).astype("<i2").tobytes()


def _append_event(encode, source_rate_hz: int) -> dict[str, object]:
    encoded = encode(_tone_pcm16(source_rate_hz))
    return {"audio": base64.b64encode(encoded).decode("ascii")}


def _assert_16khz_tone(decoded) -> None:
    samples = np.frombuffer(decoded.audio, dtype="<f4")
    assert decoded.format == "pcm_f32le"
    assert decoded.sample_rate_hz == TARGET_RATE_HZ
    assert samples.size == int(TARGET_RATE_HZ * SECONDS)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    dominant = float(np.fft.rfftfreq(samples.size, 1 / TARGET_RATE_HZ)[int(np.argmax(spectrum))])
    assert dominant == pytest.approx(TONE_HZ, abs=5.0)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_append_without_any_rate_uses_native_8khz(event_format, _session_format, encode):
    decoded = decode_audio_append(
        {**_append_event(encode, NATIVE_RATE_HZ), "format": event_format},
        defaults=RealtimeInputDefaults(),
    )

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("_event_format", "session_format", "encode"), _CODECS)
def test_g711_session_format_without_rate_uses_native_8khz(_event_format, session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(), {"audio": {"input": {"format": session_format}}}
    )

    decoded = decode_audio_append(_append_event(encode, NATIVE_RATE_HZ), defaults=defaults)

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_nested_format_rate_is_respected(event_format, _session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(),
        {"audio": {"input": {"format": {"type": event_format, "rate": DECLARED_RATE_HZ}}}},
    )

    decoded = decode_audio_append(_append_event(encode, DECLARED_RATE_HZ), defaults=defaults)

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_flat_session_rate_is_respected(event_format, _session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(),
        {"input_audio_format": event_format, "sample_rate_hz": DECLARED_RATE_HZ},
    )

    decoded = decode_audio_append(_append_event(encode, DECLARED_RATE_HZ), defaults=defaults)

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_nested_session_rate_is_respected(event_format, _session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(),
        {
            "audio": {
                "input": {"format": event_format, "sample_rate_hz": DECLARED_RATE_HZ},
            }
        },
    )

    decoded = decode_audio_append(_append_event(encode, DECLARED_RATE_HZ), defaults=defaults)

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_session_rate_is_inherited_across_updates(event_format, _session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(),
        {"audio": {"input": {"format": {"type": event_format, "rate": DECLARED_RATE_HZ}}}},
    )
    # A later update that only re-states the format keeps the accepted rate.
    defaults = apply_realtime_session_defaults(defaults, {"audio": {"input": {"format": event_format}}})

    decoded = decode_audio_append(_append_event(encode, DECLARED_RATE_HZ), defaults=defaults)

    _assert_16khz_tone(decoded)


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_g711_append_rate_overrides_the_session_rate(event_format, _session_format, encode):
    defaults = apply_realtime_session_defaults(
        RealtimeInputDefaults(),
        {"audio": {"input": {"format": {"type": event_format, "rate": DECLARED_RATE_HZ}}}},
    )

    decoded = decode_audio_append(
        {**_append_event(encode, NATIVE_RATE_HZ), "sample_rate_hz": NATIVE_RATE_HZ},
        defaults=defaults,
    )

    _assert_16khz_tone(decoded)


def _accepted_session_defaults(session_payload: dict[str, object]) -> RealtimeInputDefaults:
    """Run a session payload through the real projection and accepted-envelope update."""
    state = RealtimeProjectionState(session_id="sess-round-trip")
    state.apply_session_defaults(session_payload)
    projected = realtime_session_payload(state, session_payload)
    envelope = RealtimeEnvelope()
    envelope.apply_accepted_session(projected)
    return envelope.defaults


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_flat_rate_less_g711_survives_the_session_round_trip(event_format, _session_format, encode):
    defaults = _accepted_session_defaults({"input_audio_format": event_format})

    assert defaults.input_sample_rate_hz == NATIVE_RATE_HZ
    _assert_16khz_tone(decode_audio_append(_append_event(encode, NATIVE_RATE_HZ), defaults=defaults))


@pytest.mark.parametrize(("event_format", "_session_format", "encode"), _CODECS)
def test_explicit_g711_rate_survives_the_session_round_trip(event_format, _session_format, encode):
    defaults = _accepted_session_defaults(
        {"audio": {"input": {"format": {"type": event_format, "rate": DECLARED_RATE_HZ}}}}
    )

    assert defaults.input_sample_rate_hz == DECLARED_RATE_HZ
    _assert_16khz_tone(decode_audio_append(_append_event(encode, DECLARED_RATE_HZ), defaults=defaults))
