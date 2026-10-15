# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Decoding one ``input_audio_buffer.append``.

An append is the hottest client event on a Realtime connection and the one with
the most wire surface: base64 audio in any accepted format, an optional rate,
optional camera frames, and a set of optional hints (``duration_ms``,
``is_speech``, a VAD probability, a transcript) that clients attach and servers
are free to use or ignore.

:func:`decode_audio_append` turns that into :class:`RealtimeAudioAppend`, a
value object: bytes at 16 kHz ``pcm_f32le``, the surviving hints, and a speech
verdict. It is a pure function of the event plus the session's
:class:`~vllm_omni.protocol.realtime.session.RealtimeInputDefaults`, so any
consumer decodes an append the same way. Building a consumer's own command
object out of it is one constructor call --- see
``vllm_omni.engine.duplex.realtime_commands.build_append_audio``, which wraps it
for the duplex ``AppendAudio``.

The speech verdict here is the *client-declared* one (explicit flags, a VAD
probability the client sent, or an RMS floor as a last resort). It is not
server-side VAD; that is a runtime concern and lives with whoever owns the
session.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from vllm_omni.protocol.realtime.audio import convert_input_audio_with_rate
from vllm_omni.protocol.realtime.errors import RealtimeProtocolError
from vllm_omni.protocol.realtime.formats import (
    is_supported_realtime_input_format,
    parse_realtime_audio_format,
)
from vllm_omni.protocol.realtime.items import validate_realtime_video_frames
from vllm_omni.protocol.realtime.session import RealtimeInputDefaults

__all__ = [
    "REALTIME_INPUT_HINT_KEYS",
    "RealtimeAudioAppend",
    "copy_realtime_input_hints",
    "decode_audio_append",
    "input_explicitly_non_speech",
    "input_looks_like_speech",
]

#: Wire hint keys copied verbatim from a client append onto the internal payload.
REALTIME_INPUT_HINT_KEYS = (
    "duration_ms",
    "audio_duration_ms",
    "audio_start_ms",
    "audio_end_ms",
    "is_speech",
    "speech",
    "speech_probability",
    "vad",
    "overlap_action",
    "overlap",
    "force_barge_in",
    "force_listen",
    "text",
    "transcript",
)


@dataclass(frozen=True, slots=True)
class RealtimeAudioAppend:
    """One decoded ``input_audio_buffer.append``.

    ``audio`` is raw bytes in ``format`` at ``sample_rate_hz`` --- after
    :func:`decode_audio_append` that is 16 kHz ``pcm_f32le`` for every input
    format the codec converts.
    """

    audio: bytes
    format: str
    sample_rate_hz: int | None = None
    is_speech: bool | None = None
    video_frames: tuple[str, ...] = ()
    duration_ms: int | None = None
    audio_end_ms: int | None = None
    #: Model-neutral hints carried through from the wire (rms, vad, transcript hints ...).
    hints: dict[str, object] = field(default_factory=dict)
    #: Client correlation id (OpenAI ``event_id``), echoed on error events.
    event_id: str | None = None


# ---- speech hints ----


def input_explicitly_non_speech(event: Mapping[str, object]) -> bool:
    for key in ("is_speech", "speech"):
        value = event.get(key)
        if isinstance(value, bool):
            return not value
    vad = event.get("vad")
    if isinstance(vad, dict):
        value = vad.get("is_speech")
        if isinstance(value, bool):
            return not value
        probability = vad.get("speech_probability", vad.get("probability"))
        if isinstance(probability, int | float):
            return float(probability) < 0.5
    probability = event.get("speech_probability")
    return isinstance(probability, int | float) and float(probability) < 0.5


def input_looks_like_speech(
    event: Mapping[str, object],
    *,
    audio: object,
    fmt: object,
    overlap_silence_rms: float,
) -> bool:
    if input_explicitly_non_speech(event):
        return False
    for key in ("is_speech", "speech"):
        value = event.get(key)
        if isinstance(value, bool):
            return value
    vad = event.get("vad")
    if isinstance(vad, dict):
        probability = vad.get("speech_probability", vad.get("probability"))
        if isinstance(probability, int | float):
            return float(probability) >= 0.5
    probability = event.get("speech_probability")
    if isinstance(probability, int | float):
        return float(probability) >= 0.5
    if fmt != "pcm_f32le":
        return True
    if isinstance(audio, bytes | bytearray):
        raw = bytes(audio)
    elif isinstance(audio, str):
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return True
    else:
        return True
    if len(raw) < 4 or len(raw) % 4 != 0:
        return True
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
    threshold = event.get("overlap_silence_rms")
    if not isinstance(threshold, int | float):
        vad = event.get("vad")
        if isinstance(vad, dict):
            threshold = vad.get("silence_rms")
    silence_rms = float(threshold) if isinstance(threshold, int | float) else overlap_silence_rms
    return rms >= max(0.0, silence_rms)


def copy_realtime_input_hints(source: Mapping[str, object], target: dict[str, object]) -> None:
    for key in REALTIME_INPUT_HINT_KEYS:
        if key in source:
            target[key] = source[key]


# ---- append decoding ----


def decode_audio_append(
    event: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults,
    hints_source: Mapping[str, object] | None = None,
) -> RealtimeAudioAppend:
    """Validate and pack one input append (audio and/or video frames).

    Capability checks (required/optional modalities) run later on the session.
    Audio path converts to 16 kHz ``pcm_f32le``.

    ``hints_source`` is the enclosing payload when the audio arrives inside
    something larger than a bare append --- a ``conversation.item.create``
    audio part, say --- so its hints apply unless the part overrides them.

    Raises :class:`RealtimeProtocolError` for an unsupported format, undecodable
    audio or invalid camera frames.
    """
    event_id = cast("str", event.get("event_id")) if isinstance(event.get("event_id"), str) else None
    audio = event.get("audio") or event.get("delta")
    video_frames: tuple[str, ...] = ()
    raw_frames = event.get("video_frames")
    if raw_frames is not None:
        frames_error = validate_realtime_video_frames(raw_frames, event.get("max_slice_nums"))
        if frames_error is not None:
            raise RealtimeProtocolError(frames_error, code="invalid_video_frames", event_id=event_id)
        if isinstance(raw_frames, list):
            video_frames = tuple(frame for frame in raw_frames if isinstance(frame, str) and frame)
    has_audio_field = isinstance(audio, str) and bool(audio)
    if not has_audio_field:
        if not video_frames:
            raise RealtimeProtocolError(
                "input_audio_buffer.append requires audio and/or video_frames",
                code="bad_event",
                event_id=event_id,
            )
        hints: dict[str, object] = {}
        if hints_source is not None:
            copy_realtime_input_hints(hints_source, hints)
        copy_realtime_input_hints(event, hints)
        duration_ms = hints.get("duration_ms", hints.get("audio_duration_ms"))
        audio_end_ms = hints.get("audio_end_ms")
        return RealtimeAudioAppend(
            event_id=event_id,
            audio=b"",
            format="pcm_f32le",
            sample_rate_hz=defaults.input_sample_rate_hz,
            is_speech=False,
            video_frames=video_frames,
            duration_ms=int(duration_ms) if isinstance(duration_ms, int | float) else None,
            audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else None,
            hints=hints,
        )
    fmt, format_rate = parse_realtime_audio_format(
        event.get("format") or event.get("input_audio_format") or defaults.input_audio_format
    )
    sample_rate_hz = (
        event.get("sample_rate_hz") or event.get("sample_rate") or format_rate or defaults.input_sample_rate_hz
    )
    if not is_supported_realtime_input_format(fmt):
        raise RealtimeProtocolError(
            f"Unsupported input_audio_format: {fmt}",
            code="unsupported_audio_format",
            event_id=event_id,
        )
    try:
        audio, fmt, sample_rate_hz = convert_input_audio_with_rate(
            audio,
            fmt,
            sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int | float) else None,
        )
    except ValueError as exc:
        raise RealtimeProtocolError(str(exc), code="bad_event", event_id=event_id) from exc
    hints: dict[str, object] = {}
    if hints_source is not None:
        copy_realtime_input_hints(hints_source, hints)
    copy_realtime_input_hints(event, hints)
    looks_like_speech = input_looks_like_speech(
        {**(hints_source or {}), **event},
        audio=audio,
        fmt=fmt,
        overlap_silence_rms=defaults.overlap_silence_rms,
    )
    duration_ms = hints.get("duration_ms", hints.get("audio_duration_ms"))
    audio_end_ms = hints.get("audio_end_ms")
    try:
        audio_bytes = base64.b64decode(audio, validate=True) if isinstance(audio, str) and audio else b""
    except (binascii.Error, ValueError) as exc:
        raise RealtimeProtocolError("input audio is not valid base64", code="bad_audio", event_id=event_id) from exc
    return RealtimeAudioAppend(
        event_id=event_id,
        audio=audio_bytes,
        format=str(fmt),
        sample_rate_hz=int(sample_rate_hz) if isinstance(sample_rate_hz, int | float) else None,
        is_speech=looks_like_speech,
        video_frames=video_frames,
        duration_ms=int(duration_ms) if isinstance(duration_ms, int | float) else None,
        audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else None,
        hints=hints,
    )
