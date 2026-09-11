# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI Realtime client events -> :class:`DuplexCommand` (stateless).

This is the mapping half of the former ``entrypoints/duplex/realtime_input.py``
translator. Everything that needed per-session state in the old translator
(input-buffer emptiness for commits, response-id fallbacks for cancels,
conversation-item lookups, VAD) is resolved by the session runner through the
helpers on :class:`~vllm_omni.engine.duplex.realtime_events.RealtimeProjectionState`;
the commands produced here carry the raw client intent only.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast
from uuid import uuid4

import numpy as np

from vllm_omni.engine.duplex.audio import convert_input_audio_with_rate
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    AppendText,
    BargeIn,
    CancelInput,
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
    DeleteItem,
    DuplexCommand,
    DuplexCommandError,
    Heartbeat,
    SignalTurn,
    TruncateItem,
    UpdateSession,
)

REALTIME_INPUT_AUDIO_FORMATS = {
    "pcm16",
    "pcm_s16le",
    "s16le",
    "pcm_f32le",
    "g711_ulaw",
    "g711_alaw",
}
REALTIME_OUTPUT_AUDIO_FORMATS = {
    "pcm16",
    "pcm_s16le",
    "s16le",
    "wav",
    "pcm",
    "g711_ulaw",
    "g711_alaw",
}

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
class RealtimeInputDefaults:
    """Session-level wire defaults an append may omit (derived from the session object)."""

    input_audio_format: str = "pcm16"
    input_sample_rate_hz: int = 16000
    output_audio_format: str = "pcm16"
    output_sample_rate_hz: int | None = None
    overlap_silence_rms: float = 0.003

    def with_session_payload(self, session_payload: Mapping[str, object]) -> RealtimeInputDefaults:
        """Return defaults updated from a Realtime ``session`` object (session.update)."""
        values = apply_realtime_session_defaults(self, session_payload)
        return values


# ---- audio format helpers ----


def parse_realtime_audio_format(raw_format: object) -> tuple[object, int | None]:
    def normalize_format(fmt: str) -> str:
        normalized = fmt.lower()
        if normalized in {"audio/pcm", "pcm"}:
            return "pcm16"
        if normalized in {"audio/wav", "wav"}:
            return "wav"
        if normalized in {"audio/pcm16", "pcm16", "pcm_s16le", "s16le"}:
            return "pcm16"
        if normalized in {"audio/pcm_f32le", "pcm_f32le", "f32le"}:
            return "pcm_f32le"
        if normalized in {"audio/g711_ulaw", "g711_ulaw", "g711-ulaw", "ulaw", "mulaw"}:
            return "g711_ulaw"
        if normalized in {"audio/g711_alaw", "g711_alaw", "g711-alaw", "alaw"}:
            return "g711_alaw"
        return fmt

    if isinstance(raw_format, str):
        return normalize_format(raw_format), None
    if not isinstance(raw_format, dict):
        return raw_format, None
    rate = raw_format.get("rate") or raw_format.get("sample_rate_hz") or raw_format.get("sample_rate")
    sample_rate_hz = int(rate) if isinstance(rate, int | float) and rate > 0 else None
    fmt = raw_format.get("type") or raw_format.get("format")
    if not isinstance(fmt, str):
        return raw_format, sample_rate_hz
    return normalize_format(fmt), sample_rate_hz


def duplex_response_format(realtime_format: str) -> str:
    normalized = realtime_format.lower()
    if normalized in {"pcm16", "pcm_s16le", "s16le"}:
        return "pcm"
    if normalized in {"g711_ulaw", "g711_alaw"}:
        return "pcm"
    if normalized in {"wav", "pcm"}:
        return normalized
    return "wav"


def realtime_output_format(duplex_format: object) -> str:
    if isinstance(duplex_format, str) and duplex_format.lower() in {"g711_ulaw", "g711_alaw"}:
        return duplex_format.lower()
    if isinstance(duplex_format, str) and duplex_format.lower() == "pcm":
        return "pcm16"
    return str(duplex_format or "wav")


def is_supported_realtime_input_format(fmt: object) -> bool:
    return isinstance(fmt, str) and fmt.lower() in REALTIME_INPUT_AUDIO_FORMATS


def realtime_audio_format_object(fmt: object, *, sample_rate_hz: int | None = None) -> dict[str, object]:
    if isinstance(fmt, str) and fmt.lower() in {"pcm16", "pcm_s16le", "s16le", "pcm"}:
        payload: dict[str, object] = {"type": "audio/pcm"}
    elif isinstance(fmt, str) and fmt.lower() == "pcm_f32le":
        payload = {"type": "audio/pcm_f32le"}
    elif isinstance(fmt, str) and fmt.lower() == "g711_ulaw":
        payload = {"type": "audio/g711_ulaw"}
    elif isinstance(fmt, str) and fmt.lower() == "g711_alaw":
        payload = {"type": "audio/g711_alaw"}
    else:
        payload = {"type": "audio/wav"}
    if sample_rate_hz is not None:
        payload["rate"] = int(sample_rate_hz)
    return payload


def validate_realtime_session_audio_formats(session_payload: Mapping[str, object]) -> str | None:
    audio_config = session_payload.get("audio")
    input_format: object = session_payload.get("input_audio_format")
    if input_format is None and isinstance(audio_config, dict):
        audio_input = audio_config.get("input")
        if isinstance(audio_input, dict):
            input_format = audio_input.get("format")
    parsed_input, _ = parse_realtime_audio_format(input_format)
    if input_format is not None and not (
        isinstance(parsed_input, str) and parsed_input.lower() in REALTIME_INPUT_AUDIO_FORMATS
    ):
        return f"Unsupported input_audio_format: {input_format}"

    output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
    if output_format is None and isinstance(audio_config, dict):
        audio_output = audio_config.get("output")
        if isinstance(audio_output, dict):
            output_format = audio_output.get("format")
    parsed_output, _ = parse_realtime_audio_format(output_format)
    if output_format is not None and not (
        isinstance(parsed_output, str) and parsed_output.lower() in REALTIME_OUTPUT_AUDIO_FORMATS
    ):
        return f"Unsupported output_audio_format: {output_format}"
    return None


def validate_realtime_response_audio_formats(response_payload: Mapping[str, object]) -> str | None:
    output_format: object = response_payload.get("output_audio_format") or response_payload.get("response_format")
    audio_config = response_payload.get("audio")
    if output_format is None and isinstance(audio_config, dict):
        audio_output = audio_config.get("output")
        if isinstance(audio_output, dict):
            output_format = audio_output.get("format")
    parsed_output, _ = parse_realtime_audio_format(output_format)
    if output_format is not None and not (
        isinstance(parsed_output, str) and parsed_output.lower() in REALTIME_OUTPUT_AUDIO_FORMATS
    ):
        return f"Unsupported output_audio_format: {output_format}"
    return None


def validate_conversation_item_audio_formats(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {"input_audio", "audio"}:
            continue
        raw_format = part.get("format")
        parsed_format, _ = parse_realtime_audio_format(raw_format)
        if raw_format is not None and not (
            isinstance(parsed_format, str) and parsed_format.lower() in REALTIME_INPUT_AUDIO_FORMATS
        ):
            return f"Unsupported input_audio format in conversation.item.create: {raw_format}"
    return None


def apply_realtime_session_defaults(
    defaults: RealtimeInputDefaults,
    session_payload: Mapping[str, object],
) -> RealtimeInputDefaults:
    """Derive the wire defaults a Realtime ``session`` object declares."""
    input_format: object = session_payload.get("input_audio_format")
    audio_config = session_payload.get("audio")
    if input_format is None and isinstance(audio_config, dict):
        audio_input = audio_config.get("input")
        if isinstance(audio_input, dict):
            input_format = audio_input.get("format")
    input_format, input_rate = parse_realtime_audio_format(input_format)
    # Any: heterogeneous field values splatted into ``dataclasses.replace`` below.
    updates: dict[str, Any] = {}
    if isinstance(input_format, str) and input_format.lower() in REALTIME_INPUT_AUDIO_FORMATS:
        updates["input_audio_format"] = input_format
    output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
    output_rate_raw: object | None = None
    if output_format is None and isinstance(audio_config, dict):
        audio_output = audio_config.get("output")
        if isinstance(audio_output, dict):
            output_format = audio_output.get("format")
            output_rate_raw = audio_output.get("sample_rate_hz") or audio_output.get("sample_rate")
    output_format, output_rate = parse_realtime_audio_format(output_format)
    if output_rate is None and isinstance(output_rate_raw, int | float) and output_rate_raw > 0:
        output_rate = int(output_rate_raw)
    if isinstance(output_format, str) and output_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
        updates["output_audio_format"] = realtime_output_format(output_format)
    sample_rate = session_payload.get("sample_rate_hz") or session_payload.get("sample_rate")
    if sample_rate is None and isinstance(audio_config, dict):
        audio_input = audio_config.get("input")
        if isinstance(audio_input, dict):
            sample_rate = audio_input.get("sample_rate_hz") or audio_input.get("sample_rate")
    if sample_rate is None:
        sample_rate = input_rate
    if isinstance(sample_rate, int | float) and sample_rate > 0:
        updates["input_sample_rate_hz"] = int(sample_rate)
    if isinstance(output_rate, int | float) and output_rate > 0:
        updates["output_sample_rate_hz"] = int(output_rate)
    overlap_fields = realtime_overlap_fields(session_payload)
    overlap_silence_rms = overlap_fields.get("overlap_silence_rms")
    if isinstance(overlap_silence_rms, int | float):
        updates["overlap_silence_rms"] = max(0.0, float(overlap_silence_rms))
    return replace(defaults, **updates) if updates else defaults


def realtime_overlap_fields(session_payload: Mapping[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    if isinstance(session_payload.get("overlap_policy"), str):
        fields["overlap_policy"] = session_payload["overlap_policy"]
    for key in ("overlap_short_ack_ms", "overlap_barge_in_ms", "overlap_silence_rms"):
        value = session_payload.get(key)
        if isinstance(value, int | float):
            fields[key] = value
    if isinstance(session_payload.get("playback_commit_policy"), str):
        fields["playback_commit_policy"] = session_payload["playback_commit_policy"]
    return fields


def json_safe_realtime_payload(payload: Mapping[str, object]) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in payload.items():
        if key == "extra_body":
            continue
        if isinstance(value, str | int | float | bool) or value is None:
            clean[key] = value
        elif isinstance(value, dict):
            clean[key] = json_safe_realtime_payload(value)
        elif isinstance(value, list):
            clean[key] = [
                (json_safe_realtime_payload(item) if isinstance(item, dict) else item)
                for item in value
                if isinstance(item, str | int | float | bool | dict) or item is None
            ]
    return clean


def input_audio_transcription_config(session_payload: Mapping[str, object]) -> dict[str, object] | None:
    transcription = session_payload.get("input_audio_transcription")
    if isinstance(transcription, dict):
        return transcription
    audio_config = session_payload.get("audio")
    if not isinstance(audio_config, dict):
        return None
    audio_input = audio_config.get("input")
    if not isinstance(audio_input, dict):
        return None
    transcription = audio_input.get("transcription")
    return transcription if isinstance(transcription, dict) else None


def realtime_max_output_tokens(value: object) -> int | None:
    """Normalize Realtime max output tokens (``"inf"`` -> ``None``)."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"inf", "infinity", "unlimited"}:
        return None
    if isinstance(value, int) and value > 0:
        return int(value)
    return None


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


def validate_realtime_video_frames(video_frames: object, max_slice_nums: object) -> str | None:
    """Validate omni-duplex camera frames on input_audio_buffer.append.

    Wire contract matches the official MiniCPM-o duplex loop: one base
    base64 JPEG per ~1 s audio chunk, optionally followed by that unit's
    stacked composite tiling the sub-frames captured inside it (at most 2
    images either way). HD slicing (``max_slice_nums > 1``) is rejected
    explicitly rather than silently ignored.
    """
    if max_slice_nums not in (None, 1):
        return "max_slice_nums > 1 (HD slicing) is not implemented by the duplex Realtime adapter"
    if not isinstance(video_frames, list):
        return "video_frames must be a list of base64-encoded images"
    frames = [frame for frame in video_frames if frame is not None]
    if len(frames) > 2:
        return "video_frames carries more than 2 frames for one append; send ~1 frame per 1 s chunk"
    for frame in frames:
        if not isinstance(frame, str) or not frame:
            return "video_frames entries must be non-empty base64 strings"
        if len(frame) > 4_000_000:
            return "video_frames entry exceeds 4MB base64; reduce capture resolution or JPEG quality"
        try:
            header = base64.b64decode(frame[:64] + "=" * (-len(frame[:64]) % 4))
        except (binascii.Error, ValueError):
            return "video_frames entries must be valid base64"
        if not (header.startswith(b"\xff\xd8") or header.startswith(b"\x89PNG")):
            return "video_frames entries must be JPEG or PNG images"
    return None


# ---- conversation items ----


def normalize_conversation_item(item: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(item)
    normalized.setdefault("id", f"item_{uuid4().hex}")
    normalized.setdefault("object", "realtime.item")
    normalized.setdefault("type", "message")
    normalized.setdefault("status", "completed")
    if "role" not in normalized and normalized.get("type") == "message":
        normalized["role"] = "user"
    if not isinstance(normalized.get("content"), list):
        normalized["content"] = []
    return normalized


def text_chars_for_audio_ms_from_marks(
    audio_end_ms: int,
    text_len: int,
    marks: list[object],
    *,
    final_ms: object | None = None,
) -> int:
    if text_len <= 0:
        return 0
    clean_marks: list[tuple[int, int]] = []
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        raw_text_chars = mark.get("text_chars")
        raw_audio_end_ms = mark.get("audio_end_ms", mark.get("audio_ms"))
        if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
            continue
        clean_marks.append((max(0, int(raw_audio_end_ms)), min(text_len, max(0, int(raw_text_chars)))))
    if not clean_marks:
        return 0 if audio_end_ms <= 0 else text_len
    clean_marks.sort(key=lambda item: item[0])
    audio_end_ms = max(0, int(audio_end_ms))
    if audio_end_ms <= 0:
        return 0
    previous_ms = 0
    previous_chars = 0
    for mark_ms, mark_chars in clean_marks:
        mark_ms = max(previous_ms, mark_ms)
        mark_chars = max(previous_chars, min(text_len, mark_chars))
        if audio_end_ms <= mark_ms:
            if mark_ms <= previous_ms:
                return mark_chars
            ratio = (audio_end_ms - previous_ms) / max(1, mark_ms - previous_ms)
            return int(previous_chars + (mark_chars - previous_chars) * max(0.0, min(1.0, ratio)))
        previous_ms = mark_ms
        previous_chars = mark_chars
    if isinstance(final_ms, int | float) and int(final_ms) > previous_ms:
        if audio_end_ms >= int(final_ms):
            return text_len
        ratio = (audio_end_ms - previous_ms) / max(1, int(final_ms) - previous_ms)
        return int(previous_chars + (text_len - previous_chars) * max(0.0, min(1.0, ratio)))
    return text_len if audio_end_ms >= previous_ms else previous_chars


def truncate_realtime_item_content(item: dict[str, object], *, content_index: int, audio_end_ms: int) -> None:
    content = item.get("content")
    if not isinstance(content, list) or not content:
        return
    index = max(0, int(content_index))
    if index >= len(content):
        return
    part = content[index]
    if not isinstance(part, dict):
        return
    transcript = part.get("transcript")
    if not isinstance(transcript, str) or not transcript:
        return
    marks = part.get("audio_text_marks")
    if isinstance(marks, list):
        keep_chars = text_chars_for_audio_ms_from_marks(
            audio_end_ms,
            len(transcript),
            marks,
            final_ms=part.get("audio_duration_ms") or part.get("duration_ms") or part.get("audio_ms"),
        )
        part["transcript"] = transcript[:keep_chars].rstrip()
        return
    duration_ms = part.get("audio_duration_ms") or part.get("duration_ms") or part.get("audio_ms")
    if isinstance(duration_ms, int | float) and duration_ms > 0:
        keep_chars = int(len(transcript) * max(0.0, min(1.0, int(audio_end_ms) / float(duration_ms))))
        part["transcript"] = transcript[:keep_chars].rstrip()
    elif audio_end_ms <= 0:
        part["transcript"] = ""


def validate_realtime_item_truncate(item: Mapping[str, object], *, content_index: int, audio_end_ms: int) -> str | None:
    if item.get("type") != "message" or item.get("role") != "assistant":
        return "conversation.item.truncate only supports assistant message items"
    if audio_end_ms < 0:
        return "conversation.item.truncate requires non-negative audio_end_ms"
    content = item.get("content")
    if not isinstance(content, list) or not content:
        return None
    index = max(0, int(content_index))
    if index >= len(content):
        return f"conversation.item.truncate content_index out of range: {content_index}"
    part = content[index]
    if not isinstance(part, dict):
        return "conversation.item.truncate target content part is invalid"
    if part.get("type") not in {"audio", "output_audio"}:
        return "conversation.item.truncate target content part must be audio"
    duration_ms = part.get("audio_duration_ms") or part.get("duration_ms") or part.get("audio_ms")
    if isinstance(duration_ms, int | float) and int(duration_ms) >= 0 and audio_end_ms > int(duration_ms):
        return "conversation.item.truncate audio_end_ms exceeds item audio duration"
    return None


def input_transcript_from_item(item: Mapping[str, object]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        for key in ("transcript", "text"):
            value = part.get(key)
            if isinstance(value, str) and value:
                if part.get("type") in {"input_audio", "audio", "audio_transcript", "transcript"}:
                    parts.append(value)
                    break
    return "".join(parts).strip()


# ---- append / item audio conversion ----


def build_append_audio(
    event: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults,
    hints_source: Mapping[str, object] | None = None,
) -> AppendAudio:
    """Validate, convert (to 16 kHz ``pcm_f32le`` base64) and pack one audio append.

    Speech classification without VAD (``is_speech`` on the command) uses the
    explicit wire hints and the RMS fallback exactly like the old translator.
    """
    event_id = cast("str", event.get("event_id")) if isinstance(event.get("event_id"), str) else None
    audio = event.get("audio") or event.get("delta")
    fmt, format_rate = parse_realtime_audio_format(
        event.get("format") or event.get("input_audio_format") or defaults.input_audio_format
    )
    sample_rate_hz = (
        event.get("sample_rate_hz") or event.get("sample_rate") or format_rate or defaults.input_sample_rate_hz
    )
    if not is_supported_realtime_input_format(fmt):
        raise DuplexCommandError(
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
        raise DuplexCommandError(str(exc), code="bad_event", event_id=event_id) from exc
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
    video_frames: tuple[str, ...] = ()
    raw_frames = event.get("video_frames")
    if raw_frames is not None:
        frames_error = validate_realtime_video_frames(raw_frames, event.get("max_slice_nums"))
        if frames_error is not None:
            raise DuplexCommandError(frames_error, code="invalid_video_frames", event_id=event_id)
        if isinstance(raw_frames, list):
            video_frames = tuple(frame for frame in raw_frames if isinstance(frame, str) and frame)
    duration_ms = hints.get("duration_ms", hints.get("audio_duration_ms"))
    audio_end_ms = hints.get("audio_end_ms")
    try:
        audio_bytes = base64.b64decode(audio, validate=True) if isinstance(audio, str) and audio else b""
    except (binascii.Error, ValueError) as exc:
        raise DuplexCommandError("input audio is not valid base64", code="bad_audio", event_id=event_id) from exc
    return AppendAudio(
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


# ---- translation ----


def translate_realtime_command(
    payload: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults | None = None,
) -> DuplexCommand:
    """Map one OpenAI Realtime client event onto a :class:`DuplexCommand`.

    Raises :class:`DuplexCommandError` for malformed or unsupported payloads.
    ``session.resume``, ``session.event_ack`` and ``conversation.item.retrieve``
    are transport concerns and are rejected with ``code="unknown_event"``.
    """
    defaults = defaults or RealtimeInputDefaults()
    event_type = payload.get("type")
    event_id = cast("str", payload.get("event_id")) if isinstance(payload.get("event_id"), str) else None
    if not isinstance(event_type, str):
        raise DuplexCommandError("Duplex event missing string type", code="bad_event", event_id=event_id)

    if event_type == "session.update":
        session = payload.get("session")
        session_payload: Mapping[str, object] = session if isinstance(session, dict) else payload
        format_error = validate_realtime_session_audio_formats(session_payload)
        if format_error is not None:
            raise DuplexCommandError(format_error, code="unsupported_audio_format", event_id=event_id)
        from vllm_omni.engine.duplex.turn_detection import validate_realtime_turn_detection

        turn_detection_error = validate_realtime_turn_detection(session_payload)
        if turn_detection_error is not None:
            raise DuplexCommandError(turn_detection_error, code="unsupported_turn_detection", event_id=event_id)
        return UpdateSession(event_id=event_id, patch=dict(session_payload))

    if event_type == "conversation.item.create":
        item = payload.get("item")
        format_error = validate_conversation_item_audio_formats(item)
        if format_error is not None:
            raise DuplexCommandError(format_error, code="unsupported_audio_format", event_id=event_id)
        if not isinstance(item, dict):
            raise DuplexCommandError("conversation.item.create requires item", code="bad_event", event_id=event_id)
        previous_item_id = payload.get("previous_item_id")
        return CreateItem(
            event_id=event_id,
            item=normalize_conversation_item(item),
            previous_item_id=previous_item_id if isinstance(previous_item_id, str) else None,
        )

    if event_type == "conversation.item.delete":
        item_id = payload.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise DuplexCommandError(
                "conversation.item.delete requires item_id", code="missing_item_id", event_id=event_id
            )
        return DeleteItem(event_id=event_id, item_id=item_id)

    if event_type == "conversation.item.truncate":
        item_id = payload.get("item_id")
        audio_end_ms = payload.get("audio_end_ms")
        content_index = payload.get("content_index", 0)
        if not isinstance(item_id, str) or not item_id:
            raise DuplexCommandError(
                "conversation.item.truncate requires item_id", code="missing_item_id", event_id=event_id
            )
        if not isinstance(audio_end_ms, int | float):
            raise DuplexCommandError(
                "conversation.item.truncate requires numeric audio_end_ms", code="bad_event", event_id=event_id
            )
        return TruncateItem(
            event_id=event_id,
            item_id=item_id,
            audio_end_ms=int(audio_end_ms),
            content_index=int(content_index) if isinstance(content_index, int | float) else 0,
        )

    if event_type == "input_audio_buffer.append":
        return build_append_audio(dict(payload), defaults=defaults)

    if event_type in {"input_audio_buffer.commit", "input.commit"}:
        final = payload.get("final", True)
        create_response = payload.get("response_create", payload.get("create_response"))
        is_speech = payload.get("is_speech")
        return Commit(
            event_id=event_id,
            final=bool(final) if isinstance(final, bool) else True,
            create_response=(
                bool(create_response)
                if isinstance(create_response, bool)
                else (True if event_type == "input.commit" and create_response is None else None)
            ),
            is_speech=is_speech if isinstance(is_speech, bool) else None,
        )

    if event_type == "input_audio_buffer.clear":
        return ClearInput(event_id=event_id)

    if event_type == "output_audio_buffer.clear":
        response_id = payload.get("response_id")
        return ClearOutputAudio(
            event_id=event_id,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
        )

    if event_type == "response.cancel":
        response_id = payload.get("response_id")
        return CancelResponse(
            event_id=event_id,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
        )

    if event_type == "response.create":
        response_payload = payload.get("response")
        if isinstance(response_payload, dict):
            format_error = validate_realtime_response_audio_formats(response_payload)
            if format_error is not None:
                raise DuplexCommandError(format_error, code="unsupported_audio_format", event_id=event_id)
        return CreateResponse(
            event_id=event_id,
            options=dict(response_payload) if isinstance(response_payload, dict) else {},
        )

    if event_type in {"playback.ack", "audio.playback_ack"}:
        played_ms = payload.get("played_ms")
        if not isinstance(played_ms, int | float):
            raise DuplexCommandError("playback.ack requires numeric played_ms", code="bad_event", event_id=event_id)
        committed_ms = payload.get("committed_ms")
        response_id = payload.get("response_id")
        item_id = payload.get("item_id")
        return AckPlayback(
            event_id=event_id,
            played_ms=int(played_ms),
            committed_ms=int(committed_ms) if isinstance(committed_ms, int | float) else None,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
            item_id=item_id if isinstance(item_id, str) and item_id else None,
        )

    if event_type == "session.heartbeat":
        return Heartbeat(event_id=event_id)

    if event_type == "conversation.item.retrieve":
        # The projected conversation items live engine-side; the runner answers.
        return SignalTurn(event_id=event_id, event="conversation.item.retrieve", signal_payload=dict(payload))

    if event_type in {"session.close", "close", "close_session"}:
        reason = payload.get("reason")
        return CloseSession(event_id=event_id, reason=reason if isinstance(reason, str) and reason else "client_close")

    if event_type in {"input.text.append", "input_text.append", "push_text"}:
        text = payload.get("text")
        if not isinstance(text, str):
            raise DuplexCommandError("input.text.append requires text", code="bad_event", event_id=event_id)
        return AppendText(event_id=event_id, text=text)

    if event_type == "input.cancel":
        return CancelInput(event_id=event_id)

    if event_type == "barge_in":
        return BargeIn(event_id=event_id)

    if event_type in {"turn.signal", "signal_turn"}:
        signal_event = payload.get("event")
        if not isinstance(signal_event, str) or not signal_event:
            raise DuplexCommandError("turn.signal requires event", code="bad_event", event_id=event_id)
        signal_payload = payload.get("payload")
        signal_payload = dict(signal_payload) if isinstance(signal_payload, dict) else {}
        if signal_event == "input.cancel":
            return CancelInput(event_id=event_id)
        if signal_event == "barge_in":
            return BargeIn(event_id=event_id)
        if signal_event == "response.cancel":
            response_id = signal_payload.get("response_id", payload.get("response_id"))
            return CancelResponse(
                event_id=event_id,
                response_id=response_id if isinstance(response_id, str) and response_id else None,
            )
        if signal_event == "session.update":
            return UpdateSession(event_id=event_id, patch=signal_payload)
        if signal_event == "conversation.item.create":
            item = signal_payload.get("item")
            if not isinstance(item, dict):
                raise DuplexCommandError("conversation.item.create requires item", code="bad_event", event_id=event_id)
            previous_item_id = signal_payload.get("previous_item_id")
            return CreateItem(
                event_id=event_id,
                item=normalize_conversation_item(item),
                previous_item_id=previous_item_id if isinstance(previous_item_id, str) else None,
            )
        if signal_event == "conversation.item.delete":
            item_id = signal_payload.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise DuplexCommandError(
                    "conversation.item.delete requires item_id", code="missing_item_id", event_id=event_id
                )
            return DeleteItem(event_id=event_id, item_id=item_id)
        if signal_event == "conversation.item.truncate":
            item_id = signal_payload.get("item_id")
            audio_end_ms = signal_payload.get("audio_end_ms")
            content_index = signal_payload.get("content_index", 0)
            if not isinstance(item_id, str) or not item_id or not isinstance(audio_end_ms, int | float):
                raise DuplexCommandError(
                    "conversation.item.truncate requires item_id and numeric audio_end_ms",
                    code="bad_event",
                    event_id=event_id,
                )
            return TruncateItem(
                event_id=event_id,
                item_id=item_id,
                audio_end_ms=int(audio_end_ms),
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
            )
        return SignalTurn(event_id=event_id, event=signal_event, signal_payload=signal_payload)

    raise DuplexCommandError(f"Unknown duplex event type: {event_type}", code="unknown_event", event_id=event_id)


__all__ = [
    "REALTIME_INPUT_AUDIO_FORMATS",
    "REALTIME_INPUT_HINT_KEYS",
    "REALTIME_OUTPUT_AUDIO_FORMATS",
    "RealtimeInputDefaults",
    "apply_realtime_session_defaults",
    "build_append_audio",
    "copy_realtime_input_hints",
    "duplex_response_format",
    "input_audio_transcription_config",
    "input_explicitly_non_speech",
    "input_looks_like_speech",
    "input_transcript_from_item",
    "is_supported_realtime_input_format",
    "json_safe_realtime_payload",
    "normalize_conversation_item",
    "parse_realtime_audio_format",
    "realtime_audio_format_object",
    "realtime_max_output_tokens",
    "realtime_output_format",
    "realtime_overlap_fields",
    "text_chars_for_audio_ms_from_marks",
    "translate_realtime_command",
    "truncate_realtime_item_content",
    "validate_conversation_item_audio_formats",
    "validate_realtime_item_truncate",
    "validate_realtime_response_audio_formats",
    "validate_realtime_session_audio_formats",
    "validate_realtime_video_frames",
]
