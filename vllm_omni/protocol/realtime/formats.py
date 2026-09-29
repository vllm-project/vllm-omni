# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime audio-format negotiation: what a client may declare and how it is read.

A Realtime client declares audio formats in three places --- the session object
(``session.audio.input.format`` and the older flat ``input_audio_format``), a
``response.create`` override, and a ``conversation.item.create`` audio part ---
and may spell one format several ways (``audio/pcm``, ``pcm16``, ``s16le``).
This module is the single place that normalizes those spellings and says which
are supported. Folding a session's declaration into the per-connection append
defaults is the job of ``session.py``, which builds on this module.

Everything here is a pure function of wire payloads: no session state, no model,
no engine.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

__all__ = [
    "REALTIME_INPUT_AUDIO_FORMATS",
    "REALTIME_OUTPUT_AUDIO_FORMATS",
    "is_supported_realtime_input_format",
    "parse_realtime_audio_format",
    "realtime_audio_format_object",
    "realtime_output_format",
    "validate_conversation_item_audio_formats",
    "validate_realtime_response_audio_formats",
    "validate_realtime_session_audio_formats",
]

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


def validate_realtime_session_audio_formats(
    session_payload: Mapping[str, object],
    *,
    input_audio_formats: Collection[str] | None = None,
    output_audio_formats: Collection[str] | None = None,
) -> str | None:
    """Reject a session object that declares an audio format we cannot serve.

    The format sets default to everything the codec can decode; a consumer that
    serves a narrower set passes its own (see
    ``vllm_omni.protocol.realtime.capabilities``).
    """
    supported_input = REALTIME_INPUT_AUDIO_FORMATS if input_audio_formats is None else input_audio_formats
    supported_output = REALTIME_OUTPUT_AUDIO_FORMATS if output_audio_formats is None else output_audio_formats
    audio_config = session_payload.get("audio")
    input_format: object = session_payload.get("input_audio_format")
    if input_format is None and isinstance(audio_config, dict):
        audio_input = audio_config.get("input")
        if isinstance(audio_input, dict):
            input_format = audio_input.get("format")
    parsed_input, _ = parse_realtime_audio_format(input_format)
    if input_format is not None and not (isinstance(parsed_input, str) and parsed_input.lower() in supported_input):
        return f"Unsupported input_audio_format: {input_format}"

    output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
    if output_format is None and isinstance(audio_config, dict):
        audio_output = audio_config.get("output")
        if isinstance(audio_output, dict):
            output_format = audio_output.get("format")
    parsed_output, _ = parse_realtime_audio_format(output_format)
    if output_format is not None and not (isinstance(parsed_output, str) and parsed_output.lower() in supported_output):
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
