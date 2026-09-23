# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reading the Realtime ``session`` object.

``session.update`` carries one JSON object that a client may spell in either of
two shapes: the GA nest (``session.audio.input.format``,
``session.audio.output.sample_rate_hz``) or the older flat beta keys
(``input_audio_format``, ``sample_rate_hz``). Everything downstream --- the
append decoder, the projection that renders ``session.created`` back out, a
model's runtime configuration --- has to agree on what a given session object
means, so the readers live here once.

:class:`RealtimeInputDefaults` is the resolved answer for the input path: the
format, sample rate and silence threshold an ``input_audio_buffer.append``
inherits when it omits them. A connection keeps one of these and re-derives it
on every ``session.update``.

``overlap_policy`` / ``playback_commit_policy`` are vLLM-Omni extension fields
on the session object rather than GA OpenAI fields. They are read here because
this module owns what the wire may say; what a server *does* with them is the
consumer's business.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from vllm_omni.protocol.realtime.formats import (
    REALTIME_INPUT_AUDIO_FORMATS,
    REALTIME_OUTPUT_AUDIO_FORMATS,
    parse_realtime_audio_format,
    realtime_output_format,
)

__all__ = [
    "RealtimeInputDefaults",
    "apply_realtime_session_defaults",
    "input_audio_transcription_config",
    "json_safe_realtime_payload",
    "realtime_max_output_tokens",
    "realtime_overlap_fields",
]


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
