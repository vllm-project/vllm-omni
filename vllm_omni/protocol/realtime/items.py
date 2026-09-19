# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime conversation items: shape, truncation and the transcript inside one.

A Realtime conversation is a list of *items* --- a user message, an assistant
message, a function call --- each with a ``content`` list of typed parts. The
client creates them (``conversation.item.create``), truncates an assistant
item's audio after a barge-in (``conversation.item.truncate``), and the server
echoes them back inside ``conversation.item.*`` and ``response.done`` events.

The rules in this module are the ones both sides have to agree on: which
defaults a bare item gets, how far a transcript is cut when the client says it
only heard N milliseconds of audio, and what makes a truncate request invalid.
Item *storage* --- which items a session currently holds, in what order --- is
the consumer's state, not this module's.

``validate_realtime_video_frames`` sits here too: camera frames ride along an
``input_audio_buffer.append`` in the vLLM-Omni extension of the protocol, and
validating them is the same kind of wire-shape check.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from uuid import uuid4

__all__ = [
    "input_transcript_from_item",
    "normalize_conversation_item",
    "text_chars_for_audio_ms_from_marks",
    "truncate_realtime_item_content",
    "validate_realtime_item_truncate",
    "validate_realtime_video_frames",
]


def validate_realtime_video_frames(video_frames: object, max_slice_nums: object) -> str | None:
    """Validate omni-duplex camera frames on input_audio_buffer.append.

    Wire contract matches the official MiniCPM-o duplex loop: one base
    base64 JPEG per ~1 s audio chunk, optionally followed by that unit's
    stacked composite tiling the sub-frames captured inside it (at most 2
    images either way). A caller-supplied ``max_slice_nums`` is rejected rather
    than silently ignored: slicing is Stage 0's decision here, and Stage 0
    already applies the official HD suggestion for a stacked unit
    (``max_slice_nums=[2, 1]``). The wire simply does not let the client choose
    it.
    """
    if max_slice_nums not in (None, 1):
        return "max_slice_nums is not selectable on the wire; Stage0 slices stacked units itself"
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
