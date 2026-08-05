from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any
from uuid import uuid4

import numpy as np

from vllm_omni.experimental.fullduplex.openai.audio import (
    convert_input_audio_with_rate,
)
from vllm_omni.experimental.fullduplex.openai.realtime_state import (
    REALTIME_INPUT_AUDIO_FORMATS,
    REALTIME_OUTPUT_AUDIO_FORMATS,
)


class RealtimeInputTranslator:
    """Translate and validate client Realtime events for the duplex core."""

    async def discard_pending_input_audio(
        self,
        *,
        audio_end_ms: int | None = None,
    ) -> None:
        """Drop Realtime input-buffer state that was consumed as overlap.

        The serving overlap policy may classify a short acknowledgement such
        as "continue" as a side-channel signal instead of user turn input. The
        Realtime protocol has already observed the append and may have emitted
        ``speech_started``; reset that transient buffer so a later commit does
        not accidentally attach the acknowledgement transcript/audio to the
        next real user turn.
        """
        if self._input_speech_started and self._active_input_item_id is not None:
            await self._send_realtime_payload(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": max(0, int(audio_end_ms or 0)),
                    "item_id": self._active_input_item_id,
                }
            )
        self._input_speech_started = False
        self._active_input_item_id = None
        self._input_audio_buffer_has_audio = False
        self._input_audio_buffer_had_non_speech = False
        self._input_audio_buffer_transcript_parts.clear()

    async def _send_realtime_input_ack(self, event: dict[str, object]) -> None:
        if event.get("type") != "conversation.item.create":
            return
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item = self._normalize_conversation_item(item)
        previous_item_id = event.get("previous_item_id")
        if isinstance(previous_item_id, str):
            item["_previous_item_id"] = previous_item_id
        if item.get("role") != "user":
            return
        self._conversation_items[str(item["id"])] = item
        for payload in self._conversation_item_added_events(item):
            await self._send_realtime_payload(payload)

    async def _to_duplex_event(self, event: dict[str, object]) -> dict[str, object] | None:
        event_type = event.get("type")
        if event_type == "session.update":
            session_payload = event.get("session") if isinstance(event.get("session"), dict) else event
            format_error = self._validate_realtime_session_audio_formats(session_payload)
            if format_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_audio_format",
                        format_error,
                        event_id=event.get("event_id"),
                    )
                )
                return None
            turn_detection_error = self._validate_realtime_turn_detection(session_payload)
            if turn_detection_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_turn_detection",
                        turn_detection_error,
                        event_id=event.get("event_id"),
                        param="turn_detection",
                    )
                )
                return None
            self._apply_realtime_session_defaults(session_payload)
            session_payload.update(self._realtime_overlap_fields(session_payload))
            if not self._opened:
                self._opened = True
                self._initial_session_update = True
                return self._session_create_from_realtime(session_payload)
            return {
                "type": "turn.signal",
                "event": "session.update",
                "payload": session_payload,
            }
        if event_type == "conversation.item.create":
            item = event.get("item")
            format_error = self._validate_conversation_item_audio_formats(item)
            if format_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_audio_format",
                        format_error,
                        event_id=event.get("event_id"),
                    )
                )
                return None
            return await self._conversation_item_to_duplex(event)
        if event_type == "conversation.item.delete":
            item_id = event.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "missing_item_id",
                        "conversation.item.delete requires item_id",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            if item_id not in self._conversation_items:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "item_not_found",
                        f"Conversation item not found: {item_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            self._remove_conversation_item(item_id)
            await self._pending_outbound.put(
                {
                    "type": "turn.signal",
                    "event": "conversation.item.delete",
                    "payload": {"item_id": item_id},
                }
            )
            return None
        if event_type == "conversation.item.retrieve":
            item_id = event.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "missing_item_id",
                        "conversation.item.retrieve requires item_id",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            item = self._conversation_items.get(item_id)
            if item is None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "item_not_found",
                        f"Conversation item not found: {item_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            await self._send_realtime_payload({"type": "conversation.item.retrieved", "item": item})
            return None
        if event_type == "conversation.item.truncate":
            item_id = event.get("item_id")
            audio_end_ms = event.get("audio_end_ms")
            content_index = event.get("content_index", 0)
            if not isinstance(item_id, str) or not item_id:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "missing_item_id",
                        "conversation.item.truncate requires item_id",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            if not isinstance(audio_end_ms, int | float):
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "bad_event",
                        "conversation.item.truncate requires numeric audio_end_ms",
                        event_id=event.get("event_id"),
                        param="audio_end_ms",
                    )
                )
                return None
            item = self._conversation_items.get(item_id)
            if item is None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "item_not_found",
                        f"Conversation item not found: {item_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            truncate_error = self._validate_realtime_item_truncate(
                item,
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
                audio_end_ms=int(audio_end_ms),
            )
            if truncate_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "bad_event",
                        truncate_error,
                        event_id=event.get("event_id"),
                    )
                )
                return None
            self._item_truncation_cursors[item_id] = (
                int(content_index) if isinstance(content_index, int | float) else 0,
                int(audio_end_ms),
            )
            self._truncate_realtime_item_content(
                item,
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
                audio_end_ms=int(audio_end_ms),
            )
            await self._pending_outbound.put(
                {
                    "type": "turn.signal",
                    "event": "conversation.item.truncate",
                    "payload": {
                        "item_id": item_id,
                        "content_index": content_index,
                        "audio_end_ms": audio_end_ms,
                    },
                }
            )
            await self._pending_outbound.put(
                {
                    "type": "playback.ack",
                    "item_id": item_id,
                    "committed_ms": int(audio_end_ms),
                    "played_ms": int(audio_end_ms),
                    "truncate": True,
                }
            )
            return None
        if event_type == "input_audio_buffer.append":
            audio = event.get("audio") or event.get("delta")
            fmt, format_rate = self._parse_realtime_audio_format(
                event.get("format") or event.get("input_audio_format") or self._input_audio_format
            )
            sample_rate_hz = (
                event.get("sample_rate_hz") or event.get("sample_rate") or format_rate or self._input_sample_rate_hz
            )
            if not self._is_supported_realtime_input_format(fmt):
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_audio_format",
                        f"Unsupported input_audio_format: {fmt}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            try:
                audio, fmt, sample_rate_hz = convert_input_audio_with_rate(
                    audio,
                    fmt,
                    sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int | float) else None,
                )
            except ValueError as exc:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "bad_event",
                        str(exc),
                        event_id=event.get("event_id"),
                        param="sample_rate_hz",
                    )
                )
                return None
            looks_like_speech = self._input_looks_like_speech(event, audio=audio, fmt=fmt)
            self._input_audio_buffer_has_audio = self._input_audio_buffer_has_audio or (
                looks_like_speech and isinstance(audio, str) and bool(audio)
            )
            self._input_audio_buffer_had_non_speech = self._input_audio_buffer_had_non_speech or (
                not looks_like_speech and isinstance(audio, str) and bool(audio)
            )
            if looks_like_speech:
                await self._emit_input_speech_started(event)
                self._remember_input_transcript_hint(event)
            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio,
                "format": fmt,
                "sample_rate_hz": sample_rate_hz,
            }
            video_frames = event.get("video_frames")
            if video_frames is not None:
                frames_error = self._validate_realtime_video_frames(video_frames, event.get("max_slice_nums"))
                if frames_error is not None:
                    await self._send_realtime_payload(
                        self._realtime_error_payload(
                            "invalid_video_frames",
                            frames_error,
                            event_id=event.get("event_id"),
                            param="video_frames",
                        )
                    )
                    return None
                payload["video_frames"] = [frame for frame in video_frames if isinstance(frame, str) and frame]
            self._copy_realtime_input_hints(event, payload)
            if not looks_like_speech:
                payload["is_speech"] = False
            return payload
        if event_type == "input_audio_buffer.commit":
            if not self._input_audio_buffer_has_audio:
                if self._input_audio_buffer_had_non_speech:
                    self._input_audio_buffer_had_non_speech = False
                    self._active_input_item_id = None
                    return {
                        "type": "input_audio_buffer.commit",
                        "final": event.get("final", True),
                        "response_create": False,
                        "is_speech": False,
                    }
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "input_audio_buffer_empty",
                        "input_audio_buffer.commit requires a non-empty input audio buffer",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            item_id = self._active_input_item_id or f"item_{uuid4().hex}"
            await self._pending_commit_item_ids.put(item_id)
            await self._emit_input_speech_stopped(event, item_id=item_id)
            transcript = self._consume_input_transcript_hint()
            self._active_input_item_id = None
            self._input_audio_buffer_has_audio = False
            self._input_audio_buffer_had_non_speech = False
            payload = {
                "type": "input_audio_buffer.commit",
                "final": event.get("final", True),
                "realtime_item_id": item_id,
                "response_create": bool(event.get("response_create", False)),
            }
            if transcript:
                payload["transcript"] = transcript
            return payload
        if event_type == "input_audio_buffer.clear":
            self._input_speech_started = False
            self._active_input_item_id = None
            self._input_audio_buffer_has_audio = False
            self._input_audio_buffer_had_non_speech = False
            self._input_audio_buffer_transcript_parts.clear()
            return {"type": "input_audio_buffer.clear", "reason": event_type}
        if event_type == "output_audio_buffer.clear":
            payload = {"type": "output_audio_buffer.clear", "reason": event_type}
            response_id = event.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                response_id = self._active_response_id or self._last_response_id
            if self._response_is_done(response_id):
                await self._send_realtime_payload(
                    {
                        "type": "output_audio_buffer.cleared",
                        "response_id": response_id,
                    }
                )
                return None
            if isinstance(response_id, str) and response_id:
                payload["response_id"] = response_id
            return payload
        if event_type == "response.cancel":
            payload = {"type": "response.cancel", "reason": event_type}
            response_id = event.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                response_id = self._active_response_id or self._last_response_id
            if self._response_is_done(response_id):
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "response_not_active",
                        f"Response is already complete: {response_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            if isinstance(response_id, str) and response_id:
                payload["response_id"] = response_id
            return payload
        if event_type == "response.create":
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                format_error = self._validate_realtime_response_audio_formats(response_payload)
                if format_error is not None:
                    await self._send_realtime_payload(
                        self._realtime_error_payload(
                            "unsupported_audio_format",
                            format_error,
                            event_id=event.get("event_id"),
                        )
                    )
                    return None
            return {
                "type": "response.create",
                "response": response_payload if isinstance(response_payload, dict) else {},
            }
        if event_type in {"session.close", "close"}:
            return {"type": "session.close"}
        return event

    @staticmethod
    def _normalize_conversation_item(item: dict[str, object]) -> dict[str, object]:
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

    @staticmethod
    def _truncate_realtime_item_content(
        item: dict[str, object],
        *,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
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
            keep_chars = RealtimeInputTranslator._text_chars_for_audio_ms_from_marks(
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

    @staticmethod
    def _validate_realtime_item_truncate(
        item: dict[str, object],
        *,
        content_index: int,
        audio_end_ms: int,
    ) -> str | None:
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

    @staticmethod
    def _text_chars_for_audio_ms_from_marks(
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

    def _conversation_item_added_events(self, item: dict[str, object]) -> list[dict[str, object]]:
        item_id = item.get("id")
        explicit_previous_item_id = item.pop("_previous_item_id", None)
        previous_item_id = (
            explicit_previous_item_id if isinstance(explicit_previous_item_id, str) else self._last_conversation_item_id
        )
        if isinstance(item_id, str) and item_id:
            self._last_conversation_item_id = item_id
        return [
            {
                "type": "conversation.item.added",
                "previous_item_id": previous_item_id,
                "item": item,
            },
            {
                "type": "conversation.item.created",
                "previous_item_id": previous_item_id,
                "item": item,
            },
        ]

    def _conversation_item_done_event(self, item: dict[str, object]) -> dict[str, object]:
        item_id = item.get("id")
        previous_item_id = self._previous_item_id(item_id) if isinstance(item_id, str) else None
        if isinstance(item_id, str) and item_id:
            self._conversation_items[item_id] = item
            self._last_conversation_item_id = item_id
        return {
            "type": "conversation.item.done",
            "previous_item_id": previous_item_id,
            "item": item,
        }

    def _remove_conversation_item(self, item_id: str) -> bool:
        removed = self._conversation_items.pop(item_id, None) is not None
        if self._last_conversation_item_id == item_id:
            self._last_conversation_item_id = next(reversed(self._conversation_items), None)
        self._item_truncation_cursors.pop(item_id, None)
        return removed

    def _response_output_item_added_events(
        self,
        *,
        response_id: object,
        item: dict[str, object],
    ) -> list[dict[str, object]]:
        return [
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": item,
            },
        ]

    @staticmethod
    def _response_content_part(*, transcript: str = "") -> dict[str, object]:
        return {
            "type": "audio",
            "transcript": transcript,
        }

    @staticmethod
    def _response_text_content_part(*, text: str = "") -> dict[str, object]:
        return {
            "type": "text",
            "text": text,
        }

    @staticmethod
    def _response_item_content_part(
        *,
        transcript: str = "",
        audio_duration_ms: int | None = None,
        audio_text_marks: list[dict[str, int]] | None = None,
    ) -> dict[str, object]:
        part: dict[str, object] = {
            "type": "output_audio",
            "transcript": transcript,
        }
        if audio_duration_ms is not None:
            part["audio_duration_ms"] = int(audio_duration_ms)
        if audio_text_marks:
            part["audio_text_marks"] = [dict(mark) for mark in audio_text_marks]
        return part

    @staticmethod
    def _response_item_text_content_part(*, text: str = "") -> dict[str, object]:
        return {
            "type": "output_text",
            "text": text,
        }

    def _previous_item_id(self, item_id: str) -> str | None:
        previous: str | None = None
        for known_id in self._conversation_items:
            if known_id == item_id:
                return previous
            previous = known_id
        if self._last_conversation_item_id == item_id:
            return previous
        return self._last_conversation_item_id

    def _input_audio_buffer_committed_event(
        self,
        *,
        item_id: str,
        event: dict[str, Any],
    ) -> dict[str, object]:
        return {
            "type": "input_audio_buffer.committed",
            "previous_item_id": self._previous_item_id(item_id),
            "item_id": item_id,
            "event": event,
        }

    def _session_create_from_realtime(self, session_payload: dict[str, object]) -> dict[str, object]:
        self._apply_realtime_session_defaults(session_payload)
        model = session_payload.get("model")
        audio_config = session_payload.get("audio")
        audio_input = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        extra_body = (
            dict(session_payload.get("extra_body")) if isinstance(session_payload.get("extra_body"), dict) else {}
        )
        extra_body["realtime_session_payload"] = self._json_safe_realtime_payload(session_payload)
        if isinstance(session_payload.get("tools"), list):
            extra_body["realtime_tools"] = session_payload["tools"]
        if isinstance(session_payload.get("tool_choice"), str | dict):
            extra_body["realtime_tool_choice"] = session_payload["tool_choice"]
        if isinstance(session_payload.get("metadata"), dict):
            extra_body["realtime_metadata"] = dict(session_payload["metadata"])
        if isinstance(session_payload.get("include"), list):
            extra_body["realtime_include"] = list(session_payload["include"])
        if isinstance(session_payload.get("prompt"), dict):
            extra_body["realtime_prompt"] = dict(session_payload["prompt"])
        input_audio_transcription = self._input_audio_transcription_config(session_payload)
        if isinstance(input_audio_transcription, dict):
            extra_body["realtime_input_audio_transcription"] = dict(input_audio_transcription)
        if isinstance(session_payload.get("input_audio_noise_reduction"), dict):
            extra_body["realtime_input_audio_noise_reduction"] = dict(session_payload["input_audio_noise_reduction"])
        if isinstance(audio_input, dict) and isinstance(audio_input.get("noise_reduction"), dict):
            extra_body["realtime_input_audio_noise_reduction"] = dict(audio_input["noise_reduction"])
        if isinstance(session_payload.get("audio"), dict):
            extra_body["realtime_audio"] = dict(session_payload["audio"])
        if isinstance(session_payload.get("tracing"), str | dict):
            extra_body["realtime_tracing"] = session_payload["tracing"]
        response_format = self._duplex_response_format(self._output_audio_format)
        extra_body.setdefault("realtime_output_audio_format", self._output_audio_format)
        overlap_fields = self._realtime_overlap_fields(session_payload)
        voice = session_payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        speed = session_payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        return {
            "type": "session.create",
            "session_id": session_payload.get("session_id") or session_payload.get("id"),
            "session": {
                "model": model,
                "modalities": (
                    session_payload.get("modalities") or session_payload.get("output_modalities") or ["text", "audio"]
                ),
                "instructions": session_payload.get("instructions"),
                "voice": voice,
                "ref_audio": session_payload.get("ref_audio"),
                "response_format": response_format,
                "temperature": session_payload.get("temperature"),
                "max_tokens": self.realtime_max_output_tokens(
                    session_payload.get("max_response_output_tokens")
                    or session_payload.get("max_output_tokens")
                    or session_payload.get("max_tokens")
                ),
                "speed": speed,
                "idle_timeout_s": session_payload.get("idle_timeout_s") or 300.0,
                **overlap_fields,
                "extra_body": extra_body,
            },
        }

    def _apply_realtime_session_defaults(self, session_payload: dict[str, object]) -> None:
        input_format: object = session_payload.get("input_audio_format")
        audio_config = session_payload.get("audio")
        if input_format is None and isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict):
                input_format = audio_input.get("format")
        input_format, input_rate = self._parse_realtime_audio_format(input_format)
        if isinstance(input_format, str) and input_format.lower() in REALTIME_INPUT_AUDIO_FORMATS:
            self._input_audio_format = input_format
        output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
        output_rate_raw: object | None = None
        if output_format is None and isinstance(audio_config, dict):
            audio_output = audio_config.get("output")
            if isinstance(audio_output, dict):
                output_format = audio_output.get("format")
                output_rate_raw = audio_output.get("sample_rate_hz") or audio_output.get("sample_rate")
        output_format, output_rate = self._parse_realtime_audio_format(output_format)
        if output_rate is None and isinstance(output_rate_raw, int | float) and output_rate_raw > 0:
            output_rate = int(output_rate_raw)
        if isinstance(output_format, str) and output_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            self._output_audio_format = self._realtime_output_format(output_format)
        sample_rate = session_payload.get("sample_rate_hz") or session_payload.get("sample_rate")
        if sample_rate is None and isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict):
                sample_rate = audio_input.get("sample_rate_hz") or audio_input.get("sample_rate")
        if sample_rate is None:
            sample_rate = input_rate
        if isinstance(sample_rate, int | float) and sample_rate > 0:
            self._input_sample_rate_hz = int(sample_rate)
        if isinstance(output_rate, int | float) and output_rate > 0:
            self._output_sample_rate_hz = int(output_rate)
        overlap_fields = self._realtime_overlap_fields(session_payload)
        overlap_silence_rms = overlap_fields.get("overlap_silence_rms")
        if isinstance(overlap_silence_rms, int | float):
            self._overlap_silence_rms = max(0.0, float(overlap_silence_rms))

    @classmethod
    def _validate_realtime_session_audio_formats(cls, session_payload: dict[str, object]) -> str | None:
        audio_config = session_payload.get("audio")
        input_format: object = session_payload.get("input_audio_format")
        if input_format is None and isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict):
                input_format = audio_input.get("format")
        parsed_input, _ = cls._parse_realtime_audio_format(input_format)
        if input_format is not None and not (
            isinstance(parsed_input, str) and parsed_input.lower() in REALTIME_INPUT_AUDIO_FORMATS
        ):
            return f"Unsupported input_audio_format: {input_format}"

        output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
        if output_format is None and isinstance(audio_config, dict):
            audio_output = audio_config.get("output")
            if isinstance(audio_output, dict):
                output_format = audio_output.get("format")
        parsed_output, _ = cls._parse_realtime_audio_format(output_format)
        if output_format is not None and not (
            isinstance(parsed_output, str) and parsed_output.lower() in REALTIME_OUTPUT_AUDIO_FORMATS
        ):
            return f"Unsupported output_audio_format: {output_format}"
        return None

    @classmethod
    def _validate_realtime_response_audio_formats(cls, response_payload: dict[str, object]) -> str | None:
        output_format: object = response_payload.get("output_audio_format") or response_payload.get("response_format")
        audio_config = response_payload.get("audio")
        if output_format is None and isinstance(audio_config, dict):
            audio_output = audio_config.get("output")
            if isinstance(audio_output, dict):
                output_format = audio_output.get("format")
        parsed_output, _ = cls._parse_realtime_audio_format(output_format)
        if output_format is not None and not (
            isinstance(parsed_output, str) and parsed_output.lower() in REALTIME_OUTPUT_AUDIO_FORMATS
        ):
            return f"Unsupported output_audio_format: {output_format}"
        return None

    @classmethod
    def _validate_conversation_item_audio_formats(cls, item: object) -> str | None:
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
            parsed_format, _ = cls._parse_realtime_audio_format(raw_format)
            if raw_format is not None and not (
                isinstance(parsed_format, str) and parsed_format.lower() in REALTIME_INPUT_AUDIO_FORMATS
            ):
                return f"Unsupported input_audio format in conversation.item.create: {raw_format}"
        return None

    @staticmethod
    def _json_safe_realtime_payload(payload: dict[str, object]) -> dict[str, object]:
        clean: dict[str, object] = {}
        for key, value in payload.items():
            if key == "extra_body":
                continue
            if isinstance(value, str | int | float | bool) or value is None:
                clean[key] = value
            elif isinstance(value, dict):
                clean[key] = RealtimeInputTranslator._json_safe_realtime_payload(value)
            elif isinstance(value, list):
                clean[key] = [
                    (RealtimeInputTranslator._json_safe_realtime_payload(item) if isinstance(item, dict) else item)
                    for item in value
                    if isinstance(item, str | int | float | bool | dict) or item is None
                ]
        return clean

    @staticmethod
    def _parse_realtime_audio_format(raw_format: object) -> tuple[object, int | None]:
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

    @staticmethod
    def _duplex_response_format(realtime_format: str) -> str:
        normalized = realtime_format.lower()
        if normalized in {"pcm16", "pcm_s16le", "s16le"}:
            return "pcm"
        if normalized in {"g711_ulaw", "g711_alaw"}:
            return "pcm"
        if normalized in {"wav", "pcm"}:
            return normalized
        return "wav"

    @staticmethod
    def _realtime_output_format(duplex_format: object) -> str:
        if isinstance(duplex_format, str) and duplex_format.lower() in {"g711_ulaw", "g711_alaw"}:
            return duplex_format.lower()
        if isinstance(duplex_format, str) and duplex_format.lower() == "pcm":
            return "pcm16"
        return str(duplex_format or "wav")

    @staticmethod
    def _is_supported_realtime_input_format(fmt: object) -> bool:
        return isinstance(fmt, str) and fmt.lower() in REALTIME_INPUT_AUDIO_FORMATS

    @staticmethod
    def _input_explicitly_non_speech(event: dict[str, object]) -> bool:
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

    def _input_looks_like_speech(self, event: dict[str, object], *, audio: object, fmt: object) -> bool:
        if RealtimeInputTranslator._input_explicitly_non_speech(event):
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
        if fmt != "pcm_f32le" or not isinstance(audio, str):
            return True
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
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
        silence_rms = float(threshold) if isinstance(threshold, int | float) else self._overlap_silence_rms
        return rms >= max(0.0, silence_rms)

    @staticmethod
    def _copy_realtime_input_hints(source: dict[str, object], target: dict[str, object]) -> None:
        for key in (
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
        ):
            if key in source:
                target[key] = source[key]

    @classmethod
    def _realtime_overlap_fields(cls, session_payload: dict[str, object]) -> dict[str, object]:
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

    @staticmethod
    def _validate_realtime_video_frames(video_frames: object, max_slice_nums: object) -> str | None:
        """Validate omni-duplex camera frames on input_audio_buffer.append.

        Wire contract matches the official MiniCPM-o-Demo omni client: one
        base64 JPEG per ~1 s audio chunk. HD slicing (max_slice_nums > 1) is
        not implemented by the duplex adapter yet and is rejected explicitly
        rather than silently ignored.
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

    @staticmethod
    def _validate_realtime_turn_detection(session_payload: dict[str, object]) -> str | None:
        configured_values: list[tuple[str, object]] = []
        if "turn_detection" in session_payload:
            configured_values.append(("turn_detection", session_payload["turn_detection"]))
        audio_config = session_payload.get("audio")
        if isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict) and "turn_detection" in audio_input:
                configured_values.append(("audio.input.turn_detection", audio_input["turn_detection"]))
        for field_path, turn_detection in configured_values:
            if turn_detection is None:
                continue
            turn_detection_type = turn_detection.get("type") if isinstance(turn_detection, dict) else turn_detection
            return (
                f"{field_path}={turn_detection_type!r} is not implemented by the duplex Realtime adapter; "
                "set turn_detection to null and commit input explicitly, or use the model-owned duplex policy"
            )
        return None

    @staticmethod
    def _input_audio_transcription_config(session_payload: dict[str, object]) -> dict[str, object] | None:
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

    @staticmethod
    def realtime_max_output_tokens(value: object) -> int | None:
        """Normalize Realtime max output tokens.

        OpenAI Realtime clients commonly send ``"inf"`` for unbounded output.
        The duplex core represents that as ``None`` because model-specific
        defaults remain in force.
        """
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"inf", "infinity", "unlimited"}:
            return None
        if isinstance(value, int) and value > 0:
            return int(value)
        return None

    async def _conversation_item_to_duplex(self, event: dict[str, object]) -> dict[str, object] | None:
        item = event.get("item")
        if not isinstance(item, dict):
            return None
        item = self._normalize_conversation_item(item)
        previous_item_id = event.get("previous_item_id")
        if isinstance(previous_item_id, str):
            item["_previous_item_id"] = previous_item_id
        item_id = str(item["id"])
        item_type = item.get("type")
        role = item.get("role")
        if item_type != "message" or role in {"assistant", "system"}:
            return {
                "type": "turn.signal",
                "event": "conversation.item.create",
                "payload": {"item": item},
            }
        self._conversation_items[item_id] = item
        content = item.get("content")
        if not isinstance(content, list):
            return None
        text_chunks: list[str] = []
        audio_events: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"input_text", "text"} and isinstance(part.get("text"), str):
                text_chunks.append(str(part["text"]))
            if part.get("type") in {"input_audio", "audio"}:
                audio = part.get("audio") or part.get("data")
                fmt, format_rate = self._parse_realtime_audio_format(part.get("format") or self._input_audio_format)
                sample_rate_hz = (
                    part.get("sample_rate_hz") or part.get("sample_rate") or format_rate or self._input_sample_rate_hz
                )
                if not self._is_supported_realtime_input_format(fmt):
                    continue
                try:
                    audio, fmt, sample_rate_hz = convert_input_audio_with_rate(
                        audio,
                        fmt,
                        sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int | float) else None,
                    )
                except ValueError as exc:
                    await self._send_realtime_payload(
                        self._realtime_error_payload(
                            "bad_event",
                            str(exc),
                            event_id=event.get("event_id"),
                            param="sample_rate_hz",
                        )
                    )
                    return None
                if not isinstance(audio, str) or not audio:
                    continue
                speech_hints = dict(event)
                speech_hints.update(part)
                if not self._input_looks_like_speech(speech_hints, audio=audio, fmt=fmt):
                    continue
                self._input_speech_started = True
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": audio,
                    "format": fmt,
                    "sample_rate_hz": sample_rate_hz,
                }
                self._copy_realtime_input_hints(part, payload)
                self._copy_realtime_input_hints(event, payload)
                audio_events.append(payload)
        if audio_events:
            self._input_audio_buffer_has_audio = True
            transcript = self._input_transcript_from_item(item)
            for extra_event in audio_events[1:]:
                self._pending_outbound.put_nowait(extra_event)
            commit_payload: dict[str, object] = {
                "type": "input_audio_buffer.commit",
                "final": True,
                "realtime_item_id": item_id,
                "response_create": False,
            }
            if transcript:
                commit_payload["transcript"] = transcript
            self._pending_outbound.put_nowait(commit_payload)
            return audio_events[0]
        if not text_chunks:
            return None
        text_item = dict(item)
        text_item["status"] = "completed"
        return {
            "type": "turn.signal",
            "event": "conversation.item.create",
            "payload": {"item": text_item},
        }

    def _remember_input_transcript_hint(self, event: dict[str, object]) -> None:
        transcript = event.get("transcript")
        if not isinstance(transcript, str):
            transcript = event.get("text") if isinstance(event.get("text"), str) else None
        if not isinstance(transcript, str):
            hints = event.get("hints")
            if isinstance(hints, dict):
                transcript = hints.get("transcript")
                if not isinstance(transcript, str):
                    transcript = hints.get("text") if isinstance(hints.get("text"), str) else None
        if isinstance(transcript, str) and transcript:
            if (
                self._input_audio_buffer_transcript_parts
                and self._input_audio_buffer_transcript_parts[-1] == transcript
            ):
                return
            self._input_audio_buffer_transcript_parts.append(transcript)

    def _consume_input_transcript_hint(self) -> str:
        transcript = "".join(self._input_audio_buffer_transcript_parts).strip()
        self._input_audio_buffer_transcript_parts.clear()
        return transcript

    @staticmethod
    def _input_transcript_from_item(item: dict[str, object]) -> str:
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

    @staticmethod
    def _input_audio_transcription_completed_event(
        item_id: str,
        item: dict[str, object],
    ) -> dict[str, object] | None:
        content = item.get("content")
        if not isinstance(content, list):
            return None
        transcript_parts: list[str] = []
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "input_audio":
                continue
            transcript = part.get("transcript")
            if isinstance(transcript, str) and transcript:
                transcript_parts.append(transcript)
                return {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": index,
                    "transcript": "".join(transcript_parts),
                }
        return None

    def _pop_pending_commit_item_id(self) -> str:
        try:
            return self._pending_commit_item_ids.get_nowait()
        except asyncio.QueueEmpty:
            return f"item_{uuid4().hex}"

    async def _emit_input_speech_started(self, event: dict[str, object]) -> None:
        if self._input_speech_started:
            return
        self._input_speech_started = True
        if self._active_input_item_id is None:
            self._active_input_item_id = f"item_{uuid4().hex}"
        await self._send_realtime_payload(
            {
                "type": "input_audio_buffer.speech_started",
                "audio_start_ms": int(event.get("audio_start_ms", 0) or 0),
                "item_id": self._active_input_item_id,
            }
        )

    async def _emit_input_speech_stopped(self, event: dict[str, object], *, item_id: str) -> None:
        if not self._input_speech_started:
            return
        self._input_speech_started = False
        audio_end_ms = event.get("audio_end_ms", event.get("audio_ms", 0))
        await self._send_realtime_payload(
            {
                "type": "input_audio_buffer.speech_stopped",
                "audio_end_ms": int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                "item_id": item_id,
            }
        )
