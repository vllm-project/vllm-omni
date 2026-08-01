from __future__ import annotations

from typing import Any
from uuid import uuid4

from vllm_omni.experimental.fullduplex.openai.audio import convert_output_audio
from vllm_omni.experimental.fullduplex.openai.realtime_state import (
    _RealtimeResponseState,
)


class RealtimeOutputProjector:
    """Project internal duplex events onto the OpenAI Realtime schema."""

    def _from_duplex_event(self, event: dict[str, Any]) -> list[dict[str, object]]:
        event_type = event.get("type")
        if event_type == "session.created":
            session = self._realtime_session_payload(event.get("session"))
            created: dict[str, object] = {"type": "session.created", "session": session}
            for key in (
                "incarnation",
                "attachment_generation",
                "resume_token",
            ):
                if key in event:
                    created[key] = event[key]
            payloads: list[dict[str, object]] = [created]
            if self._initial_session_update:
                payloads.append({"type": "session.updated", "session": session})
                self._initial_session_update = False
            self._hold_realtime_output_until_session_created = False
            if self._held_realtime_payloads:
                payloads.extend(self._held_realtime_payloads)
                self._held_realtime_payloads = []
            return payloads
        if event_type == "session.updated":
            session = self._realtime_session_payload(event.get("session"))
            return [{"type": "session.updated", "session": session}]
        if event_type in {
            "session.resumed",
            "session.heartbeat_ack",
            "session.replaced",
            "session.expired",
            "session.resync_required",
        }:
            if event_type == "session.resumed":
                self._hold_realtime_output_until_session_created = False
            return [dict(event)]
        if event_type == "response.created":
            response_id = event.get("response_id")
            if isinstance(response_id, str) and response_id:
                self._active_response_id = response_id
                self._last_response_id = response_id
            item_id = self._response_item_id(response_id)
            modalities = event.get("modalities")
            has_audio_modality = not isinstance(modalities, list) or "audio" in modalities
            item = {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
            self._conversation_items[item_id] = item
            payloads = [
                self._response_created_event(event),
                *self._conversation_item_added_events(item),
                *self._response_output_item_added_events(response_id=response_id, item=item),
            ]
            if has_audio_modality:
                payloads.extend(self._ensure_response_audio_part_added(response_id))
            return payloads
        if event_type == "response.listen":
            return [
                {
                    "type": "response.listen",
                    "session_id": event.get("session_id"),
                    "epoch": event.get("epoch"),
                    "response": {
                        "object": "realtime.response",
                        "status": "listening",
                        "metadata": event,
                    },
                }
            ]
        if event_type == "response.speak":
            response_id = event.get("response_id")
            state = self._response_state(response_id)
            if state is not None:
                if state.speak_emitted:
                    return []
                state.speak_emitted = True
            return [
                {
                    "type": "response.speak",
                    "response_id": response_id,
                    "item_id": self._response_item_id(response_id),
                    "output_index": 0,
                    "content_index": 0,
                    "metadata": self._response_speak_metadata(event),
                }
            ]
        if event_type == "overlap.decision":
            return [
                {
                    "type": "overlap.decision",
                    "session_id": event.get("session_id"),
                    "epoch": event.get("epoch"),
                    "policy": event.get("policy"),
                    "action": event.get("action"),
                    "reason": event.get("reason"),
                    "metadata": event,
                }
            ]
        if event_type == "response.output_audio.delta":
            response_id = event.get("response_id")
            audio = event.get("audio", "")
            payloads: list[dict[str, object]] = []
            if isinstance(audio, str) and audio:
                payloads.extend(self._ensure_response_audio_part_added(response_id))
                payloads.extend(self._realtime_audio_delta_events(event, response_id, audio))
                self._refresh_in_progress_response_item(response_id)
            text = event.get("text")
            has_text = isinstance(text, str) and bool(text)
            if has_text:
                self._append_response_transcript(response_id, text)
                self._refresh_in_progress_response_item(response_id)
            # Keep the audio.delta + transcript.delta pair invariant even for
            # text-less units (deduplicated continuations, turn-end flush):
            # clients that treat the pair as unit-complete would otherwise
            # wait on a transcript that never comes.
            emit_transcript = has_text or (isinstance(audio, str) and bool(audio))
            if emit_transcript:
                if not has_text:
                    text = ""
                payloads.append(
                    {
                        "type": "response.audio_transcript.delta",
                        "response_id": response_id,
                        "item_id": self._response_item_id(response_id),
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    }
                )
            if event.get("end_of_turn") is True:
                payloads.extend(self._realtime_audio_done_events(event, response_id))
                payloads.extend(
                    self._realtime_response_terminal_events(
                        event,
                        response_id,
                        status="completed",
                        status_details={
                            "type": "completed",
                            "reason": event.get("finish_reason") or "stop",
                        },
                    )
                )
            return payloads
        if event_type == "response.text.delta":
            response_id = event.get("response_id")
            text = event.get("delta", "")
            state = self._response_state(response_id)
            if state is not None and isinstance(text, str) and text:
                state.text_parts.append(text)
                self._refresh_in_progress_response_item(response_id)
            payloads = self._ensure_response_text_part_added(response_id)
            payloads.append(
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "item_id": self._response_item_id(response_id),
                    "output_index": 0,
                    "content_index": 1 if state is not None and state.audio_part_added else 0,
                    "delta": text,
                }
            )
            return payloads
        if event_type == "response.done":
            response_id = event.get("response_id")
            status = event.get("status") if isinstance(event.get("status"), str) else "completed"
            status_details = event.get("status_details") if isinstance(event.get("status_details"), dict) else None
            return [
                *self._realtime_audio_done_events(event, response_id),
                *self._realtime_response_terminal_events(
                    event,
                    response_id,
                    status=status,
                    status_details=status_details,
                ),
            ]
        if event_type == "input.committed":
            event_item_id = event.get("realtime_item_id")
            item_id = (
                event_item_id
                if isinstance(event_item_id, str) and event_item_id
                else self._pop_pending_commit_item_id()
            )
            item = self._conversation_items.get(item_id)
            created_payload: list[dict[str, object]] = []
            if item is None:
                message = event.get("message")
                no_response = event.get("no_response") is True
                is_speech = event.get("is_speech")
                item = {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                    "content": (
                        [{"type": "input_audio", "transcript": "", "is_speech": False}]
                        if no_response and is_speech is False
                        else self._user_item_content_from_duplex_message(message)
                    ),
                }
                self._conversation_items[item_id] = item
                created_payload.extend(self._conversation_item_added_events(item))
            item["status"] = "completed"
            payloads = created_payload + [
                self._input_audio_buffer_committed_event(item_id=item_id, event=event),
            ]
            transcription_event = self._input_audio_transcription_completed_event(item_id, item)
            if transcription_event is not None:
                payloads.append(transcription_event)
            payloads.append(self._conversation_item_done_event(item))
            return payloads
        if event_type == "input.cancelled":
            return [{"type": "input_audio_buffer.cleared"}]
        if event_type == "input_audio_buffer.cleared":
            return [{"type": "input_audio_buffer.cleared"}]
        if event_type == "audio.cancelled":
            response_id = event.get("response_id")
            payloads: list[dict[str, object]] = []
            if event.get("reason") == "output_audio_buffer_clear":
                if not isinstance(response_id, str) or not response_id:
                    response_id = self._active_response_id or self._last_response_id
                payloads.append(
                    {
                        "type": "output_audio_buffer.cleared",
                        "response_id": response_id,
                    }
                )
                if not isinstance(response_id, str) or not response_id:
                    return payloads
            elif not isinstance(response_id, str) or not response_id:
                response_id = self._active_response_id
            if not isinstance(response_id, str) or not response_id:
                return payloads
            if self._response_is_done(response_id):
                return payloads
            committed_ms = event.get("committed_ms")
            if isinstance(committed_ms, int | float):
                item_id = self._response_item_id(response_id)
                committed_audio_ms = max(0, int(committed_ms))
                self._item_truncation_cursors[item_id] = (0, committed_audio_ms)
                item = self._conversation_items.get(item_id)
                if item is not None:
                    self._truncate_realtime_item_content(
                        item,
                        content_index=0,
                        audio_end_ms=committed_audio_ms,
                    )
                payloads.append(
                    {
                        "type": "conversation.item.truncated",
                        "item_id": item_id,
                        "content_index": 0,
                        "audio_end_ms": committed_audio_ms,
                        "event": event,
                    }
                )
            payloads.extend(self._realtime_audio_done_events(event, response_id))
            payloads.extend(
                self._realtime_response_terminal_events(
                    event,
                    response_id,
                    status="cancelled",
                    status_details={
                        "type": "cancelled",
                        "reason": event.get("reason") or "client_cancelled",
                    },
                )
            )
            if isinstance(response_id, str) and response_id == self._active_response_id:
                self._active_response_id = None
            return payloads
        if event_type == "playback.acknowledged":
            return [{"type": "playback.acknowledged", "event": event}]
        if event_type == "conversation.item.created":
            item = event.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                item_id = str(item["id"])
                already_known = item_id in self._conversation_items
                self._conversation_items[item_id] = item
                if already_known:
                    if item.get("status") == "completed":
                        return [self._conversation_item_done_event(item)]
                    return []
                payloads = self._conversation_item_added_events(item)
                if item.get("status") == "completed":
                    payloads.append(self._conversation_item_done_event(item))
                return payloads
            return [{"type": "conversation.item.created", "item": item, "event": event}]
        if event_type == "conversation.item.deleted":
            item_id = event.get("item_id")
            if isinstance(item_id, str):
                self._remove_conversation_item(item_id)
            return [
                {
                    "type": "conversation.item.deleted",
                    "item_id": item_id,
                    "event": event,
                }
            ]
        if event_type == "conversation.item.truncated":
            item_id = event.get("item_id")
            audio_end_ms = event.get("audio_end_ms")
            content_index = event.get("content_index", 0)
            if isinstance(item_id, str):
                item = self._conversation_items.get(item_id)
                if item is not None:
                    self._truncate_realtime_item_content(
                        item,
                        content_index=int(content_index) if isinstance(content_index, int | float) else 0,
                        audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                    )
            return [
                {
                    "type": "conversation.item.truncated",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                    "event": event,
                }
            ]
        if event_type == "response.output_item.done":
            response_id = event.get("response_id")
            return self._realtime_response_terminal_events(
                event,
                response_id,
                status="completed",
                status_details={
                    "type": "completed",
                    "reason": event.get("finish_reason") or "stop",
                },
            )
        if event_type == "error":
            raw_error = event.get("error")
            if isinstance(raw_error, dict):
                return [event]
            message = str(raw_error or event.get("message") or "Duplex runtime error")
            code = str(event.get("code") or "duplex_error")
            return [self._realtime_error_payload(code, message)]
        if event_type == "session.closed":
            return [{"type": "session.closed", "event": event}]
        return [{"type": f"duplex.{event_type}", "event": event}]

    @staticmethod
    def _user_item_content_from_duplex_message(message: object) -> list[dict[str, object]]:
        if not isinstance(message, dict):
            return [{"type": "input_audio", "transcript": ""}]
        message_transcript = message.get("transcript")
        content = message.get("content")
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
        if not isinstance(content, list):
            return [
                {
                    "type": "input_audio",
                    "transcript": message_transcript if isinstance(message_transcript, str) else "",
                }
            ]
        parts: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
                parts.append({"type": "input_text", "text": part["text"]})
                continue
            if part_type == "audio_url":
                transcript = (
                    part.get("transcript")
                    if isinstance(part.get("transcript"), str)
                    else message_transcript
                    if isinstance(message_transcript, str)
                    else ""
                )
                parts.append({"type": "input_audio", "transcript": transcript})
                continue
            if part_type in {"audio", "input_audio"}:
                transcript = part.get("transcript") if isinstance(part.get("transcript"), str) else ""
                parts.append({"type": "input_audio", "transcript": transcript})
        return parts or [{"type": "input_audio", "transcript": ""}]

    def _realtime_session_payload(self, session: object) -> object:
        if not isinstance(session, dict):
            return session
        payload = dict(session)
        payload.setdefault("object", "realtime.session")
        payload.setdefault("type", "realtime")
        payload.setdefault("id", payload.get("id") or self._default_session_id)
        payload.setdefault("model", payload.get("model") or self._default_model)
        payload.setdefault("input_audio_format", self._input_audio_format)
        payload.setdefault("output_audio_format", self._output_audio_format)
        payload.setdefault("modalities", payload.get("modalities") or ["text", "audio"])
        payload.setdefault("output_modalities", payload.get("output_modalities") or payload.get("modalities"))
        payload.setdefault("object", "realtime.session")
        payload.setdefault(
            "audio",
            {
                "input": {
                    "format": self._realtime_audio_format_object(
                        self._input_audio_format,
                        sample_rate_hz=self._input_sample_rate_hz,
                    ),
                    "sample_rate_hz": self._input_sample_rate_hz,
                },
                "output": {
                    "format": self._realtime_audio_format_object(
                        self._output_audio_format,
                        sample_rate_hz=self._output_sample_rate_hz,
                    ),
                },
            },
        )
        payload.setdefault("turn_detection", payload.get("turn_detection"))
        payload.setdefault("input_audio_transcription", payload.get("input_audio_transcription"))
        payload.setdefault("tracing", payload.get("tracing"))
        return payload

    @staticmethod
    def _realtime_audio_format_object(fmt: object, *, sample_rate_hz: int | None = None) -> dict[str, object]:
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

    def _response_state(
        self,
        response_id: object,
        *,
        event: dict[str, Any] | None = None,
        create: bool = True,
    ) -> _RealtimeResponseState | None:
        if isinstance(response_id, str) and response_id:
            key: str | int = response_id
            item_id = f"item_{response_id}"
        elif event is not None:
            key = id(event)
            item_id = f"item_{uuid4().hex}"
        else:
            return None
        state = self._response_states.get(key)
        if state is None and create:
            state = _RealtimeResponseState(item_id=item_id)
            self._response_states[key] = state
        return state

    def _response_is_done(self, response_id: object) -> bool:
        state = self._response_state(response_id, create=False)
        return state is not None and state.done_emitted

    def _response_item_id(self, response_id: object) -> str:
        state = self._response_state(response_id)
        if state is not None:
            return state.item_id
        return f"item_{uuid4().hex}"

    def _response_done_output_item(
        self,
        response_id: object,
        *,
        status: str,
    ) -> dict[str, object]:
        item_id = self._response_item_id(response_id)
        state = self._response_state(response_id)
        transcript = state.transcript if state is not None else ""
        text = state.text if state is not None else ""
        audio_duration_ms = state.audio_duration_ms if state is not None else None
        audio_text_marks = state.audio_text_marks if state is not None else None
        content: list[dict[str, object]] = []
        if (state is not None and state.audio_part_added) or transcript or audio_duration_ms is not None:
            content.append(
                self._response_item_content_part(
                    transcript=transcript,
                    audio_duration_ms=audio_duration_ms,
                    audio_text_marks=audio_text_marks,
                )
            )
        if text:
            content.append(self._response_item_text_content_part(text=text))
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": content,
        }
        self._apply_pending_item_truncation(item)
        return item

    def _apply_pending_item_truncation(self, item: dict[str, object]) -> None:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            return
        cursor = self._item_truncation_cursors.get(item_id)
        if cursor is None:
            return
        content_index, audio_end_ms = cursor
        self._truncate_realtime_item_content(
            item,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        )

    def _refresh_in_progress_response_item(self, response_id: object) -> None:
        if not isinstance(response_id, str) or not response_id:
            return
        item_id = self._response_item_id(response_id)
        item = self._conversation_items.get(item_id)
        if not isinstance(item, dict):
            return
        content = item.get("content")
        if not isinstance(content, list):
            content = []
            item["content"] = content

        state = self._response_state(response_id)
        if state is None:
            return
        transcript = state.transcript
        audio_duration_ms = state.audio_duration_ms
        audio_text_marks = state.audio_text_marks
        has_audio = state.audio_part_added or bool(transcript) or audio_duration_ms is not None
        if has_audio:
            audio_part = self._response_item_content_part(
                transcript=transcript,
                audio_duration_ms=audio_duration_ms,
                audio_text_marks=audio_text_marks,
            )
            if content and isinstance(content[0], dict) and content[0].get("type") in {"audio", "output_audio"}:
                content[0] = audio_part
            else:
                content.insert(0, audio_part)

        text = state.text
        if text:
            text_index = (
                1
                if content
                and isinstance(content[0], dict)
                and content[0].get("type")
                in {
                    "audio",
                    "output_audio",
                }
                else 0
            )
            text_part = self._response_item_text_content_part(text=text)
            if (
                len(content) > text_index
                and isinstance(content[text_index], dict)
                and content[text_index].get("type") in {"text", "output_text"}
            ):
                content[text_index] = text_part
            else:
                content.insert(text_index, text_part)

        self._apply_pending_item_truncation(item)

    def _append_response_transcript(self, response_id: object, text: str) -> None:
        state = self._response_state(response_id)
        if state is None or not text:
            return
        state.transcript_parts.append(text)

    def _ensure_response_text_part_added(self, response_id: object) -> list[dict[str, object]]:
        state = self._response_state(response_id)
        if state is None:
            return []
        if state.text_part_added:
            return []
        state.text_part_added = True
        content_index = 1 if state.audio_part_added else 0
        return [
            {
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": self._response_item_id(response_id),
                "output_index": 0,
                "content_index": content_index,
                "part": self._response_text_content_part(),
            }
        ]

    def _ensure_response_audio_part_added(self, response_id: object) -> list[dict[str, object]]:
        state = self._response_state(response_id)
        if state is None:
            return []
        if state.audio_part_added:
            return []
        state.audio_part_added = True
        return [
            {
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": self._response_item_id(response_id),
                "output_index": 0,
                "content_index": 0,
                "part": self._response_content_part(),
            }
        ]

    def _remember_response_audio_metadata(self, response_id: object, event: dict[str, Any]) -> None:
        state = self._response_state(response_id)
        if state is None:
            return
        duration = event.get("audio_duration_ms")
        playback = event.get("playback")
        if not isinstance(duration, int | float) and isinstance(playback, dict):
            duration = playback.get("sent_ms") or playback.get("generated_ms")
        if isinstance(duration, int | float):
            state.audio_duration_ms = max(state.audio_duration_ms or 0, int(duration))
        marks = event.get("audio_text_marks")
        if not isinstance(marks, list):
            return
        clean_marks: list[dict[str, int]] = []
        for mark in marks:
            if not isinstance(mark, dict):
                continue
            text_chars = mark.get("text_chars")
            audio_end_ms = mark.get("audio_end_ms", mark.get("audio_ms"))
            if not isinstance(text_chars, int | float) or not isinstance(audio_end_ms, int | float):
                continue
            clean_marks.append(
                {
                    "text_chars": max(0, int(text_chars)),
                    "audio_end_ms": max(0, int(audio_end_ms)),
                }
            )
        if clean_marks:
            merged = list(state.audio_text_marks)
            merged.extend(clean_marks)
            deduped: dict[tuple[int, int], dict[str, int]] = {}
            for mark in merged:
                deduped[(int(mark["audio_end_ms"]), int(mark["text_chars"]))] = mark
            state.audio_text_marks = sorted(
                deduped.values(),
                key=lambda mark: (mark["audio_end_ms"], mark["text_chars"]),
            )

    def _response_created_event(self, event: dict[str, Any]) -> dict[str, object]:
        response_id = event.get("response_id")
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = {**metadata, "duplex_event": event}
        return {
            "type": "response.created",
            "response_id": response_id,
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "status_details": None,
                "output": [],
                "modalities": event.get("modalities") or ["audio", "text"],
                "metadata": metadata,
            },
        }

    def _realtime_audio_delta_events(
        self,
        event: dict[str, Any],
        response_id: object,
        audio: str,
    ) -> list[dict[str, object]]:
        item_id = self._response_item_id(response_id)
        fmt, format_rate = self._parse_realtime_audio_format(event.get("format", "wav"))
        source_fmt = self._realtime_output_format(fmt)
        source_sample_rate_hz = event.get("sample_rate_hz") or format_rate
        target_sample_rate_hz = (
            self._output_sample_rate_hz if self._output_audio_format in {"g711_ulaw", "g711_alaw"} else None
        )
        audio, fmt, converted_sample_rate_hz = convert_output_audio(
            audio,
            source_fmt=source_fmt,
            target_fmt=self._output_audio_format,
            source_sample_rate_hz=(
                int(source_sample_rate_hz) if isinstance(source_sample_rate_hz, int | float) else None
            ),
            target_sample_rate_hz=target_sample_rate_hz,
        )
        sample_rate_hz = (
            converted_sample_rate_hz
            if fmt in {"g711_ulaw", "g711_alaw"}
            else event.get("sample_rate_hz") or format_rate or self._output_sample_rate_hz
        )
        metadata: dict[str, object] = {}
        for key in ("session_id", "epoch", "model_speak", "end_of_turn", "playback", "vllm_omni"):
            if key in event:
                metadata[key] = event[key]
        duration_ms = event.get("audio_duration_ms")
        if isinstance(duration_ms, int | float):
            metadata["audio_duration_ms"] = int(duration_ms)
        marks = event.get("audio_text_marks")
        if isinstance(marks, list):
            metadata["audio_text_marks"] = marks
        state = self._response_state(response_id)
        if state is not None:
            state.audio_delta_emitted = True
            self._remember_response_audio_metadata(response_id, event)
        payloads: list[dict[str, object]] = []
        if state is not None and not state.speak_emitted and metadata.get("model_speak") is True:
            state.speak_emitted = True
            payloads.insert(
                0,
                {
                    "type": "response.speak",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "metadata": self._response_speak_metadata(event),
                },
            )
        payloads.append(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": audio,
                "format": fmt,
                **({"sample_rate_hz": int(sample_rate_hz)} if isinstance(sample_rate_hz, int | float) else {}),
                **({"metadata": metadata} if metadata else {}),
            }
        )
        return payloads

    @staticmethod
    def _response_speak_metadata(event: dict[str, Any]) -> dict[str, object]:
        return {key: event[key] for key in ("session_id", "epoch", "model_speak") if key in event}

    def _realtime_audio_done_events(
        self,
        event: dict[str, Any],
        response_id: object,
    ) -> list[dict[str, object]]:
        item_id = self._response_item_id(response_id)
        state = self._response_state(response_id, event=event)
        transcript = state.transcript if state is not None else ""
        payloads: list[dict[str, object]] = []
        if isinstance(response_id, str) and state is not None and not state.audio_delta_emitted and not transcript:
            return payloads
        if state is None or not state.audio_done_emitted:
            if state is not None:
                state.audio_done_emitted = True
            payloads.append(
                {
                    "type": "response.audio.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                }
            )
            if transcript:
                payloads.append(
                    {
                        "type": "response.audio_transcript.done",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "transcript": transcript,
                    }
                )
        return payloads

    def _realtime_response_terminal_events(
        self,
        event: dict[str, Any],
        response_id: object,
        *,
        status: str = "completed",
        status_details: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        item_id = self._response_item_id(response_id)
        state = self._response_state(response_id, event=event)
        transcript = state.transcript if state is not None else ""
        payloads: list[dict[str, object]] = []
        if state is not None and state.audio_part_added and not state.audio_part_done:
            state.audio_part_done = True
            payloads.append(
                {
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": self._response_content_part(transcript=transcript),
                }
            )
        if state is not None and state.text_parts and not state.output_text_done:
            state.output_text_done = True
            content_index = 1 if state.audio_part_added else 0
            payloads.append(
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": content_index,
                    "text": state.text,
                }
            )
        if state is not None and state.text_parts and not state.text_part_done:
            state.text_part_done = True
            content_index = 1 if state.audio_part_added else 0
            payloads.append(
                {
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": content_index,
                    "part": self._response_text_content_part(text=state.text),
                }
            )
        if state is None or not state.output_item_done:
            if state is not None:
                state.output_item_done = True
            item = self._response_done_output_item(response_id, status=status)
            payloads.append(
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": item,
                }
            )
            if state is None or not state.conversation_item_done:
                if state is not None:
                    state.conversation_item_done = True
                payloads.append(self._conversation_item_done_event(item))
        done_event = self._realtime_response_done_event(
            {**event, "response_id": response_id},
            state=state,
            status=status,
            status_details=status_details,
        )
        if done_event is not None:
            payloads.append(done_event)
            payloads.append(self._rate_limits_updated_event())
            if isinstance(response_id, str) and response_id == self._active_response_id:
                self._active_response_id = None
        return payloads

    def _realtime_response_done_event(
        self,
        event: dict[str, Any],
        *,
        state: _RealtimeResponseState | None = None,
        status: str = "completed",
        status_details: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        response_id = event.get("response_id")
        if state is None:
            state = self._response_state(response_id, event=event)
        if state is not None:
            if state.done_emitted:
                return None
            state.done_emitted = True
        return {
            "type": "response.done",
            "response_id": response_id,
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": status,
                "status_details": status_details,
                "output": [self._response_done_output_item(response_id, status=status)],
                "metadata": event,
            },
        }

    @staticmethod
    def _rate_limits_updated_event() -> dict[str, object]:
        # vLLM-Omni does not yet expose a Realtime-specific quota budget. Emit
        # the terminal event with an empty list so clients that sequence on
        # rate_limits.updated do not need a private vLLM branch.
        return {"type": "rate_limits.updated", "rate_limits": []}
