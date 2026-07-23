from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

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
REALTIME_ERROR_TYPES_BY_CODE = {
    "bad_event": "invalid_request_error",
    "bad_audio": "invalid_request_error",
    "config_timeout": "invalid_request_error",
    "invalid_json": "invalid_request_error",
    "event_too_large": "invalid_request_error",
    "unknown_event": "invalid_request_error",
    "internal_error": "server_error",
    "runtime_open_failed": "server_error",
    "runtime_open_unsupported": "server_error",
    "runtime_append_failed": "server_error",
    "runtime_append_task_failed": "server_error",
    "runtime_signal_failed": "server_error",
    "runtime_close_failed": "server_error",
    "runtime_abort_failed": "server_error",
    "runtime_data_plane_stream_failed": "server_error",
    "runtime_data_plane_text_without_audio": "server_error",
    "response_error": "server_error",
    "chat_error": "server_error",
    "duplex_session_busy": "rate_limit_error",
    "resource_exhausted": "rate_limit_error",
    "response_already_active": "invalid_request_error",
    "response_not_active": "invalid_request_error",
    "response_create_without_input": "invalid_request_error",
    "input_audio_buffer_empty": "invalid_request_error",
    "missing_item_id": "invalid_request_error",
    "item_not_found": "invalid_request_error",
    "unsupported_audio_format": "invalid_request_error",
    "unsupported_ref_audio_path": "invalid_request_error",
    "ref_audio_required": "invalid_request_error",
    "model_update_unsupported": "invalid_request_error",
    "voice_update_after_audio_unsupported": "invalid_request_error",
    "ref_audio_update_unsupported": "invalid_request_error",
    "native_text_append_unsupported": "invalid_request_error",
    "runtime_native_stage_role_required": "server_error",
    "runtime_native_runner_kv_required": "server_error",
}


@dataclass(slots=True)
class _RealtimeResponseState:
    item_id: str
    transcript_parts: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    audio_duration_ms: int | None = None
    audio_text_marks: list[dict[str, int]] = field(default_factory=list)
    audio_delta_emitted: bool = False
    audio_done_emitted: bool = False
    audio_part_added: bool = False
    audio_part_done: bool = False
    text_part_added: bool = False
    text_part_done: bool = False
    output_text_done: bool = False
    output_item_done: bool = False
    conversation_item_done: bool = False
    speak_emitted: bool = False
    done_emitted: bool = False

    @property
    def transcript(self) -> str:
        return "".join(self.transcript_parts)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


@dataclass(slots=True)
class RealtimeSessionState:
    """Single mutable authority for one Realtime protocol session."""

    _opened: bool = False
    _autostarted_default_session: bool = False
    _resume_only: bool = False
    _pending_outbound: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    _held_realtime_payloads: list[dict[str, object]] = field(default_factory=list)
    _hold_realtime_output_until_session_created: bool = True
    _default_model: object | None = None
    _default_session_id: object | None = None
    _default_extra_body: dict[str, object] = field(default_factory=dict)
    _input_audio_format: str = "pcm16"
    _input_sample_rate_hz: int = 16000
    _output_audio_format: str = "pcm16"
    _overlap_silence_rms: float = 0.003
    _send_realtime_json: Any = None
    _initial_session_update: bool = False
    _input_speech_started: bool = False
    _response_states: dict[str | int, _RealtimeResponseState] = field(default_factory=dict)
    _item_truncation_cursors: dict[str, tuple[int, int]] = field(default_factory=dict)
    _active_response_id: str | None = None
    _last_response_id: str | None = None
    _conversation_items: dict[str, dict[str, object]] = field(default_factory=dict)
    _pending_commit_item_ids: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    _last_conversation_item_id: str | None = None
    _output_sample_rate_hz: int | None = None
    _active_input_item_id: str | None = None
    _input_audio_buffer_has_audio: bool = False
    _input_audio_buffer_had_non_speech: bool = False
    _input_audio_buffer_transcript_parts: list[str] = field(default_factory=list)

    @classmethod
    def from_query_params(cls, query_params: Any) -> RealtimeSessionState:
        if hasattr(query_params, "query_params"):
            query_params = query_params.query_params
        getter = query_params.get if hasattr(query_params, "get") else None
        native_duplex = getter("minicpmo45_native_duplex") if getter is not None else None
        resume_only = getter("resume") if getter is not None else None
        autostart = getter("autostart") if getter is not None else None
        default_extra_body = {}
        if str(native_duplex).strip().lower() in {"1", "true", "yes", "on"}:
            default_extra_body["minicpmo45_native_duplex"] = True
        return cls(
            _resume_only=(
                str(resume_only).strip().lower() in {"1", "true", "yes", "on"}
                or str(autostart).strip().lower() in {"0", "false", "no", "off"}
            ),
            _default_model=getter("model") if getter is not None else None,
            _default_session_id=(getter("session_id") if getter is not None else None),
            _default_extra_body=default_extra_body,
        )


_StateValue = TypeVar("_StateValue")


class _RealtimeStateField(Generic[_StateValue]):
    """Explicit descriptor binding one protocol attribute to session state."""

    def __set_name__(self, owner: type, name: str) -> None:
        del owner
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None) -> _StateValue | _RealtimeStateField[_StateValue]:
        del owner
        if instance is None:
            return self
        return getattr(instance._state, self._name)

    def __set__(self, instance: Any, value: _StateValue) -> None:
        setattr(instance._state, self._name, value)


class RealtimeStateOwner:
    """Declare the mutable state surface shared by input/output projectors."""

    _state: RealtimeSessionState
    _opened: bool = _RealtimeStateField()
    _autostarted_default_session: bool = _RealtimeStateField()
    _resume_only: bool = _RealtimeStateField()
    _pending_outbound: asyncio.Queue[dict[str, object]] = _RealtimeStateField()
    _held_realtime_payloads: list[dict[str, object]] = _RealtimeStateField()
    _hold_realtime_output_until_session_created: bool = _RealtimeStateField()
    _default_model: object | None = _RealtimeStateField()
    _default_session_id: object | None = _RealtimeStateField()
    _default_extra_body: dict[str, object] = _RealtimeStateField()
    _input_audio_format: str = _RealtimeStateField()
    _input_sample_rate_hz: int = _RealtimeStateField()
    _output_audio_format: str = _RealtimeStateField()
    _overlap_silence_rms: float = _RealtimeStateField()
    _send_realtime_json: Any = _RealtimeStateField()
    _initial_session_update: bool = _RealtimeStateField()
    _input_speech_started: bool = _RealtimeStateField()
    _response_states: dict[str | int, _RealtimeResponseState] = _RealtimeStateField()
    _item_truncation_cursors: dict[str, tuple[int, int]] = _RealtimeStateField()
    _active_response_id: str | None = _RealtimeStateField()
    _last_response_id: str | None = _RealtimeStateField()
    _conversation_items: dict[str, dict[str, object]] = _RealtimeStateField()
    _pending_commit_item_ids: asyncio.Queue[str] = _RealtimeStateField()
    _last_conversation_item_id: str | None = _RealtimeStateField()
    _output_sample_rate_hz: int | None = _RealtimeStateField()
    _active_input_item_id: str | None = _RealtimeStateField()
    _input_audio_buffer_has_audio: bool = _RealtimeStateField()
    _input_audio_buffer_had_non_speech: bool = _RealtimeStateField()
    _input_audio_buffer_transcript_parts: list[str] = _RealtimeStateField()
