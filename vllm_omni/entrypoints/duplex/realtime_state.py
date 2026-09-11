# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, overload

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
REALTIME_PCM_DEFAULT_SAMPLE_RATE_HZ = 24_000
REALTIME_PCM_F32_DEFAULT_SAMPLE_RATE_HZ = 16_000
REALTIME_G711_DEFAULT_SAMPLE_RATE_HZ = 8_000

_RealtimeProtocolConfig = tuple[str, int, str, int | None, float]
_PendingTurnDetectionUpdate = tuple[
    bool,
    dict[str, object] | None,
    asyncio.Event,
    _RealtimeProtocolConfig,
]


def realtime_default_sample_rate_hz(fmt: object) -> int | None:
    if not isinstance(fmt, str):
        return None
    normalized = fmt.lower()
    if normalized in {"pcm16", "pcm_s16le", "s16le", "pcm", "audio/pcm"}:
        return REALTIME_PCM_DEFAULT_SAMPLE_RATE_HZ
    if normalized in {"pcm_f32le", "audio/pcm_f32le"}:
        return REALTIME_PCM_F32_DEFAULT_SAMPLE_RATE_HZ
    if normalized in {"g711_ulaw", "g711_alaw", "audio/g711_ulaw", "audio/g711_alaw"}:
        return REALTIME_G711_DEFAULT_SAMPLE_RATE_HZ
    return None


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
    "server_vad_initialization_failed": "server_error",
    "server_vad_inference_failed": "server_error",
    "duplex_session_busy": "rate_limit_error",
    "resource_exhausted": "rate_limit_error",
    "input_backpressure": "rate_limit_error",
    "response_already_active": "invalid_request_error",
    "response_not_active": "invalid_request_error",
    "response_create_without_input": "invalid_request_error",
    "input_audio_buffer_empty": "invalid_request_error",
    "server_vad_manual_commit_unsupported": "invalid_request_error",
    "missing_item_id": "invalid_request_error",
    "item_not_found": "invalid_request_error",
    "playback_item_mismatch": "invalid_request_error",
    "playback_item_not_found": "invalid_request_error",
    "playback_ack_too_late": "invalid_request_error",
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
    _input_sample_rate_hz: int = REALTIME_PCM_DEFAULT_SAMPLE_RATE_HZ
    _output_audio_format: str = "pcm16"
    _overlap_silence_rms: float = 0.003
    _turn_detection: dict[str, object] | None = None
    _native_input_append: bool = False
    _pending_turn_detection_update: _PendingTurnDetectionUpdate | None = None
    _send_realtime_json: Any = None
    _initial_session_update: bool = False
    _input_speech_started: bool = False
    _response_states: dict[str | int, _RealtimeResponseState] = field(default_factory=dict)
    _completed_response_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    _has_response_lifecycle: bool = False
    _item_truncation_cursors: dict[str, tuple[int, int]] = field(default_factory=dict)
    _active_response_id: str | None = None
    _last_response_id: str | None = None
    _conversation_items: dict[str, dict[str, object]] = field(default_factory=dict)
    _last_conversation_item_id: str | None = None
    _output_sample_rate_hz: int | None = None
    _active_input_item_id: str | None = None
    _input_audio_buffer_has_audio: bool = False
    _input_audio_buffer_had_non_speech: bool = False
    _input_audio_buffer_transcript_parts: list[str] = field(default_factory=list)
    _turn_detection_configured: bool = False

    @classmethod
    def from_query_params(cls, query_params: Any) -> RealtimeSessionState:
        if hasattr(query_params, "query_params"):
            query_params = query_params.query_params
        getter = query_params.get if hasattr(query_params, "get") else None
        # Canonical query param plus the deprecated model-prefixed alias;
        # both seed the canonical extra_body key.
        native_duplex = None
        if getter is not None:
            native_duplex = getter("native_duplex")
            if native_duplex is None:
                native_duplex = getter("minicpmo45_native_duplex")
        resume_only = getter("resume") if getter is not None else None
        autostart = getter("autostart") if getter is not None else None
        default_extra_body: dict[str, object] = {}
        if str(native_duplex).strip().lower() in {"1", "true", "yes", "on"}:
            default_extra_body["native_duplex"] = True
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

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> _RealtimeStateField[_StateValue]: ...

    @overload
    def __get__(self, instance: RealtimeStateOwner, owner: type | None = None) -> _StateValue: ...

    def __get__(
        self, instance: RealtimeStateOwner | None, owner: type | None = None
    ) -> _StateValue | _RealtimeStateField[_StateValue]:
        del owner
        if instance is None:
            return self
        return getattr(instance._state, self._name)

    def __set__(self, instance: RealtimeStateOwner, value: _StateValue) -> None:
        setattr(instance._state, self._name, value)


class RealtimeStateOwner:
    """Declare the mutable state surface shared by input/output projectors."""

    _state: RealtimeSessionState
    _opened = _RealtimeStateField[bool]()
    _autostarted_default_session = _RealtimeStateField[bool]()
    _resume_only = _RealtimeStateField[bool]()
    _pending_outbound = _RealtimeStateField[asyncio.Queue[dict[str, object]]]()
    _held_realtime_payloads = _RealtimeStateField[list[dict[str, object]]]()
    _hold_realtime_output_until_session_created = _RealtimeStateField[bool]()
    _default_model = _RealtimeStateField[object | None]()
    _default_session_id = _RealtimeStateField[object | None]()
    _default_extra_body = _RealtimeStateField[dict[str, object]]()
    _input_audio_format = _RealtimeStateField[str]()
    _input_sample_rate_hz = _RealtimeStateField[int]()
    _output_audio_format = _RealtimeStateField[str]()
    _overlap_silence_rms = _RealtimeStateField[float]()
    _turn_detection = _RealtimeStateField[dict[str, object] | None]()
    _native_input_append = _RealtimeStateField[bool]()
    _pending_turn_detection_update = _RealtimeStateField[_PendingTurnDetectionUpdate | None]()
    _send_realtime_json = _RealtimeStateField[Any]()
    _initial_session_update = _RealtimeStateField[bool]()
    _input_speech_started = _RealtimeStateField[bool]()
    _response_states = _RealtimeStateField[dict[str | int, _RealtimeResponseState]]()
    _completed_response_ids = _RealtimeStateField[OrderedDict[str, None]]()
    _has_response_lifecycle = _RealtimeStateField[bool]()
    _item_truncation_cursors = _RealtimeStateField[dict[str, tuple[int, int]]]()
    _active_response_id = _RealtimeStateField[str | None]()
    _last_response_id = _RealtimeStateField[str | None]()
    _conversation_items = _RealtimeStateField[dict[str, dict[str, object]]]()
    _last_conversation_item_id = _RealtimeStateField[str | None]()
    _output_sample_rate_hz = _RealtimeStateField[int | None]()
    _active_input_item_id = _RealtimeStateField[str | None]()
    _input_audio_buffer_has_audio = _RealtimeStateField[bool]()
    _input_audio_buffer_had_non_speech = _RealtimeStateField[bool]()
    _input_audio_buffer_transcript_parts = _RealtimeStateField[list[str]]()
    _turn_detection_configured = _RealtimeStateField[bool]()
