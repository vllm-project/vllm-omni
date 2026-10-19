# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Live client-event parsing and server-event construction."""

from __future__ import annotations

import json

import pytest
from openai.types.live import (
    ErrorEvent,
    InfoEvent,
    InputAudioAppendEvent,
    InputAudioMutedEvent,
    InputAudioUnmutedEvent,
    OutputAudioDeltaEvent,
    OutputTranscriptDeltaEvent,
    ServerEvent,
    SessionCloseEvent,
    SessionStartEvent,
    SessionUpdateEvent,
    SessionUsageUpdatedEvent,
)
from pydantic import BaseModel, TypeAdapter

from vllm_omni.entrypoints.openai.live import protocol
from vllm_omni.entrypoints.openai.live.protocol import (
    LiveProtocolError,
    dump_event,
    error_event,
    parse_client_json,
    validate_client_event,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SERVER_EVENT = TypeAdapter(ServerEvent)


class TestParseClientJson:
    def test_valid(self) -> None:
        text = json.dumps({"type": "session.close", "event_id": "c1"})
        assert parse_client_json(text) == {"type": "session.close", "event_id": "c1"}

    def test_event_id_optional(self) -> None:
        assert parse_client_json('{"type": "session.close"}') == {"type": "session.close"}

    @pytest.mark.parametrize("text", ["{", "", "not json", None])
    def test_invalid_json(self, text: str | None) -> None:
        with pytest.raises(LiveProtocolError) as exc_info:
            parse_client_json(text)
        assert exc_info.value.code == "invalid_json"

    @pytest.mark.parametrize("text", ["[]", '"session.close"', "{}", '{"type": 1}', '{"type": null}'])
    def test_needs_object_with_string_type(self, text: str) -> None:
        with pytest.raises(LiveProtocolError, match="string 'type'") as exc_info:
            parse_client_json(text)
        assert exc_info.value.code == "invalid_event"

    def test_rejects_non_string_event_id(self) -> None:
        with pytest.raises(LiveProtocolError) as exc_info:
            parse_client_json('{"type": "session.close", "event_id": 7}')
        assert (exc_info.value.code, exc_info.value.param) == ("invalid_event", "event_id")


class TestValidateClientEvent:
    @pytest.mark.parametrize(
        ("payload", "model"),
        [
            ({"type": "session.close"}, SessionCloseEvent),
            ({"type": "session.input_audio.append", "audio": "AAAA", "event_id": "c1"}, InputAudioAppendEvent),
            ({"type": "session.start", "session": {"model": "m"}}, SessionStartEvent),
            ({"type": "session.update", "session": {}}, SessionUpdateEvent),
        ],
    )
    def test_valid(self, payload: dict, model: type[BaseModel]) -> None:
        assert isinstance(validate_client_event(payload), model)

    @pytest.mark.parametrize(
        ("payload", "param", "message"),
        [
            ({"type": "session.input_audio.append"}, "audio", "Field required"),
            ({"type": "session.input_audio.append", "audio": 5}, "audio", "valid string"),
            ({"type": "session.start"}, "session", "Field required"),
        ],
    )
    def test_malformed(self, payload: dict, param: str, message: str) -> None:
        with pytest.raises(LiveProtocolError, match=message) as exc_info:
            validate_client_event(payload)
        assert exc_info.value.code == "invalid_value"
        assert exc_info.value.param == param
        assert payload["type"] in exc_info.value.message

    def test_known_events_cover_handled_types(self) -> None:
        assert set(protocol.CLIENT_EVENT_MODELS) == {
            "session.start",
            "session.update",
            "session.input_audio.append",
            "session.input_audio.mute",
            "session.input_audio.unmute",
            "session.close",
        }


class TestErrorEvent:
    def test_echoes_client_event_id(self) -> None:
        payload = json.loads(dump_event(error_event(protocol.invalid_value("bad", param="x"), "c1")))
        assert payload["type"] == "error"
        assert payload["client_event_id"] == "c1"
        assert payload["error"] == {
            "code": "invalid_value",
            "message": "bad",
            "type": "invalid_request_error",
            "param": "x",
            "client_event_id": "c1",
        }

    def test_omits_unset_fields(self) -> None:
        payload = json.loads(dump_event(error_event(protocol.session_limit_reached())))
        assert "client_event_id" not in payload
        assert payload["error"] == {
            "code": "session_limit_reached",
            "message": protocol.session_limit_reached().message,
            "type": "server_error",
        }

    @pytest.mark.parametrize(
        ("err", "code", "error_type"),
        [
            (protocol.unsupported_session_config("tools"), "unsupported_session_config", "invalid_request_error"),
            (protocol.internal_error(), "internal_server_error", "server_error"),
            (protocol.session_limit_reached(), "session_limit_reached", "server_error"),
        ],
    )
    def test_error_helpers(self, err: LiveProtocolError, code: str, error_type: str) -> None:
        assert (err.code, err.error_type) == (code, error_type)


class TestServerEventsMatchSchema:
    @pytest.mark.parametrize(
        ("event", "model"),
        [
            (error_event(protocol.invalid_value("bad", param="x"), "c1"), ErrorEvent),
            (error_event(protocol.internal_error()), ErrorEvent),
            (protocol.info("context_trimmed", "trimmed"), InfoEvent),
            (protocol.input_audio_muted("c2"), InputAudioMutedEvent),
            (protocol.input_audio_unmuted(None), InputAudioUnmutedEvent),
            (protocol.output_audio_delta("AAAA"), OutputAudioDeltaEvent),
            (protocol.output_transcript_delta("hi", 10, 20), OutputTranscriptDeltaEvent),
            (protocol.usage_updated(1.5), SessionUsageUpdatedEvent),
        ],
    )
    def test_dump_parses_as_server_event(self, event: BaseModel, model: type[BaseModel]) -> None:
        parsed = SERVER_EVENT.validate_json(dump_event(event))
        assert isinstance(parsed, model)
        assert parsed == event

    def test_transcript_end_never_before_start(self) -> None:
        event = protocol.output_transcript_delta("hi", 10, 5)
        assert (event.start_ms, event.end_ms) == (10, 10)

    def test_event_ids_are_unique(self) -> None:
        assert protocol.info("c", "m").event_id != protocol.info("c", "m").event_id
