# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Typed duplex events: pure ``to_realtime()`` rendering and the stateful Realtime projection."""

from __future__ import annotations

import base64
import inspect

import pytest

from vllm_omni.engine.duplex import events as events_module
from vllm_omni.engine.duplex.events import (
    REALTIME_ERROR_TYPES_BY_CODE,
    AudioDelta,
    DuplexEvent,
    DuplexRawEvent,
    ErrorEvent,
    InputCommitted,
    ItemAdded,
    ItemCreated,
    ItemDone,
    Listen,
    OutputItemAdded,
    OverlapDecision,
    ResponseDone,
    SessionClosed,
    SessionCreated,
    SessionExpired,
    SessionHeartbeatAck,
    SessionReplaced,
    SessionResumed,
    SessionResyncRequired,
    TextDelta,
    TranscriptDone,
    TurnEvent,
    error_event,
)
from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    project_internal_event,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: Classes whose wire object carries ``session_id`` by contract.
_SESSION_ID_ON_WIRE = {
    SessionClosed,
    SessionExpired,
    SessionHeartbeatAck,
    Listen,
    TurnEvent,
    OverlapDecision,
    SessionResumed,
    SessionReplaced,
    SessionResyncRequired,
}
#: Classes whose wire object carries ``epoch`` by contract.
_EPOCH_ON_WIRE = {Listen, TurnEvent, OverlapDecision}


def _public_event_classes() -> list[type[DuplexEvent]]:
    classes = [
        getattr(events_module, name)
        for name in events_module.__all__
        if inspect.isclass(getattr(events_module, name)) and issubclass(getattr(events_module, name), DuplexEvent)
    ]
    assert len(classes) > 40
    return [cls for cls in classes if cls is not DuplexEvent]


@pytest.mark.parametrize("event_cls", _public_event_classes(), ids=lambda cls: cls.__name__)
def test_to_realtime_renders_wire_type_event_id_and_documented_fields(event_cls: type[DuplexEvent]):
    event = event_cls(session_id="duplex-hidden", epoch=7)

    wire = event.to_realtime()

    assert wire["type"] == event.type
    assert wire["event_id"] == event.event_id
    assert wire["event_id"].startswith("event_")
    if event_cls is DuplexRawEvent:
        assert wire["type"] == "duplex."
    else:
        assert wire["type"] == event_cls.wire_type
    if event_cls in _SESSION_ID_ON_WIRE:
        assert wire["session_id"] == "duplex-hidden"
    else:
        assert "session_id" not in wire
    if event_cls in _EPOCH_ON_WIRE:
        assert wire["epoch"] == 7
    else:
        assert "epoch" not in wire
    # Every value is plain JSON (mappings / sequences are copied into dict / list).
    for value in wire.values():
        assert isinstance(value, str | int | float | bool | dict | list) or value is None


def test_optional_wire_fields_are_omitted_when_none():
    created = SessionCreated(session={"id": "sid"})
    assert created.optional_wire_fields == frozenset({"attachment_generation", "resume_token"})
    wire = created.to_realtime()
    assert "attachment_generation" not in wire
    assert "resume_token" not in wire
    assert wire["session"] == {"id": "sid"}

    with_resume = SessionCreated(session={"id": "sid"}, attachment_generation=2, resume_token="tok").to_realtime()
    assert with_resume["attachment_generation"] == 2
    assert with_resume["resume_token"] == "tok"

    delta = AudioDelta(response_id="resp_1", item_id="item_resp_1", delta="AAAA").to_realtime()
    assert "sample_rate_hz" not in delta
    assert "metadata" not in delta
    assert delta["format"] == "pcm16"
    assert delta["output_index"] == 0
    assert delta["content_index"] == 0

    anonymous_listen = Listen(session_id="sid", epoch=0, details={"reason": "silence"}).to_realtime()
    assert "response_id" not in anonymous_listen
    assert "id" not in anonymous_listen["response"]
    assert anonymous_listen["response"]["status"] == "listening"
    assert anonymous_listen["response"]["metadata"] == {"reason": "silence"}

    bound_listen = Listen(session_id="sid", epoch=0, response_id="resp_1").to_realtime()
    assert bound_listen["response_id"] == "resp_1"
    assert bound_listen["response"]["id"] == "resp_1"


def test_error_event_maps_codes_to_openai_error_types_and_echoes_client_event_id():
    assert REALTIME_ERROR_TYPES_BY_CODE["resource_exhausted"] == "rate_limit_error"
    assert REALTIME_ERROR_TYPES_BY_CODE["internal_error"] == "server_error"
    assert REALTIME_ERROR_TYPES_BY_CODE["bad_event"] == "invalid_request_error"

    error = ErrorEvent(
        code="resource_exhausted",
        message="too many sessions",
        related_event_id="evt_client",
        param="session",
        extra={"retryable": True},
    )

    assert error.error_type == "rate_limit_error"
    assert error.text == "too many sessions"
    assert error.error == {
        "type": "rate_limit_error",
        "code": "resource_exhausted",
        "message": "too many sessions",
        "event_id": "evt_client",
        "param": "session",
        "retryable": True,
    }
    wire = error.to_realtime()
    assert wire["type"] == "error"
    assert wire["error"] == error.error
    # The server event id and the echoed client event id are distinct.
    assert wire["event_id"] != "evt_client"
    assert ErrorEvent(code="never-seen-code").error_type == "invalid_request_error"


def test_error_event_helper_normalizes_optional_client_event_id_and_param():
    event = error_event("bad_event", "nope", event_id="evt_1", param="item", extra={"retryable": False})
    assert isinstance(event, ErrorEvent)
    assert event.related_event_id == "evt_1"
    assert event.param == "item"
    assert event.error["retryable"] is False

    anonymous = error_event("bad_event", "nope", event_id=42, param="")
    assert anonymous.related_event_id is None
    assert anonymous.param is None
    assert "event_id" not in anonymous.error
    assert "param" not in anonymous.error


def test_audio_delta_decodes_base64_and_invalid_payloads_yield_none():
    raw = bytes(range(16))
    delta = AudioDelta(delta=base64.b64encode(raw).decode("ascii"), format="pcm16", sample_rate_hz=24000)

    assert delta.audio == raw
    assert delta.to_realtime()["sample_rate_hz"] == 24000
    assert AudioDelta(delta="@@not base64@@").audio is None


def test_session_closed_and_expired_are_the_only_terminal_events():
    closed = SessionClosed(session_id="sid", reason="client_close", details={"type": "session.closed"})
    expired = SessionExpired(session_id="sid", reason="idle_timeout")

    assert closed.is_terminal is True
    assert expired.is_terminal is True
    assert closed.to_realtime() == {
        "type": "session.closed",
        "event_id": closed.event_id,
        "session_id": "sid",
        "reason": "client_close",
        "event": {"type": "session.closed"},
    }
    assert expired.to_realtime() == {
        "type": "session.expired",
        "event_id": expired.event_id,
        "session_id": "sid",
        "reason": "idle_timeout",
    }
    non_terminal = [cls for cls in _public_event_classes() if cls().is_terminal]
    assert non_terminal == [SessionClosed, SessionExpired]


def test_generic_accessors_expose_response_item_text_and_status():
    assert TextDelta(response_id="resp_1", item_id="item_resp_1", delta="hi").text == "hi"
    assert TranscriptDone(transcript="done").text == "done"
    assert OutputItemAdded(item={"id": "item_1"}).item_id == "item_1"
    assert ItemAdded(item={"id": 3}).item_id is None
    assert ResponseDone(response={"status": "cancelled"}).status == "cancelled"
    assert ResponseDone().status is None
    assert TurnEvent(event="user_started", turn_state="user_speaking").response_id is None
    assert TurnEvent().audio is None


def test_raw_event_prefixes_internal_type_and_wraps_details():
    raw = DuplexRawEvent(internal_type="model.custom", details={"type": "model.custom", "value": (1, 2)})

    assert raw.type == "duplex.model.custom"
    assert raw.to_realtime() == {
        "type": "duplex.model.custom",
        "event_id": raw.event_id,
        "event": {"type": "model.custom", "value": [1, 2]},
    }


def test_turn_event_and_overlap_decision_carry_session_identity_on_the_wire():
    turn = TurnEvent(session_id="sid", epoch=2, event="barge_in", turn_state="barge_in").to_realtime()
    assert turn == {
        "type": "turn.event",
        "event_id": turn["event_id"],
        "session_id": "sid",
        "event": "barge_in",
        "turn_state": "barge_in",
        "epoch": 2,
    }
    decision = OverlapDecision(
        session_id="sid", epoch=2, policy="listen_only", action="ignore", reason="short", details={"ms": 300}
    ).to_realtime()
    assert decision["policy"] == "listen_only"
    assert decision["action"] == "ignore"
    assert decision["metadata"] == {"ms": 300}
    assert decision["session_id"] == "sid"


# ---- RealtimeProjectionState / project_internal_event ----


def _types(events: list[DuplexEvent]) -> list[str]:
    return [event.type for event in events]


def test_projection_of_the_main_internal_event_sequence():
    state = RealtimeProjectionState(session_id="duplex-proj", model="test-model", initial_session_update=True)
    pcm = base64.b64encode(b"\x00\x10" * 8).decode("ascii")

    created = project_internal_event(state, {"type": "session.created", "session": {"voice": "alloy"}})
    assert _types(created) == ["session.created", "session.updated"]
    assert isinstance(created[0], SessionCreated)
    session_payload = created[0].session
    assert session_payload["id"] == "duplex-proj"
    assert session_payload["model"] == "test-model"
    assert session_payload["voice"] == "alloy"
    assert session_payload["input_audio_format"] == "pcm16"
    assert session_payload["output_audio_format"] == "pcm16"
    assert state.initial_session_update is False
    # The extra session.updated is a one-shot for the opening session.update.
    assert _types(project_internal_event(state, {"type": "session.created", "session": {}})) == ["session.created"]

    response_created = project_internal_event(
        state, {"type": "response.created", "response_id": "resp_1", "modalities": ["audio", "text"]}
    )
    assert _types(response_created) == [
        "response.created",
        "conversation.item.added",
        "conversation.item.created",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert state.active_response_id == "resp_1"
    assert response_created[0].response_id == "resp_1"
    assert response_created[0].to_realtime()["response"]["status"] == "in_progress"
    assert isinstance(response_created[1], ItemAdded)
    assert isinstance(response_created[2], ItemCreated)
    assert response_created[1].item_id == "item_resp_1"
    assert response_created[3].item_id == "item_resp_1"
    assert response_created[4].to_realtime()["part"] == {"type": "audio", "transcript": ""}

    audio_delta = project_internal_event(
        state,
        {"type": "response.output_audio.delta", "response_id": "resp_1", "audio": pcm, "text": "hi", "format": "pcm16"},
    )
    assert _types(audio_delta) == ["response.output_audio.delta", "response.output_audio_transcript.delta"]
    assert isinstance(audio_delta[0], AudioDelta)
    assert audio_delta[0].delta == pcm
    assert audio_delta[0].format == "pcm16"
    assert audio_delta[0].item_id == "item_resp_1"
    assert audio_delta[1].text == "hi"
    assert state.conversation_items["item_resp_1"]["content"][0]["transcript"] == "hi"

    done = project_internal_event(state, {"type": "response.done", "response_id": "resp_1"})
    assert _types(done) == [
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.content_part.done",
        "response.output_item.done",
        "conversation.item.done",
        "response.done",
        "rate_limits.updated",
    ]
    assert done[1].text == "hi"
    assert isinstance(done[4], ItemDone)
    assert done[4].item["status"] == "completed"
    assert isinstance(done[5], ResponseDone)
    assert done[5].status == "completed"
    assert done[5].response["output"][0]["content"][0] == {"type": "output_audio", "transcript": "hi"}
    assert state.active_response_id is None
    # A duplicate terminal for the same response is idempotent.
    assert project_internal_event(state, {"type": "response.done", "response_id": "resp_1"}) == []

    committed = project_internal_event(
        state, {"type": "input.committed", "message": {"role": "user", "content": "hello"}, "input_commit_seq": 1}
    )
    assert _types(committed) == [
        "conversation.item.added",
        "conversation.item.created",
        "input_audio_buffer.committed",
        "conversation.item.done",
    ]
    assert isinstance(committed[2], InputCommitted)
    assert committed[2].previous_item_id == "item_resp_1"
    assert committed[2].item_id == committed[0].item_id
    assert committed[2].to_realtime()["event"]["input_commit_seq"] == 1
    assert committed[0].item["content"] == [{"type": "input_text", "text": "hello"}]

    project_internal_event(state, {"type": "response.created", "response_id": "resp_2"})
    project_internal_event(state, {"type": "response.output_audio.delta", "response_id": "resp_2", "audio": pcm})
    cancelled = project_internal_event(
        state, {"type": "audio.cancelled", "response_id": "resp_2", "reason": "barge_in", "committed_ms": 0}
    )
    assert _types(cancelled) == [
        "response.output_audio.done",
        "response.content_part.done",
        "response.output_item.done",
        "conversation.item.done",
        "response.done",
        "rate_limits.updated",
    ]
    assert cancelled[4].status == "cancelled"
    assert cancelled[4].response["status_details"] == {"type": "cancelled", "reason": "barge_in"}
    assert state.active_response_id is None

    closed = project_internal_event(state, {"type": "session.closed", "reason": "client_close"})
    assert _types(closed) == ["session.closed"]
    assert closed[0].is_terminal is True
    assert closed[0].to_realtime()["reason"] == "client_close"


def test_projection_of_output_audio_buffer_clear_emits_cleared_before_terminals():
    state = RealtimeProjectionState(session_id="duplex-clear")
    pcm = base64.b64encode(b"\x00\x10" * 8).decode("ascii")
    project_internal_event(state, {"type": "response.created", "response_id": "resp_1"})
    project_internal_event(state, {"type": "response.output_audio.delta", "response_id": "resp_1", "audio": pcm})

    cleared = project_internal_event(
        state, {"type": "audio.cancelled", "reason": "output_audio_buffer_clear", "committed_ms": 250}
    )

    assert _types(cleared)[:2] == ["output_audio_buffer.cleared", "response.output_audio.done"]
    assert cleared[0].response_id == "resp_1"
    assert _types(cleared)[-2:] == ["response.done", "rate_limits.updated"]
    assert state.item_truncation_cursors["item_resp_1"] == (0, 250)


def test_projection_leaves_error_to_typed_emit_sites():
    # Errors are constructed typed (``error_event``) at the emit site; the projector
    # has no ``error`` branch and falls back to the raw wrapper for such a dict.
    state = RealtimeProjectionState(session_id="duplex-err")

    projected = project_internal_event(state, {"type": "error", "code": "bad_event", "error": "nope"})

    assert len(projected) == 1
    assert isinstance(projected[0], DuplexRawEvent)
    assert projected[0].type == "duplex.error"
    assert projected[0].to_realtime()["event"]["code"] == "bad_event"
