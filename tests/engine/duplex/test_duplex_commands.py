# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI Realtime client events -> typed ``DuplexCommand`` objects and their mailbox payloads."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.engine.duplex.commands import (
    REALTIME_COMMAND_TYPES,
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
    command_from_realtime,
)
from vllm_omni.engine.duplex.realtime_commands import (
    RealtimeInputDefaults,
    translate_realtime_command,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LOUD_PCM16 = base64.b64encode(b"\x00\x10" * 8).decode("ascii")
_SILENT_PCM16 = base64.b64encode(b"\x00\x00" * 8).decode("ascii")
_JPEG_FRAME = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 12).decode("ascii")

_MINIMAL_PAYLOADS: dict[str, tuple[dict[str, object], type[DuplexCommand]]] = {
    "input_audio_buffer.append": ({"audio": _LOUD_PCM16}, AppendAudio),
    "input_audio_buffer.commit": ({}, Commit),
    "input_audio_buffer.clear": ({}, ClearInput),
    "output_audio_buffer.clear": ({}, ClearOutputAudio),
    "response.create": ({}, CreateResponse),
    "response.cancel": ({}, CancelResponse),
    "conversation.item.create": ({"item": {"type": "message", "role": "user", "content": []}}, CreateItem),
    "conversation.item.delete": ({"item_id": "item_1"}, DeleteItem),
    "conversation.item.truncate": ({"item_id": "item_1", "audio_end_ms": 100}, TruncateItem),
    "session.update": ({"session": {"instructions": "hi"}}, UpdateSession),
    "playback.ack": ({"played_ms": 10}, AckPlayback),
    "session.heartbeat": ({}, Heartbeat),
    "session.close": ({}, CloseSession),
    "turn.signal": ({"event": "user_started"}, SignalTurn),
    "input.text.append": ({"text": "hello"}, AppendText),
    "input.cancel": ({}, CancelInput),
    "barge_in": ({}, BargeIn),
}


def test_every_realtime_command_type_has_a_mapping_case():
    assert set(_MINIMAL_PAYLOADS) == set(REALTIME_COMMAND_TYPES)


@pytest.mark.parametrize("event_type", sorted(REALTIME_COMMAND_TYPES))
def test_command_from_realtime_maps_each_type_to_its_dataclass_and_propagates_event_id(event_type: str):
    body, expected_cls = _MINIMAL_PAYLOADS[event_type]

    command = command_from_realtime({"type": event_type, "event_id": f"evt-{event_type}", **body})

    assert type(command) is expected_cls
    assert command.event_id == f"evt-{event_type}"
    payload = command.payload()
    assert payload["type"] == expected_cls.type
    assert payload["realtime_event_id"] == f"evt-{event_type}"
    assert "event_id" not in payload

    anonymous = command_from_realtime({"type": event_type, **body})
    assert anonymous.event_id is None
    assert "realtime_event_id" not in anonymous.payload()


@pytest.mark.parametrize(
    ("alias", "expected_cls", "body"),
    [
        ("push_text", AppendText, {"text": "hello"}),
        ("input_text.append", AppendText, {"text": "hello"}),
        ("signal_turn", SignalTurn, {"event": "user_started"}),
        ("close_session", CloseSession, {}),
        ("close", CloseSession, {}),
        ("audio.playback_ack", AckPlayback, {"played_ms": 10}),
        ("input.commit", Commit, {}),
    ],
)
def test_pre_realtime_client_event_aliases_map_to_their_canonical_command(
    alias: str, expected_cls: type[DuplexCommand], body: dict[str, object]
):
    assert type(command_from_realtime({"type": alias, **body})) is expected_cls


def test_append_audio_decodes_and_converts_pcm16_to_16k_float32():
    command = command_from_realtime(
        {"type": "input_audio_buffer.append", "audio": _LOUD_PCM16, "format": "pcm16", "sample_rate_hz": 16000}
    )

    assert isinstance(command, AppendAudio)
    assert command.format == "pcm_f32le"
    assert command.sample_rate_hz == 16000
    samples = np.frombuffer(command.audio, dtype="<f4")
    assert samples.shape == (8,)
    assert np.allclose(samples, 4096 / 32768.0)
    assert command.is_speech is True
    assert command.video_frames == ()
    assert command.duration_ms is None
    assert command.audio_end_ms is None


def test_append_audio_defaults_come_from_the_session_payload():
    defaults = RealtimeInputDefaults().with_session_payload({"input_audio_format": "pcm16", "sample_rate_hz": 8000})
    assert defaults.input_audio_format == "pcm16"
    assert defaults.input_sample_rate_hz == 8000

    command = translate_realtime_command({"type": "input_audio_buffer.append", "audio": _LOUD_PCM16}, defaults=defaults)

    assert isinstance(command, AppendAudio)
    assert command.format == "pcm_f32le"
    # 8 kHz input is resampled to the model's 16 kHz float stream.
    assert command.sample_rate_hz == 16000
    assert len(command.audio) > 8 * 4

    passthrough = translate_realtime_command(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(np.zeros(4, dtype="<f4").tobytes()).decode()},
        defaults=RealtimeInputDefaults(input_audio_format="pcm_f32le", input_sample_rate_hz=16000),
    )
    assert passthrough.format == "pcm_f32le"
    assert passthrough.audio == np.zeros(4, dtype="<f4").tobytes()


def test_append_audio_speech_classification_uses_hints_then_rms():
    silent = command_from_realtime({"type": "input_audio_buffer.append", "audio": _SILENT_PCM16})
    assert silent.is_speech is False

    forced = command_from_realtime({"type": "input_audio_buffer.append", "audio": _SILENT_PCM16, "is_speech": True})
    assert forced.is_speech is True
    assert forced.hints == {"is_speech": True}

    vad = command_from_realtime(
        {"type": "input_audio_buffer.append", "audio": _LOUD_PCM16, "vad": {"speech_probability": 0.1}}
    )
    assert vad.is_speech is False
    assert vad.hints["vad"] == {"speech_probability": 0.1}


def test_append_audio_carries_wire_hints_and_video_frames():
    command = command_from_realtime(
        {
            "type": "input_audio_buffer.append",
            "event_id": "evt-append",
            "audio": _LOUD_PCM16,
            "transcript": "hi there",
            "duration_ms": 1000,
            "audio_end_ms": 3000,
            "video_frames": [_JPEG_FRAME],
            "unknown_key": "dropped",
        }
    )

    assert isinstance(command, AppendAudio)
    assert command.duration_ms == 1000
    assert command.audio_end_ms == 3000
    assert command.video_frames == (_JPEG_FRAME,)
    assert dict(command.hints) == {"transcript": "hi there", "duration_ms": 1000, "audio_end_ms": 3000}

    payload = command.payload()
    assert payload["type"] == "input_audio_buffer.append"
    assert payload["realtime_event_id"] == "evt-append"
    assert payload["audio"] == base64.b64encode(command.audio).decode("ascii")
    assert payload["format"] == "pcm_f32le"
    assert payload["sample_rate_hz"] == 16000
    assert payload["is_speech"] is True
    assert payload["transcript"] == "hi there"
    assert payload["duration_ms"] == 1000
    assert payload["audio_end_ms"] == 3000
    assert payload["video_frames"] == [_JPEG_FRAME]
    assert "hints" not in payload
    assert "unknown_key" not in payload


def test_append_audio_payload_re_encodes_bytes_and_flattens_hints():
    command = AppendAudio(audio=b"\x01\x02", hints={"vad": {"is_speech": True}}, is_speech=True)

    assert command.payload() == {
        "type": "input_audio_buffer.append",
        "audio": "AQI=",
        "format": "pcm16",
        "is_speech": True,
        "vad": {"is_speech": True},
    }


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"audio": _LOUD_PCM16, "format": "mp3"}, "unsupported_audio_format"),
        ({"audio": "@@@@", "format": "pcm_f32le"}, "bad_audio"),
        ({"audio": _LOUD_PCM16, "video_frames": ["aGVsbG8="]}, "invalid_video_frames"),
        ({"audio": _LOUD_PCM16, "video_frames": [_JPEG_FRAME], "max_slice_nums": 4}, "invalid_video_frames"),
        ({"audio": _LOUD_PCM16, "format": "pcm16", "sample_rate_hz": 3}, "bad_event"),
    ],
)
def test_malformed_append_raises_command_error_with_code(body: dict[str, object], code: str):
    with pytest.raises(DuplexCommandError) as excinfo:
        command_from_realtime({"type": "input_audio_buffer.append", "event_id": "evt-bad", **body})

    assert excinfo.value.code == code
    assert excinfo.value.event_id == "evt-bad"


# ---- other malformed payloads ----


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "bad_event"),
        ({"type": 7}, "bad_event"),
        ({"type": "session.resume", "session_id": "s", "resume_token": "t"}, "unknown_event"),
        ({"type": "session.event_ack", "server_event_seq": 1}, "unknown_event"),
        ({"type": "conversation.item.frobnicate", "item_id": "item_1"}, "unknown_event"),
        ({"type": "conversation.item.create"}, "bad_event"),
        (
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_audio", "format": "mp3"}]},
            },
            "unsupported_audio_format",
        ),
        ({"type": "conversation.item.delete"}, "missing_item_id"),
        ({"type": "conversation.item.truncate", "audio_end_ms": 1}, "missing_item_id"),
        ({"type": "conversation.item.truncate", "item_id": "item_1"}, "bad_event"),
        ({"type": "session.update", "session": {"output_audio_format": "opus"}}, "unsupported_audio_format"),
        (
            {"type": "session.update", "session": {"turn_detection": {"type": "semantic_vad"}}},
            "unsupported_turn_detection",
        ),
        ({"type": "response.create", "response": {"output_audio_format": "opus"}}, "unsupported_audio_format"),
        ({"type": "playback.ack"}, "bad_event"),
        ({"type": "input.text.append"}, "bad_event"),
        ({"type": "turn.signal"}, "bad_event"),
        ({"type": "turn.signal", "event": "conversation.item.delete", "payload": {}}, "missing_item_id"),
        ({"type": "turn.signal", "event": "conversation.item.truncate", "payload": {"item_id": "i"}}, "bad_event"),
    ],
)
def test_malformed_payloads_raise_command_error_with_code(payload: dict[str, object], code: str):
    with pytest.raises(DuplexCommandError) as excinfo:
        command_from_realtime({"event_id": "evt-bad", **payload})

    assert excinfo.value.code == code
    assert excinfo.value.event_id == "evt-bad"


# ---- payload() rendering ----


def test_commit_maps_create_response_to_response_create():
    command = command_from_realtime(
        {"type": "input_audio_buffer.commit", "event_id": "evt-commit", "response_create": True, "final": False}
    )

    assert command == Commit(event_id="evt-commit", final=False, create_response=True)
    assert command.payload() == {
        "type": "input_audio_buffer.commit",
        "realtime_event_id": "evt-commit",
        "final": False,
        "response_create": True,
    }
    assert Commit().payload() == {"type": "input_audio_buffer.commit", "final": True}
    assert Commit(is_speech=False, realtime_item_id="item_1").payload() == {
        "type": "input_audio_buffer.commit",
        "final": True,
        "is_speech": False,
        "realtime_item_id": "item_1",
    }
    assert (
        command_from_realtime({"type": "input_audio_buffer.commit", "create_response": False}).create_response is False
    )


def test_create_response_renders_options_as_response_object():
    command = command_from_realtime({"type": "response.create", "response": {"modalities": ["text"]}})

    assert isinstance(command, CreateResponse)
    assert dict(command.options) == {"modalities": ["text"]}
    assert command.payload() == {"type": "response.create", "response": {"modalities": ["text"]}}
    assert command_from_realtime({"type": "response.create"}).payload() == {"type": "response.create", "response": {}}


def test_cancel_and_clear_commands_keep_optional_response_id():
    assert command_from_realtime({"type": "response.cancel", "response_id": ""}).response_id is None
    cancel = command_from_realtime({"type": "response.cancel", "response_id": "resp_1"})
    assert cancel == CancelResponse(response_id="resp_1")
    assert cancel.payload() == {"type": "response.cancel", "response_id": "resp_1"}
    clear = command_from_realtime({"type": "output_audio_buffer.clear", "response_id": "resp_2"})
    assert clear == ClearOutputAudio(response_id="resp_2")
    assert ClearInput().payload() == {"type": "input_audio_buffer.clear"}


def test_update_session_renders_as_turn_signal_payload():
    command = command_from_realtime(
        {"type": "session.update", "session": {"instructions": "be brief", "voice": "alloy"}}
    )

    assert command == UpdateSession(patch={"instructions": "be brief", "voice": "alloy"})
    assert command.payload() == {
        "type": "turn.signal",
        "event": "session.update",
        "payload": {"instructions": "be brief", "voice": "alloy"},
    }


def test_create_item_normalizes_item_and_renders_as_turn_signal_payload():
    command = command_from_realtime(
        {
            "type": "conversation.item.create",
            "previous_item_id": "item_0",
            "item": {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        }
    )

    assert isinstance(command, CreateItem)
    assert command.previous_item_id == "item_0"
    assert command.item["id"].startswith("item_")
    assert command.item["object"] == "realtime.item"
    assert command.item["status"] == "completed"
    payload = command.payload()
    assert payload["type"] == "turn.signal"
    assert payload["event"] == "conversation.item.create"
    assert payload["payload"] == {"item": dict(command.item), "previous_item_id": "item_0"}
    assert "item" not in payload
    assert "previous_item_id" not in payload

    without_previous = command_from_realtime({"type": "conversation.item.create", "item": {"content": "x"}})
    assert without_previous.item["role"] == "user"
    assert without_previous.item["content"] == []
    assert without_previous.payload()["payload"] == {"item": dict(without_previous.item)}


def test_delete_and_truncate_render_as_turn_signal_payloads():
    delete = command_from_realtime({"type": "conversation.item.delete", "item_id": "item_1"})
    assert delete == DeleteItem(item_id="item_1")
    assert delete.payload() == {
        "type": "turn.signal",
        "event": "conversation.item.delete",
        "payload": {"item_id": "item_1"},
    }

    truncate = command_from_realtime(
        {"type": "conversation.item.truncate", "item_id": "item_1", "audio_end_ms": 1500.0, "content_index": 1}
    )
    assert truncate == TruncateItem(item_id="item_1", audio_end_ms=1500, content_index=1)
    assert truncate.payload() == {
        "type": "turn.signal",
        "event": "conversation.item.truncate",
        "payload": {"item_id": "item_1", "audio_end_ms": 1500, "content_index": 1},
    }


def test_playback_ack_close_and_text_append_payloads():
    ack = command_from_realtime(
        {"type": "playback.ack", "played_ms": 1200.0, "committed_ms": 1000, "response_id": "resp_1", "item_id": ""}
    )
    assert ack == AckPlayback(played_ms=1200, committed_ms=1000, response_id="resp_1")
    assert ack.payload() == {
        "type": "playback.ack",
        "played_ms": 1200,
        "committed_ms": 1000,
        "response_id": "resp_1",
    }

    assert command_from_realtime({"type": "session.close"}) == CloseSession(reason="client_close")
    assert command_from_realtime({"type": "session.close", "reason": "done"}).payload() == {
        "type": "session.close",
        "reason": "done",
    }
    assert command_from_realtime({"type": "input.text.append", "text": "hello"}).payload() == {
        "type": "input.text.append",
        "text": "hello",
    }
    assert Heartbeat().payload() == {"type": "session.heartbeat"}


def test_turn_signal_renders_payload_only_when_present_and_dispatches_known_events():
    signal = command_from_realtime({"type": "turn.signal", "event": "user_started", "payload": {"source": "client"}})
    assert signal == SignalTurn(event="user_started", signal_payload={"source": "client"})
    assert signal.payload() == {"type": "turn.signal", "event": "user_started", "payload": {"source": "client"}}
    assert command_from_realtime({"type": "turn.signal", "event": "user_started"}).payload() == {
        "type": "turn.signal",
        "event": "user_started",
    }

    assert command_from_realtime({"type": "turn.signal", "event": "barge_in"}) == BargeIn()
    assert command_from_realtime({"type": "turn.signal", "event": "input.cancel"}) == CancelInput()
    assert command_from_realtime(
        {"type": "turn.signal", "event": "response.cancel", "payload": {"response_id": "resp_1"}}
    ) == CancelResponse(response_id="resp_1")
    assert command_from_realtime(
        {"type": "turn.signal", "event": "session.update", "payload": {"voice": "alloy"}}
    ) == UpdateSession(patch={"voice": "alloy"})
    assert command_from_realtime(
        {"type": "turn.signal", "event": "conversation.item.delete", "payload": {"item_id": "item_1"}}
    ) == DeleteItem(item_id="item_1")
    assert command_from_realtime(
        {"type": "turn.signal", "event": "conversation.item.truncate", "payload": {"item_id": "i", "audio_end_ms": 5}}
    ) == TruncateItem(item_id="i", audio_end_ms=5)
    created = command_from_realtime(
        {"type": "turn.signal", "event": "conversation.item.create", "payload": {"item": {"role": "system"}}}
    )
    assert isinstance(created, CreateItem)
    assert created.item["role"] == "system"
