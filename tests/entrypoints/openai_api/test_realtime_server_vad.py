# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from openai.types import realtime as types
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from vllm_omni.engine.duplex.turn_detection import (
    ServerTurnDetector,
    ServerVADUnavailableError,
    TurnDetectionConfig,
    TurnDetectionResult,
)
from vllm_omni.entrypoints.openai.realtime.connection import (
    SAMPLE_RATE_HZ,
    OpenAIFullDuplexConnection,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _synchronous_turn_detector_offload(monkeypatch):
    async def _to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


class _FakeTurnDetector(ServerTurnDetector):
    def __init__(
        self,
        results: list[TurnDetectionResult],
        *,
        config: TurnDetectionConfig | None = None,
    ) -> None:
        super().__init__(config or TurnDetectionConfig())
        self.results = results
        self.reset_count = 0

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        return self.results.pop(0)

    def reset(self) -> None:
        self.reset_count += 1
        super().reset()


class _UnavailableTurnDetector(ServerTurnDetector):
    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        raise ServerVADUnavailableError("fake VAD backend is unavailable")


class _ResponseRecordingConnection(OpenAIFullDuplexConnection):
    response_creates: list[types.ResponseCreateEvent]

    async def _handle_response_create(self, event: types.ResponseCreateEvent) -> None:
        self.response_creates.append(event)


def _make_connection(websocket: _FakeWebSocket) -> _ResponseRecordingConnection:
    engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    chat_handler = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: object()))
    connection = _ResponseRecordingConnection(
        websocket=websocket,
        engine=engine,
        model_name="test-model",
        chat_handler=chat_handler,
    )
    connection.response_creates = []
    return connection


def _audio_event(audio: bytes) -> types.InputAudioBufferAppendEvent:
    return types.InputAudioBufferAppendEvent(
        type="input_audio_buffer.append",
        audio=base64.b64encode(audio).decode("ascii"),
    )


def _session_update_event(turn_detection: object) -> types.SessionUpdateEvent:
    session = types.RealtimeSessionCreateRequest.model_validate(
        {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "turn_detection": turn_detection,
                }
            },
        }
    )
    return types.SessionUpdateEvent(type="session.update", session=session)


def _speech_started_result() -> TurnDetectionResult:
    return TurnDetectionResult(
        is_speech=True,
        speech_active=True,
        speech_started=True,
        speech_probability=0.9,
        audio_start_ms=100,
    )


@pytest.mark.asyncio
async def test_server_vad_session_update_creates_turn_detector() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    session = types.RealtimeSessionCreateRequest.model_validate(
        {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                },
                "output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ}},
            },
        }
    )

    await connection._handle_session_update(types.SessionUpdateEvent(type="session.update", session=session))

    events = [json.loads(message) for message in websocket.messages]
    assert events[-1]["type"] == "session.updated"
    assert isinstance(connection.session.config.audio.input.turn_detection, ServerVad)
    assert connection._turn_detector is not None


@pytest.mark.asyncio
async def test_server_vad_commits_and_creates_response() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _FakeTurnDetector(
        [
            TurnDetectionResult(
                is_speech=True,
                speech_active=True,
                speech_started=True,
                speech_probability=0.9,
                audio_start_ms=100,
            ),
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                speech_probability=0.1,
                audio_end_ms=500,
                should_commit=True,
                create_response=True,
            ),
        ]
    )

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))
    await connection._handle_audio_append(_audio_event(b"\x00\x40"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
    ]
    started, stopped, committed = events[:3]
    assert started["audio_start_ms"] == 100
    assert stopped["audio_end_ms"] == 500
    assert started["item_id"] == stopped["item_id"] == committed["item_id"]
    assert connection.session.items[0].id == started["item_id"]
    assert connection.session.input_audio_buffer == bytearray()
    assert connection._turn_detector.reset_count == 1
    assert len(connection.response_creates) == 1


@pytest.mark.asyncio
async def test_manual_commit_reuses_speech_item_id() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _FakeTurnDetector(
        [
            TurnDetectionResult(
                is_speech=True,
                speech_active=True,
                speech_started=True,
                speech_probability=0.9,
                audio_start_ms=100,
            )
        ]
    )

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))
    started = json.loads(websocket.messages[0])
    await connection._handle_audio_commit(types.InputAudioBufferCommitEvent(type="input_audio_buffer.commit"))

    events = [json.loads(message) for message in websocket.messages]
    committed = next(event for event in events if event["type"] == "input_audio_buffer.committed")
    assert committed["item_id"] == started["item_id"]
    assert connection.session.items[0].id == started["item_id"]
    assert connection._turn_detector.reset_count == 1
    assert connection._speech_item_id is None


@pytest.mark.asyncio
async def test_server_vad_disable_clears_speech_state() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    detector = _FakeTurnDetector([])
    connection._turn_detector = detector
    connection._speech_item_id = "item_existing"

    await connection._handle_session_update(_session_update_event(None))

    events = [json.loads(message) for message in websocket.messages]
    assert events[-1]["type"] == "session.updated"
    assert connection.session.config.audio.input.turn_detection is None
    assert connection._turn_detector is None
    assert connection._speech_item_id is None
    assert detector.reset_count == 1


@pytest.mark.asyncio
async def test_unsupported_turn_detection_rejects_update_without_mutating_state() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    detector = _FakeTurnDetector([])
    previous_turn_detection = ServerVad(type="server_vad")
    connection.session.config.audio.input.turn_detection = previous_turn_detection
    connection._turn_detector = detector
    connection._speech_item_id = "item_existing"

    await connection._handle_session_update(_session_update_event({"type": "semantic_vad", "eagerness": "high"}))

    events = [json.loads(message) for message in websocket.messages]
    assert events[-1]["type"] == "error"
    assert events[-1]["error"]["code"] == "unsupported_turn_detection"
    assert connection.session.config.audio.input.turn_detection is previous_turn_detection
    assert connection._turn_detector is detector
    assert connection._speech_item_id == "item_existing"
    assert detector.reset_count == 0


@pytest.mark.asyncio
async def test_server_vad_unavailable_clears_detector_and_rejects_append() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _UnavailableTurnDetector(TurnDetectionConfig())
    connection._speech_item_id = "item_existing"

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))

    events = [json.loads(message) for message in websocket.messages]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error"] == {
        "type": "invalid_request_error",
        "code": "server_vad_unavailable",
        "message": "fake VAD backend is unavailable",
    }
    assert connection._turn_detector is None
    assert connection._speech_item_id is None
    assert connection.session.input_audio_buffer == bytearray()


@pytest.mark.asyncio
async def test_server_vad_speech_started_interrupts_active_response() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _FakeTurnDetector(
        [_speech_started_result()],
        config=TurnDetectionConfig(interrupt_response=True),
    )
    cancellations = []

    async def _cancel_active_response() -> None:
        cancellations.append(True)

    connection._cancel_active_response = _cancel_active_response

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == ["input_audio_buffer.speech_started"]
    assert cancellations == [True]


@pytest.mark.asyncio
async def test_server_vad_commit_without_response_creation() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _FakeTurnDetector(
        [
            _speech_started_result(),
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                speech_probability=0.1,
                audio_end_ms=500,
                should_commit=True,
                create_response=False,
            ),
        ]
    )

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))
    await connection._handle_audio_append(_audio_event(b"\x00\x40"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
    ]
    assert connection.response_creates == []
    assert connection.session.input_audio_buffer == bytearray()
    assert connection._speech_item_id is None


@pytest.mark.asyncio
async def test_server_vad_clear_resets_detector_state() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    detector = _FakeTurnDetector([])
    connection._turn_detector = detector
    connection._speech_item_id = "item_existing"
    connection.session.input_audio_buffer.extend(b"\x00\x00")

    await connection._handle_audio_clear(types.InputAudioBufferClearEvent(type="input_audio_buffer.clear"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == ["input_audio_buffer.cleared"]
    assert connection.session.input_audio_buffer == bytearray()
    assert connection._speech_item_id is None
    assert detector.reset_count == 1
