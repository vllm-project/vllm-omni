# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect
from openai.types import realtime as types
from openai.types.realtime.realtime_audio_input_turn_detection import SemanticVad, ServerVad

from vllm_omni.engine.duplex.turn_detection import (
    SemanticTurnDetector,
    SemanticVADUnavailableError,
    ServerTurnDetector,
    ServerVADUnavailableError,
    SmartTurnBackendProvider,
    TurnDetectionConfig,
    TurnDetectionResult,
)
from vllm_omni.engine.duplex.vad import SileroStreamingVAD
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


class _InboundFakeWebSocket(_FakeWebSocket):
    def __init__(self, incoming_messages: list[str]) -> None:
        super().__init__()
        self.incoming_messages = incoming_messages

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if not self.incoming_messages:
            raise WebSocketDisconnect()
        return self.incoming_messages.pop(0)


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


class _FakeSemanticTurnDetector(SemanticTurnDetector):
    def __init__(
        self,
        results: list[TurnDetectionResult],
        *,
        config: TurnDetectionConfig | None = None,
    ) -> None:
        super().__init__(
            config or TurnDetectionConfig(type="semantic_vad"),
            turn_scorer=lambda _audio: 0.9,
            frame_scorer=lambda _frame: 0.9,
        )
        self.results = results

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        return self.results.pop(0)


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


class _UnavailableSemanticTurnDetector(SemanticTurnDetector):
    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        raise SemanticVADUnavailableError("fake Smart Turn backend is unavailable")


class _ResponseRecordingConnection(OpenAIFullDuplexConnection):
    response_creates: list[types.ResponseCreateEvent]

    async def _handle_response_create(self, event: types.ResponseCreateEvent) -> None:
        self.response_creates.append(event)


def _make_connection(
    websocket: _FakeWebSocket,
    *,
    semantic_vad_backend_provider: SmartTurnBackendProvider | None = None,
) -> _ResponseRecordingConnection:
    engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    chat_handler = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: object()))
    connection = _ResponseRecordingConnection(
        websocket=websocket,
        engine=engine,
        model_name="test-model",
        chat_handler=chat_handler,
        semantic_vad_backend_provider=semantic_vad_backend_provider,
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
async def test_semantic_vad_session_update_creates_turn_detector() -> None:
    websocket = _FakeWebSocket()
    provider = SmartTurnBackendProvider()
    connection = _make_connection(websocket, semantic_vad_backend_provider=provider)

    await connection._handle_session_update(_session_update_event({"type": "semantic_vad", "eagerness": "high"}))

    events = [json.loads(message) for message in websocket.messages]
    assert events[-1]["type"] == "session.updated"
    assert isinstance(connection.session.config.audio.input.turn_detection, SemanticVad)
    assert isinstance(connection._turn_detector, SemanticTurnDetector)
    assert connection._turn_detector.config.type == "semantic_vad"
    assert connection._turn_detector.config.eagerness == "high"


@pytest.mark.asyncio
async def test_semantic_vad_build_detector_receives_smart_turn_provider(monkeypatch) -> None:
    websocket = _FakeWebSocket()
    provider = SmartTurnBackendProvider()
    connection = _make_connection(websocket, semantic_vad_backend_provider=provider)
    captured: list[tuple[object, object]] = []

    def build_detector(
        self,
        backend_provider=None,
        semantic_backend_provider=None,
    ):
        captured.append((backend_provider, semantic_backend_provider))
        return _FakeTurnDetector([], config=self)

    monkeypatch.setattr(TurnDetectionConfig, "build_detector", build_detector)
    await connection._handle_session_update(_session_update_event({"type": "semantic_vad", "eagerness": "high"}))

    assert captured == [(None, provider)]


@pytest.mark.asyncio
async def test_server_vad_extra_field_is_ignored_before_consecutive_session_updates() -> None:
    first_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "turn_detection": {"type": "server_vad", "min_speech_duration_ms": 250},
                }
            },
        },
    }
    second_update = {
        "type": "session.update",
        "session": {"type": "realtime", "instructions": "Answer briefly."},
    }
    websocket = _InboundFakeWebSocket([json.dumps(first_update), json.dumps(second_update)])
    connection = _make_connection(websocket)

    await connection.handle_connection()

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == ["session.created", "session.updated", "session.updated"]
    assert connection.session.config.instructions == "Answer briefly."

    audio = connection.session.config.audio
    assert audio is not None and audio.input is not None
    turn_detection = audio.input.turn_detection
    assert isinstance(turn_detection, ServerVad)
    assert not turn_detection.model_extra
    assert "min_speech_duration_ms" not in turn_detection.model_fields_set


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
@pytest.mark.parametrize("create_response", [False, True])
async def test_server_vad_appends_with_real_detector(monkeypatch, create_response: bool) -> None:
    # Script only acoustic inference; keep the detector and its process API real.
    monkeypatch.setattr(SileroStreamingVAD, "_score_frame", lambda _self, frame: 0.9 if abs(frame).max() > 0.1 else 0.0)
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    await connection._handle_session_update(
        _session_update_event(
            {
                "type": "server_vad",
                "prefix_padding_ms": 0,
                "silence_duration_ms": 64,
                "create_response": create_response,
            }
        )
    )
    assert type(connection._turn_detector) is ServerTurnDetector

    speech = b"\x00\x20" * (SAMPLE_RATE_HZ * 96 // 1000)
    silence = b"\x00\x00" * (SAMPLE_RATE_HZ * 64 // 1000)
    for _ in range(2):
        await connection._handle_audio_append(_audio_event(speech))
        await connection._handle_audio_append(_audio_event(silence))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == ["session.updated"] + [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
    ] * 2
    starts = [event for event in events if event["type"] == "input_audio_buffer.speech_started"]
    stops = [event for event in events if event["type"] == "input_audio_buffer.speech_stopped"]
    commits = [event for event in events if event["type"] == "input_audio_buffer.committed"]
    assert [event["audio_start_ms"] for event in starts] == [0, 160]
    assert [event["audio_end_ms"] for event in stops] == [160, 320]
    assert starts[0]["item_id"] != starts[1]["item_id"]
    for start, stop, commit, item in zip(starts, stops, commits, connection.session.items, strict=True):
        assert start["item_id"] == stop["item_id"] == commit["item_id"] == item.id
        assert base64.b64decode(item.content[0].audio) == speech + silence
    assert not connection.session.input_audio_buffer
    assert len(connection.response_creates) == (2 if create_response else 0)


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
async def test_server_vad_disable_resets_state_and_commits_pending_audio_with_new_id() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    detector = _FakeTurnDetector([_speech_started_result()])
    connection._turn_detector = detector

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))
    started = json.loads(websocket.messages[0])
    assert started["type"] == "input_audio_buffer.speech_started"

    await connection._handle_session_update(_session_update_event(None))

    events = [json.loads(message) for message in websocket.messages]
    assert events[-1]["type"] == "session.updated"
    assert connection.session.config.audio.input.turn_detection is None
    assert connection._turn_detector is None
    assert connection._speech_item_id is None
    assert connection.session.input_audio_buffer == bytearray(b"\x00\x00")
    assert detector.reset_count == 1

    await connection._handle_audio_commit(types.InputAudioBufferCommitEvent(type="input_audio_buffer.commit"))

    events = [json.loads(message) for message in websocket.messages]
    committed = next(event for event in events if event["type"] == "input_audio_buffer.committed")
    assert committed["item_id"] != started["item_id"]
    assert connection.session.items[0].id == committed["item_id"]
    assert connection.session.input_audio_buffer == bytearray()


@pytest.mark.asyncio
async def test_semantic_vad_without_interrupt_response_is_rejected_without_mutating_state() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    detector = _FakeTurnDetector([])
    previous_turn_detection = ServerVad(type="server_vad")
    connection.session.config.audio.input.turn_detection = previous_turn_detection
    connection._turn_detector = detector
    connection._speech_item_id = "item_existing"

    await connection._handle_session_update(
        _session_update_event({"type": "semantic_vad", "eagerness": "high", "interrupt_response": False})
    )

    events = [json.loads(message) for message in websocket.messages]
    assert events[-1]["type"] == "error"
    assert events[-1]["error"]["code"] == "unsupported_turn_detection"
    assert connection.session.config.audio.input.turn_detection is previous_turn_detection
    assert connection._turn_detector is detector
    assert connection._speech_item_id == "item_existing"
    assert detector.reset_count == 0


@pytest.mark.asyncio
async def test_server_and_semantic_detectors_share_the_realtime_append_path() -> None:
    result = _speech_started_result()
    detectors = [
        _FakeTurnDetector([result], config=TurnDetectionConfig(create_response=False)),
        _FakeSemanticTurnDetector([result], config=TurnDetectionConfig(type="semantic_vad", create_response=False)),
    ]
    event_batches = []

    for detector in detectors:
        websocket = _FakeWebSocket()
        connection = _make_connection(websocket)
        connection._turn_detector = detector

        await connection._handle_audio_append(_audio_event(b"\x00\x00"))

        event_batches.append([json.loads(message) for message in websocket.messages])

    normalized_batches = [
        [{key: value for key, value in event.items() if key not in {"event_id", "item_id"}} for event in events]
        for events in event_batches
    ]
    assert normalized_batches[0] == normalized_batches[1]
    assert all([event["type"] for event in events] == ["input_audio_buffer.speech_started"] for events in event_batches)


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
async def test_semantic_vad_unavailable_clears_detector_and_rejects_append() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _UnavailableSemanticTurnDetector(TurnDetectionConfig(type="semantic_vad"))
    connection._speech_item_id = "item_existing"

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))

    events = [json.loads(message) for message in websocket.messages]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error"] == {
        "type": "invalid_request_error",
        "code": "semantic_vad_unavailable",
        "message": "fake Smart Turn backend is unavailable",
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
async def test_server_vad_coalesced_stop_restart_without_item_stays_open_until_later_stop() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    connection._turn_detector = _FakeTurnDetector(
        [
            TurnDetectionResult(
                is_speech=True,
                speech_active=True,
                speech_started=True,
                speech_stopped=True,
                speech_probability=0.9,
                audio_start_ms=300,
                audio_end_ms=200,
            ),
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                speech_probability=0.1,
                audio_end_ms=500,
                should_commit=True,
                create_response=False,
            ),
        ],
        config=TurnDetectionConfig(interrupt_response=True),
    )
    cancellations = []

    async def _cancel_active_response() -> None:
        cancellations.append(True)

    connection._cancel_active_response = _cancel_active_response

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == ["input_audio_buffer.speech_started"]
    started = events[0]
    item_id = started["item_id"]
    assert connection._speech_item_id == item_id
    assert connection.session.input_audio_buffer == bytearray(b"\x00\x00")
    assert cancellations == [True]

    await connection._handle_audio_append(_audio_event(b"\x00\x40"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
    ]
    stopped, committed = events[1:3]
    assert stopped["item_id"] == committed["item_id"] == item_id
    assert stopped["audio_end_ms"] == 500
    assert connection.session.items[0].id == item_id
    assert connection.session.input_audio_buffer == bytearray()
    assert connection._speech_item_id is None
    assert cancellations == [True]


@pytest.mark.asyncio
async def test_server_vad_coalesced_stop_restart_with_open_item_stays_open_until_later_stop() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    detector = _FakeTurnDetector(
        [
            _speech_started_result(),
            TurnDetectionResult(
                is_speech=True,
                speech_active=True,
                speech_started=True,
                speech_stopped=True,
                speech_probability=0.9,
                audio_start_ms=300,
                audio_end_ms=200,
            ),
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
    connection._turn_detector = detector

    await connection._handle_audio_append(_audio_event(b"\x00\x00"))
    started = json.loads(websocket.messages[0])
    assert started["type"] == "input_audio_buffer.speech_started"
    item_id = started["item_id"]
    assert connection._speech_item_id == item_id

    await connection._handle_audio_append(_audio_event(b"\x00\x20"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == ["input_audio_buffer.speech_started"]
    assert connection._speech_item_id == item_id
    assert connection.session.input_audio_buffer == bytearray(b"\x00\x00\x00\x20")

    await connection._handle_audio_append(_audio_event(b"\x00\x40"))

    events = [json.loads(message) for message in websocket.messages]
    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
    ]
    stopped, committed = events[1:3]
    assert stopped["item_id"] == committed["item_id"] == item_id
    assert stopped["audio_end_ms"] == 500
    assert connection.session.items[0].id == item_id
    assert connection.session.input_audio_buffer == bytearray()
    assert connection._speech_item_id is None
    assert detector.reset_count == 1


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
