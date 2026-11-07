# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import base64
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect
from openai.types import realtime as types
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from vllm.sampling_params import SamplingParams

from tests.helpers.serving_chat import make_text_omni_output
from vllm_omni.engine.duplex.turn_detection import (
    ServerTurnDetector,
    ServerVADUnavailableError,
    TurnDetectionConfig,
    TurnDetectionResult,
)
from vllm_omni.engine.duplex.vad import SileroStreamingVAD
from vllm_omni.entrypoints.openai.realtime.connection import (
    SAMPLE_RATE_HZ,
    OpenAIFullDuplexConnection,
)
from vllm_omni.entrypoints.transcription import SameModelTranscriber, TranscriptionStage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def transcription_logs(caplog, monkeypatch):
    logger = logging.getLogger(f"{__name__}.transcription")
    monkeypatch.setattr("vllm_omni.entrypoints.openai.realtime.connection.logger", logger)
    caplog.set_level(logging.INFO, logger=logger.name)
    return caplog


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


class _FakeTranscriber:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls = []
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.transcript = "Wie geht es dir?"

    def validate_settings(self, settings):
        if settings.model != "test-model":
            raise ValueError("Unknown transcription model")
        return settings.model_copy(deep=True)

    async def transcribe_pcm(self, audio, sample_rate, settings, request_id):
        self.calls.append((audio, sample_rate, settings, request_id))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.failure:
            raise self.failure
        return self.transcript


async def _enable_transcription(connection, transcriber, *, model="test-model", language="de"):
    connection._transcriber = transcriber
    session = types.RealtimeSessionCreateRequest.model_validate(
        {"type": "realtime", "audio": {"input": {"transcription": {"model": model, "language": language}}}}
    )
    await connection._handle_session_update(types.SessionUpdateEvent(type="session.update", session=session))


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "fail", "delete", "disconnect"])
async def test_chunked_recognition_keeps_one_realtime_item_and_one_terminal_result(mocker, monkeypatch, outcome):
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    model_cls = mocker.Mock()
    model_cls.supported_languages = {"de": "German"}
    model_cls.no_space_languages = {"ja", "zh"}
    model_cls.get_speech_to_text_config.return_value = SimpleNamespace(sample_rate=24000, max_audio_clip_s=1)
    model_cls.get_generation_prompt.return_value = {"prompt": "recognize"}
    model_cls.post_process_output.side_effect = lambda text: text
    renderer = SimpleNamespace(render_cmpl_async=mocker.AsyncMock(return_value=[{"prompt_token_ids": [1]}]))
    engine = SimpleNamespace(renderer=renderer, default_sampling_params_list=[SamplingParams()])
    transcriber = SameModelTranscriber(
        engine, TranscriptionStage(0, SimpleNamespace(max_model_len=100), model_cls), {"test-model"}
    )
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.parse_model_prompt", lambda config, prompt: prompt)
    second_started = asyncio.Event()
    release = asyncio.Event()
    submitted = []
    closed = []

    async def generate(**kwargs):
        submitted.append(kwargs["request_id"])
        try:
            if len(submitted) == 2:
                second_started.set()
                await release.wait()
                if outcome == "fail":
                    raise RuntimeError("later clip failed")
            output = make_text_omni_output(text=f"part{len(submitted)}", finish_reason="stop")
            output.stage_id = 0
            yield output
        finally:
            closed.append(kwargs["request_id"])

    engine.generate = generate
    await _enable_transcription(connection, transcriber)
    original_audio = b"\x01\x00" * (24000 * 3 + 100)
    connection.session.input_audio_buffer.extend(original_audio)
    item = await connection._commit_audio_buffer()
    task = connection._transcription_tasks[item.id]
    await second_started.wait()
    assert len(connection.session.items) == 1
    assert base64.b64decode(item.content[0].audio) == original_audio
    assert not item.content[0].transcript
    assert not any("input_audio_transcription.completed" in message for message in websocket.messages)
    assert connection.response_creates == []
    if outcome == "delete":
        await connection._handle_item_delete(
            types.ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item.id)
        )
    elif outcome == "disconnect":
        await connection._cleanup()
    else:
        release.set()
        await task
    terminal = [
        json.loads(message)
        for message in websocket.messages
        if "conversation.item.input_audio_transcription." in message
    ]
    if outcome == "complete":
        assert len(terminal) == 1
        assert terminal[0]["type"] == "conversation.item.input_audio_transcription.completed"
        assert terminal[0]["item_id"] == item.id
        assert terminal[0]["transcript"] == " ".join(f"part{i + 1}" for i in range(len(submitted)))
        retained = connection.session.find_item(item.id)
        assert retained.content[0].transcript == terminal[0]["transcript"]
        assert base64.b64decode(retained.content[0].audio) == original_audio
    elif outcome == "fail":
        assert len(terminal) == 1
        assert terminal[0]["type"] == "conversation.item.input_audio_transcription.failed"
        assert terminal[0]["item_id"] == item.id
        assert len(submitted) == 2
    else:
        assert terminal == []
        assert task.cancelled()
        assert len(submitted) == 2
    assert closed == submitted
    assert not connection._transcription_tasks
    assert connection.response_creates == []
    assert connection.session.active_response is None


@pytest.mark.asyncio
async def test_transcription_follows_commit_without_creating_response(transcription_logs):
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    await _enable_transcription(connection, transcriber)
    assert json.loads(websocket.messages[-1])["session"]["audio"]["input"]["transcription"]["model"] == "test-model"
    original_audio = b"\x01\x02" * 100
    connection.session.input_audio_buffer.extend(original_audio)
    await connection._handle_audio_commit(types.InputAudioBufferCommitEvent(type="input_audio_buffer.commit"))
    item = connection.session.items[-1]
    task = connection._transcription_tasks[item.id]
    await transcriber.started.wait()
    assert [json.loads(message)["type"] for message in websocket.messages][-3:] == [
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
    ]
    connection.session.input_audio_buffer.extend(b"\x03\x04")
    await _enable_transcription(connection, transcriber, language="en")
    transcriber.release.set()
    await task
    event = json.loads(websocket.messages[-1])
    assert event["type"] == "conversation.item.input_audio_transcription.completed"
    assert event["item_id"] == item.id
    assert event["content_index"] == 0
    assert event["transcript"] == "Wie geht es dir?"
    assert event["usage"] == {"type": "duration", "seconds": len(original_audio) / 48000}
    types.ConversationItemInputAudioTranscriptionCompletedEvent.model_validate(event)
    retained = connection.session.find_item(item.id)
    assert base64.b64decode(retained.content[0].audio) == original_audio
    assert retained.content[0].transcript == event["transcript"]
    assert transcriber.calls[0][:2] == (original_audio, 24000)
    assert transcriber.calls[0][2].language == "de"
    assert connection.session.input_audio_buffer == b"\x03\x04"
    assert connection.response_creates == []
    assert connection.session.active_response is None
    assert not connection._transcription_tasks
    records = transcription_logs.records
    assert len(records) == 3
    assert all(record.levelno == logging.INFO and record.exc_info is None for record in records)
    assert "Audio committed" in records[0].getMessage()
    assert "audio_bytes=200 transcription_scheduled=True" in records[0].getMessage()
    assert "ASR submit" in records[1].getMessage()
    assert "ASR completed" in records[2].getMessage()
    assert "elapsed_ms=" in records[2].getMessage()
    assert all(item.id in record.getMessage() for record in records)
    assert transcriber.transcript not in transcription_logs.text
    assert base64.b64encode(original_audio).decode() not in transcription_logs.text


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError, AttributeError])
async def test_transcription_failure_is_item_specific_and_does_not_leak_details(transcription_logs, error_type):
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    failure = error_type("private recognition details")
    transcriber = _FakeTranscriber(failure=failure)
    transcriber.release.set()
    await _enable_transcription(connection, transcriber)
    connection.session.input_audio_buffer.extend(b"\x01\x02")
    item = await connection._commit_audio_buffer()
    await connection._transcription_tasks[item.id]
    event = json.loads(websocket.messages[-1])
    assert event["type"] == "conversation.item.input_audio_transcription.failed"
    assert event["item_id"] == item.id
    assert event["content_index"] == 0
    assert "private" not in websocket.messages[-1]
    assert connection._connected
    assert connection.response_creates == []
    errors = [record for record in transcription_logs.records if record.levelno >= logging.WARNING]
    assert len(errors) == 1
    record = errors[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info[1] is failure
    assert record.exc_info[2] is not None
    assert "ASR failed" in record.getMessage()
    assert f"request_id=asr-{connection.session.session_id}-{item.id}" in record.getMessage()
    assert f"session_id={connection.session.session_id}" in record.getMessage()
    assert f"item_id={item.id}" in record.getMessage()
    assert "private recognition details" in transcription_logs.text
    assert "transcribe_pcm" in transcription_logs.text


@pytest.mark.asyncio
async def test_empty_recognition_is_a_failed_item_not_a_fabricated_question():
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    transcriber.transcript = "  "
    transcriber.release.set()
    await _enable_transcription(connection, transcriber)
    connection.session.input_audio_buffer.extend(b"\x00\x00")
    item = await connection._commit_audio_buffer()
    await connection._transcription_tasks[item.id]
    event = json.loads(websocket.messages[-1])
    assert event["type"] == "conversation.item.input_audio_transcription.failed"
    assert event["item_id"] == item.id
    assert event["error"]["code"] == "transcription_empty"
    assert connection.response_creates == []


@pytest.mark.asyncio
async def test_committed_transcriptions_keep_order_and_survive_input_clear():
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    await _enable_transcription(connection, transcriber)
    connection.session.input_audio_buffer.extend(b"\x01\x02")
    first = await connection._commit_audio_buffer()
    connection.session.input_audio_buffer.extend(b"\x03\x04")
    second = await connection._commit_audio_buffer()
    await transcriber.started.wait()
    assert len(transcriber.calls) == 1
    tasks = list(connection._transcription_tasks.values())
    await connection._handle_audio_clear(types.InputAudioBufferClearEvent(type="input_audio_buffer.clear"))
    await connection._handle_response_cancel(types.ResponseCancelEvent(type="response.cancel"))
    transcriber.release.set()
    await asyncio.gather(*tasks)
    completed = [
        json.loads(message) for message in websocket.messages if "input_audio_transcription.completed" in message
    ]
    assert [event["item_id"] for event in completed] == [first.id, second.id]
    assert [call[0] for call in transcriber.calls] == [b"\x01\x02", b"\x03\x04"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_item", [True, False])
async def test_removed_audio_does_not_deliver_late_transcription(delete_item, transcription_logs):
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    await _enable_transcription(connection, transcriber)
    connection.session.input_audio_buffer.extend(b"\x01\x02")
    item = await connection._commit_audio_buffer()
    await transcriber.started.wait()
    if delete_item:
        await connection._handle_item_delete(
            types.ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item.id)
        )
    else:
        await connection._cleanup()
    assert transcriber.cancelled.is_set()
    assert not connection._transcription_tasks
    assert connection.session.find_item(item.id) is None
    assert not any("input_audio_transcription.completed" in message for message in websocket.messages)
    assert not any("input_audio_transcription.failed" in message for message in websocket.messages)
    assert all(record.levelno < logging.WARNING and record.exc_info is None for record in transcription_logs.records)
    assert "ASR cancelled request_id=" in transcription_logs.text


@pytest.mark.asyncio
async def test_unknown_transcription_model_does_not_change_session(transcription_logs):
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    await _enable_transcription(connection, _FakeTranscriber(), model="unavailable-asr")
    assert json.loads(websocket.messages[-1])["type"] == "error"
    assert connection.session.config.audio.input.transcription is None
    assert len(transcription_logs.records) == 1
    assert transcription_logs.records[0].levelno == logging.WARNING
    assert transcription_logs.records[0].exc_info is None


@pytest.mark.asyncio
async def test_transcription_can_be_disabled_without_cancelling_committed_audio():
    connection = _make_connection(_FakeWebSocket())
    transcriber = _FakeTranscriber()
    await _enable_transcription(connection, transcriber)
    connection.session.input_audio_buffer.extend(b"\x01\x02")
    first = await connection._commit_audio_buffer()
    task = connection._transcription_tasks[first.id]
    await connection._handle_session_update(
        types.SessionUpdateEvent.model_validate(
            {"type": "session.update", "session": {"type": "realtime", "audio": {"input": {"transcription": None}}}}
        )
    )
    connection.session.input_audio_buffer.extend(b"\x03\x04")
    second = await connection._commit_audio_buffer()
    assert second.id not in connection._transcription_tasks
    transcriber.release.set()
    await task
    assert len(transcriber.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "history"])
async def test_transcription_limit_failure_has_terminal_event(monkeypatch, failure, transcription_logs):
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    await _enable_transcription(connection, transcriber)
    if failure == "timeout":
        monkeypatch.setattr("vllm_omni.entrypoints.openai.realtime.connection.TRANSCRIPTION_TIMEOUT_S", 0)
    connection.session.input_audio_buffer.extend(b"\x01\x02")
    item = await connection._commit_audio_buffer()
    if failure == "history":
        monkeypatch.setattr(
            "vllm_omni.entrypoints.openai.realtime.session.MAX_HISTORY_BYTES", connection.session._history_size()
        )
        transcriber.release.set()
    await connection._transcription_tasks[item.id]
    event = json.loads(websocket.messages[-1])
    assert event["type"] == "conversation.item.input_audio_transcription.failed"
    assert event["item_id"] == item.id
    assert not connection._transcription_tasks
    warnings = [record for record in transcription_logs.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert warnings[0].exc_info is None
    assert item.id in warnings[0].getMessage()
    assert ("TimeoutError" if failure == "timeout" else "HistoryLimitError") in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_transcription_queue_limit_reports_item_failure(monkeypatch, transcription_logs):
    monkeypatch.setattr("vllm_omni.entrypoints.openai.realtime.connection.MAX_PENDING_TRANSCRIPTIONS", 1)
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    await _enable_transcription(connection, transcriber)
    connection.session.input_audio_buffer.extend(b"\x01\x02")
    first = await connection._commit_audio_buffer()
    connection.session.input_audio_buffer.extend(b"\x03\x04")
    second = await connection._commit_audio_buffer()
    event = json.loads(websocket.messages[-1])
    assert event["type"] == "conversation.item.input_audio_transcription.failed"
    assert event["item_id"] == second.id
    assert event["error"]["code"] == "transcription_queue_full"
    assert list(connection._transcription_tasks) == [first.id]
    task = connection._transcription_tasks[first.id]
    transcriber.release.set()
    await task
    warnings = [record for record in transcription_logs.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert warnings[0].exc_info is None
    assert f"item_id={second.id} pending=1 limit=1" in warnings[0].getMessage()
    commits = [record.getMessage() for record in transcription_logs.records if "Audio committed" in record.getMessage()]
    assert f"item_id={second.id} audio_bytes=2 transcription_scheduled=False" in commits[-1]


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
@pytest.mark.parametrize("transcription", [False, True])
async def test_server_vad_commit_without_response_creation(transcription) -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket)
    transcriber = _FakeTranscriber()
    if transcription:
        await _enable_transcription(connection, transcriber)
        websocket.messages.clear()
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

    if transcription:
        tasks = list(connection._transcription_tasks.values())
        transcriber.release.set()
        await asyncio.gather(*tasks)
        assert len(transcriber.calls) == 1
        assert json.loads(websocket.messages[-1])["type"] == "conversation.item.input_audio_transcription.completed"
    else:
        assert not connection._transcription_tasks


@pytest.mark.asyncio
async def test_server_vad_clear_resets_detector_state(transcription_logs) -> None:
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
    assert len(transcription_logs.records) == 1
    record = transcription_logs.records[0]
    assert record.levelno == logging.INFO
    assert record.exc_info is None
    assert "Audio buffer cleared by client" in record.getMessage()
    assert connection.session.session_id in record.getMessage()
    assert "discarded_bytes=2 speech_active=True" in record.getMessage()
    transcription_logs.clear()
    await connection._handle_audio_clear(types.InputAudioBufferClearEvent(type="input_audio_buffer.clear"))
    assert not transcription_logs.records
