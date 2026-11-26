# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime transcription sessions (#11074).

``session.type: "transcription"`` turns a conversation realtime session
into a transcription-only one: every committed audio turn — by server VAD or
by ``input_audio_buffer.commit`` — becomes a single ASR request whose text
streams back as ``conversation.item.input_audio_transcription.delta`` and
finishes with ``.completed``. The client-event adapter rejects the session
type itself (``Literal["realtime"]``), so the connection absorbs it from the
raw JSON and normalizes the frame before validation.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect
from openai.types import realtime as types
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.turn_detection import (
    ServerTurnDetector,
    TurnDetectionConfig,
    TurnDetectionResult,
)
from vllm_omni.entrypoints.openai.realtime.connection import OpenAIFullDuplexConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


class _InboundFakeWebSocket(_FakeWebSocket):
    """Feeds scripted client frames to ``handle_connection``'s read loop."""

    def __init__(self, incoming_messages: list[str]) -> None:
        super().__init__()
        self.incoming_messages = list(incoming_messages)

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if not self.incoming_messages:
            raise WebSocketDisconnect()
        return self.incoming_messages.pop(0)


class _FakeTurnDetector(ServerTurnDetector):
    def __init__(self, results: list[TurnDetectionResult]) -> None:
        super().__init__(TurnDetectionConfig())
        self.results = list(results)

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        return self.results.pop(0)


class _TranscribingEngine(SimpleNamespace):
    """``engine.generate`` returns a scripted text-chunk stream."""

    def __init__(self, text_chunks: list[str]) -> None:
        super().__init__(
            model_config=SimpleNamespace(max_model_len=100),
            default_sampling_params_list=[SamplingParams()],
        )
        self.text_chunks = list(text_chunks)

    def generate(self, *, prompt, request_id, sampling_params_list, output_modalities):
        chunks = self.text_chunks

        async def _stream():
            for chunk in chunks:
                yield SimpleNamespace(outputs=[SimpleNamespace(text=chunk)])

        return _stream()


class _TranscriptionConnection(OpenAIFullDuplexConnection):
    """Stub only the chat prompt build; the transcription path owns the rest."""

    async def _build_full_prompt(self, tools=None, *, instructions=None, items=None, **_kwargs):
        return {"prompt": "transcription-input"}


def _make_connection(websocket: _FakeWebSocket, engine) -> _TranscriptionConnection:
    chat_handler = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: object()))
    return _TranscriptionConnection(
        websocket=websocket,
        engine=engine,
        model_name="test-model",
        chat_handler=chat_handler,
    )


def _transcription_update() -> str:
    """A client's raw session.update asking for a transcription session."""
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "input_audio_transcription": {"language": "en", "prompt": "board meeting"},
            },
        }
    )


def _audio_append(audio: bytes) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode("ascii"),
        }
    )


def _sent_events(websocket: _FakeWebSocket) -> list[dict]:
    return [json.loads(message) for message in websocket.messages]


@pytest.mark.asyncio
async def test_transcription_session_over_wire() -> None:
    """The full loop: update, append, commit, stream, complete."""
    audio = b"\x01\x02" * 100
    websocket = _InboundFakeWebSocket(
        [
            _transcription_update(),
            _audio_append(audio),
            json.dumps({"type": "input_audio_buffer.commit"}),
        ]
    )
    connection = _make_connection(websocket, _TranscribingEngine(["hello ", "world"]))

    await connection.handle_connection()

    events = _sent_events(websocket)
    event_types = [e["type"] for e in events]
    # The update was accepted — the normalized session type validates.
    assert "session.updated" in event_types
    assert "error" not in event_types
    deltas = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.delta"]
    completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert [d["delta"] for d in deltas] == ["hello ", "world"]
    assert len(completed) == 1
    assert completed[0]["transcript"] == "hello world"
    assert completed[0]["item_id"] == deltas[0]["item_id"]
    # The ASR knobs ride along on the same update.
    assert connection._transcription_language == "en"
    assert connection._transcription_prompt == "board meeting"


@pytest.mark.asyncio
async def test_server_vad_committed_turn_transcribes() -> None:
    """A VAD end-of-speech commit routes to ASR, not response.create."""
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, _TranscribingEngine(["hello"]))
    connection._absorb_transcription_session(_transcription_update())
    connection._turn_detector = _FakeTurnDetector(
        [
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                should_commit=True,
                create_response=True,
            )
        ]
    )

    await connection._handle_audio_append(
        types.InputAudioBufferAppendEvent(
            type="input_audio_buffer.append",
            audio=base64.b64encode(b"\x01\x02" * 100).decode("ascii"),
        )
    )

    events = _sent_events(websocket)
    completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert len(completed) == 1
    assert completed[0]["transcript"] == "hello"
    # No response lifecycle events leak into a transcription session.
    assert not [e for e in events if e["type"].startswith("response.")]


@pytest.mark.asyncio
async def test_response_create_rejected_in_transcription_session() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, _TranscribingEngine([]))
    connection._absorb_transcription_session(_transcription_update())

    await connection._handle_response_create(
        types.ResponseCreateEvent(type="response.create", event_id="evt-1", response=None)
    )

    errors = [e for e in _sent_events(websocket) if e["type"] == "error"]
    assert len(errors) == 1
    assert "transcription session" in errors[0]["error"]["message"]


@pytest.mark.asyncio
async def test_transcription_engine_failure_emits_failed() -> None:
    class _FailingEngine(SimpleNamespace):
        def __init__(self) -> None:
            super().__init__(
                model_config=SimpleNamespace(max_model_len=100),
                default_sampling_params_list=[SamplingParams()],
            )

        def generate(self, **_kwargs):
            async def _stream():
                raise RuntimeError("engine exploded")
                yield  # pragma: no cover

            return _stream()

    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, _FailingEngine())
    connection._absorb_transcription_session(_transcription_update())
    audio = base64.b64encode(b"\x01\x02" * 100).decode("ascii")
    item = types.RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        status="completed",
        content=[{"type": "input_audio", "audio": audio}],
    )

    await connection._run_transcription(item)

    failed = [e for e in _sent_events(websocket) if e["type"] == "conversation.item.input_audio_transcription.failed"]
    assert len(failed) == 1
    assert failed[0]["error"]["code"] == "transcription_error"


def test_absorb_transcription_session_normalizes_for_the_adapter() -> None:
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))

    out = connection._absorb_transcription_session(_transcription_update())

    assert connection._transcription_mode is True
    normalized = json.loads(out)
    assert normalized["session"]["type"] == "realtime"
    # Everything else carries through untouched.
    assert normalized["session"]["input_audio_transcription"] == {"language": "en", "prompt": "board meeting"}


def test_absorb_transcription_session_without_type_keeps_the_mode() -> None:
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))
    connection._absorb_transcription_session(_transcription_update())

    # A later update without a session type (the common case) must not
    # silently switch the transcription session back off.
    connection._absorb_transcription_session(json.dumps({"type": "session.update", "session": {"voice": "alloy"}}))
    assert connection._transcription_mode is True
    # An explicit realtime type does switch back to a conversation session.
    connection._absorb_transcription_session(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
    assert connection._transcription_mode is False


def test_absorb_transcription_session_passes_other_frames_through() -> None:
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))

    for text in (
        json.dumps({"type": "input_audio_buffer.append", "audio": "dGVzdA=="}),
        "not json at all",
        json.dumps({"type": "session.update", "session": "not-a-dict"}),
    ):
        assert connection._absorb_transcription_session(text) == text
    assert connection._transcription_mode is False
