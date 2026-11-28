# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime transcription sessions (#11074).

``session.type: "transcription"`` turns a conversation realtime session
into a transcription-only one: every committed audio turn — by server VAD or
by ``input_audio_buffer.commit`` — becomes a single ASR request whose text
streams back as ``conversation.item.input_audio_transcription.delta`` and
finishes with ``.completed``. The client-event adapter rejects the session
type itself (``Literal["realtime"]``), so the connection absorbs it from the
raw JSON and normalizes the frame before validation; the parsed mode and
knobs are staged as a candidate and committed only when the update is
accepted, so a rejected update never changes the live mode. ASR runs on a
worker off the read loop, so frames keep being received while a turn
transcribes and a disconnect aborts the in-flight request.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
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
    """Feeds scripted client frames to ``handle_connection``'s read loop.

    Once the scripted frames are exhausted the socket stays open until
    ``disconnect_event`` is set, then raises ``WebSocketDisconnect`` — the
    client choosing when to hang up while the session is otherwise idle.
    """

    def __init__(self, incoming_messages: list[str], disconnect_event: asyncio.Event | None = None) -> None:
        super().__init__()
        self.incoming_messages = list(incoming_messages)
        self.disconnect_event = disconnect_event

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if self.incoming_messages:
            return self.incoming_messages.pop(0)
        if self.disconnect_event is not None:
            await self.disconnect_event.wait()
        raise WebSocketDisconnect()


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
        if instructions is not None:
            self.transcription_instructions.append(instructions)
        return {"prompt": "transcription-input"}


def _make_connection(websocket: _FakeWebSocket, engine) -> _TranscriptionConnection:
    chat_handler = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: object()))
    connection = _TranscriptionConnection(
        websocket=websocket,
        engine=engine,
        model_name="test-model",
        chat_handler=chat_handler,
    )
    connection.transcription_instructions = []
    return connection


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


def _events_of_type(websocket: _FakeWebSocket, event_type: str) -> list[dict]:
    return [e for e in _sent_events(websocket) if e.get("type") == event_type]


async def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


@pytest.mark.asyncio
async def test_transcription_session_over_wire() -> None:
    """The full loop: update, append, commit, stream, complete, disconnect."""
    hold = asyncio.Event()
    websocket = _InboundFakeWebSocket(
        [
            _transcription_update(),
            _audio_append(b"\x01\x02" * 100),
            json.dumps({"type": "input_audio_buffer.commit"}),
        ],
        disconnect_event=hold,
    )
    connection = _make_connection(websocket, _TranscribingEngine(["hello ", "world"]))
    conn_task = asyncio.create_task(connection.handle_connection())

    assert await _wait_for(
        lambda: bool(_events_of_type(websocket, "conversation.item.input_audio_transcription.completed"))
    ), f"transcription never completed; got: {[e.get('type') for e in _sent_events(websocket)]}"
    hold.set()
    await asyncio.wait_for(conn_task, timeout=5.0)

    events = _sent_events(websocket)
    event_types = [e["type"] for e in events]
    # The update was accepted — the normalized session type validates.
    assert "session.updated" in event_types
    assert "error" not in event_types
    deltas = _events_of_type(websocket, "conversation.item.input_audio_transcription.delta")
    completed = _events_of_type(websocket, "conversation.item.input_audio_transcription.completed")
    assert [d["delta"] for d in deltas] == ["hello ", "world"]
    assert len(completed) == 1
    assert completed[0]["transcript"] == "hello world"
    assert completed[0]["item_id"] == deltas[0]["item_id"]
    # The ASR knobs ride along on the same accepted update.
    assert connection._transcription_language == "en"
    assert connection._transcription_prompt == "board meeting"


@pytest.mark.asyncio
async def test_rejected_update_keeps_live_mode_and_knobs() -> None:
    """A rejected session.update must not change the live mode or knobs.

    The second update is rejected at session-level validation (semantic_vad
    requires interrupt_response=true) while carrying an explicit realtime
    type and replacement ASR knobs; the following committed turn still
    transcribes with the original settings.
    """
    hold = asyncio.Event()
    rejected_update = json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "input_audio_transcription": {"language": "fr", "prompt": "stadium tour"},
                "audio": {"input": {"turn_detection": {"type": "semantic_vad", "interrupt_response": False}}},
            },
        }
    )
    websocket = _InboundFakeWebSocket(
        [
            _transcription_update(),
            rejected_update,
            _audio_append(b"\x01\x02" * 100),
            json.dumps({"type": "input_audio_buffer.commit"}),
        ],
        disconnect_event=hold,
    )
    connection = _make_connection(websocket, _TranscribingEngine(["hello"]))
    conn_task = asyncio.create_task(connection.handle_connection())

    assert await _wait_for(
        lambda: bool(_events_of_type(websocket, "conversation.item.input_audio_transcription.completed"))
    ), f"transcription never completed; got: {[e.get('type') for e in _sent_events(websocket)]}"
    hold.set()
    await asyncio.wait_for(conn_task, timeout=5.0)

    # The second update was rejected, not silently applied.
    errors = _events_of_type(websocket, "error")
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "unsupported_turn_detection"
    # The live mode and knobs are the ones from the accepted update.
    assert connection._transcription_mode is True
    assert connection._transcription_language == "en"
    assert connection._transcription_prompt == "board meeting"
    instructions = connection.transcription_instructions[-1]
    assert "The language is en." in instructions
    assert "Context: board meeting" in instructions
    assert "fr" not in instructions
    assert "stadium" not in instructions


@pytest.mark.asyncio
async def test_read_loop_stays_live_during_asr_and_abort_on_disconnect() -> None:
    """Frames are received while ASR is pending; disconnect aborts it once.

    The engine holds the generator open, so the committed turn's ASR never
    finishes on its own: the read loop must still serve a following
    ``input_audio_buffer.clear``, and the disconnect must abort the
    in-flight request exactly once and stop the worker.
    """
    hold = asyncio.Event()
    release_asr = asyncio.Event()
    aborted: list[str] = []

    class _HeldEngine(SimpleNamespace):
        def __init__(self) -> None:
            super().__init__(
                model_config=SimpleNamespace(max_model_len=100),
                default_sampling_params_list=[SamplingParams()],
            )

        def generate(self, *, prompt, request_id, sampling_params_list, output_modalities):
            async def _stream():
                await release_asr.wait()
                yield SimpleNamespace(outputs=[SimpleNamespace(text="too late")])

            return _stream()

        async def abort(self, request_id):
            aborted.append(request_id)

    websocket = _InboundFakeWebSocket(
        [
            _transcription_update(),
            _audio_append(b"\x01\x02" * 100),
            json.dumps({"type": "input_audio_buffer.commit"}),
            json.dumps({"type": "input_audio_buffer.clear"}),
        ],
        disconnect_event=hold,
    )
    connection = _make_connection(websocket, _HeldEngine())
    conn_task = asyncio.create_task(connection.handle_connection())

    # The clear is served while the first ASR is still held — the read loop
    # did not block on the generator.
    assert await _wait_for(lambda: bool(_events_of_type(websocket, "input_audio_buffer.cleared"))), (
        "the read loop stayed blocked on the ASR: input_audio_buffer.clear was never served"
    )
    assert not _events_of_type(websocket, "conversation.item.input_audio_transcription.completed")

    # Disconnect while the ASR is in flight: cleanup aborts it exactly once.
    hold.set()
    await asyncio.wait_for(conn_task, timeout=5.0)
    assert len(aborted) == 1
    assert aborted[0].startswith("transcription")
    assert connection._transcription_task is None or connection._transcription_task.done()
    # No transcript events leak to a client that is already gone.
    assert not _events_of_type(websocket, "conversation.item.input_audio_transcription.completed")


@pytest.mark.asyncio
async def test_vad_committed_turn_transcribes_without_create_response() -> None:
    """A VAD turn committed under create_response=false still reaches ASR.

    The knob only gates the conversation response a transcription session
    never creates; without this routing the committed audio would be
    dropped from ASR with nothing but input_audio_buffer.committed sent.
    """
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, _TranscribingEngine(["hello"]))
    connection._absorb_transcription_session(_transcription_update())
    connection._commit_transcription_session()
    connection._turn_detector = _FakeTurnDetector(
        [
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                should_commit=True,
                create_response=False,
            )
        ]
    )

    await connection._handle_audio_append(
        types.InputAudioBufferAppendEvent(
            type="input_audio_buffer.append",
            audio=base64.b64encode(b"\x01\x02" * 100).decode("ascii"),
        )
    )
    await asyncio.wait_for(connection._transcription_queue.join(), timeout=5.0)
    await connection._cancel_active_transcription()

    events = _sent_events(websocket)
    completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert len(completed) == 1
    assert completed[0]["transcript"] == "hello"


@pytest.mark.asyncio
async def test_server_vad_committed_turn_transcribes() -> None:
    """A VAD end-of-speech commit routes to ASR, not response.create."""
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, _TranscribingEngine(["hello"]))
    connection._absorb_transcription_session(_transcription_update())
    connection._commit_transcription_session()
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
    await asyncio.wait_for(connection._transcription_queue.join(), timeout=5.0)
    await connection._cancel_active_transcription()

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
    connection._commit_transcription_session()

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
    connection._commit_transcription_session()
    audio = base64.b64encode(b"\x01\x02" * 100).decode("ascii")
    item = types.RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        status="completed",
        content=[{"type": "input_audio", "audio": audio}],
    )

    await connection._run_transcription(item, "en", "board meeting")

    failed = [e for e in _sent_events(websocket) if e["type"] == "conversation.item.input_audio_transcription.failed"]
    assert len(failed) == 1
    assert failed[0]["content_index"] == 0
    assert failed[0]["error"]["code"] == "transcription_error"


@pytest.mark.asyncio
async def test_committed_turns_snapshot_their_asr_options() -> None:
    """Options are snapshotted at commit time, not dequeue time.

    With the first turn's ASR held, a second turn is committed under the
    original knobs, then an accepted session.update changes the language
    before the first turn releases: the queued turn still transcribes with
    the options it was committed under, while a later turn — committed after
    the update — uses the new ones.
    """
    hold = asyncio.Event()
    release_first = asyncio.Event()

    class _HoldFirstEngine(SimpleNamespace):
        def __init__(self) -> None:
            super().__init__(
                model_config=SimpleNamespace(max_model_len=100),
                default_sampling_params_list=[SamplingParams()],
            )
            self.calls = 0

        def generate(self, *, prompt, request_id, sampling_params_list, output_modalities):
            first = self.calls == 0
            self.calls += 1

            async def _stream():
                if first:
                    await release_first.wait()
                yield SimpleNamespace(outputs=[SimpleNamespace(text="t")])

            return _stream()

    fr_update = json.dumps(
        {
            "type": "session.update",
            "session": {"type": "transcription", "input_audio_transcription": {"language": "fr"}},
        }
    )
    websocket = _InboundFakeWebSocket(
        [
            _transcription_update(),
            _audio_append(b"\x01\x02" * 100),
            json.dumps({"type": "input_audio_buffer.commit"}),
            _audio_append(b"\x03\x04" * 100),
            json.dumps({"type": "input_audio_buffer.commit"}),
            fr_update,
            _audio_append(b"\x05\x06" * 100),
            json.dumps({"type": "input_audio_buffer.commit"}),
        ],
        disconnect_event=hold,
    )
    connection = _make_connection(websocket, _HoldFirstEngine())
    conn_task = asyncio.create_task(connection.handle_connection())

    def completed() -> list[dict]:
        return _events_of_type(websocket, "conversation.item.input_audio_transcription.completed")

    # Let every commit land (the worker stays held on turn A), then release.
    assert await _wait_for(lambda: connection._transcription_queue.qsize() == 2)
    release_first.set()
    assert await _wait_for(lambda: len(completed()) == 3), (
        f"not all turns completed; got: {[e.get('type') for e in _sent_events(websocket)]}"
    )
    hold.set()
    await asyncio.wait_for(conn_task, timeout=5.0)

    assert len(connection.transcription_instructions) == 3
    turn_a, turn_b, turn_c = connection.transcription_instructions
    assert "The language is en." in turn_a
    # B was committed under the en knobs — the later fr update does not
    # change how it transcribes.
    assert "The language is en." in turn_b
    assert "The language is fr." not in turn_b
    assert "Context: board meeting" in turn_b
    # C was committed after the update: it uses the new language.
    assert "The language is fr." in turn_c


@pytest.mark.asyncio
async def test_ga_session_shape_knobs_reach_the_asr() -> None:
    """The GA session shape carries the knobs under audio.input.transcription.

    The OpenAI SDK's ``type: "transcription"`` session sends language and
    prompt there with no flat key; both shapes must reach the ASR.
    """
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine(["hello"]))
    ga_update = json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {"transcription": {"language": "en", "prompt": "board meeting"}}},
            },
        }
    )

    out = connection._absorb_transcription_session(ga_update)
    connection._commit_transcription_session()

    assert connection._transcription_mode is True
    assert connection._transcription_language == "en"
    assert connection._transcription_prompt == "board meeting"
    # The session type is normalized for the adapter (its Literal is
    # "realtime"); the GA knobs ride audio.input.transcription, a known
    # field the adapter validates as-is.
    normalized = json.loads(out)
    assert normalized["session"]["type"] == "realtime"
    assert normalized["session"]["audio"]["input"]["transcription"] == {"language": "en", "prompt": "board meeting"}


@pytest.mark.asyncio
async def test_prompt_preparation_failure_is_a_per_item_event() -> None:
    """A prompt-preparation failure ends with a per-item .failed event.

    The committed item is already announced and turns run asynchronously,
    so a generic session error without an item_id would leave the client
    unable to tell which item failed.
    """
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, _TranscribingEngine([]))
    connection._absorb_transcription_session(_transcription_update())
    connection._commit_transcription_session()

    async def _failing_prompt(tools=None, *, instructions=None, items=None, **_kwargs):
        raise RuntimeError("prompt rendering exploded")

    connection._build_full_prompt = _failing_prompt
    audio = base64.b64encode(b"\x01\x02" * 100).decode("ascii")
    item = types.RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        status="completed",
        id="item-prompt-fail",
        content=[{"type": "input_audio", "audio": audio}],
    )

    await connection._run_transcription(item, "en", "board meeting")

    failed = [e for e in _sent_events(websocket) if e["type"] == "conversation.item.input_audio_transcription.failed"]
    assert len(failed) == 1
    assert failed[0]["item_id"] == item.id
    assert failed[0]["content_index"] == 0
    assert failed[0]["error"]["code"] == "transcription_error"
    assert "prompt rendering exploded" in failed[0]["error"]["message"]
    # No generic session-level error was sent instead.
    assert not _events_of_type(websocket, "error")


def test_session_acknowledgement_echo_keeps_transcription_on() -> None:
    """session.updated reports the effective transcription session.

    A client that echoes the acknowledged settings back as its next update
    must keep the mode (and the knobs) instead of switching it off.
    """
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))
    connection._absorb_transcription_session(_transcription_update())
    connection._commit_transcription_session()

    session_obj = connection._build_session_object()
    assert session_obj["type"] == "transcription"
    assert session_obj["input_audio_transcription"] == {"language": "en", "prompt": "board meeting"}

    # Echo the acknowledged session back as the next update.
    echo = json.dumps({"type": "session.update", "session": session_obj})
    connection._absorb_transcription_session(echo)
    connection._commit_transcription_session()
    assert connection._transcription_mode is True
    assert connection._transcription_language == "en"
    assert connection._transcription_prompt == "board meeting"


def test_absorb_stages_a_candidate_and_normalizes_for_the_adapter() -> None:
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))

    out = connection._absorb_transcription_session(_transcription_update())

    # Staged, not applied: the live mode is untouched until the update is
    # accepted and the candidate committed.
    assert connection._pending_transcription_session == {
        "mode": True,
        "language": "en",
        "prompt": "board meeting",
    }
    assert connection._transcription_mode is False
    normalized = json.loads(out)
    assert normalized["session"]["type"] == "realtime"
    # Everything else carries through untouched.
    assert normalized["session"]["input_audio_transcription"] == {"language": "en", "prompt": "board meeting"}


def test_absorb_without_type_keeps_the_mode() -> None:
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))
    connection._absorb_transcription_session(_transcription_update())
    connection._commit_transcription_session()

    # A later update without a session type (the common case) stages an empty
    # candidate: committing it must not switch the transcription session off.
    connection._absorb_transcription_session(json.dumps({"type": "session.update", "session": {"voice": "alloy"}}))
    connection._commit_transcription_session()
    assert connection._transcription_mode is True
    # An explicit realtime type does switch back to a conversation session.
    connection._absorb_transcription_session(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
    connection._commit_transcription_session()
    assert connection._transcription_mode is False


def test_absorb_passes_other_frames_through() -> None:
    connection = _make_connection(_FakeWebSocket(), _TranscribingEngine([]))

    for text in (
        json.dumps({"type": "input_audio_buffer.append", "audio": "dGVzdA=="}),
        "not json at all",
        json.dumps({"type": "session.update", "session": "not-a-dict"}),
    ):
        assert connection._absorb_transcription_session(text) == text
    assert connection._pending_transcription_session is None
    assert connection._transcription_mode is False


class _FakeRawFramingCleaner:
    """Strips Qwen3-ASR-style ``language X<asr_text>`` framing, incrementally."""

    def __init__(self) -> None:
        self.raw_text = ""
        self.emitted_text = ""

    def process_delta(self, text_delta: str, finished: bool) -> str:
        self.raw_text += text_delta
        processed = self.raw_text.split("<asr_text>", 1)[1] if "<asr_text>" in self.raw_text else ""
        new_text = processed[len(self.emitted_text) :]
        self.emitted_text = processed
        return new_text


class _FakeInterfaceASR:
    """A stage-0 model implementing the vLLM speech-to-text interface."""

    last_stt_params = None

    @classmethod
    def get_speech_to_text_config(cls, model_config, task_type):
        assert task_type == "transcribe"
        return SimpleNamespace(sample_rate=16000.0)

    @classmethod
    def get_generation_prompt(cls, stt_params):
        cls.last_stt_params = stt_params
        return {"prompt_token_ids": [1, 2, 3], "multi_modal_data": {"audio": stt_params.audio}}

    @classmethod
    def get_streaming_post_processor_cls(cls):
        return _FakeRawFramingCleaner


class _PartialInterfaceASR:
    """Only the prompt builder — not enough to leave the fallback path."""

    @classmethod
    def get_generation_prompt(cls, stt_params):
        raise AssertionError("must not be called")


class _RecordingEngine(SimpleNamespace):
    """``engine.generate`` returns a scripted stream and records the prompt."""

    def __init__(self, text_chunks: list[str]) -> None:
        super().__init__(
            model_config=SimpleNamespace(max_model_len=100),
            default_sampling_params_list=[SamplingParams()],
        )
        self.text_chunks = list(text_chunks)
        self.prompts = []

    def generate(self, *, prompt, request_id, sampling_params_list, output_modalities):
        self.prompts.append(prompt)
        chunks = self.text_chunks

        async def _stream():
            for chunk in chunks:
                yield SimpleNamespace(outputs=[SimpleNamespace(text=chunk)])

        return _stream()


@pytest.mark.asyncio
async def test_model_transcription_interface_drives_the_turn() -> None:
    """#11074 follow-up: a stage-0 model with the vLLM speech-to-text
    interface drives the turn like ``/v1/audio/transcriptions`` — the knobs
    ride the model's prompt protocol, the audio is resampled to the model's
    rate, and the raw framing never reaches the client deltas."""
    hold = asyncio.Event()
    websocket = _InboundFakeWebSocket(
        [
            _transcription_update(),
            _audio_append(b"\x01\x02" * 12000),
            json.dumps({"type": "input_audio_buffer.commit"}),
        ],
        disconnect_event=hold,
    )
    engine = _RecordingEngine(["language English<asr_text>He", "llo ", "world"])
    connection = _make_connection(websocket, engine)
    connection._transcription_model_cls = _FakeInterfaceASR
    conn_task = asyncio.create_task(connection.handle_connection())

    assert await _wait_for(
        lambda: bool(_events_of_type(websocket, "conversation.item.input_audio_transcription.completed"))
    ), f"transcription never completed; got: {[e.get('type') for e in _sent_events(websocket)]}"
    hold.set()
    await asyncio.wait_for(conn_task, timeout=5.0)

    # The streamed deltas and the final transcript are cleaned: the raw
    # ``language English<asr_text>`` framing never reaches the client.
    deltas = _events_of_type(websocket, "conversation.item.input_audio_transcription.delta")
    assert [d["delta"] for d in deltas] == ["He", "llo ", "world"]
    completed = _events_of_type(websocket, "conversation.item.input_audio_transcription.completed")
    assert completed[0]["transcript"] == "Hello world"
    # The instructions fallback was not used.
    assert connection.transcription_instructions == []
    # The model's interface received the snapshotted knobs and the
    # resampled audio, and its prompt reached the engine unchanged.
    stt = _FakeInterfaceASR.last_stt_params
    assert stt.language == "en"
    assert stt.request_prompt == "board meeting"
    assert stt.task_type == "transcribe"
    assert engine.prompts == [{"prompt_token_ids": [1, 2, 3], "multi_modal_data": {"audio": stt.audio}}]
    # 0.5 s of 24 kHz session audio was resampled to the model's 16 kHz rate.
    assert abs(len(stt.audio) - 8000) <= 4


def test_resolve_transcription_model_requires_the_full_interface(mocker) -> None:
    """Only a complete speech-to-text interface leaves the fallback path."""
    get_model_cls = mocker.patch("vllm.model_executor.model_loader.get_model_cls")

    get_model_cls.return_value = _PartialInterfaceASR
    partial = _make_connection(_FakeWebSocket(), _TranscribingEngine(["x"]))
    assert partial._resolve_transcription_model() is None
    get_model_cls.assert_called_once_with(partial.engine.model_config)

    get_model_cls.return_value = _FakeInterfaceASR
    complete = _make_connection(_FakeWebSocket(), _TranscribingEngine(["x"]))
    assert complete._resolve_transcription_model() is _FakeInterfaceASR
    # The resolution is cached — the registry is consulted once per
    # connection, not once per committed turn.
    assert complete._resolve_transcription_model() is _FakeInterfaceASR
    assert get_model_cls.call_count == 2
