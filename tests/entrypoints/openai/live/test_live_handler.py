# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``LiveSessionHandler`` against a fake WebSocket and a scripted engine."""

import asyncio
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import WebSocketDisconnect
from vllm import SamplingParams

from vllm_omni.config.live_session import (
    LIVE_PREEMPTED_KEY,
    LiveSessionConfig,
    LiveSessionDeployConfig,
    UnsupportedFeatures,
)
from vllm_omni.entrypoints.openai.live import protocol as proto
from vllm_omni.entrypoints.openai.live.handler import LiveSessionHandler, SessionSlots
from vllm_omni.entrypoints.openai.live.processor import LiveSessionProcessor, ProcessorContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "test-model"
RATE = 16_000
UNIT_MS = 100
UNIT_SAMPLES = RATE * UNIT_MS // 1000
BACKLOG_S = 1.0


class _FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[dict] = []
        self.close_code: int | None = None

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        text = await self.incoming.get()
        if text is None:
            raise WebSocketDisconnect()
        return text

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000) -> None:
        self.close_code = code

    def send(self, event: dict) -> None:
        self.incoming.put_nowait(json.dumps(event))

    def of_type(self, event_type: str) -> list[dict]:
        return [event for event in self.sent if event["type"] == event_type]


class _FakeRequest:
    def __init__(self) -> None:
        self.inputs: list = []
        self.outputs: asyncio.Queue = asyncio.Queue()
        self.closed = False

    def emit(self, output) -> None:
        self.outputs.put_nowait(output)


class _FakeEngine:
    """Each ``generate`` call is one request whose outputs the test scripts."""

    def __init__(self) -> None:
        self.default_sampling_params_list = [SamplingParams(max_tokens=8)]
        self.requests: list[_FakeRequest] = []

    def generate(self, *, prompt, request_id, sampling_params_list):
        request = _FakeRequest()
        self.requests.append(request)
        return self._run(request, prompt)

    async def _run(self, request: _FakeRequest, prompt):
        async def feed() -> None:
            async for item in prompt:
                request.inputs.append(item.prompt)

        feeder = asyncio.create_task(feed())
        try:
            while True:
                item = await request.outputs.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            feeder.cancel()
            request.closed = True


class _UnitProcessor(LiveSessionProcessor):
    """100 ms units at 16 kHz, each costing 100 of a 300-token context."""

    input_sample_rate = RATE
    output_sample_rate = RATE
    unit_ms = UNIT_MS

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def render_unit(self, state, unit, *, first, last_sampled_token):
        return {"unit": unit.start_ms, "first": first}

    def render_resume(self, state):
        return {"resume": [unit.start_ms for unit in state.units]}

    def prefix_tokens(self, state):
        return 0

    def unit_tokens(self, unit):
        return 100


def _stage0(token_ids, finish_reason=None, text="", multimodal_output=None):
    completion = SimpleNamespace(
        token_ids=token_ids, text=text, finish_reason=finish_reason, multimodal_output=multimodal_output
    )
    return SimpleNamespace(stage_id=0, outputs=[completion], final_output_type="text", multimodal_output=None)


def _start_event(**session) -> dict:
    return {
        "type": "session.start",
        "event_id": "evt_start",
        "session": {
            "model": MODEL,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": "default"}},
            **session,
        },
    }


def _append_event(units: int = 1) -> dict:
    pcm = np.zeros(UNIT_SAMPLES * units, dtype="<i2").tobytes()
    return {"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()}


async def _until(predicate, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


class _Session:
    def __init__(self, slots: SessionSlots | None = None, unsupported=None) -> None:
        self.ws = _FakeWebSocket()
        self.engine = _FakeEngine()
        live_config = LiveSessionConfig("unused:Proc", unsupported_features=unsupported or UnsupportedFeatures())
        self.handler = LiveSessionHandler(
            self.ws,
            engine=self.engine,
            model_name=MODEL,
            live_config=live_config,
            deploy_config=LiveSessionDeployConfig(
                session_lifetime_s=1800, max_sessions=4, max_input_backlog_s=BACKLOG_S
            ),
            processor_cls=_UnitProcessor,
            processor_context=ProcessorContext(tokenizer=None, max_model_len=300, model_name=MODEL, model_path=MODEL),
            slots=slots,
        )
        self.task: asyncio.Task | None = None

    async def __aenter__(self) -> "_Session":
        self.task = asyncio.create_task(self.handler.run())
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.ws.incoming.put_nowait(None)
        assert self.task is not None
        await asyncio.wait_for(self.task, 5)

    async def start(self) -> None:
        self.ws.send(_start_event())
        await _until(lambda: self.ws.of_type("session.started"))

    def errors(self) -> list[str]:
        return [event["error"]["code"] for event in self.ws.of_type("error")]

    async def finished(self) -> None:
        assert self.task is not None
        await asyncio.wait_for(asyncio.shield(self.task), 5)


async def test_session_start_reports_started_and_capabilities():
    async with _Session() as session:
        await session.start()
        await _until(lambda: session.ws.of_type("info"))

        started = session.ws.of_type("session.started")[0]
        info = session.ws.of_type("info")[0]
        assert started["session"]["model"] == MODEL
        assert json.loads(info["message"])["vad"] == "native"


async def test_session_limit_rejects_with_try_again_later():
    slots = SessionSlots(1)
    async with _Session(slots) as first:
        await first.start()

        async with _Session(slots) as second:
            second.ws.send(_start_event())
            await second.finished()
            assert second.errors() == ["session_limit_reached"]
            assert second.ws.close_code == proto.CLOSE_TRY_AGAIN_LATER

        first.ws.send({"type": "session.close"})
        await first.finished()

    assert slots.active == 0
    async with _Session(slots) as third:
        await third.start()
        assert slots.active == 1


@pytest.mark.parametrize(
    ("text", "code"),
    [
        pytest.param("{not json", "invalid_json", id="invalid-json"),
        pytest.param(json.dumps({"type": "session.unknown"}), "unsupported_event", id="unknown-event"),
        pytest.param(json.dumps({"type": "response.create"}), "unsupported_event", id="server-unsupported"),
        pytest.param(json.dumps(_append_event()), "session_not_started", id="append-before-start"),
    ],
)
async def test_protocol_errors_keep_the_connection_open(text, code):
    async with _Session() as session:
        session.ws.incoming.put_nowait(text)
        await _until(lambda: session.ws.of_type("error"))
        await session.start()

        assert session.errors() == [code]
        assert session.ws.close_code is None


async def test_model_specific_unsupported_event():
    async with _Session(unsupported=UnsupportedFeatures(events=("session.input_audio.mute",))) as session:
        await session.start()
        session.ws.send({"type": "session.input_audio.mute"})
        await _until(lambda: session.ws.of_type("error"))

        assert session.errors() == ["unsupported_event"]


async def test_second_session_start_is_rejected():
    async with _Session() as session:
        await session.start()
        session.ws.send(_start_event())
        await _until(lambda: session.ws.of_type("error"))

        assert session.errors() == ["session_already_started"]
        assert len(session.ws.of_type("session.started")) == 1


async def test_ignored_event_gets_no_reply():
    async with _Session() as session:
        await session.start()
        session.ws.send({"type": "session.commentary.append", "event_id": "evt_c"})
        session.ws.send({"type": "session.input_audio.mute", "event_id": "evt_m"})
        await _until(lambda: session.ws.of_type("session.input_audio.muted"))

        assert session.errors() == []


@pytest.mark.parametrize(
    ("session_config", "code", "param"),
    [
        pytest.param({"store": True}, "unsupported_session_config", "store", id="store"),
        pytest.param({"model": "other"}, "invalid_value", "model", id="wrong-model"),
        pytest.param(
            {"audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": "alloy"}}},
            "invalid_value",
            "audio.output.voice",
            id="openai-voice",
        ),
        pytest.param(
            {"delegation": {"type": "responses", "responses": {"tools": [{"type": "function"}]}}},
            "unsupported_session_config",
            "delegation.responses.tools",
            id="tools",
        ),
    ],
)
async def test_invalid_session_start_closes_with_policy_violation(session_config, code, param):
    slots = SessionSlots(1)
    async with _Session(slots) as session:
        session.ws.send(_start_event(**session_config))
        await session.finished()

        error = session.ws.of_type("error")[0]["error"]
        assert (error["code"], error.get("param")) == (code, param)
        assert session.ws.close_code == proto.CLOSE_POLICY_VIOLATION
    assert slots.active == 0


async def test_units_stream_as_updates_of_one_request():
    async with _Session() as session:
        await session.start()
        session.ws.send(_append_event(units=2))
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)
        request = session.engine.requests[0]

        request.emit(_stage0([7], finish_reason="stop"))
        await _until(lambda: len(request.inputs) == 2)

        assert request.inputs == [{"unit": 0, "first": True}, {"unit": UNIT_MS, "first": False}]
        assert len(session.engine.requests) == 1


async def test_context_overflow_trims_and_resubmits(mocker):
    async with _Session() as session:
        await session.start()
        trims = mocker.spy(session.handler.metrics, "context_trim")
        session.ws.send(_append_event(units=3))
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)
        first = session.engine.requests[0]
        for expected in (2, 3):
            first.emit(_stage0([7], finish_reason="stop"))
            await _until(lambda expected=expected: len(first.inputs) == expected)

        first.emit(RuntimeError("context_length_exceeded: the next update does not fit"))
        await _until(lambda: len(session.engine.requests) == 2 and session.engine.requests[1].inputs)

        # budget = 300 * 0.5 = 150 tokens: only the overflowing unit is kept
        assert session.engine.requests[1].inputs == [{"resume": [2 * UNIT_MS]}]
        assert first.closed
        assert trims.call_count == 1
        assert session.ws.close_code is None
        assert session.errors() == []


async def test_context_overflow_on_the_first_unit_fails_the_session():
    async with _Session() as session:
        await session.start()
        session.ws.send(_append_event())
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)

        session.engine.requests[0].emit(RuntimeError("context_length_exceeded"))
        await session.finished()

        assert session.errors() == ["internal_server_error"]
        assert session.ws.close_code == proto.CLOSE_INTERNAL_ERROR


async def test_engine_error_fails_the_session():
    async with _Session() as session:
        await session.start()
        session.ws.send(_append_event())
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)

        session.engine.requests[0].emit(RuntimeError("CUDA out of memory"))
        await session.finished()

        assert session.errors() == ["internal_server_error"]
        assert session.ws.close_code == proto.CLOSE_INTERNAL_ERROR


async def test_session_close_aborts_the_request():
    slots = SessionSlots(4)
    async with _Session(slots) as session:
        await session.start()
        session.ws.send(_append_event())
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)

        session.ws.send({"type": "session.close", "event_id": "evt_close"})
        await session.finished()

        closed = session.ws.of_type("session.closed")[0]
        assert closed["reason"] == "close_requested"
        assert session.engine.requests[0].closed
        assert session.ws.close_code == proto.CLOSE_NORMAL
    assert slots.active == 0


async def test_input_backlog_over_the_limit_closes_with_try_again_later(mocker):
    async with _Session() as session:
        await session.start()
        errors = mocker.spy(session.handler.metrics, "error")
        # The model never finishes a segment: one unit is in flight, the rest queue.
        session.ws.send(_append_event(units=5))
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)
        assert session.errors() == []

        session.ws.send(_append_event(units=7))
        await session.finished()

        assert session.errors() == ["internal_server_error"]
        assert session.ws.close_code == proto.CLOSE_TRY_AGAIN_LATER
        assert session.engine.requests[0].closed
        errors.assert_called_once_with("backlog")


async def test_preempted_request_cancels_the_session(mocker):
    async with _Session() as session:
        await session.start()
        errors = mocker.spy(session.handler.metrics, "error")
        session.ws.send(_append_event())
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)

        session.engine.requests[0].emit(_stage0([7], text="hello", multimodal_output={LIVE_PREEMPTED_KEY: True}))
        await session.finished()

        assert session.errors() == ["internal_server_error"]
        assert session.ws.close_code == proto.CLOSE_TRY_AGAIN_LATER
        assert session.ws.of_type("session.output_transcript.delta") == []
        assert session.engine.requests[0].closed
        errors.assert_called_once_with("preempted")


async def test_unpreempted_stage0_output_is_forwarded():
    async with _Session() as session:
        await session.start()
        session.ws.send(_append_event())
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)

        session.engine.requests[0].emit(_stage0([7], text="hello", multimodal_output={LIVE_PREEMPTED_KEY: False}))
        await _until(lambda: session.ws.of_type("session.output_transcript.delta"))

        assert session.errors() == []
        assert session.ws.close_code is None


async def test_audio_output_is_sent_and_counted():
    async with _Session() as session:
        await session.start()
        session.ws.send(_append_event())
        await _until(lambda: session.engine.requests and session.engine.requests[0].inputs)
        audio = np.full(RATE // 10, 0.25, dtype=np.float32)

        session.engine.requests[0].emit(
            SimpleNamespace(stage_id=1, final_output_type="audio", multimodal_output={"audio": [audio], "sr": RATE})
        )
        await _until(lambda: session.ws.of_type("session.output_audio.delta"))

        delta = session.ws.of_type("session.output_audio.delta")[0]["delta"]
        assert np.frombuffer(base64.b64decode(delta), dtype="<i2").size == audio.size
        assert session.handler.state.output_audio_ms == pytest.approx(100.0)
