# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Incremental TTS model input and ordinary-request WebSocket lifecycle."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import aclosing
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai import serving_speech_stream as stream_module
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest, StreamingSpeechSessionConfig
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    SpeechServingContext,
    TextCommitmentCapabilities,
)
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import (
    QWEN3_TTS_AUTO_BUDGET_KEY,
    Qwen3TTSAdapter,
)
from vllm_omni.model_executor.models.qwen3_tts.input_stream import Qwen3TTSTextInput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NATIVE_CONFIG = {
    "type": "session.config",
    "text_input_mode": "commitment",
    "model_input_mode": "incremental",
    "stream_audio": True,
    "response_format": "pcm",
    "voice": "vivian",
    "language": "English",
}


@pytest.fixture
def text_tokenizer(mocker):
    """Stub model tokenization; these tests cover delivery, not BPE behavior."""
    tokenizer = mocker.MagicMock()
    tokenizer.encode.side_effect = lambda text, **_kwargs: list(text.encode("utf-8"))
    tokenizer.__len__.return_value = 256
    return tokenizer


@pytest.fixture
def native_app(text_tokenizer, mocker):
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server._tts_model_type = "qwen3_tts"
    server._diffusion_mode = False
    server._tts_stage = None
    server._tts_tokenizer = text_tokenizer
    server._tts_executor = None
    server._max_instructions_length = 512
    server.uploaded_speakers = {}
    mocker.patch.object(server, "_refresh_uploaded_speakers")
    mocker.patch.object(server, "_get_available_speakers", return_value={"vivian"})
    mocker.patch.object(server, "_voice_created_at", return_value=0)
    mocker.patch.object(
        server,
        "_count_usage_text_tokens",
        side_effect=lambda text: len(text_tokenizer.encode(text, add_special_tokens=False)),
    )
    server._check_model = mocker.AsyncMock(return_value=None)
    params = SamplingParams(max_tokens=1024)
    engine = SimpleNamespace(
        num_stages=1,
        errored=False,
        model_config=SimpleNamespace(
            model="test-CustomVoice",
            async_chunk=True,
            max_model_len=2048,
            hf_config=SimpleNamespace(
                tts_model_type="custom_voice",
                talker_config=SimpleNamespace(codec_language_id={"english": 2050}, codec_eos_token_id=2150),
            ),
        ),
        default_sampling_params_list=[params],
        active={},
        streams={},
        submissions=[],
        appends=[],
        cancelled=[],
        submitting=threading.Event(),
        draining=threading.Event(),
        submission_gate=None,
        drain_gate=None,
        abort=mocker.AsyncMock(),
    )
    eof = object()

    def result(value):
        return SimpleNamespace(multimodal_output={"audio": value, "sr": 24000})

    async def generate(**kwargs):
        request_id = kwargs["request_id"]
        drained = False
        engine.submissions.append(kwargs)
        engine.submitting.set()
        try:
            if engine.submission_gate is not None:
                await engine.submission_gate.wait()
            stream = kwargs["prompt"]["additional_information"]["input_stream"]
            engine.streams[request_id] = Qwen3TTSTextInput(
                stream["token_ids"],
                min(
                    kwargs["sampling_params_list"][0].max_tokens - 1,
                    engine.model_config.max_model_len - len(kwargs["prompt"]["prompt_token_ids"]) - 1,
                ),
                len(text_tokenizer),
                final=stream["final"],
            )
            queue: asyncio.Queue[object] = asyncio.Queue()
            engine.active[request_id] = queue
            kwargs["submitted"].set_result(None)
            yield result(np.ones(8, dtype=np.float32) * 0.1)
            if not kwargs["prompt"]["additional_information"]["input_stream"]["final"]:
                while True:
                    item = await queue.get()
                    if item is eof:
                        break
                    yield result(item)
            if engine.drain_gate is not None:
                engine.draining.set()
                await engine.drain_gate.wait()
            drained = True
        finally:
            engine.active.pop(request_id, None)
            engine.streams.pop(request_id, None)
            if not drained:
                engine.cancelled.append(request_id)

    async def append(request_id, token_ids, *, final=False):
        engine.appends.append((request_id, list(token_ids), final))
        engine.streams[request_id].append(token_ids, final=final)
        queue = engine.active[request_id]
        if token_ids:
            queue.put_nowait(np.ones(8, dtype=np.float32) * 0.2)
        if final:
            queue.put_nowait(eof)

    engine.generate = generate
    engine.append_text = mocker.AsyncMock(side_effect=append)
    server.engine_client = engine
    server.model_config = engine.model_config
    server._adapter = Qwen3TTSAdapter(SpeechServingContext(server=server, engine_client=engine))
    mocker.patch.object(server._adapter, "_estimate_prompt_len_async", new=mocker.AsyncMock(return_value=10))
    handler = OmniStreamingSpeechHandler(server)
    app = FastAPI()

    @app.websocket("/v1/audio/speech/stream")
    async def endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return SimpleNamespace(app=app, server=server, engine=engine, handler=handler)


def _finish_utterance(ws):
    events = []
    audio = []
    while True:
        message = ws.receive()
        if "bytes" in message:
            audio.append(message["bytes"])
        else:
            event = json.loads(message["text"])
            events.append(event)
            if event["type"] == "session.done":
                return events, audio


def test_native_ws_streams_before_eof_and_reuses_connection(native_app, text_tokenizer):
    engine = native_app.engine
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        start = ws.receive_json()
        assert start["type"] == "audio.start"
        assert "text_complete" not in start and "sentence_text" not in start
        assert (start["sample_rate"], start["channels"]) == (24000, 1)
        assert ws.receive_bytes()
        ws.send_json(
            {"type": "session.config", **{key: value for key, value in _NATIVE_CONFIG.items() if key != "type"}}
        )
        assert "utterance" in ws.receive_json()["message"]
        ws.send_json({"type": "input.text", "text": "world "})
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert [event["type"] for event in events] == ["audio.done", "session.done"]
        assert not events[0]["error"]
        assert events[1] == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}
        assert len(engine.submissions) == 1
        submitted = engine.submissions[0]
        ids = submitted["prompt"]["additional_information"]["input_stream"]["token_ids"]
        ids = ids + [token for _, tokens, _ in engine.appends for token in tokens]
        assert ids == text_tokenizer.encode("Hello ", add_special_tokens=False) + text_tokenizer.encode(
            "world ", add_special_tokens=False
        )
        assert {request_id for request_id, _, _ in engine.appends} == {submitted["request_id"]}
        assert engine.appends[-1][2]
        assert submitted["sampling_params_list"][0].max_tokens == 1024
        assert QWEN3_TTS_AUTO_BUDGET_KEY not in submitted["prompt"]["additional_information"]

        ws.send_json({"type": "input.text", "text": "A second utterance!"})
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert events[-1]["utterance_index"] == 1
        assert sum(event["type"] == "audio.start" for event in events) == 1
        assert len(engine.submissions) == 2
    assert not engine.active and not engine.cancelled
    engine.abort.assert_not_awaited()


@pytest.mark.parametrize("text", ["", " \r\n\t  "])
def test_native_empty_utterance_does_not_submit(native_app, text):
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": text})
        ws.send_json({"type": "input.done"})
        assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}
    assert not native_app.engine.submissions


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("admitted", [False, True])
def test_native_disconnect_cancels_generation_before_or_after_admission(native_app, close, admitted):
    if not admitted:
        native_app.engine.submission_gate = asyncio.Event()
    with TestClient(native_app.app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json(_NATIVE_CONFIG)
            ws.send_json({"type": "input.text", "text": "Hello "})
            if admitted:
                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes()
            else:
                assert native_app.engine.submitting.wait(3)
            if close:
                ws.send_json({"type": "session.close"})
                assert ws.receive()["type"] == "websocket.close"
    assert len(native_app.engine.cancelled) == 1
    assert not native_app.engine.active
    native_app.engine.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_cleanup_survives_asgi_cancellation(native_app):
    import anyio

    entered = asyncio.Event()
    cancelled = asyncio.Event()
    cleaned = asyncio.Event()

    async def worker():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
            await asyncio.sleep(0)
            cleaned.set()

    task = asyncio.create_task(worker())
    await entered.wait()
    config = StreamingSpeechSessionConfig(**_NATIVE_CONFIG)
    state = stream_module._IncrementalUtterance(
        index=0,
        config=config,
        frontend=native_app.handler._create_commitment_frontend(config),
        worker=task,
    )
    state.inputs.put_nowait("discard")
    with anyio.CancelScope() as scope:
        scope.cancel()
        await native_app.handler._cleanup_incremental_utterance(state)
    assert cancelled.is_set() and cleaned.is_set()
    assert state.inputs.empty()
    assert task.cancelled()


def test_native_append_error_cancels_audio_and_waits_for_input_done(native_app):
    append = native_app.engine.append_text.side_effect
    native_app.engine.append_text.side_effect = ValueError("append refused")
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "world "})
        error = ws.receive_json()
        assert "append refused" in error["message"]
        assert error["partial_audio"] and error["action"] == "discard"
        assert "utterance_index" not in error
        assert ws.receive_json()["error"]
        ws.send_json({"type": "input.text", "text": "rejected"})
        assert "failed" in ws.receive_json()["message"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
        native_app.engine.append_text.side_effect = append
        ws.send_json({"type": "input.text", "text": "Another "})
        assert ws.receive_json()["utterance_index"] == 1
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"] and events[-1]["utterance_index"] == 1
    assert len(native_app.engine.cancelled) == 1
    assert len(native_app.engine.submissions) == 2
    assert not native_app.engine.active


@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_native_generation_ending_before_eof_reports_error_and_rejects_more_text(native_app, finish_reason):
    generate = native_app.engine.generate

    async def end_after_first_chunk(**kwargs):
        async with aclosing(generate(**kwargs)) as outputs:
            yield await anext(outputs)
            if finish_reason is not None:
                yield SimpleNamespace(metrics={"stage_metrics": {"0": {"finish_reason": finish_reason}}})

    native_app.engine.generate = end_after_first_chunk
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        error = ws.receive_json()
        if finish_reason == "length":
            assert "token budget" in error["message"]
        else:
            assert "ended before closing its text input" in error["message"]
            if finish_reason is not None:
                assert f"finish_reason={finish_reason}" in error["message"]
            else:
                assert "finish_reason=" not in error["message"]
        assert set(error) == {"type", "message", "partial_audio", "action"}
        assert error["partial_audio"] and error["action"] == "discard"
        done = ws.receive_json()
        assert done["type"] == "audio.done" and done["error"]
        ws.send_json({"type": "input.text", "text": "more text"})
        assert "failed" in ws.receive_json()["message"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
        native_app.engine.generate = generate
        ws.send_json({"type": "input.text", "text": "Another "})
        assert ws.receive_json()["utterance_index"] == 1
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert [event["type"] for event in events] == ["audio.done", "session.done"]
        assert not events[0]["error"] and events[-1]["utterance_index"] == 1
    assert len(native_app.engine.submissions) == 2
    assert not native_app.engine.active
    assert native_app.engine.appends == [(native_app.engine.submissions[1]["request_id"], [], True)]


def test_native_utterance_limit_is_enforced_before_submission(native_app, monkeypatch):
    monkeypatch.setattr(stream_module, "_MAX_COMMITMENT_UTTERANCE_CHARS", 16)
    native_app.engine.submission_gate = asyncio.Event()
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert native_app.engine.submitting.wait(3)
        ws.send_json({"type": "input.text", "text": "x" * 17})
        assert "utterance exceeds 16 characters" in ws.receive_json()["message"]
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_json()["error"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
    assert len(native_app.engine.cancelled) == 1
    assert not native_app.engine.active


def test_native_eof_before_add_is_ordered_after_submission(native_app, text_tokenizer):
    engine = native_app.engine
    engine.submission_gate = asyncio.Event()
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert engine.submitting.wait(3)
        ws.send_json({"type": "input.text", "text": "world "})
        ws.send_json({"type": "input.text", "text": "again."})
        ws.send_json({"type": "input.done"})
        ws.send_json(_NATIVE_CONFIG)
        assert "utterance" in ws.receive_json()["message"]
        assert not engine.appends
        client.portal.call(engine.submission_gate.set)
        events, audio = _finish_utterance(ws)
        assert audio and not any(event.get("error") for event in events)
    initial = engine.submissions[0]["prompt"]["additional_information"]["input_stream"]["token_ids"]
    assert initial + [token for _, tokens, _ in engine.appends for token in tokens] == (
        text_tokenizer.encode("Hello ", add_special_tokens=False)
        + text_tokenizer.encode("world again.", add_special_tokens=False)
    )
    assert len(engine.appends) == 1
    assert sum(final for _, _, final in engine.appends) == 1


def test_native_drain_allows_a_silent_client_until_session_done(native_app):
    engine = native_app.engine
    engine.drain_gate = asyncio.Event()
    native_app.handler._idle_timeout = 0.03
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        assert engine.draining.wait(3)
        client.portal.call(asyncio.sleep, 0.12)
        assert engine.active and not engine.cancelled
        client.portal.call(engine.drain_gate.set)
        events, _ = _finish_utterance(ws)
        assert not any(event["type"] == "error" or event.get("error") for event in events)
        assert len(engine.submissions) == 1


def test_native_completed_utterance_idle_uses_normal_session_timeout(native_app, mocker):
    engine = native_app.engine
    engine.drain_gate = asyncio.Event()
    native_app.handler._idle_timeout = 0.05
    failure = mocker.spy(native_app.handler, "_fail_incremental_input")
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        assert engine.draining.wait(3)
        client.portal.call(engine.drain_gate.set)
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
        assert ws.receive_json() == {"type": "error", "message": "Idle timeout: no message received"}
    failure.assert_not_awaited()
    assert len(engine.submissions) == 1
    assert not engine.active and not engine.cancelled


@pytest.mark.parametrize("input_closed", [False, True])
def test_native_total_deadline_cancels_the_request(native_app, monkeypatch, input_closed):
    monkeypatch.setattr(stream_module, "_MAX_INCREMENTAL_UTTERANCE_SECONDS", 0.1)
    if input_closed:
        native_app.engine.drain_gate = asyncio.Event()
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        ws.receive_bytes()
        if input_closed:
            ws.send_json({"type": "input.done"})
            assert native_app.engine.draining.wait(3)
        error = ws.receive_json()
        assert "exceeded 0.1 seconds" in error["message"]
        assert error["partial_audio"]
        assert ws.receive_json()["error"]
        if not input_closed:
            ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
    assert len(native_app.engine.cancelled) == 1
    assert not native_app.engine.active


def test_native_slow_audio_client_is_bounded(native_app, monkeypatch):
    async def blocked_send(self, data):
        await asyncio.Event().wait()

    monkeypatch.setattr(stream_module._SerializedWebSocket, "send_bytes", blocked_send)
    monkeypatch.setattr(stream_module, "_INCREMENTAL_SEND_TIMEOUT_SECONDS", 0.03)
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_json()["type"] == "error"
        assert ws.receive_json()["error"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
    assert len(native_app.engine.cancelled) == 1
    assert not native_app.engine.active


def test_native_input_token_limit_cancels_without_dropping_text(native_app):
    native_app.engine.default_sampling_params_list[0].max_tokens = 16
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "world. " * 30})
        assert "token capacity" in ws.receive_json()["message"]
        assert ws.receive_json()["error"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
    assert len(native_app.engine.cancelled) == 1
    assert len(native_app.engine.appends) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"text_input_mode": "raw"},
        {"stream_audio": False},
        {"response_format": "wav"},
        {"split_granularity": "sentence"},
        {"word_timestamps": True},
        {"speed": 1.5},
    ],
)
def test_native_rejects_unsupported_session_options(native_app, overrides):
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({**_NATIVE_CONFIG, **overrides})
        assert ws.receive_json()["type"] == "error"
    assert not native_app.engine.submissions


def test_native_preserves_normal_sampling_options(native_app):
    native_app.engine.default_sampling_params_list[0].stop_token_ids = [42]
    native_app.engine.default_sampling_params_list[0].ignore_eos = True
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not any(event["type"] == "error" for event in events)
    params = native_app.engine.submissions[0]["sampling_params_list"][0]
    assert params.stop_token_ids == [42] and params.ignore_eos


@pytest.mark.asyncio
async def test_incremental_reuses_ordinary_validation_before_build(native_app, mocker):
    model_input = SimpleNamespace(build=mocker.AsyncMock())
    request = OpenAICreateSpeechRequest(input="Hello ", voice="vivian", task_type="Base")
    with pytest.raises(ValueError, match="checkpoint does not support"):
        await native_app.server._prepare_speech_generation(request, incremental_input=model_input)
    model_input.build.assert_not_awaited()
    assert not native_app.engine.submissions


def test_native_ignores_unknown_fields_without_forwarding_independent_ids(native_app, text_tokenizer):
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({**_NATIVE_CONFIG, "extra_params": {"stop_token_ids": [42]}, "future_option": True})
        ws.send_json({"type": "input.text", "text": "Hello ", "token_ids": [42]})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
    submission = native_app.engine.submissions[0]
    assert submission["prompt"]["additional_information"]["input_stream"]["token_ids"] == text_tokenizer.encode(
        "Hello ", add_special_tokens=False
    )
    assert not submission["sampling_params_list"][0].stop_token_ids
    assert "future_option" not in submission["prompt"]["additional_information"]


def test_native_admission_capacity_error_does_not_leave_the_input_pump_waiting(native_app):
    native_app.engine.default_sampling_params_list[0].max_tokens = 4
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert "token capacity" in ws.receive_json()["message"]
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_json()["error"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
    assert not native_app.engine.active
    assert not native_app.engine.appends


def test_native_close_cancels_an_inflight_append(native_app):
    append_started = threading.Event()

    async def blocked_append(*_args, **_kwargs):
        append_started.set()
        await asyncio.Event().wait()

    native_app.engine.append_text.side_effect = blocked_append
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "world "})
        assert append_started.wait(3)
        ws.send_json({"type": "session.close"})
        assert ws.receive()["type"] == "websocket.close"
    assert len(native_app.engine.cancelled) == 1
    assert not native_app.engine.active


def test_native_empty_audio_is_not_a_success(native_app):
    async def empty_generate(**kwargs):
        kwargs["submitted"].set_result(None)
        yield SimpleNamespace(multimodal_output={"audio": np.empty(0, dtype=np.float32), "sr": 24000})

    native_app.engine.generate = empty_generate
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello"})
        ws.send_json({"type": "input.done"})
        assert "no audio" in ws.receive_json()["message"]
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_json()["error"]
        assert ws.receive_json()["type"] == "session.done"


def test_incremental_commitment_starts_before_a_sentence_boundary(native_app, text_tokenizer):
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "你"})
        assert native_app.engine.submitting.wait(3)
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
    params = native_app.engine.submissions[0]["prompt"]["additional_information"]
    assert params["input_stream"]["token_ids"] == text_tokenizer.encode("你", add_special_tokens=False)
    assert native_app.engine.appends[-1][1:] == ([], True)


@pytest.mark.parametrize("text_policy", [{}, {"text_input_mode": "buffered"}])
def test_incremental_requires_explicit_commitment_policy(native_app, text_policy):
    config = {key: value for key, value in _NATIVE_CONFIG.items() if key != "text_input_mode"}
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({**config, **text_policy})
        assert "requires text_input_mode='commitment'" in ws.receive_json()["message"]
    assert not native_app.engine.submissions


def test_incremental_commitment_appends_releases_without_closing_the_request(native_app, text_tokenizer):
    engine = native_app.engine
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        config = _NATIVE_CONFIG
        ws.send_json(config)
        ws.send_json({"type": "input.text", "text": "good"})
        ws.send_json(config)
        assert "utterance" in ws.receive_json()["message"]
        assert not engine.submissions
        ws.send_json({"type": "input.text", "text": "bye world "})
        assert engine.submitting.wait(3)
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "Next sentence. "})
        assert ws.receive_bytes()
        assert len(engine.submissions) == 1
        assert engine.appends and not any(final for _, _, final in engine.appends)
        ws.send_json({"type": "input.text", "text": "Third sentence. "})
        assert ws.receive_bytes()
        assert not any(final for _, _, final in engine.appends)
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
        assert events[-1]["total_sentences"] == 1
    params = engine.submissions[0]["prompt"]["additional_information"]
    assert params["input_stream"]["token_ids"] == text_tokenizer.encode("goodbye world ", add_special_tokens=False)
    assert [token for _, tokens, _ in engine.appends for token in tokens] == (
        text_tokenizer.encode("Next sentence. ", add_special_tokens=False)
        + text_tokenizer.encode("Third sentence. ", add_special_tokens=False)
    )
    assert sum(final for _, _, final in engine.appends) == 1


def test_incremental_commitment_withholds_special_text_and_flushes_pending_at_eof(native_app, text_tokenizer):
    engine = native_app.engine
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({**_NATIVE_CONFIG, "language": "Chinese"})
        ws.send_json({"type": "input.text", "text": "金额为2026"})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "元"})
        ws.send_json(_NATIVE_CONFIG)
        assert "utterance" in ws.receive_json()["message"]
        assert not engine.appends
        ws.send_json({"type": "input.text", "text": "，继续"})
        assert ws.receive_bytes()
        assert not any(final for _, _, final in engine.appends)
        ws.send_json({"type": "input.text", "text": "最后123"})
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
        assert events[-1]["total_sentences"] == 1
    assert len(engine.submissions) == 1
    initial = engine.submissions[0]["prompt"]["additional_information"]["input_stream"]
    assert initial["token_ids"] == text_tokenizer.encode("金额为", add_special_tokens=False)
    assert not initial["final"]
    assert [token for _, tokens, _ in engine.appends for token in tokens] == [
        token
        for release in ("2026元，继续", "最后", "123")
        for token in text_tokenizer.encode(release, add_special_tokens=False)
    ]
    assert engine.appends[-1][1:] == (text_tokenizer.encode("123", add_special_tokens=False), True)
    assert sum(final for _, _, final in engine.appends) == 1


def test_incremental_commitment_can_start_from_pending_text_at_eof(native_app, text_tokenizer):
    engine = native_app.engine
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "2026"})
        ws.send_json(_NATIVE_CONFIG)
        assert "utterance" in ws.receive_json()["message"]
        assert not engine.submissions
        ws.send_json({"type": "input.done"})
        events, audio = _finish_utterance(ws)
        assert audio
        assert "text_complete" not in events[0]
        assert not events[-2]["error"]
    assert len(engine.submissions) == 1
    stream = engine.submissions[0]["prompt"]["additional_information"]["input_stream"]
    assert stream["token_ids"] == text_tokenizer.encode("2026", add_special_tokens=False)
    assert stream["final"]
    assert not engine.appends


def test_incremental_commitment_pending_limit_cancels_the_active_request(native_app, monkeypatch):
    monkeypatch.setattr(stream_module, "_MAX_COMMITMENT_PENDING_CHARS", 3)
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "1234"})
        error = ws.receive_json()
        assert "max_pending_chars=3" in error["message"]
        assert error["partial_audio"] and error["action"] == "discard"
        assert ws.receive_json()["error"]
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "session.done"
    assert len(native_app.engine.cancelled) == 1
    assert not native_app.engine.active and not native_app.engine.appends


def test_incremental_handler_uses_a_non_qwen_adapter_contract(native_app, mocker):
    engine = native_app.engine
    received = []
    requests = []

    class OtherInput:
        async def build(self, text, *, final):
            received.append(text)
            return PreparedRequest(
                prompt={
                    "prompt_token_ids": [1],
                    "additional_information": {"input_stream": {"token_ids": [7], "final": final}},
                },
                model_type="other",
            )

        async def append(self, request_id, text, *, final):
            received.append(text)
            await engine.append_text(request_id, [8] if text else [], final=final)

    class OtherAdapter(ARTTSAdapter):
        name = "other"
        supports_incremental_input = True
        text_commitment = TextCommitmentCapabilities(
            profile="zh_en_special_v1",
            languages=frozenset({"English"}),
            independent_segments=False,
            preserves_cross_segment_context=True,
            streaming_audio=True,
            word_timestamps=False,
            retry_failed_segments=False,
        )

        async def build(self, request, sampling_params_list, has_inline_ref_audio):
            raise AssertionError("Complete-request builder must not handle incremental input")

        async def create_incremental_input(self, request):
            requests.append(request)
            return OtherInput()

    adapter = OtherAdapter(SpeechServingContext(server=native_app.server, engine_client=engine))
    mocker.patch.object(native_app.server, "_get_tts_adapter", return_value=adapter)
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({**_NATIVE_CONFIG, "task_type": "Base", "non_streaming_mode": True})
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": "world "})
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
    assert received == ["Hello ", "world ", ""]
    assert len(engine.submissions) == 1
    assert all(request.task_type == "Base" and request.non_streaming_mode is True for request in requests)


def test_incremental_input_is_opt_in_for_each_adapter(native_app, mocker):
    class CompleteOnlyAdapter(ARTTSAdapter):
        name = "complete_only"

        async def build(self, request, sampling_params_list, has_inline_ref_audio):
            raise AssertionError("Unsupported incremental input must not submit a request")

    adapter = CompleteOnlyAdapter(SpeechServingContext(server=native_app.server, engine_client=native_app.engine))
    mocker.patch.object(native_app.server, "_get_tts_adapter", return_value=adapter)
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        assert "not supported" in ws.receive_json()["message"]
    assert not native_app.engine.submissions


def test_incremental_session_config_does_not_create_model_input(native_app, mocker):
    factory = mocker.spy(native_app.server._adapter, "create_incremental_input")
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "session.close"})
        assert ws.receive()["type"] == "websocket.close"
    factory.assert_not_awaited()
    assert not native_app.engine.submissions


@pytest.mark.parametrize("overlap", [{"type": "input.text", "text": "Rejected "}, {"type": "input.done"}])
def test_incremental_rejects_overlap_until_session_done_without_advancing_the_index(native_app, overlap):
    engine = native_app.engine
    engine.drain_gate = asyncio.Event()
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["utterance_index"] == 0
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        assert engine.draining.wait(3)
        ws.send_json(overlap)
        error = ws.receive_json()
        assert "wait for session.done" in error["message"]
        assert set(error) == {"type", "message"}
        assert "partial_audio" not in error and "action" not in error
        assert len(engine.submissions) == 1 and not engine.cancelled
        client.portal.call(engine.drain_gate.set)
        events, _ = _finish_utterance(ws)
        assert events[-1]["utterance_index"] == 0
        assert not events[-2]["error"]
        ws.send_json({"type": "input.text", "text": "Second "})
        assert ws.receive_json()["utterance_index"] == 1
        assert ws.receive_bytes()
        second_id = engine.submissions[1]["request_id"]
        ws.send_json({"type": "input.text", "text": "continues "})
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
        assert events[-1]["utterance_index"] == 1
        assert engine.appends[-1] == (second_id, [], True)
        ws.send_json({"type": "input.done"})
        assert ws.receive_json() == {"type": "session.done", "utterance_index": 2, "total_sentences": 0}
    assert len(engine.submissions) == 2
    assert not engine.active and not engine.cancelled


@pytest.mark.parametrize("close", [False, True])
def test_incremental_disconnect_during_drain_cleans_the_current_utterance(native_app, mocker, close):
    engine = native_app.engine
    engine.drain_gate = asyncio.Event()
    cleanup = mocker.spy(native_app.handler, "_cleanup_incremental_utterance")
    with TestClient(native_app.app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json(_NATIVE_CONFIG)
            ws.send_json({"type": "input.text", "text": "Hello "})
            assert ws.receive_json()["type"] == "audio.start"
            assert ws.receive_bytes()
            ws.send_json({"type": "input.done"})
            assert engine.draining.wait(3)
            ws.send_json({"type": "input.text", "text": "Second "})
            assert "wait for session.done" in ws.receive_json()["message"]
            if close:
                ws.send_json({"type": "session.close"})
                assert ws.receive()["type"] == "websocket.close"
    states = [call.args[0] for call in cleanup.call_args_list if call.args[0] is not None]
    assert states and all(state.index == 0 and state.inputs.empty() and not state.text for state in states)
    assert all(state.worker.done() and state.closed.done() and state.failure.done() for state in states)
    assert len(engine.submissions) == 1 and len(engine.cancelled) == 1
    assert not engine.active


def test_incremental_pending_text_does_not_start_the_processing_deadline(native_app, monkeypatch):
    monkeypatch.setattr(stream_module, "_MAX_INCREMENTAL_UTTERANCE_SECONDS", 0.1)
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Good"})
        client.portal.call(asyncio.sleep, 0.15)
        ws.send_json(_NATIVE_CONFIG)
        assert "utterance" in ws.receive_json()["message"]
        assert not native_app.engine.submissions
        ws.send_json({"type": "input.text", "text": "bye "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not events[-2]["error"]
    assert len(native_app.engine.submissions) == 1
    assert not native_app.engine.active and not native_app.engine.cancelled


@pytest.mark.parametrize("text", ["Good", "Hello "])
def test_incremental_open_input_idle_uses_normal_session_timeout(native_app, text):
    native_app.handler._idle_timeout = 0.05
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": text})
        if text == "Hello ":
            assert ws.receive_json()["type"] == "audio.start"
            assert ws.receive_bytes()
        assert ws.receive_json() == {"type": "error", "message": "Idle timeout: no message received"}
    assert len(native_app.engine.submissions) == len(native_app.engine.cancelled) == int(text == "Hello ")
    assert not native_app.engine.active


@pytest.mark.parametrize("close", [False, True])
def test_incremental_disconnect_before_first_release_cleans_input_without_submitting(native_app, mocker, close):
    cleanup = mocker.spy(native_app.handler, "_cleanup_incremental_utterance")
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Good"})
        ws.send_json(_NATIVE_CONFIG)
        assert "utterance" in ws.receive_json()["message"]
        if close:
            ws.send_json({"type": "session.close"})
            assert ws.receive()["type"] == "websocket.close"
    states = [call.args[0] for call in cleanup.call_args_list if call.args[0] is not None]
    assert states and all(state.worker is None and state.inputs.empty() and not state.text for state in states)
    assert all(state.closed.cancelled() and state.failure.cancelled() for state in states)
    assert not native_app.engine.submissions


def test_incremental_large_ingress_reuses_the_utterance_total_limit(native_app, text_tokenizer):
    engine = native_app.engine
    text = "ordinary text " * 1500
    text_tokenizer.encode.side_effect = lambda text, **_kwargs: [31] * len(text.split())
    engine.default_sampling_params_list[0].max_tokens = 4096
    engine.model_config.max_model_len = 4096
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        assert ws.receive_json()["type"] == "audio.start"
        assert ws.receive_bytes()
        ws.send_json({"type": "input.text", "text": text})
        assert ws.receive_bytes()
        ws.send_json({"type": "input.done"})
        events, _ = _finish_utterance(ws)
        assert not any(event.get("error") for event in events)
    assert len(engine.submissions) == 1
    assert engine.appends[0][1] == text_tokenizer.encode(text, add_special_tokens=False)
    assert not engine.active and not engine.cancelled


def test_incremental_configuration_can_change_after_session_done(native_app):
    with TestClient(native_app.app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(_NATIVE_CONFIG)
        ws.send_json({"type": "input.text", "text": "Hello "})
        ws.send_json({"type": "input.done"})
        first_events, _ = _finish_utterance(ws)
        assert first_events[-1]["utterance_index"] == 0
        ws.send_json({**_NATIVE_CONFIG, "language": "Chinese"})
        ws.send_json({"type": "input.text", "text": "你好"})
        ws.send_json({"type": "input.done"})
        second_events, _ = _finish_utterance(ws)
        assert not any(event["type"] == "error" or event.get("error") for event in second_events)
        assert second_events[-1]["utterance_index"] == 1
    params = native_app.engine.submissions[1]["prompt"]["additional_information"]
    assert params["language"] == ["Chinese"]


def test_incremental_state_and_input_indices_are_connection_local(native_app):
    with TestClient(native_app.app) as client:
        with (
            client.websocket_connect("/v1/audio/speech/stream") as first,
            client.websocket_connect("/v1/audio/speech/stream") as second,
        ):
            for websocket in (first, second):
                websocket.send_json(_NATIVE_CONFIG)
                websocket.send_json({"type": "input.text", "text": "Hello "})
                assert websocket.receive_json()["utterance_index"] == 0
                assert websocket.receive_bytes()
            assert len(native_app.engine.active) == 2
            for websocket in (first, second):
                websocket.send_json({"type": "input.done"})
                events, _ = _finish_utterance(websocket)
                assert events[-1]["utterance_index"] == 0
                assert not events[-2]["error"]
    assert len(native_app.engine.submissions) == 2
    assert not native_app.engine.active and not native_app.engine.cancelled
