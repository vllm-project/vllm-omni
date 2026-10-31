# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native extensions inherit vLLM authentication before body validation."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

from vllm_omni.entrypoints.openai.drvexo_routes import router, stream_payload
from vllm_omni.entrypoints.openai.stream_lifecycle import owned_chat_stream

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PATHS = ["/v1/drvexo/decision", "/v1/drvexo/speech", "/v1/drvexo/speech/stream", "/v1/drvexo/chat/stream"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic test-key"])
def test_extensions_reject_unauthenticated_calls(path, authorization):
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    headers = {"Authorization": authorization} if authorization else {}
    with TestClient(app) as client:
        response = client.post(path, json={}, headers=headers)
    assert response.status_code == 401


@pytest.mark.parametrize("path", PATHS)
def test_extensions_validate_authenticated_calls(path):
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    with TestClient(app) as client:
        response = client.post(path, json={}, headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 422


def test_authenticated_disabled_decision_bypasses_model():
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    with TestClient(app) as client:
        response = client.post(
            "/v1/drvexo/decision",
            headers={"Authorization": "Bearer test-key"},
            json={
                "general_decision_head_enabled": False,
                "task": "choice",
                "instruction": "Choose a color.",
                "state": {},
                "candidates": [{"id": "a", "text": "blue"}, {"id": "b", "text": "red"}],
            },
        )
        assert client.post("/drvexo/decision", json={}).status_code == 404
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "probabilities": None, "executable": False}


def test_stream_payload_preserves_committed_history():
    messages = [{"role": "system", "content": "Product facts"}, {"role": "user", "content": "Hello"}]
    request = {"messages": messages, "stream": False, "modalities": ["audio"], "audio": {"format": "wav"}}
    payload = stream_payload(request)
    assert payload["messages"] is messages
    assert request["stream"] is False
    assert request["audio"] == {"format": "wav"}
    assert payload["stream"] is True
    assert "audio" not in payload


@pytest.mark.asyncio
async def test_early_stream_close_aborts_only_owned_request():
    response_closed, result_closed = [], []

    async def response():
        try:
            yield "frame"
            yield "later"
        finally:
            response_closed.append(True)

    async def results():
        try:
            yield "result"
        finally:
            result_closed.append(True)

    result_stream = results()
    await anext(result_stream)
    engine = AsyncMock()
    stream = owned_chat_stream(response(), result_stream, engine, "owned")
    assert await anext(stream) == "frame"
    await stream.aclose()
    assert response_closed and result_closed
    engine.abort.assert_awaited_once_with("owned")


@pytest.mark.asyncio
async def test_completed_stream_does_not_abort():
    async def response():
        yield "done"

    async def results():
        yield "result"

    engine = AsyncMock()
    assert [chunk async for chunk in owned_chat_stream(response(), results(), engine, "owned")] == ["done"]
    engine.abort.assert_not_awaited()


@pytest.mark.parametrize("n", [None, 1])
def test_stream_payload_accepts_optional_single_choice(n):
    assert stream_payload({"messages": [{"role": "user", "content": "Hi"}], "n": n})["n"] in (None, 1)


@pytest.mark.parametrize("path", ["/v1/drvexo/speech", "/v1/drvexo/speech/stream"])
def test_whitespace_speech_rejected_before_model_access(path):
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post(path, json={"text": "   "}).status_code == 400


@pytest.mark.parametrize(
    "server,scheme,expected",
    [
        (("127.0.0.1", 8000), "http", "http://127.0.0.1:8000"),
        (("10.1.2.3", 8000), "http", "http://10.1.2.3:8000"),
        (("0.0.0.0", 8000), "http", "http://127.0.0.1:8000"),
        (("::", 8000), "http", "http://[::1]:8000"),
        (("2001:db8::1", 8000), "http", "http://[2001:db8::1]:8000"),
    ],
)
def test_backend_uses_actual_socket(server, scheme, expected):
    from vllm_omni.entrypoints.openai.drvexo_routes import local_backend

    assert local_backend({"server": server, "scheme": scheme}) == expected


@pytest.mark.parametrize("server,scheme", [(None, "http"), (("127.0.0.1", 8000), "https")])
def test_chat_stream_unsupported_transport_is_server_error(server, scheme):
    from fastapi import HTTPException

    from vllm_omni.entrypoints.openai.drvexo_routes import local_backend

    with pytest.raises(HTTPException) as error:
        local_backend({"server": server, "scheme": scheme})
    assert error.value.status_code == 503


def test_backend_hop_preserves_auth_and_trace_but_rebuilds_transport_headers():
    from starlette.requests import Request

    from vllm_omni.entrypoints.openai.drvexo_routes import backend_headers

    headers = {
        "host": "untrusted.example",
        "content-length": "7",
        "connection": "x-hop",
        "x-hop": "drop",
        "x-forwarded-host": "external.example",
        "authorization": "Bearer key",
        "x-custom-auth": "middleware-value",
        "x-request-id": "trace-1",
    }
    request = Request({"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})
    assert backend_headers(request) == {
        "authorization": "Bearer key",
        "x-custom-auth": "middleware-value",
        "x-request-id": "trace-1",
    }


@pytest.mark.asyncio
async def test_sentence_stream_disconnect_closes_owned_engine_request(mocker):
    from unittest.mock import Mock

    import anyio
    from starlette.requests import Request
    from vllm import SamplingParams

    from vllm_omni.entrypoints.openai.drvexo_routes import SpeechText, sentence_speech_stream

    started, closed = anyio.Event(), []
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        started.set()
        try:
            await anyio.sleep_forever()
            yield None
        finally:
            closed.append(True)

    engine = Mock(
        num_stages=3, default_sampling_params_list=[SamplingParams(max_tokens=1)], generate=generate, abort=AsyncMock()
    )
    app = FastAPI()
    app.state.openai_serving_chat = Mock(engine_client=engine)
    app.state.drvexo_speech_tokenizer = Mock(
        all_special_ids=[],
        encode=lambda text, **kw: list(text.encode()),
        decode=lambda ids, **kw: bytes(ids).decode(),
    )
    raw = Request({"type": "http", "app": app, "headers": []})
    response = await sentence_speech_stream(SpeechText(text="Speak this sentence."), raw)

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, AsyncMock())
    assert closed == [True]
    engine.abort.assert_awaited_once_with(calls[0]["request_id"])


@pytest.mark.parametrize("path", ["/v1/drvexo/speech", "/v1/drvexo/speech/stream"])
@pytest.mark.parametrize("invalid", ["Say <|im_start|> aloud.", "unroundtrippable"])
def test_invalid_speech_is_rejected_before_engine_and_valid_request_recovers(mocker, path, invalid):
    from unittest.mock import Mock

    import pybase64

    from vllm_omni.entrypoints.openai import drvexo_routes as routes

    tokenizer = Mock(all_special_ids=[1])

    def encode(text, **kwargs):
        return [1] if "<|im_start|>" in text else [ord(c) + 100 for c in text]

    def decode(ids, **kwargs):
        text = "".join(chr(i - 100) for i in ids)
        return "changed" if text == "unroundtrippable" else text

    tokenizer.encode.side_effect = encode
    tokenizer.decode.side_effect = decode
    app = FastAPI()
    app.state.drvexo_speech_tokenizer = tokenizer
    wav = b"RIFF" + bytes(4) + b"WAVE" + bytes(32)
    audio = Mock(data=pybase64.b64encode(wav).decode())
    serving = Mock()
    serving.models.model_name.return_value = "test"
    serving._create_audio_choice.return_value = [Mock(message=Mock(audio=audio))]
    app.state.openai_serving_chat = serving
    collect = mocker.patch.object(routes, "collect_sentence", new_callable=AsyncMock)
    calls = []

    async def stream(*args):
        calls.append(args)
        yield {"wav": wav, "chunk_index": 0, "samples": 1, "sample_rate": 24000}

    mocker.patch.object(routes, "stream_sentence", side_effect=stream)
    app.include_router(router)
    with TestClient(app) as client:
        bad = client.post(path, json={"text": invalid})
        assert bad.status_code == 400
        collect.assert_not_called()
        assert calls == []
        valid = client.post(path, json={"text": "A normal sentence."})
        assert valid.status_code == 200
    assert collect.call_count + len(calls) == 1


@pytest.mark.parametrize("path", ["/v1/drvexo/speech", "/v1/drvexo/speech/stream"])
def test_speech_without_talker_tokenizer_returns_unavailable(path):
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post(path, json={"text": "hello"}).status_code == 503


def test_admission_uses_real_local_talker_tokenizer(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from vllm_omni.entrypoints.openai.drvexo_speech import load_speech_tokenizer
    from vllm_omni.model_executor.models.drvexo_speech_8b.lexical import fixed_text_ids

    native = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "<|im_start|>": 3}, unk_token="[UNK]"))
    native.pre_tokenizer = Whitespace()
    target = PreTrainedTokenizerFast(
        tokenizer_object=native, unk_token="[UNK]", additional_special_tokens=["<|im_start|>"]
    )
    target.save_pretrained(tmp_path / "talker")
    loaded = load_speech_tokenizer(tmp_path)
    assert loaded.get_vocab() == target.get_vocab()
    assert loaded.padding_side == "left"
    assert fixed_text_ids("hello world", loaded) == [1, 2]
    with pytest.raises(ValueError, match="control tokens"):
        fixed_text_ids("hello <|im_start|>", loaded)
    with pytest.raises(ValueError, match="changed"):
        fixed_text_ids("hello   world", loaded)
