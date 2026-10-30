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
