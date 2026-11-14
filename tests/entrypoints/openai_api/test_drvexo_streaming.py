# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Backpressure and backend failure must not leak stream owners."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from vllm_omni.entrypoints.openai import drvexo_streaming as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_disconnect_during_backpressured_error_terminates_owner(mocker):
    blocked = asyncio.Event()
    queues = []

    class ObservedQueue(asyncio.Queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queues.append(self)

        async def put(self, value):
            if isinstance(value, dict) and value.get("type") == "error" and self.full():
                blocked.set()
            await super().put(value)

    mocker.patch.object(module.asyncio, "Queue", ObservedQueue)
    closed, client_closed = [], []

    class Response:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for _ in range(33):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]})
            raise RuntimeError("backend failed with a full event queue")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await asyncio.sleep(0)
            client_closed.append(True)

        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            try:
                yield Response()
            finally:
                closed.append(True)

    from fastapi import FastAPI
    from starlette.requests import Request

    from vllm_omni.entrypoints.openai import drvexo_routes as routes

    mocker.patch.object(routes.httpx, "AsyncClient", return_value=Client())
    request = Request({"type": "http", "app": FastAPI(), "headers": [], "server": ("127.0.0.1", 8000)})
    response = await routes.chat_stream(
        routes.StreamChat(request={"messages": [{"role": "user", "content": "hi"}]}, audio_output=False), request
    )
    source = response.body_iterator
    assert json.loads(await anext(source))["type"] == "text"
    await asyncio.wait_for(blocked.wait(), 2)
    close = asyncio.create_task(source.aclose())
    done, _ = await asyncio.wait({close}, timeout=0.25)
    terminated = close in done
    if not terminated:
        # Release the old implementation's stuck terminal put so fail-before
        # leaves no background task behind.
        queues[0].get_nowait()
    await asyncio.wait_for(close, 2)
    assert closed == [True]
    assert client_closed == [True]
    assert terminated, "stream owner deadlocked while cancelling a full error queue"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_terminal_event_is_delivered_while_consumer_drains(failure):
    class Response:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for _ in range(40):
                yield 'data: {"choices": [{"delta": {"content": "x"}}]}'
            if failure:
                raise RuntimeError("failed backend")
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'
            yield "data: [DONE]"

    class Client:
        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            yield Response()

    source = module.stream_reply(Client(), "http://local", {}, False)

    async def consume():
        return [json.loads(line) async for line in source]

    events = await asyncio.wait_for(consume(), 2)
    assert [event["type"] for event in events].count("text") == 40
    assert events[-1]["type"] == ("error" if failure else "done")
