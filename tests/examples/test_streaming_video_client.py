# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import importlib
import json
from pathlib import Path

import pytest
from websockets.exceptions import ConnectionClosed

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("closed", [False, True])
def test_heartbeat_uses_session_protocol_and_stops_on_disconnect(monkeypatch, mocker, closed):
    directory = Path(__file__).resolve().parents[2] / "examples/online_serving/streaming_video_generation"
    monkeypatch.syspath_prepend(str(directory))
    client = importlib.import_module("streaming_video_client")
    delays = []

    async def sleep(delay):
        delays.append(delay)
        if len(delays) > 1:
            raise asyncio.CancelledError

    websocket = mocker.Mock(send=mocker.AsyncMock(side_effect=ConnectionClosed(None, None) if closed else None))
    monkeypatch.setattr(client.asyncio, "sleep", sleep)

    async def run():
        if closed:
            await client._send_heartbeats(websocket, asyncio.Lock())
        else:
            with pytest.raises(asyncio.CancelledError):
                await client._send_heartbeats(websocket, asyncio.Lock())

    asyncio.run(run())
    websocket.send.assert_awaited_once()
    assert json.loads(websocket.send.call_args.args[0]) == {"type": "session.ping"}
    assert delays == ([10] if closed else [10, 10])
