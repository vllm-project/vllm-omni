# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transport deadline checks for the duplex client's HTTP upgrade."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from vllm_omni.clients.duplex import DuplexClient, DuplexConnectionError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("timeout_s", [30.0, 300.0])
def test_websocket_upgrade_uses_configured_handshake_timeout(monkeypatch, timeout_s):
    socket = object()
    calls = []

    async def connect(url, **kwargs):
        calls.append((url, kwargs))
        return socket

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    client = DuplexClient("ws://test-host:8099", model="test-model", handshake_timeout_s=timeout_s)
    assert asyncio.run(client._default_connect("ws://test-host:8099/v1/realtime")) is socket
    assert calls[0][1]["open_timeout"] == timeout_s


def test_websocket_upgrade_timeout_keeps_connection_error(monkeypatch):
    async def connect(url, **kwargs):
        raise TimeoutError("timed out during opening handshake")

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    client = DuplexClient("ws://test-host:8099", model="test-model", handshake_timeout_s=0.01)
    with pytest.raises(DuplexConnectionError, match="timed out during opening handshake") as caught:
        asyncio.run(client._default_connect("ws://test-host:8099/v1/realtime"))
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_websocket_upgrade_cancellation_propagates(monkeypatch):
    async def connect(url, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    client = DuplexClient("ws://test-host:8099", model="test-model", handshake_timeout_s=300.0)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(client._default_connect("ws://test-host:8099/v1/realtime"))


@pytest.mark.parametrize("timeout_s,succeeds", [(0.5, True), (0.01, False)])
def test_delayed_http_upgrade_obeys_configured_timeout(timeout_s, succeeds):
    from websockets.asyncio.server import serve

    async def scenario():
        async def delay_upgrade(connection, request):
            await asyncio.sleep(0.05)

        async def handler(connection):
            await connection.wait_closed()

        async with serve(handler, "127.0.0.1", 0, process_request=delay_upgrade) as server:
            port = server.sockets[0].getsockname()[1]
            url = f"ws://127.0.0.1:{port}/v1/realtime"
            client = DuplexClient(url, model="test-model", handshake_timeout_s=timeout_s)
            if succeeds:
                socket = await client._default_connect(url)
                await socket.close()
            else:
                with pytest.raises(DuplexConnectionError, match="timed out during opening handshake"):
                    await client._default_connect(url)

    asyncio.run(scenario())
