# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session cleanup may temporarily retain capacity after session.closed."""

import asyncio
import json

import pytest
import websockets

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.clients.duplex import DuplexProtocolError

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _session(url):
    return runner._open_duplex_session(
        url=url,
        model="test-model",
        additional_headers={"Authorization": "Bearer test-token"},
        ref_audio=None,
        instructions="test instructions",
        extra_body={"return_stage_metrics": True},
        turn_detection="none",
    )


def _url(server):
    return f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"


async def _reject(socket, *, code="resource_exhausted", retryable=True):
    await socket.send(json.dumps({"type": "error", "error": {"code": code, "retryable": retryable}}))
    await socket.close()


async def test_admission_reconnects_before_sending_any_sample_input():
    sockets = []
    configurations = []

    async def handler(socket):
        sockets.append(socket)
        assert socket.request.headers["Authorization"] == "Bearer test-token"
        config = json.loads(await socket.recv())
        assert config["type"] == "session.update"
        configurations.append(config)
        if len(sockets) <= 2:
            await _reject(socket)
            return
        await socket.send(json.dumps({"type": "session.created"}))
        # No sample input is sent by admission, including on refused attempts.
        await socket.wait_closed()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        async with _session(_url(server)) as (client, rejections, elapsed_s):
            assert client.events.count("session.created") == 1
            assert client.events.errors() == []
            assert rejections == 2
            assert elapsed_s >= 0.75
            assert len(sockets) == 3
            assert all(socket.close_code is not None for socket in sockets[:2])
            assert configurations[0] == configurations[1] == configurations[2]
    assert all(socket.close_code is not None for socket in sockets)


@pytest.mark.parametrize(
    ("code", "retryable"),
    [("resource_exhausted", False), ("resource_exhausted", None), ("unsupported_turn_detection", True)],
)
async def test_admission_does_not_retry_other_rejections(code, retryable):
    attempts = 0

    async def handler(socket):
        nonlocal attempts
        attempts += 1
        await socket.recv()
        await _reject(socket, code=code, retryable=retryable)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        with pytest.raises(DuplexProtocolError) as caught:
            async with _session(_url(server)):
                pytest.fail("rejected session was admitted")
    assert caught.value.code == code
    assert attempts == 1


@pytest.mark.parametrize("reject", [True, False])
async def test_admission_wait_has_a_total_deadline(monkeypatch, reject):
    monkeypatch.setattr(runner, "_SESSION_ADMISSION_TIMEOUT_S", 0.1)
    sockets = []

    async def handler(socket):
        sockets.append(socket)
        await socket.recv()
        if reject:
            await _reject(socket)
        else:
            await socket.wait_closed()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:

        async def open_session():
            with pytest.raises(TimeoutError):
                async with _session(_url(server)):
                    pytest.fail("permanently full server was admitted")

        await asyncio.wait_for(open_session(), timeout=2)
    assert len(sockets) == 1
    assert sockets[0].close_code is not None


async def test_cancelling_admission_closes_the_pending_transport():
    configuring = asyncio.Event()
    closed = asyncio.Event()

    async def handler(socket):
        await socket.recv()
        configuring.set()
        await socket.wait_closed()
        closed.set()

    async def open_session(url):
        async with _session(url):
            pytest.fail("server never admitted the session")

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        task = asyncio.create_task(open_session(_url(server)))
        await asyncio.wait_for(configuring.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), timeout=2)


@pytest.mark.parametrize("phase", ["configure", "sample"])
async def test_admission_never_retries_an_already_started_session(phase):
    attempts = 0

    async def handler(socket):
        nonlocal attempts
        attempts += 1
        await socket.recv()
        await socket.send(json.dumps({"type": "session.created"}))
        if phase == "configure":
            await _reject(socket)
        await socket.wait_closed()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        with pytest.raises(DuplexProtocolError):
            async with _session(_url(server)):
                assert phase == "sample", "configure error was lost"
                raise DuplexProtocolError(
                    "failure after admission",
                    code="resource_exhausted",
                    raw={"error": {"retryable": True}},
                )
    assert attempts == 1
