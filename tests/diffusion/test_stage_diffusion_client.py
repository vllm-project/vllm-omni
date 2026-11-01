# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading
from types import SimpleNamespace

import pytest
import zmq
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _client_without_subprocess() -> StageDiffusionClient:
    # A PUSH socket with no peer blocks on send, which is what the client is
    # left with after its subprocess died.
    client = object.__new__(StageDiffusionClient)
    client._zmq_ctx = zmq.Context()
    client._request_socket = client._zmq_ctx.socket(zmq.PUSH)
    client.request_address = get_open_zmq_ipc_path()
    client._request_socket.bind(client.request_address)
    client._response_socket = client._zmq_ctx.socket(zmq.PULL)
    client._response_socket.bind(get_open_zmq_ipc_path())
    client._encoder = OmniMsgpackEncoder()
    client._proc_manager = None
    client._engine_dead = False
    client.stage_id, client.replica_id = 0, 0
    client._pending_rpcs = set()
    return client


def _with_proc(client: StageDiffusionClient, alive: bool) -> StageDiffusionClient:
    client._proc_manager = SimpleNamespace(proc=SimpleNamespace(is_alive=lambda: alive, exitcode=None if alive else -9))
    return client


def _run_in_thread(coro) -> tuple[threading.Thread, dict]:
    outcome = {}

    def run():
        try:
            outcome["result"] = asyncio.run(coro)
        except Exception as e:
            outcome["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _close(client: StageDiffusionClient, thread: threading.Thread) -> None:
    # Left to garbage collection, the context can be terminated at exit before its sockets are
    # closed and hang. A thread still stuck in a send owns the socket, so leave it alone then.
    if thread.is_alive():
        return
    client._request_socket.close(linger=0)
    client._response_socket.close(linger=0)
    client._zmq_ctx.term()


def test_shutdown_returns_when_subprocess_is_gone():
    client = _client_without_subprocess()

    shutdown = threading.Thread(target=client.shutdown, daemon=True)
    shutdown.start()
    shutdown.join(timeout=5)

    assert not shutdown.is_alive()
    # Terminating the context is the last step, so shutdown() ran to the end.
    assert client._zmq_ctx.closed


def test_abort_returns_when_engine_is_dead():
    client = _client_without_subprocess()
    client._engine_dead = True

    thread, outcome = _run_in_thread(client.abort_requests_async(["req-0"]))
    thread.join(timeout=5)
    try:
        assert not thread.is_alive()
        assert "error" not in outcome
    finally:
        _close(client, thread)


def test_abort_returns_when_subprocess_is_gone_before_engine_dead_is_set():
    client = _client_without_subprocess()

    thread, outcome = _run_in_thread(client.abort_requests_async(["req-0"]))
    thread.join(timeout=5)
    try:
        assert not thread.is_alive()
        assert "error" not in outcome
    finally:
        _close(client, thread)


@pytest.mark.parametrize("call", ["add_request", "collective_rpc"])
def test_send_raises_when_subprocess_died_before_engine_dead_is_set(call):
    client = _with_proc(_client_without_subprocess(), alive=False)
    if call == "add_request":
        coro = client.add_request_async("req-0", "a cat", OmniDiffusionSamplingParams())
    else:
        coro = client.collective_rpc_async("sleep")

    thread, outcome = _run_in_thread(coro)
    thread.join(timeout=5)
    try:
        assert not thread.is_alive()
        assert isinstance(outcome.get("error"), EngineDeadError)
        assert client._engine_dead
        assert not client._pending_rpcs
    finally:
        _close(client, thread)


def test_add_request_waits_for_a_live_subprocess_to_connect():
    client = _with_proc(_client_without_subprocess(), alive=True)
    thread, outcome = _run_in_thread(client.add_request_async("req-0", "a cat", OmniDiffusionSamplingParams()))
    thread.join(timeout=0.5)
    assert thread.is_alive()

    peer = client._zmq_ctx.socket(zmq.PULL)
    peer.connect(client.request_address)
    try:
        assert peer.poll(5000)
        message = OmniMsgpackDecoder().decode(peer.recv())
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert "error" not in outcome
        assert message["type"] == "add_request"
        assert message["request_id"] == "req-0"
    finally:
        peer.close(linger=0)
        _close(client, thread)
