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
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProcManager
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


@pytest.mark.asyncio
async def test_async_shutdown_keeps_socket_operations_on_owner_loop(mocker, monkeypatch):
    client = _client_without_subprocess()
    manager = mocker.Mock(spec=StageDiffusionProcManager)
    manager.proc = mocker.Mock()
    manager.proc.is_alive.return_value = True
    manager.distributed_executor_backend = "mp"
    client._proc_manager = manager
    owner_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    socket_threads = []
    request_shutdown = client._request_shutdown
    close_sockets = client._close_sockets

    def request():
        socket_threads.append(threading.get_ident())
        return request_shutdown()

    def close():
        socket_threads.append(threading.get_ident())
        close_sockets()

    def join(*, timeout):
        assert threading.get_ident() != owner_thread
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(client, "_request_shutdown", request)
    monkeypatch.setattr(client, "_close_sockets", close)
    manager.shutdown.side_effect = join
    task = asyncio.create_task(client.shutdown_async())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not task.done()
    finally:
        release.set()
        await task
    assert socket_threads == [owner_thread, owner_thread]
    assert client._zmq_ctx.closed
    manager.shutdown.assert_called_once_with(timeout=10.0)


@pytest.mark.parametrize("backend", ["mp", "ray"])
def test_shutdown_waits_for_normal_cleanup_before_signaling(mocker, backend):
    client = _client_without_subprocess()
    manager = mocker.Mock(spec=StageDiffusionProcManager)
    manager.proc = mocker.Mock()
    manager.proc.is_alive.return_value = True
    manager.distributed_executor_backend = backend
    manager.wait_for_shutdown.return_value = True
    client._proc_manager = manager
    try:
        client._shutdown_process(shutdown_requested=True)
        manager.wait_for_shutdown.assert_called_once_with(timeout=20.0)
        manager.shutdown.assert_not_called()
    finally:
        client._close_sockets()


@pytest.mark.parametrize("request_sent", [True, False])
def test_shutdown_signals_when_normal_cleanup_cannot_finish(mocker, request_sent):
    client = _client_without_subprocess()
    manager = mocker.Mock(spec=StageDiffusionProcManager)
    manager.proc = mocker.Mock()
    manager.proc.is_alive.return_value = True
    manager.distributed_executor_backend = "mp"
    manager.wait_for_shutdown.return_value = False
    client._proc_manager = manager
    try:
        client._shutdown_process(shutdown_requested=request_sent)
        if request_sent:
            assert manager.method_calls == [
                mocker.call.proc.is_alive(),
                mocker.call.wait_for_shutdown(timeout=20.0),
                mocker.call.shutdown(timeout=10.0),
            ]
        else:
            manager.wait_for_shutdown.assert_not_called()
            manager.shutdown.assert_called_once_with(timeout=10.0)
    finally:
        client._close_sockets()


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


@pytest.mark.asyncio
async def test_opt_in_send_remains_cancellable_without_a_peer():
    client = _client_without_subprocess()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client._send_request_async(b"control"), 0.05)
    finally:
        client._request_socket.close(linger=0)
        client._response_socket.close(linger=0)
        client._zmq_ctx.term()
