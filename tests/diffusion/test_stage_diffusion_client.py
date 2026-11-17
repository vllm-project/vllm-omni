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


@pytest.mark.parametrize("npu,exits", [(True, True), (True, False), (False, False)])
def test_mp_shutdown_preserves_npu_worker_grace(monkeypatch, npu, exits):
    from unittest.mock import Mock

    from vllm_omni.diffusion import stage_diffusion_client as module

    client = object.__new__(StageDiffusionClient)
    client._request_socket = Mock()
    client._response_socket = Mock()
    client._encoder = OmniMsgpackEncoder()
    client._zmq_ctx = Mock()
    manager = Mock(distributed_executor_backend="mp", uses_multiproc=True)
    manager.proc.is_alive.return_value = True
    manager.wait_for_shutdown.return_value = exits
    client._proc_manager = manager
    monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(is_npu=lambda: npu))
    client.shutdown()
    if npu:
        manager.wait_for_shutdown.assert_called_once_with(timeout=80.0)
    else:
        manager.wait_for_shutdown.assert_not_called()
    if npu and exits:
        manager.shutdown.assert_not_called()
    else:
        manager.shutdown.assert_called_once_with(timeout=10.0)
    client._zmq_ctx.term.assert_called_once()


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize(
    "backend,num_gpus,uses_multiproc",
    [
        (None, 1, False),
        (None, 4, True),
        ("uni", 1, False),
        ("mp", 1, True),
        ("vllm_omni.diffusion.executor.multiproc_executor.MultiprocDiffusionExecutor", 1, True),
    ],
)
def test_shutdown_grace_uses_resolved_executor(mocker, backend, num_gpus, uses_multiproc, headless):
    """Only an executor with child workers needs the extended NPU grace."""
    from vllm_omni.diffusion import stage_diffusion_client as client_module
    from vllm_omni.diffusion import stage_diffusion_proc as proc_module
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    process = mocker.Mock()
    process.is_alive.return_value = True
    mocker.patch.object(proc_module, "get_mp_context").return_value.Process.return_value = process
    mocker.patch.object(proc_module.StageDiffusionProcManager, "_wait_until_started")
    config = OmniDiffusionConfig(num_gpus=num_gpus, distributed_executor_backend=backend)
    kwargs = dict(
        model="test-model",
        od_config=config,
        handshake_address="inproc://shutdown-test",
        addresses=proc_module.EngineZmqAddresses(inputs=["input"], outputs=["output"]),
    )
    if headless:
        manager = proc_module.StageDiffusionProcManager.launch_headless(
            **kwargs, omni_coordinator_address=None, omni_stage_id=0, omni_replica_id=0
        )
    else:
        manager = proc_module.StageDiffusionProcManager(**kwargs, stage_init_timeout=1)
    mocker.patch.object(manager, "wait_for_shutdown", return_value=False)
    mocker.patch.object(manager, "shutdown")
    mocker.patch.object(client_module.current_omni_platform, "is_npu", return_value=True)
    client = object.__new__(StageDiffusionClient)
    client._request_socket = mocker.Mock()
    client._response_socket = mocker.Mock()
    client._encoder = OmniMsgpackEncoder()
    client._zmq_ctx = mocker.Mock()
    client._proc_manager = manager
    client.shutdown()
    if uses_multiproc:
        manager.wait_for_shutdown.assert_called_once_with(timeout=80.0)
    else:
        manager.wait_for_shutdown.assert_not_called()
    manager.shutdown.assert_called_once_with(timeout=10.0)


def test_npu_mp_shutdown_does_not_wait_if_request_cannot_be_sent(mocker):
    """A failed shutdown send must fall back directly to bounded termination."""
    from vllm_omni.diffusion import stage_diffusion_client as module

    client = object.__new__(StageDiffusionClient)
    client._request_socket = mocker.Mock()
    client._request_socket.send.side_effect = zmq.Again()
    client._response_socket = mocker.Mock()
    client._encoder = OmniMsgpackEncoder()
    client._zmq_ctx = mocker.Mock()
    manager = mocker.Mock(distributed_executor_backend="mp", uses_multiproc=True)
    manager.proc.is_alive.return_value = True
    client._proc_manager = manager
    mocker.patch.object(module.current_omni_platform, "is_npu", return_value=True)
    client.shutdown()
    manager.wait_for_shutdown.assert_not_called()
    manager.shutdown.assert_called_once_with(timeout=10.0)
