# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""TCP completion must mean that the destination has applied the write."""

import multiprocessing
import os
import signal
import threading
import time

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import TransferEngine

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.core_model,
    pytest.mark.skipif(TransferEngine is None, reason="Mooncake TransferEngine not installed"),
    pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process suspension"),
]


def _tcp_receiver(pipe):
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
    destination = torch.zeros(65536, dtype=torch.uint8).pin_memory()
    assert engine.register_memory(destination.data_ptr(), destination.numel()) == 0
    pipe.send((engine.get_rpc_port(), destination.data_ptr()))
    while True:
        value = pipe.recv()
        if value is None:
            return
        deadline = time.monotonic() + 10
        while not bool((destination == value).all()):
            assert time.monotonic() < deadline, "Destination did not receive the expected bytes"
        pipe.send(True)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_tcp_write_completion_waits_for_receiver(monkeypatch):
    # Also exercise TCP on hosts where Mooncake would otherwise auto-select RDMA.
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    receiver = context.Process(target=_tcp_receiver, args=(child,))
    receiver.start()
    receiver_pid = receiver.pid
    assert receiver_pid is not None
    stopped = False
    try:
        assert parent.poll(120), "Receiver did not initialize"
        port, address = parent.recv()
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        source = torch.ones(65536, dtype=torch.uint8).pin_memory()
        assert engine.register_memory(source.data_ptr(), source.numel()) == 0
        remote = f"127.0.0.1:{port}"

        # Establish the data connection before suspending the receiver, so the
        # second call tests data completion rather than RPC/connection setup.
        assert engine.batch_transfer_sync_write(remote, [source.data_ptr()], [address], [source.numel()]) == 0
        parent.send(1)
        assert parent.poll(30) and parent.recv() is True
        os.kill(receiver_pid, signal.SIGSTOP)
        stopped = True
        _, status = os.waitpid(receiver_pid, os.WUNTRACED)
        assert os.WIFSTOPPED(status)
        source.fill_(173)
        completed = threading.Event()
        result = []

        def transfer():
            result.append(engine.batch_transfer_sync_write(remote, [source.data_ptr()], [address], [source.numel()]))
            completed.set()

        thread = threading.Thread(target=transfer, daemon=True)
        thread.start()
        completed_while_stopped = completed.wait(0.5)
        os.kill(receiver_pid, signal.SIGCONT)
        stopped = False
        thread.join(30)
        assert result == [0], "Write did not complete successfully after resuming the receiver"
        parent.send(173)
        assert parent.poll(30) and parent.recv() is True
        assert not completed_while_stopped, "TCP reported completion before destination memory could be updated"
    finally:
        if stopped:
            os.kill(receiver_pid, signal.SIGCONT)
        if receiver.is_alive():
            parent.send(None)
            receiver.join(10)
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(10)
        parent.close()
        child.close()
