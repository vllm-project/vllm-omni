# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading

import pytest
import zmq
from vllm.utils.network_utils import get_open_zmq_ipc_path

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _client_without_subprocess() -> StageDiffusionClient:
    # A PUSH socket with no peer blocks on send, which is what the client is
    # left with after its subprocess died.
    client = object.__new__(StageDiffusionClient)
    client._zmq_ctx = zmq.Context()
    client._request_socket = client._zmq_ctx.socket(zmq.PUSH)
    client._request_socket.bind(get_open_zmq_ipc_path())
    client._response_socket = client._zmq_ctx.socket(zmq.PULL)
    client._response_socket.bind(get_open_zmq_ipc_path())
    client._encoder = OmniMsgpackEncoder()
    client._proc_manager = None
    client._engine_dead = False
    return client


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

    abort = threading.Thread(target=asyncio.run, args=(client.abort_requests_async(["req-0"]),), daemon=True)
    abort.start()
    abort.join(timeout=5)

    assert not abort.is_alive()
