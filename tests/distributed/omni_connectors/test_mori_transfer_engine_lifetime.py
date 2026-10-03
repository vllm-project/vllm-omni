# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Mori sender buffers must remain allocated until every RDMA write completes."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors import mori_transfer_engine_connector as mori
from vllm_omni.distributed.omni_connectors.utils.memory_pool import BufferAllocator, ManagedBuffer

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


@pytest.fixture
def sender(monkeypatch):
    connector = mori.MoriTransferEngineConnector.__new__(mori.MoriTransferEngineConnector)
    connector._local_buffers_lock = threading.Lock()
    connector._in_flight_buffers = {}
    connector._cleanup_pending = set()
    connector.allocator = BufferAllocator(total_size=64, alignment=64)
    connector.pool = torch.full((64,), 173, dtype=torch.uint8)
    holder = ManagedBuffer(connector.allocator, connector.allocator.alloc(64), 64, connector.pool)
    connector._local_buffers = {"request": (0, 64, holder, True, False, 0.0)}
    connector.pool_mem_desc = object()
    connector._ensure_remote_registered = Mock()
    connector._notify_listener = Mock()
    connector.engine = Mock()
    monkeypatch.setattr(mori, "MemoryDesc", SimpleNamespace(unpack=lambda value: value), raising=False)
    monkeypatch.setattr(mori._time_mod, "monotonic", lambda: mori._BUFFER_TTL_SECONDS + 1)
    try:
        yield connector
    finally:
        # The fixture owns no transport or listener threads.
        connector._closed = True


def _start_pull(sender, *, failed=False, wait_error=False):
    started, finish = threading.Event(), threading.Event()

    def wait():
        started.set()
        assert finish.wait(10), "Test did not release the transfer"
        if wait_error:
            raise RuntimeError("Transfer completion failed")

    status = SimpleNamespace(Wait=wait, Failed=lambda: failed, Message=lambda: "failed")
    sender.engine.batch_write.return_value = [status]
    request = mori.MoriPullRequest("request", b"engine", b"pool", 0, 64)
    responses: queue.Queue[tuple[bytes, bytes]] = queue.Queue()
    thread = threading.Thread(
        target=sender._handle_pull_request,
        args=(responses, "notify", b"receiver", msgspec.msgpack.encode(request)),
        daemon=True,
    )
    thread.start()
    assert started.wait(10), "Transfer did not reach its completion wait"
    return thread, finish, responses


@pytest.mark.parametrize("cleanup", ["ttl", "explicit"])
@pytest.mark.parametrize("failed", [False, True])
def test_cleanup_cannot_recycle_an_in_flight_buffer(sender, cleanup, failed):
    thread, finish, responses = _start_pull(sender, failed=failed)
    try:
        if cleanup == "ttl":
            sender._cleanup_stale_buffers()
        else:
            sender.cleanup("request")
        with pytest.raises(MemoryError):
            sender.allocator.alloc(64)
    finally:
        finish.set()
        thread.join(10)
    assert not thread.is_alive()
    expected = mori.TRANS_ERROR if failed else mori.TRANS_DONE
    assert responses.get_nowait() == (b"receiver", expected)
    if failed and cleanup == "ttl":
        # A failed write remains retryable, and TTL may reclaim it once idle.
        assert "request" in sender._local_buffers
        sender._cleanup_stale_buffers()
    assert sender.allocator.alloc(64) == 0


def test_overlapping_pulls_hold_the_buffer_until_both_complete(sender):
    first, finish_first, _ = _start_pull(sender)
    second, finish_second, _ = _start_pull(sender)
    try:
        finish_first.set()
        first.join(10)
        assert not first.is_alive()
        sender._cleanup_stale_buffers()
        with pytest.raises(MemoryError):
            sender.allocator.alloc(64)
    finally:
        finish_first.set()
        finish_second.set()
        first.join(10)
        second.join(10)
    assert not second.is_alive()
    assert sender.allocator.alloc(64) == 0


def test_completion_exception_releases_the_transfer_pin(sender):
    thread, finish, responses = _start_pull(sender, wait_error=True)
    finish.set()
    thread.join(10)
    assert not thread.is_alive()
    assert responses.get_nowait() == (b"receiver", mori.TRANS_ERROR)
    sender._cleanup_stale_buffers()
    assert sender.allocator.alloc(64) == 0


def test_pending_cleanup_is_hidden_from_new_receivers(sender):
    thread, finish, _ = _start_pull(sender)
    try:
        sender.cleanup("request")
        responses: queue.Queue[tuple[bytes, bytes]] = queue.Queue()
        sender._handle_query_request(
            responses, "notify", b"receiver", msgspec.msgpack.encode(mori.QueryRequest("request"))
        )
        assert responses.get_nowait() == (b"receiver", mori.INFO_NOT_FOUND)
        request = mori.MoriPullRequest("request", b"engine", b"pool", 0, 64)
        sender._handle_pull_request(responses, "notify", b"receiver", msgspec.msgpack.encode(request))
        assert responses.get_nowait() == (b"receiver", mori.TRANS_ERROR)
        with pytest.raises(MemoryError):
            sender.allocator.alloc(64)
    finally:
        finish.set()
        thread.join(10)
    assert not thread.is_alive()
    assert sender.allocator.alloc(64) == 0


def test_put_cannot_replace_an_in_flight_buffer(sender, monkeypatch):
    old_pool = sender.pool
    holder = sender._local_buffers["request"][2]
    sender.pool = torch.full((128,), 173, dtype=torch.uint8)
    sender.pool[:64].copy_(old_pool)
    holder.pool_tensor = sender.pool
    sender.allocator.total_size = 128
    sender.allocator.free_blocks = [(64, 64)]
    sender._closed = False
    sender.can_put = True
    sender._metrics = {"puts": 0, "bytes_transferred": 0, "errors": 0}
    sender.host, sender.zmq_port = "localhost", 1234
    monkeypatch.setattr(sender, "_make_key", lambda key, from_stage, to_stage: key)
    thread, finish, _ = _start_pull(sender)
    try:
        assert sender.put("0", "1", "request", torch.zeros(64, dtype=torch.uint8)) == (False, 0, None)
        assert sender._local_buffers["request"][2] is holder
        # The rejected replacement releases its staging allocation only.
        assert sender.allocator.alloc(64) == 64
        with pytest.raises(MemoryError):
            sender.allocator.alloc(64)
    finally:
        finish.set()
        thread.join(10)
    assert not thread.is_alive()
    assert sender.allocator.alloc(64) == 0
