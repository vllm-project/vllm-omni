# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Receive deadlines and ambiguous WRITE ownership; no native SDK or GPU needed."""

import ctypes
import socket
import threading
import time

import msgspec
import pytest
import torch
import zmq

from vllm_omni.distributed.omni_connectors.connectors import mooncake_transfer_engine_connector as module

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class Engine:
    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()
        self.unregistered = threading.Event()
        self.mode = "success"
        self.calls = []

    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 1

    def unregister_memory(self, *args):
        self.unregistered.set()

    def finish_write(self):
        for sources, destinations, lengths in self.calls:
            for src, dst, size in zip(sources, destinations, lengths):
                ctypes.memmove(dst, src, size)

    def batch_transfer_sync_write(self, remote, sources, destinations, lengths):
        self.calls.append((sources, destinations, lengths))
        self.started.set()
        assert self.gate.wait(5), "test did not release native operation"
        if self.mode == "write_raise":
            raise RuntimeError("unknown completion")
        if self.mode == "write_error":
            return -1
        self.finish_write()
        return 0


def eventually(predicate):
    until = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < until, "operation did not finish"
        time.sleep(0.005)


@pytest.fixture
def connectors(monkeypatch):
    made = []
    monkeypatch.setattr(module, "TransferEngine", Engine)

    def pool(c):
        c.pool = torch.empty(65536, dtype=torch.uint8)
        c.pool_size = c.pool.numel()
        c.base_ptr = c.pool.data_ptr()
        c.allocator = module.BufferAllocator(c.pool_size, alignment=64)

    monkeypatch.setattr(module.MooncakeTransferEngineConnector, "_init_pool", pool)

    def make(role="receiver", **config):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        c = module.MooncakeTransferEngineConnector({"host": "127.0.0.1", "role": role, "zmq_port": port, **config})
        made.append(c)
        return c

    yield make
    for c in made:
        c.engine.gate.set()
    for c in made:
        eventually(lambda: c._active_handlers == 0)
        # These are mock writers, now quiescent. Explicitly discard intentionally
        # retained unknown operations; production must retain them until exit.
        with c._state_lock:
            for buffer, sock in c._pending_receives.values():
                assert sock is None, "test left a live completion waiter"
                buffer.release()
            c._pending_receives.clear()
            c._in_flight_buffers.clear()
        c.close()
        eventually(lambda: c.pool is None)
        assert c not in module._RETAINED_CONNECTORS


def publish(sender, key="key"):
    ok, _, metadata = sender.put("0", "1", key, {"value": 42})
    assert ok
    return metadata


def receive(receiver, metadata, key="key", budget=1):
    return receiver.get_with_deadline("0", "1", key, metadata, deadline=time.monotonic() + budget)


def test_late_success_retains_both_ends_and_defers_close(connectors):
    sender, receiver = connectors("sender"), connectors()
    metadata = publish(sender)
    key = sender._make_key("key", "0", "1")
    source = sender._local_buffers[key][2]
    sender.engine.gate.clear()
    start = time.monotonic()
    assert receive(receiver, metadata, budget=0.1) is None
    assert time.monotonic() - start < 0.75
    assert sender.engine.started.wait(1)
    buffer = receiver._pending_receives[key][0]
    assert not buffer._released
    assert not source._released
    assert receive(receiver, metadata) is None
    assert len(sender.engine.calls) == 1
    sender.close()
    receiver.close()
    assert not sender.engine.unregistered.is_set()
    assert not receiver.engine.unregistered.is_set()
    sender.engine.gate.set()
    eventually(lambda: sender.pool is None and receiver.pool is None)
    assert source._released and buffer._released


@pytest.mark.parametrize("mode", ["write_error", "write_raise", "legacy_error"])
@pytest.mark.parametrize("late", [False, True])
def test_ambiguous_native_result_pins_sources_destinations_and_capacity(connectors, monkeypatch, mode, late):
    sender = connectors("sender", max_inflight_sends=1, buffer_ttl_seconds=0.001)
    receiver = connectors(max_inflight_receives=1)
    metadata = publish(sender)
    key = sender._make_key("key", "0", "1")
    source = sender._local_buffers[key][2]
    # Emulate a peer returning the old wire error, which also carries no
    # remote completion proof. The receiver must treat it as ambiguous.
    if mode == "legacy_error":
        monkeypatch.setattr(module, "TRANS_UNKNOWN", module.TRANS_ERROR)
    sender.engine.mode = "write_error" if mode == "legacy_error" else mode
    if late:
        sender.engine.gate.clear()
    assert receive(receiver, metadata, budget=0.1 if late else 2) is None
    if late:
        assert sender.engine.started.wait(1)
        assert not source._released
        assert not receiver._pending_receives[key][0]._released
        sender.engine.gate.set()
        eventually(lambda: receiver._pending_receives[key][1] is None)
    buffer, sock = receiver._pending_receives[key]
    assert sock is None
    assert not buffer._released
    assert not source._released
    sender.cleanup(key)
    with sender._state_lock:
        item = sender._local_buffers[key]
        sender._local_buffers[key] = (*item[:5], time.monotonic() - 10)
    sender._cleanup_stale_buffers()
    assert sender._local_buffers[key][2] is source
    assert not sender.put("0", "1", "key", b"replacement")[0]
    assert receive(receiver, metadata, key="different") is None
    # Sender capacity is also exhausted: a different receiver gets a safe
    # pre-submission rejection, not another native operation.
    other = connectors()
    second = publish(sender, "second")
    assert receive(other, second, key="second") is None
    assert not other._pending_receives
    assert len(sender.engine.calls) == 1
    sender.close()
    receiver.close()
    assert not sender.engine.unregistered.is_set()
    assert not receiver.engine.unregistered.is_set()
    # A late native write after an SDK error still targets its original live
    # allocation. Nothing has unregistered or recycled either pool.
    sender.engine.finish_write()
    assert not buffer._released and not source._released


def test_expired_deadline_never_allocates_or_submits(connectors):
    receiver = connectors()
    assert receiver.get_with_deadline("0", "1", "key", deadline=time.monotonic() - 1) is None
    assert not receiver._receiving_keys
    assert not hasattr(receiver._req_local, "deadline")


def test_tcp_requires_explicit_acknowledged_native_capability(connectors):
    with pytest.raises(ValueError, match="native_tcp_proto_version=2"):
        connectors(protocol="tcp")
    connector = connectors(protocol="tcp", native_tcp_proto_version=2)
    assert connector.native_tcp_proto_version == 2


def test_tcp_capability_mismatch_rejects_before_native_submission(connectors):
    sender = connectors("sender", protocol="tcp", native_tcp_proto_version=2)
    receiver = connectors(protocol="tcp", native_tcp_proto_version=2)
    metadata = publish(sender)
    metadata["native_tcp_proto_version"] = 1
    with pytest.raises(RuntimeError, match="acknowledged completion"):
        receive(receiver, metadata)
    assert not sender.engine.calls


def test_tcp_sender_rejects_unsupported_peer_before_native_submission(connectors):
    sender = connectors("sender", protocol="tcp", native_tcp_proto_version=2)
    publish(sender)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    try:
        sock.connect(f"tcp://127.0.0.1:{sender.zmq_port}")
        request = module.MooncakeAgentMetadata(
            "127.0.0.1", 1, "key@0_1", [123], [8], protocol_version=2, native_tcp_proto_version=1
        )
        sock.send(msgspec.msgpack.encode(request))
        assert sock.recv() == module.TRANS_NOT_SUBMITTED
        assert not sender.engine.calls
    finally:
        sock.close(linger=0)
        ctx.term()


def test_legacy_metadata_fails_before_write(connectors):
    sender, receiver = connectors("sender"), connectors()
    metadata = publish(sender)
    metadata.pop("protocol_version")
    with pytest.raises(RuntimeError, match="protocol v2"):
        receive(receiver, metadata)
    assert not sender.engine.calls
    assert not receiver._receiving_keys


def test_legacy_receiver_rejected_before_native_submission(connectors):
    sender = connectors("sender")
    publish(sender)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    try:
        sock.connect(f"tcp://127.0.0.1:{sender.zmq_port}")
        sock.send(msgspec.msgpack.encode(module.MooncakeAgentMetadata("127.0.0.1", 1, "key@0_1", [123], [8])))
        assert sock.recv() == module.TRANS_ERROR
        assert not sender.engine.calls
    finally:
        sock.close(linger=0)
        ctx.term()


def test_metadata_send_and_recv_share_one_budget(connectors, monkeypatch):
    receiver = connectors()
    clock = [10.0]
    options = {}

    class Socket:
        def setsockopt(self, option, value):
            options[option] = value

        def send(self, data):
            assert options[zmq.SNDTIMEO] == 1000
            clock[0] += 0.75

        def recv(self):
            assert options[zmq.RCVTIMEO] == 250
            return module.INFO_NOT_FOUND

    receiver._req_local.cache = {"tcp://127.0.0.1:1": Socket()}
    receiver._req_local.deadline = 11.0
    monkeypatch.setattr(module._time_mod, "monotonic", lambda: clock[0])
    try:
        assert receiver._query_metadata_at("key", "127.0.0.1", 1) is None
    finally:
        receiver._req_local.cache.clear()


def test_managed_source_is_owned_by_sender(connectors):
    sender, receiver = connectors("sender"), connectors()
    offset = sender.allocator.alloc(8)
    external = module.ManagedBuffer(sender.allocator, offset, 8, sender.pool)
    external.tensor.fill_(73)
    ok, _, metadata = sender.put("0", "1", "key", external)
    assert ok
    external.release()
    sender.pool[offset : offset + 8].fill_(2)
    result, size = receive(receiver, metadata)
    try:
        assert size == 8
        assert result.to_bytes() == bytes([73]) * 8
    finally:
        result.release()


def test_missing_source_is_a_safe_rejection(connectors, monkeypatch):
    sender, receiver = connectors("sender"), connectors()
    allocated = []
    allocate = receiver._alloc_recv_buffer

    def capture(*args):
        buffer, pointer = allocate(*args)
        allocated.append(buffer)
        return buffer, pointer

    monkeypatch.setattr(receiver, "_alloc_recv_buffer", capture)
    metadata = publish(sender)
    sender.cleanup(sender._make_key("key", "0", "1"))
    assert receive(receiver, metadata) is None
    assert not sender.engine.calls
    assert not receiver._pending_receives
    assert len(allocated) == 1 and allocated[0]._released
