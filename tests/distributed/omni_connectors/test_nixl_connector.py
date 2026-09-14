# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for the NixlConnector metadata handshake.

NIXL itself is stubbed out, so these tests exercise the ZMQ control plane and
the payload normalisation rather than any actual RDMA transfer.
"""

import ctypes
import sys
import time
import types

import msgspec
import pytest
import torch
import zmq

from vllm_omni.data_entry_keys import HiddenStatesStruct, MetaStruct, OmniPayloadStruct

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]

PORT = 47431


def _wait_for_metadata(consumer, key, metadata=None, *, timeout=2.0):
    """Retry outside the connector's deliberately bounded metadata query."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resolved = consumer._resolve_metadata(key, metadata)
        if resolved is not None:
            return resolved
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for metadata for {key!r}")


@pytest.mark.parametrize("misses", [0, 2])
def test_wait_for_metadata_retries_until_success(monkeypatch, misses):
    expected = {"claim_id": "claim"}
    forwarded = {"source_host": "127.0.0.1", "source_port": PORT}
    calls = []

    def resolve(key, metadata):
        calls.append((key, metadata))
        return None if len(calls) <= misses else expected

    ticks = iter(range(10))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    consumer = types.SimpleNamespace(_resolve_metadata=resolve)
    assert _wait_for_metadata(consumer, "retry", forwarded, timeout=5) is expected
    assert calls == [("retry", forwarded)] * (misses + 1)


def test_wait_for_metadata_stops_at_deadline(monkeypatch):
    calls = []
    ticks = iter([0.0, 0.0, 1.0, 2.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    consumer = types.SimpleNamespace(_resolve_metadata=lambda *args: calls.append(args))
    with pytest.raises(pytest.fail.Exception, match="Timed out waiting for metadata for 'missing'"):
        _wait_for_metadata(consumer, "missing")
    assert calls == [("missing", None)] * 2


@pytest.mark.parametrize("async_chunk", [False, True])
def test_three_stage_incoming_and_outgoing_endpoints(nixl_connector_cls, async_chunk):
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
    from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_connector_spec
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

    config = OmniTransferConfig(
        connectors={
            ("0", "1"): ConnectorSpec(name="NixlConnector", extra={"host": "127.0.0.1", "zmq_port": PORT}),
            ("1", "2"): ConnectorSpec(name="NixlConnector", extra={"host": "127.0.0.1", "zmq_port": PORT + 10}),
        }
    )
    connectors = []
    try:
        for stage in range(3):
            spec = get_stage_connector_spec(config, stage, async_chunk)
            resolved = resolve_connector_spec(ConnectorSpec(**spec), stage_id=stage, role=spec["extra"]["role"])
            connectors.append(nixl_connector_cls(resolved.extra))
        first, middle, last = connectors
        assert middle._sender_zmq_port == PORT
        assert middle._serving_handshake
        assert middle._zmq_port == PORT + 11
        first.put("0", "1", "incoming", torch.ones(1))
        incoming = _wait_for_metadata(middle, "incoming")
        assert incoming["sender_zmq_port"] == PORT
        assert incoming["generation"] == first._published["incoming"]["generation"]
        middle.put("1", "2", "outgoing", torch.ones(1))
        outgoing = _wait_for_metadata(last, "outgoing")
        assert outgoing["sender_zmq_port"] == PORT + 11
        assert outgoing["generation"] == middle._published["outgoing"]["generation"]
    finally:
        for connector in reversed(connectors):
            for pending in connector._pending.values():
                pending.claims.clear()  # No DMA in this metadata-only probe.
            connector.close()


@pytest.mark.parametrize("name", ["NixlConnector", "MooncakeTransferEngineConnector"])
def test_explicit_sender_port_is_not_derived(name):
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
    from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_connector_spec

    spec = ConnectorSpec(name=name, extra={"sender_zmq_port": 55000, "from_stage": 0})
    resolved = resolve_connector_spec(spec, stage_id=1, role="receiver", replica_id=3, local_rank=2)
    assert resolved.extra["sender_zmq_port"] == 55000
    assert spec.extra == {"sender_zmq_port": 55000, "from_stage": 0}
    spec.extra["zmq_port"] = "${UNUSED_NIXL_BASE_PORT}"
    assert resolve_connector_spec(spec, stage_id=1, role="receiver").extra["sender_zmq_port"] == 55000


@pytest.mark.parametrize("need_recv_cache", [False, True])
@pytest.mark.parametrize("update_before_init", [False, True])
def test_manager_receiver_does_not_bind_shared_producer_port(producer, need_recv_cache, update_before_init):
    """Payload-only and KV receivers must dial, not bind, the incoming edge."""
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVCacheConfig,
        OmniKVTransferManager,
    )

    manager = OmniKVTransferManager(
        OmniKVCacheConfig(
            connector_config={
                "type": "NixlConnector",
                "role": "receiver",
                "host": "127.0.0.1",
                "zmq_port": PORT,
                "backends": ["UCX"],
            },
            from_stage="0",
            to_stage="1",
            stage_id=1,
            need_recv_cache=need_recv_cache,
        )
    )
    sender_info = {"host": producer.host, "zmq_port": producer._zmq_port}
    if update_before_init:
        manager.update_sender_info(sender_info)
    receiver = manager.connector
    try:
        assert receiver is not None, "Receiver incorrectly tried to bind the producer's occupied port"
        assert receiver._zmq_port is None
        assert not receiver._serving_handshake
        if not update_before_init:
            assert receiver._sender_zmq_port is None
            manager.update_sender_info(sender_info)
        assert receiver._sender_zmq_port == producer._zmq_port
        assert manager.config.connector_config["zmq_port"] == PORT  # Do not mutate the shared specification.
        success, _, _ = producer.put("0", "1", "manager-receive", torch.ones(1))
        assert success
        metadata = _wait_for_metadata(receiver, "manager-receive")
        assert metadata["generation"] == producer._published["manager-receive"]["generation"]
        receiver._notify_transfer_done("manager-receive", metadata)
        assert not producer._pending
    finally:
        if receiver is not None:
            receiver.close()


@pytest.mark.parametrize("need_recv_cache", [False, True])
def test_manager_receiver_preserves_explicit_standalone_sender(producer, need_recv_cache):
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVCacheConfig,
        OmniKVTransferManager,
    )

    manager = OmniKVTransferManager(
        OmniKVCacheConfig(
            connector_config={
                "type": "NixlConnector",
                "role": "receiver",
                "zmq_port": PORT,
                "sender_host": producer.host,
                "sender_zmq_port": producer._zmq_port,
            },
            need_recv_cache=need_recv_cache,
        )
    )
    receiver = manager.connector
    try:
        assert receiver is not None
        assert receiver._zmq_port is None and not receiver._serving_handshake
        assert (receiver._sender_host, receiver._sender_zmq_port) == (producer.host, producer._zmq_port)
        success, _, _ = producer.put("0", "1", "standalone-receive", torch.ones(1))
        assert success
        metadata = _wait_for_metadata(receiver, "standalone-receive")
        assert metadata["generation"] == producer._published["standalone-receive"]["generation"]
        receiver._notify_transfer_done("standalone-receive", metadata)
        assert not producer._pending
    finally:
        if receiver is not None:
            receiver.close()


def test_middle_stage_advertises_its_outgoing_replica_endpoint():
    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 1
    client.replica_id = 3
    client.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            stage_connector_config={
                "name": "NixlConnector",
                "extra": {
                    "role": "receiver",
                    "host": "10.0.0.1",
                    "zmq_port": 47000,
                    "from_stage": 0,
                    "outgoing": {"host": "10.0.0.2", "zmq_port": 48000, "from_stage": 1},
                },
            }
        )
    )
    assert client._build_payload_sender_info() == {"host": "10.0.0.2", "zmq_port": 51073}


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.usefixtures("reliable_claim_queries")
def test_claimed_source_survives_expiry_and_cleanup(producer, consumer, direct):
    _, _, metadata = producer.put("0", "1", "claimed", torch.ones(1))
    resolved = consumer._resolve_metadata("claimed", metadata if direct else None)
    pending = producer._pending["claimed"]
    pending.deadline = 0
    producer._cleanup_expired_pending()
    assert producer._pending.get("claimed") is pending
    producer.cleanup("claimed")
    assert producer._pending.get("claimed") is pending
    assert producer._agent.registered
    consumer._notify_transfer_done("claimed", resolved)
    consumer._notify_transfer_done("claimed", resolved)
    assert producer._pending == {}
    assert producer._agent.registered == []


@pytest.mark.usefixtures("reliable_claim_queries")
def test_claims_are_generation_scoped_and_duplicate_ack_cannot_release_sibling(producer, consumer):
    _, _, original = producer.put("0", "1", "owners", torch.ones(1))
    first = consumer._resolve_metadata("owners", original)
    second = consumer._resolve_metadata("owners", original)
    assert first["claim_id"] != second["claim_id"]
    assert not producer.put("0", "1", "owners", torch.zeros(1))[0]
    consumer._notify_transfer_done("owners", first)
    consumer._notify_transfer_done("owners", first)
    assert len(producer._pending["owners"].claims) == 1
    consumer._notify_transfer_done("owners", second)
    assert not producer._pending
    producer.put("0", "1", "owners", torch.zeros(1))
    consumer._notify_transfer_done("owners", second)
    assert "owners" in producer._pending
    assert consumer._resolve_metadata("owners", original) is None


@pytest.mark.parametrize("get_metadata", [None, {"schema_version": 1, "tensor_specs": [], "descriptor_groups": []}])
@pytest.mark.usefixtures("reliable_claim_queries")
def test_abandoned_claim_survives_close_and_late_completion(producer, consumer, get_metadata):
    _, _, metadata = producer.put("0", "1", "abandoned", torch.ones(1))
    claimed = consumer._resolve_metadata("abandoned", metadata)
    producer._pending["abandoned"].deadline = 0
    producer.close()
    assert producer._closing and not producer._closed
    assert producer._agent.registered
    assert producer._listener_thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        producer.put("0", "1", "new", torch.ones(1))
    metrics = dict(producer._metrics)
    with pytest.raises(RuntimeError, match="Cannot get data: NixlConnector is closed"):
        producer.get("0", "1", "new", get_metadata)
    assert producer._metrics == metrics
    assert producer._pending["abandoned"].claims == {claimed["claim_id"]}
    assert consumer._resolve_metadata("abandoned", metadata) is None
    consumer._notify_transfer_done("abandoned", claimed)
    assert not producer._pending
    assert not producer._agent.registered
    assert producer._listener_thread.is_alive()
    producer.close()
    assert not producer._agent.registered
    assert producer._closed


@pytest.fixture
def ownership_copying_agent(consumer, monkeypatch):
    """Strict fake descriptors and actual CPU copies; not native NIXL evidence."""
    calls = []

    def descriptors(regions, memory_type):
        assert regions and all(region[0] > 0 and region[1] > 0 for region in regions)
        return regions

    def prepare(operation, local, local_ids, remote, remote_ids):
        assert operation == "READ"
        assert local_ids == remote_ids == list(range(len(local)))
        return list(zip(local, remote, strict=True))

    def transfer(pairs):
        calls.append(pairs)
        for destination, source in pairs:
            assert destination[1] == source[1] > 0
            ctypes.memmove(destination[0], source[0], source[1])

    agent = consumer._agent
    monkeypatch.setattr(agent, "add_remote_agent", lambda metadata: "producer", raising=False)
    monkeypatch.setattr(agent, "get_xfer_descs", descriptors, raising=False)
    monkeypatch.setattr(agent, "prep_xfer_dlist", lambda agent, descs: descs, raising=False)
    monkeypatch.setattr(agent, "make_prepped_xfer", prepare, raising=False)
    monkeypatch.setattr(agent, "transfer", transfer, raising=False)
    monkeypatch.setattr(agent, "check_xfer_state", lambda handle: "DONE", raising=False)
    for method in ("release_xfer_handle", "release_dlist_handle", "remove_remote_agent"):
        monkeypatch.setattr(agent, method, lambda handle: None, raising=False)
    return calls


@pytest.mark.parametrize("case", ["empty", "all_empty", "mixed", "scalar", "structured_empty", "structured_mixed"])
def test_zero_byte_leaves_complete_source_claim(producer, consumer, ownership_copying_agent, case):
    empty = torch.empty((2, 0, 3), dtype=torch.float64)
    other = torch.empty((0,), dtype=torch.int64)
    scalar = torch.tensor(7)
    vector = torch.arange(6, dtype=torch.float32)
    payload = {
        "empty": empty,
        "all_empty": [empty, other],
        "mixed": [empty, scalar, other, vector],
        "scalar": scalar,
        "structured_empty": {"leaves": (empty, [other]), "label": "empty"},
        "structured_mixed": {"leaves": (empty, [scalar, other, vector]), "label": "mixed"},
    }[case]
    ok, size, metadata = producer.put("0", "1", "empty-slots", payload)
    assert ok
    indices = [index for group in metadata["descriptor_groups"] for index in group["tensor_indices"]]
    assert sorted(indices) == [i for i, spec in enumerate(metadata["tensor_specs"]) if spec["size"]]
    actual, received_size = consumer.get("0", "1", "empty-slots", metadata)
    if isinstance(payload, dict):
        assert actual["label"] == payload["label"]
        torch.testing.assert_close(actual["leaves"], payload["leaves"], rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, payload, rtol=0, atol=0)
    assert size == received_size
    if case in ("empty", "all_empty"):
        assert not ownership_copying_agent
    assert not producer._pending and not producer._agent.registered
    assert not consumer._agent.registered


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("outcome", ["done", "error", "timeout", "unknown"])
def test_read_ownership_through_terminal_and_deferred_paths(
    producer, consumer, ownership_copying_agent, monkeypatch, direct, outcome
):
    import threading

    _, _, metadata = producer.put("0", "1", "active-read", torch.ones(2))
    entered, proceed = threading.Event(), threading.Event()
    state = ["PROC"]
    monkeypatch.setattr(consumer._agent, "check_xfer_state", lambda handle: state[0])

    def wait(handle, key):
        entered.set()
        assert proceed.wait(5)
        if outcome == "done":
            state[0] = "DONE"
        elif outcome == "error":
            state[0] = "ERR"
            raise RuntimeError("terminal error")
        else:
            if outcome == "unknown":
                state[0] = "UNKNOWN"
            raise TimeoutError("still owned")

    monkeypatch.setattr(consumer, "_wait_for_transfer", wait)
    worker = threading.Thread(target=consumer.get, args=("0", "1", "active-read", metadata if direct else None))
    worker.start()
    try:
        assert entered.wait(5)
        pending = producer._pending["active-read"]
        pending.deadline = 0
        producer._cleanup_expired_pending()
        producer.cleanup("active-read")
        assert producer._pending["active-read"] is pending
        assert producer._agent.registered
    finally:
        proceed.set()
        worker.join(5)
    assert not worker.is_alive()
    if outcome in ("timeout", "unknown"):
        assert producer._agent.registered
        assert consumer._deferred_transfers
        consumer._reap_deferred_transfers()
        assert producer._agent.registered
        state[0] = "DONE"
        deadline = time.monotonic() + 5
        while producer._pending and time.monotonic() < deadline:
            time.sleep(0.01)
    assert not producer._pending
    assert not producer._agent.registered


class _FakeNixlAgent:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config
        self.registered = []

    def get_reg_descs(self, regions, memory_type):
        assert regions and all(region[0] > 0 and region[1] > 0 for region in regions)
        return ("reg", tuple(regions), memory_type)

    def register_memory(self, descs, backends=None):
        self.registered.append(descs)

    def deregister_memory(self, descs):
        if descs in self.registered:
            self.registered.remove(descs)

    def get_agent_metadata(self):
        return f"agent-metadata-{self.name}".encode()


@pytest.fixture
def nixl_connector_cls(monkeypatch):
    """Import NixlConnector with vLLM's optional NIXL dependency stubbed out."""
    stub = types.ModuleType("vllm.distributed.nixl_utils")
    stub.NixlWrapper = _FakeNixlAgent  # type: ignore[attr-defined]
    config_module = types.ModuleType("test_nixl_config_api")
    config_sync_type = types.SimpleNamespace(NIXL_THREAD_SYNC_STRICT="strict")
    config_module.nixl_thread_sync_t = config_sync_type  # type: ignore[attr-defined]

    def agent_config(**kwargs):
        return kwargs

    agent_config.__module__ = config_module.__name__
    config_module.nixl_agent_config = agent_config  # type: ignore[attr-defined]
    stub.nixl_agent_config = agent_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.distributed.nixl_utils", stub)
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    sync_type = types.SimpleNamespace(NIXL_THREAD_SYNC_STRICT="wrong-public-enum")
    nixl_module = types.ModuleType("nixl")
    nixl_module.nixl_thread_sync_t = sync_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nixl", nixl_module)

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    return NixlConnector


@pytest.fixture
def producer(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "sender", "host": "127.0.0.1", "zmq_port": PORT})
    yield connector
    # These control-plane tests do not submit DMA. Explicitly complete claims
    # made by metadata-only probes before tearing down the shared port.
    for key, pending in list(connector._pending.items()):
        for claim in list(pending.claims):
            from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _XFER_DONE_MSG

            connector._handle_handshake_message(
                _XFER_DONE_MSG
                + msgspec.msgpack.encode({"key": key, "generation": pending.generation, "claim_id": claim})
            )
    connector.close()


@pytest.fixture
def consumer(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "receiver", "sender_host": "127.0.0.1", "sender_zmq_port": PORT})
    yield connector
    connector.close()


@pytest.fixture
def reliable_claim_queries(producer, consumer, monkeypatch):
    """Ownership probes need exact claims, not claims retained after lost replies.

    Exercise the real resolver, wire encoding and producer handler synchronously;
    discovery tests separately cover bounded queries over real ZMQ sockets.
    """
    import uuid

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG, _META_NOT_FOUND

    def query(key, host, port, *, generation=None):
        assert (host, port) == (producer.host, producer._zmq_port)
        request = {"key": key, "generation": generation, "claim_id": uuid.uuid4().hex}
        reply = producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(request))
        return None if reply == _META_NOT_FOUND else msgspec.msgpack.decode(reply)

    monkeypatch.setattr(consumer, "_query_metadata_at", query)


def test_put_publishes_its_handshake_endpoint(producer):
    ok, size, metadata = producer.put("0", "1", "req-0", torch.arange(8, dtype=torch.float32))

    assert ok is True
    assert size == 32
    assert metadata["sender_host"] == "127.0.0.1"
    assert metadata["sender_zmq_port"] == PORT


def test_agent_uses_strict_thread_synchronization(producer):
    assert producer._agent.config["sync_mode"] == "strict"


def test_handshake_serves_metadata_when_caller_has_none(producer, consumer):
    _, _, published = producer.put("0", "1", "req-1", torch.arange(4, dtype=torch.float32))

    resolved = _wait_for_metadata(consumer, "req-1")

    # msgpack has no tuple type, so region descriptors arrive as lists; get()
    # re-tuples them before handing them to NIXL.
    assert resolved.pop("claim_id")
    assert resolved == msgspec.msgpack.decode(msgspec.msgpack.encode(published))
    assert [tuple(region) for group in resolved["descriptor_groups"] for region in group["regions"]] == [
        region for group in published["descriptor_groups"] for region in group["regions"]
    ]


def test_legacy_metadata_without_generation_skips_the_handshake(consumer):
    """Externally owned metadata without a generation needs no ownership claim."""
    direct = {"schema_version": 1, "kind": "tensors", "tensor_specs": []}

    assert consumer._resolve_metadata("req-2", direct) is direct


def test_source_endpoint_metadata_overrides_configured_sender(consumer, monkeypatch):
    queried = []
    expected = {"schema_version": 1, "kind": "tensors", "tensor_specs": []}

    def query(get_key, host, port):
        queried.append((get_key, host, port))
        return expected

    monkeypatch.setattr(consumer, "_query_metadata_at", query)

    resolved = consumer._resolve_metadata(
        "req-tp4-rank3",
        {"source_host": "10.0.0.3", "source_port": PORT + 3 * 16},
    )

    assert resolved is expected
    assert queried == [("req-tp4-rank3", "10.0.0.3", PORT + 3 * 16)]


def test_unknown_key_is_queried_once(producer, consumer, monkeypatch):
    socket = consumer._get_req_socket(f"tcp://127.0.0.1:{PORT}")
    send_count = 0
    original_send = socket.send

    def count_send(message):
        nonlocal send_count
        send_count += 1
        return original_send(message)

    monkeypatch.setattr(socket, "send", count_send)

    assert consumer._resolve_metadata("never-published", None) is None
    assert send_count == 1
    assert socket.getsockopt(zmq.RCVTIMEO) == 10


@pytest.mark.usefixtures("reliable_claim_queries")
def test_transfer_done_releases_the_producer_buffer(producer, consumer):
    _, _, metadata = producer.put("0", "1", "req-3", torch.arange(4, dtype=torch.float32))
    assert producer._pending and producer._agent.registered

    metadata = consumer._resolve_metadata("req-3", metadata)
    consumer._notify_transfer_done("req-3", metadata)

    assert producer._pending == {}
    assert producer._published == {}
    assert producer._agent.registered == []


def test_direct_metadata_has_ephemeral_ownership_endpoint(nixl_connector_cls):
    connector = nixl_connector_cls({})
    try:
        assert connector._zmq_ctx is not None
        assert connector._listener_thread is not None
        _, _, metadata = connector.put("0", "1", "req-4", torch.zeros(2))
        assert metadata["sender_host"]
        assert metadata["sender_zmq_port"] > 0
    finally:
        connector.close()


def test_idle_producer_lease_expires_without_another_put(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "sender", "lease_seconds": 0.01})
    try:
        connector.put("0", "1", "req-expire", torch.zeros(2))

        deadline = time.monotonic() + 1.0
        while connector._pending and time.monotonic() < deadline:
            time.sleep(0.01)

        assert connector._pending == {}
        assert connector._published == {}
        assert connector._agent.registered == []
        assert connector._registered_descs == []
    finally:
        connector.close()


def test_close_stops_lease_reaper(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "sender"})
    lease_thread = connector._lease_thread

    connector.close()

    assert lease_thread is not None
    assert not lease_thread.is_alive()
    assert connector._lease_thread is None


def test_deferred_transfer_is_retained_while_active(nixl_connector_cls):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _DeferredTransfer

    connector = nixl_connector_cls({"role": "receiver"})
    released = []
    connector._agent.check_xfer_state = lambda handle: "PROC"
    connector._agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
    connector._agent.release_dlist_handle = lambda handle: released.append(("dlist", handle))
    connector._agent.remove_remote_agent = lambda agent: released.append(("agent", agent))
    transfer = _DeferredTransfer(
        tensors=[torch.zeros(1)],
        registrations=["registration"],
        dlists=["local", "remote"],
        handles=["transfer"],
        remote_agent="producer",
    )
    connector._defer_transfer(transfer)
    try:
        connector._reap_deferred_transfers()

        assert released == []
        assert transfer in connector._deferred_transfers
        assert transfer.tensors
    finally:
        connector._agent.check_xfer_state = lambda handle: "DONE"
        connector.close()


def test_deferred_transfer_releases_exactly_once_after_done(nixl_connector_cls):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _DeferredTransfer

    connector = nixl_connector_cls({"role": "receiver"})
    released = []
    connector._agent.check_xfer_state = lambda handle: "DONE"
    connector._agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
    connector._agent.release_dlist_handle = lambda handle: released.append(("dlist", handle))
    connector._agent.remove_remote_agent = lambda agent: released.append(("agent", agent))
    connector._agent.deregister_memory = lambda descs: released.append(("registration", descs))
    transfer = _DeferredTransfer(
        tensors=[torch.zeros(1)],
        registrations=["registration"],
        dlists=["local", "remote"],
        handles=["transfer"],
        remote_agent="producer",
    )
    connector._defer_transfer(transfer)
    try:
        connector._reap_deferred_transfers()
        connector._reap_deferred_transfers()

        assert released == [
            ("handle", "transfer"),
            ("dlist", "local"),
            ("dlist", "remote"),
            ("agent", "producer"),
            ("registration", "registration"),
        ]
        assert connector._deferred_transfers == []
        assert transfer.tensors == []
    finally:
        connector.close()


def test_active_sibling_transfer_requires_deferred_ownership(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "receiver"})
    connector._agent.check_xfer_state = lambda handle: {"failed": "ERR", "active": "PROC"}[handle]
    try:
        assert connector._transfers_may_be_active(["failed", "active"]) is True
        assert connector._transfers_may_be_active(["failed"]) is False
    finally:
        connector.close()


@pytest.mark.parametrize("with_empty", [False, True])
def test_get_defers_complete_ownership_when_sibling_transfer_is_active(nixl_connector_cls, with_empty):
    connector = nixl_connector_cls({"role": "receiver"})
    connector._agent.add_remote_agent = lambda metadata: "producer"
    connector._agent.get_xfer_descs = lambda regions, memory_type: (regions, memory_type)
    connector._agent.prep_xfer_dlist = lambda agent, descs: (agent, tuple(descs[0]))
    handles = iter(["failed", "active"])
    connector._agent.make_prepped_xfer = lambda *_args: next(handles)
    connector._agent.transfer = lambda handle: None
    connector._agent.check_xfer_state = lambda handle: {"failed": "ERR", "active": "PROC"}[handle]
    released = []
    connector._agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
    connector._agent.release_dlist_handle = lambda handle: released.append(("dlist", handle))
    connector._agent.remove_remote_agent = lambda agent: released.append(("agent", agent))
    connector._agent.deregister_memory = lambda descs: released.append(("registration", descs))
    metadata = {
        "schema_version": 1,
        "kind": "tensors",
        "agent_metadata": b"producer-metadata",
        "tensor_specs": [
            {"shape": [1], "dtype": "torch.float32", "device": "cpu", "size": 4},
            {"shape": [1], "dtype": "torch.float32", "device": "cpu", "size": 4},
        ],
        "descriptor_groups": [
            {"memory_type": "DRAM", "tensor_indices": [0], "regions": [(1, 4, 0, "")]},
            {"memory_type": "DRAM", "tensor_indices": [1], "regions": [(2, 4, 0, "")]},
        ],
        "size": 8,
    }
    if with_empty:
        metadata["tensor_specs"].insert(1, {"shape": [2, 0, 3], "dtype": "torch.int64", "device": "cpu", "size": 0})
        metadata["descriptor_groups"][1]["tensor_indices"] = [2]

    try:
        assert connector.get("0", "1", "req-active-sibling", metadata) is None
        assert released == []
        assert len(connector._deferred_transfers) == 1
        deferred = connector._deferred_transfers[0]
        assert deferred.handles == ["failed", "active"]
        assert len(deferred.registrations) == 2
        assert len(deferred.dlists) == 4
        assert deferred.remote_agent == "producer"
        assert len(deferred.tensors) == 2 + with_empty
        if with_empty:
            assert deferred.tensors[1].shape == (2, 0, 3)
    finally:
        connector._agent.check_xfer_state = lambda handle: "DONE"
        connector.close()


@pytest.mark.parametrize(
    "payload,expected_kind",
    [
        (torch.zeros(4), "tensors"),
        ([torch.zeros(2), torch.ones(3)], "tensors"),
        ({"hidden": torch.zeros(4), "meta": {"token_role_ids": [1, 2]}}, "structured"),
        ({"prompt": "a cat", "steps": 8}, "object"),
    ],
)
def test_payload_kinds_round_trip_through_the_handshake(producer, consumer, payload, expected_kind):
    _, _, published = producer.put("0", "1", "req-kind", payload)
    assert published["kind"] == expected_kind

    resolved = _wait_for_metadata(consumer, "req-kind")

    assert resolved["kind"] == expected_kind
    assert resolved["tensor_specs"] == published["tensor_specs"]


def test_structured_payload_groups_descriptors_by_memory_type(producer, monkeypatch):
    monkeypatch.setattr(
        producer,
        "_resolve_memory_type",
        lambda tensor: "DRAM" if tensor.dtype == torch.uint8 else "VRAM",
    )

    _, _, metadata = producer.put(
        "0",
        "1",
        "req-mixed",
        {"meta": "value", "hidden": torch.ones(4, dtype=torch.float32)},
    )

    assert metadata["schema_version"] == 1
    assert metadata["descriptor_groups"] == [
        {
            "memory_type": "DRAM",
            "tensor_indices": [0],
            "regions": metadata["descriptor_groups"][0]["regions"],
        },
        {
            "memory_type": "VRAM",
            "tensor_indices": [1],
            "regions": metadata["descriptor_groups"][1]["regions"],
        },
    ]
    assert len(producer._agent.registered) == 2


def test_omni_payload_struct_uses_structured_tensor_path(producer):
    payload = OmniPayloadStruct(
        hidden_states=HiddenStatesStruct(output=torch.ones(4, dtype=torch.float32)),
        meta=MetaStruct(left_context_size=2),
    )

    _, _, metadata = producer.put("0", "1", "req-struct", payload)
    skeleton, tensors = producer._extract_tensor_leaves(payload)

    assert metadata["kind"] == "structured"
    assert len(tensors) == 1
    assert torch.equal(tensors[0], payload.hidden_states.output)
    assert skeleton["meta"]["left_context_size"] == 2


def test_partial_group_registration_failure_rolls_back(producer, monkeypatch):
    monkeypatch.setattr(
        producer,
        "_resolve_memory_type",
        lambda tensor: "DRAM" if tensor.dtype == torch.uint8 else "VRAM",
    )
    original_register = producer._agent.register_memory

    def fail_second_group(descs, backends=None):
        if descs[2] == "VRAM":
            raise RuntimeError("registration failed")
        original_register(descs, backends=backends)

    monkeypatch.setattr(producer._agent, "register_memory", fail_second_group)

    success, _, metadata = producer.put(
        "0",
        "1",
        "req-partial",
        {"meta": "value", "hidden": torch.ones(4, dtype=torch.float32)},
    )

    assert success is False
    assert metadata is None
    assert producer._agent.registered == []
    assert producer._registered_descs == []
    assert "req-partial" not in producer._pending


def test_reusing_put_key_releases_previous_registration(producer):
    producer.put("0", "1", "req-reuse", torch.zeros(2))
    previous_registration = producer._agent.registered[0]

    producer.put("0", "1", "req-reuse", torch.ones(2))

    assert previous_registration not in producer._agent.registered
    assert len(producer._agent.registered) == 1
    assert len(producer._registered_descs) == 1


def test_expired_snapshot_cannot_claim_replacement(producer):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _PendingPayload

    old = _PendingPayload([torch.zeros(1)], ["old"], 0.0)
    replacement = _PendingPayload([torch.ones(1)], ["new"], time.monotonic() + 60)
    producer._pending["req-race"] = replacement

    assert producer._take_pending("req-race", expected=old) is None
    assert producer._pending["req-race"] is replacement


@pytest.mark.parametrize(
    "indices",
    [
        [0, 0],
        [0],
        [0, 2],
        [-1, 0],
    ],
)
def test_descriptor_groups_require_exact_tensor_index_partition(nixl_connector_cls, indices):
    metadata = {
        "schema_version": 1,
        "descriptor_groups": [
            {
                "memory_type": "DRAM",
                "tensor_indices": indices,
                "regions": [(1, 4, 0, "")] * len(indices),
            }
        ],
    }

    with pytest.raises(RuntimeError, match="exact partition"):
        nixl_connector_cls._validated_descriptor_groups(
            metadata, [{"shape": [1], "dtype": "torch.float32", "size": 4}] * 2
        )


def test_receive_device_ignores_the_producer_index(nixl_connector_cls):
    """A producer on cuda:3 must not dictate the consumer's card."""
    connector = nixl_connector_cls({})
    try:
        assert connector._resolve_receive_device("cpu") == torch.device("cpu")
        assert connector._resolve_receive_device(None) == torch.device("cpu")
    finally:
        connector.close()


def test_receive_device_config_wins(nixl_connector_cls):
    connector = nixl_connector_cls({"receive_device": "cpu"})
    try:
        assert connector._resolve_receive_device("cuda:3") == torch.device("cpu")
    finally:
        connector.close()


@pytest.fixture
def copying_native_agent(consumer, monkeypatch):
    """Enforce native nonzero descriptors and copy only the submitted regions."""
    agent = consumer._agent
    calls = {"agents": [], "descs": [], "transfers": [], "released": []}

    def add_remote(metadata):
        calls["agents"].append(metadata)
        return "producer"

    def descriptors(regions, memory_type):
        assert regions and all(len(region) == 3 and region[0] > 0 and region[1] > 0 for region in regions)
        calls["descs"].append((regions, memory_type))
        return regions

    def prepare(operation, local, local_ids, remote, remote_ids):
        assert operation == "READ"
        assert local_ids == remote_ids == list(range(len(local)))
        assert len(local) == len(remote)
        pairs = list(zip(local, remote, strict=True))
        for destination, source in pairs:
            assert destination[1] == source[1] > 0
        calls["transfers"].append(pairs)
        return pairs

    def transfer(pairs):
        for destination, source in pairs:
            ctypes.memmove(destination[0], source[0], source[1])

    monkeypatch.setattr(agent, "add_remote_agent", add_remote, raising=False)
    monkeypatch.setattr(agent, "get_xfer_descs", descriptors, raising=False)
    monkeypatch.setattr(agent, "prep_xfer_dlist", lambda agent, descs: descs, raising=False)
    monkeypatch.setattr(agent, "make_prepped_xfer", prepare, raising=False)
    monkeypatch.setattr(agent, "transfer", transfer, raising=False)
    monkeypatch.setattr(agent, "check_xfer_state", lambda handle: "DONE", raising=False)
    for method in ("release_xfer_handle", "release_dlist_handle", "remove_remote_agent"):
        monkeypatch.setattr(
            agent, method, lambda handle, method=method: calls["released"].append((method, handle)), raising=False
        )
    return calls


@pytest.mark.parametrize("case", ["empty", "all_empty", "mixed", "scalar", "structured_empty", "structured_mixed"])
def test_zero_byte_leaves_roundtrip_without_native_descriptors(
    producer, consumer, copying_native_agent, monkeypatch, case
):
    empty = torch.empty((2, 0, 3), dtype=torch.float64)
    other_empty = torch.empty((0,), dtype=torch.int64)
    scalar = torch.tensor(7, dtype=torch.int64)
    vector = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    payloads = {
        "empty": empty,
        "all_empty": [empty, other_empty],
        "mixed": [empty, scalar, other_empty, vector, empty],
        "scalar": scalar,
        "structured_empty": {"leaves": (empty, [other_empty]), "label": "empty slots"},
        "structured_mixed": {"leaves": (empty, [scalar, other_empty, vector, empty]), "label": "mixed slots"},
    }
    payload = payloads[case]
    # Simulate mixed source memory groups without requiring a GPU in unit tests.
    monkeypatch.setattr(producer, "_resolve_memory_type", lambda t: "VRAM" if t.dtype == torch.int64 else "DRAM")
    # Splitting the producer's DRAM group on receipt also exercises local IDs:
    # IDs are group-relative, whereas spec/skeleton indices remain global.
    monkeypatch.setattr(consumer, "_resolve_memory_type", lambda t: "VRAM" if t.dtype == torch.float32 else "DRAM")
    consumer._receive_device = torch.device("cpu")
    success, size, metadata = producer.put("0", "1", "req-empty-slots", payload)
    assert success
    tensors = producer._pending["req-empty-slots"].tensors
    specs = metadata["tensor_specs"]
    assert len(specs) == len(tensors)
    assert size == sum(t.numel() * t.element_size() for t in tensors)
    for tensor, spec in zip(tensors, specs, strict=True):
        assert spec["shape"] == list(tensor.shape)
        assert spec["dtype"] == str(tensor.dtype)
        assert spec["device"] == str(tensor.device)
    indices = [i for group in metadata["descriptor_groups"] for i in group["tensor_indices"]]
    assert sorted(indices) == [i for i, tensor in enumerate(tensors) if tensor.numel()]
    # Verify receiver override applies to empty specs too, without trusting a
    # producer's device index. Real CUDA placement is covered by the native suite.
    for spec in specs:
        if spec["size"] == 0:
            spec["device"] = "cuda:7"
    received = consumer.get("0", "1", "req-empty-slots", metadata)
    assert received is not None
    actual, received_size = received
    assert received_size == size
    if isinstance(payload, dict):
        assert actual.keys() == payload.keys()
        assert actual["label"] == payload["label"]
        assert isinstance(actual["leaves"], tuple)
        assert isinstance(actual["leaves"][1], list)
        torch.testing.assert_close(actual["leaves"], payload["leaves"], rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, payload, rtol=0, atol=0)
    calls = copying_native_agent
    assert sum(len(pairs) for pairs in calls["transfers"]) == len(indices)
    if case in ("empty", "all_empty"):
        assert metadata["descriptor_groups"] == []
        assert calls == {"agents": [], "descs": [], "transfers": [], "released": []}
        assert size == 0
    else:
        assert len(calls["agents"]) == 1
        assert len(calls["released"]) == 3 * len(calls["transfers"]) + 1
    assert producer._pending == producer._published == {}
    assert producer._registered_descs == producer._agent.registered == []
    assert consumer._agent.registered == consumer._remote_agents == consumer._deferred_transfers == []


@pytest.mark.parametrize(
    "defect",
    [
        "missing_groups",
        "missing_nonempty",
        "duplicate",
        "empty_index",
        "zero_region",
        "wrong_region_size",
        "null_pointer",
        "negative_device",
        "short_region",
        "bad_region_type",
        "bool_index",
        "float_index",
        "empty_group",
        "unequal_lengths",
        "missing_memory_type",
        "fake_empty_size",
        "fake_nonempty_size",
        "negative_shape",
        "float_shape",
        "bad_dtype",
        "bad_spec",
    ],
)
def test_invalid_empty_tensor_metadata_rejected_before_native_calls(producer, consumer, copying_native_agent, defect):
    success, _, metadata = producer.put(
        "0", "1", "req-invalid-empty", [torch.empty((0, 2)), torch.tensor(3.0), torch.empty((1, 0))]
    )
    assert success
    group = metadata["descriptor_groups"][0]
    region = group["regions"][0]
    if defect == "missing_groups":
        metadata.pop("descriptor_groups")
    elif defect == "missing_nonempty":
        metadata["descriptor_groups"] = []
    elif defect == "duplicate":
        group["tensor_indices"] *= 2
        group["regions"] *= 2
    elif defect == "empty_index":
        group["tensor_indices"] = [0]
    elif defect in ("zero_region", "wrong_region_size"):
        group["regions"] = [(region[0], 0 if defect == "zero_region" else 8, 0, "")]
    elif defect == "null_pointer":
        group["regions"] = [(0, 4, 0, "")]
    elif defect == "negative_device":
        group["regions"] = [(region[0], 4, -1, "")]
    elif defect == "short_region":
        group["regions"] = [(region[0], 4)]
    elif defect == "bad_region_type":
        group["regions"] = [None]
    elif defect in ("bool_index", "float_index"):
        group["tensor_indices"] = [True if defect == "bool_index" else 1.0]
    elif defect == "empty_group":
        metadata["descriptor_groups"].append({"memory_type": "DRAM", "tensor_indices": [], "regions": []})
    elif defect == "unequal_lengths":
        group["regions"] = []
    elif defect == "missing_memory_type":
        group.pop("memory_type")
    elif defect == "fake_empty_size":
        metadata["tensor_specs"][1]["size"] = 0
        metadata["descriptor_groups"] = []
    elif defect == "fake_nonempty_size":
        metadata["tensor_specs"][0]["size"] = 4
    elif defect == "negative_shape":
        metadata["tensor_specs"][0]["shape"] = [-1, 0]
    elif defect == "float_shape":
        metadata["tensor_specs"][0]["shape"] = [0.0, 2]
    elif defect == "bad_dtype":
        metadata["tensor_specs"][0]["dtype"] = "torch.not_a_dtype"
    elif defect == "bad_spec":
        metadata["tensor_specs"][0] = None
    assert consumer.get("0", "1", "req-invalid-empty", metadata) is None
    assert copying_native_agent == {"agents": [], "descs": [], "transfers": [], "released": []}
    assert consumer._agent.registered == []
    # Metadata acquisition now claims the source before validation. With no
    # submitted DMA, failure safely completes that claim instead of leaking it;
    # it must still report no successful payload receive.
    assert producer._pending == producer._published == {}
    assert producer._registered_descs == producer._agent.registered == []
    assert consumer._metrics["gets"] == 0
    assert consumer._metrics["errors"] == 1
