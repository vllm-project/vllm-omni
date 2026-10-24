# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for the NixlConnector metadata handshake.

NIXL itself is stubbed out, so these tests exercise the ZMQ control plane and
the payload normalisation rather than any actual RDMA transfer.
"""

import ctypes
import subprocess
import sys
import threading
import time
import types
import weakref
from pathlib import Path
from typing import Any

import msgspec
import pytest
import torch
import zmq

from vllm_omni.data_entry_keys import HiddenStatesStruct, MetaStruct, OmniPayloadStruct

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]

PORT = 47431


@pytest.mark.parametrize(
    "fixture,arguments,returncode,expected",
    [
        ("benchmark_nixl_publication.py", ["--help"], 0, "--prefixes"),
        ("cross_node_nixl_cancellation.py", ["--help"], 0, "--control-port"),
        (
            "benchmark_nixl_publication.py",
            ["--prefixes", "0", "--output", "unused.json"],
            2,
            "--prefixes must be positive",
        ),
        (
            "benchmark_nixl_publication.py",
            ["--prefixes", "1", "--num-threads", "-1", "--output", "unused.json"],
            2,
            "--num-threads must be non-negative",
        ),
        (
            "cross_node_nixl_cancellation.py",
            ["producer", "--host", "localhost", "--output", "unused.json"],
            2,
            "producer requires --endpoint",
        ),
        (
            "cross_node_nixl_cancellation.py",
            ["receiver", "--host", "localhost", "--port", "0", "--control-port", "1", "--output", "unused.json"],
            2,
            "receiver requires --port and --control-port in 1..65535",
        ),
        (
            "cross_node_nixl_cancellation.py",
            ["receiver", "--host", "localhost", "--cancel-delay-ms", "-1", "--output", "unused.json"],
            2,
            "--cancel-delay-ms must be non-negative",
        ),
    ],
)
def test_nixl_fixture_cli_validates_before_loading_native_sdk(fixture, arguments, returncode, expected):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-I", str(root / "benchmarks/distributed/omni_connectors" / fixture), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == returncode, result.stderr
    assert expected in result.stdout + result.stderr


def _wait_for_metadata(consumer, key, metadata=None, *, timeout=2.0):
    """Retry outside the connector's deliberately bounded metadata query."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resolved = consumer._resolve_metadata(key, metadata)
        if resolved is not None:
            return resolved
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for metadata for {key!r}")


def _claim(producer, key):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG, _META_NOT_FOUND

    reply = producer._handle_handshake_message(
        _GET_META_MSG + msgspec.msgpack.encode({"key": key, "claim_id": "reader"})
    )
    return None if reply == _META_NOT_FOUND else msgspec.msgpack.decode(reply)


def _fail_metadata():
    raise RuntimeError("injected metadata publication failure")


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
@pytest.mark.usefixtures("reliable_claim_queries")
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
                "metadata_query_timeout_ms": 5000,
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
@pytest.mark.usefixtures("reliable_claim_queries")
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
    assert producer._registered_descs
    consumer._notify_transfer_done("claimed", resolved)
    consumer._notify_transfer_done("claimed", resolved)
    assert producer._pending == {}
    assert producer._registered_descs == []


@pytest.mark.usefixtures("reliable_claim_queries")
@pytest.mark.parametrize("other_request_id", ["req-other", "req_0_other", "req_0_1_other"])
def test_request_prefix_cleanup_retires_claimed_context_and_preserves_other_requests(
    producer, consumer, other_request_id
):
    other_key = f"{other_request_id}_0_0"
    for key in ("req_0_0", "req_0_1", other_key):
        producer.put("0", "2", key, torch.ones(1))
        assert consumer._resolve_metadata(key, None)
    metadata = consumer._resolve_metadata("req_0_1", None)
    assert producer._pending["req_0_0"].agent is producer._pending["req_0_1"].agent
    native = weakref.ref(producer._pending["req_0_0"].agent)
    other = producer._pending[other_key].agent
    assert producer.cleanup_prefix("req_0_") == 2
    assert native() is None
    assert set(producer._pending) == {other_key}
    assert producer._pending[other_key].agent is other
    assert len(producer._registered_descs) == 1
    assert producer.cleanup_prefix("req_0_") == 0
    consumer._notify_transfer_done("req_0_1", metadata)
    assert set(producer._pending) == {other_key}
    assert len(producer._registered_descs) == 1
    producer.wait_for_cleanup("req_0_", deadline=time.monotonic())
    assert set(producer._pending) == {other_key}


def test_cleanup_retires_claimed_source_without_ack_and_allows_key_reuse(producer, consumer):
    producer.put("0", "1", "req_0_0", torch.ones(4))
    metadata = _wait_for_metadata(consumer, "req_0_0")
    assert producer._pending["req_0_0"].claims == {metadata["claim_id"]}
    producer.wait_for_cleanup("req_0_", deadline=time.monotonic())
    assert not producer._pending
    assert not producer._registered_descs
    assert producer.put("0", "1", "req_0_0", torch.zeros(4))[0]
    consumer._notify_transfer_done("req_0_0", metadata)
    assert "req_0_0" in producer._pending


def test_cleanup_destroys_shared_context_before_releasing_all_source_storage(producer, monkeypatch):
    tensors: list[weakref.ReferenceType[torch.Tensor]] = []
    destroyed = []

    class ObservedAgent(_FakeNixlAgent):
        def __del__(self):
            destroyed.append(all(tensor() is not None for tensor in tensors))

    monkeypatch.setattr(producer, "_agent_factory", lambda: ObservedAgent("observed"))
    for chunk in range(2):
        tensor = torch.ones(4)
        assert producer.put("0", "2", f"req_0_{chunk}", tensor)[0]
        tensors.append(weakref.ref(producer._pending[f"req_0_{chunk}"].tensors[0]))
        del tensor
        assert _claim(producer, f"req_0_{chunk}")
    producer.cleanup_prefix("req_0_")
    assert destroyed == [True]
    assert all(tensor() is None for tensor in tensors)


@pytest.mark.usefixtures("reliable_claim_queries")
def test_completed_context_reused_only_after_last_chunk_and_with_fresh_generation(producer, consumer):
    for chunk in range(2):
        assert producer.put("0", "2", f"req_0_{chunk}", torch.ones(4))[0]
    first = consumer._resolve_metadata("req_0_0", None)
    second = consumer._resolve_metadata("req_0_1", None)
    agent = producer._pending["req_0_0"].agent
    assert consumer._notify_transfer_done("req_0_0", first)
    idle_while_chunk_pending = producer._idle_producer_agent
    assert idle_while_chunk_pending is None
    assert producer._pending["req_0_1"].agent is agent
    assert consumer._notify_transfer_done("req_0_1", second)
    assert producer._idle_producer_agent is agent
    assert agent.registered == []
    assert producer.put("0", "2", "req_0_1", torch.zeros(4))[0]
    replacement = producer._pending["req_0_1"]
    assert replacement.agent is None
    assert replacement.generation != second["generation"]
    assert producer._idle_producer_agent is agent
    assert consumer._notify_transfer_done("req_0_1", second)
    assert producer._pending["req_0_1"] is replacement
    assert consumer._resolve_metadata("req_0_1", None)
    assert replacement.agent is agent and producer._idle_producer_agent is None


def test_cuda_payload_ready_before_registration_and_publication(producer, monkeypatch):
    devices = [torch.device(name) for name in ("cuda:0", "cuda:0", "cuda:1", "cpu")]
    synchronized: list[torch.device] = []
    copied: list[torch.device] = []

    def contiguous():
        assert set(synchronized) == {torch.device("cuda:0"), torch.device("cuda:1")}
        return torch.ones(4)

    tensors = [
        types.SimpleNamespace(
            device=device,
            numel=lambda: 4,
            is_contiguous=lambda: False,
            detach=lambda: types.SimpleNamespace(contiguous=contiguous),
        )
        for device in devices
    ]
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronized.append)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: types.SimpleNamespace(synchronize=lambda: copied.append(device)),
    )
    normalized, _ = producer._normalize_tensor_payload(tensors)
    agent = _FakeNixlAgent("ready-gpu-storage")
    register = agent.register_memory

    def register_ready(descs, backends=None):
        assert len(synchronized) == 2
        assert set(synchronized) == {torch.device("cuda:0"), torch.device("cuda:1")}
        assert set(copied) == set(synchronized)
        assert "req_0_0" in producer._published
        register(descs, backends)

    monkeypatch.setattr(agent, "register_memory", register_ready)
    monkeypatch.setattr(producer, "_agent_factory", lambda: agent)
    assert producer.put("0", "2", "req_0_0", normalized)[0]
    assert "req_0_0" in producer._published
    assert producer._pending["req_0_0"].agent is None
    assert _claim(producer, "req_0_0")


@pytest.mark.parametrize("event_device", [None, "cuda:1", "not-an-event", "cuda:0"])
@pytest.mark.parametrize("contiguous", [False, True])
def test_ready_event_contract_and_copy_fence(producer, monkeypatch, event_device, contiguous):
    calls: list[str] = []
    tensor: types.SimpleNamespace = types.SimpleNamespace(
        device=torch.device("cuda:0"),
        numel=lambda: 4,
        is_contiguous=lambda: contiguous,
        detach=lambda: tensor,
        contiguous=lambda: normalize(),
    )

    class Event:
        device = torch.device(event_device) if event_device in ("cuda:0", "cuda:1") else None

        def synchronize(self):
            calls.append("source")

    def normalize():
        assert calls == ["source"]
        calls.append("normalize")
        return torch.ones(4)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: types.SimpleNamespace(synchronize=lambda: calls.append("copy"))
    )
    monkeypatch.setattr(producer, "_is_tensor_payload", lambda data: True)
    event = object() if event_device == "not-an-event" else Event()
    success, _, metadata = producer.put("0", "2", "event_0_0", [tensor], ready_event=event)
    if event_device != "cuda:0":
        assert not success and metadata is None
        assert calls == [] and not producer._pending
    else:
        assert success and metadata is not None
        assert calls == ["source", "normalize"] + ([] if contiguous else ["copy"])


@pytest.mark.usefixtures("reliable_claim_queries")
def test_only_one_completed_context_retained_and_close_releases_it(producer, consumer):
    contexts = []
    claims = {}
    for key in ("first_0_0", "second_0_0"):
        assert producer.put("0", "2", key, torch.ones(4))[0]
        claims[key] = consumer._resolve_metadata(key, None)
        contexts.append(weakref.ref(producer._pending[key].agent))
    assert contexts[0]() is not contexts[1]()
    for key in ("first_0_0", "second_0_0"):
        assert consumer._notify_transfer_done(key, claims[key])
    assert producer._idle_producer_agent is contexts[0]()
    assert contexts[1]() is None
    assert producer._pending == producer._published == {}
    assert producer._registered_descs == []
    producer.close()
    assert contexts[0]() is None
    assert producer._idle_producer_agent is None


@pytest.mark.usefixtures("reliable_claim_queries")
@pytest.mark.parametrize("claimed", [False, True])
def test_cancelled_context_is_destroyed_instead_of_reused(producer, consumer, claimed):
    assert producer.put("0", "2", "req_0_0", torch.ones(4))[0]
    if claimed:
        assert consumer._resolve_metadata("req_0_0", None) is not None
        context = weakref.ref(producer._pending["req_0_0"].agent)
    else:
        assert producer._pending["req_0_0"].agent is None
    producer.cleanup_prefix("req_0_")
    if claimed:
        assert context() is None
    assert producer._idle_producer_agent is None


@pytest.mark.usefixtures("reliable_claim_queries")
def test_ack_deregistration_failure_retains_context_without_reuse(producer, consumer, monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _XFER_DONE_MSG

    assert producer.put("0", "2", "req_0_0", torch.ones(4))[0]
    pending = producer._pending["req_0_0"]
    metadata = consumer._resolve_metadata("req_0_0", None)

    def fail_release(descs):
        raise RuntimeError("registration release failed")

    monkeypatch.setattr(pending.agent, "deregister_memory", fail_release)
    with pytest.raises(RuntimeError, match="registration release failed"):
        producer._handle_handshake_message(
            _XFER_DONE_MSG
            + msgspec.msgpack.encode(
                {"key": "req_0_0", "generation": metadata["generation"], "claim_id": metadata["claim_id"]}
            )
        )
    assert producer._pending["req_0_0"] is pending
    assert pending.tensors and pending.registrations
    assert producer._idle_producer_agent is None
    producer.cleanup_prefix("req_0_")
    assert producer._pending == {}
    assert producer._idle_producer_agent is None


@pytest.mark.parametrize("prefix", ["req", "req_0"])
def test_cleanup_rejects_partial_request_prefix(producer, prefix):
    producer.put("0", "2", "req_0_0", torch.ones(1))
    with pytest.raises(ValueError, match="prefixes must end"):
        producer.cleanup_prefix(prefix)
    assert "req_0_0" in producer._pending


def test_cleanup_registration_failure_preserves_owner_for_retry(producer, monkeypatch):
    agent = _FakeNixlAgent("failed-claim")
    monkeypatch.setattr(agent, "get_agent_metadata", _fail_metadata)
    monkeypatch.setattr(producer, "_agent_factory", lambda: agent)
    producer.put("0", "1", "req_0_0", torch.ones(4))
    assert _claim(producer, "req_0_0") is None
    original = agent.deregister_memory

    def fail_registration(descs):
        raise RuntimeError("registration release failed")

    monkeypatch.setattr(agent, "deregister_memory", fail_registration)
    with pytest.raises(RuntimeError, match="registration release failed"):
        producer.cleanup("req_0_0")
    assert producer._pending["req_0_0"].tensors
    assert producer._pending["req_0_0"].registrations == producer._registered_descs
    assert "req_0_0" not in producer._published
    monkeypatch.setattr(agent, "deregister_memory", original)
    producer.cleanup("req_0_0")
    assert not producer._pending
    assert not producer._registered_descs


@pytest.mark.parametrize("registration_groups", [1, 2])
def test_final_ack_retries_failed_registration_release(producer, monkeypatch, registration_groups):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
        _ACK,
        _GET_META_MSG,
        _META_NOT_FOUND,
        _XFER_DONE_MSG,
    )

    key = "ack-retry_0_0"
    data = torch.ones(4)
    if registration_groups == 2:
        monkeypatch.setattr(
            producer, "_resolve_memory_type", lambda tensor: "DRAM" if tensor.dtype == torch.uint8 else "VRAM"
        )
        data = {"meta": "value", "audio": data}
    assert producer.put("0", "2", key, data)[0]
    metadata = msgspec.msgpack.decode(
        producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode({"key": key, "claim_id": "reader"}))
    )
    pending = producer._pending[key]
    original_release = pending.agent.deregister_memory
    attempts = []

    def fail_once(descs):
        attempts.append(descs)
        if len(attempts) == registration_groups:
            raise RuntimeError("injected deregistration failure")
        original_release(descs)

    monkeypatch.setattr(pending.agent, "deregister_memory", fail_once)
    completion = _XFER_DONE_MSG + msgspec.msgpack.encode(
        {"key": key, "generation": metadata["generation"], "claim_id": "reader"}
    )
    with pytest.raises(RuntimeError, match="injected deregistration failure"):
        producer._handle_handshake_message(completion)
    assert producer._pending[key] is pending
    assert pending.tensors and pending.registrations
    assert key not in producer._published
    assert (
        producer._handle_handshake_message(
            _GET_META_MSG + msgspec.msgpack.encode({"key": key, "claim_id": "late-reader"})
        )
        == _META_NOT_FOUND
    )
    assert producer._handle_handshake_message(completion) == _ACK
    assert len(attempts) == registration_groups + 1
    assert not producer._pending and not producer._registered_descs
    assert producer.put("0", "2", key, torch.zeros(4))[0]
    replacement = producer._pending[key]
    assert producer._handle_handshake_message(completion) == _ACK
    assert producer._pending[key] is replacement and replacement.agent is None


@pytest.mark.parametrize("operation", ["put", "get", "ack"])
def test_context_retirement_allows_unrelated_request_progress(producer, monkeypatch, operation):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG, _XFER_DONE_MSG

    entered = threading.Event()
    retire = threading.Event()
    finished = threading.Event()
    errors = []
    storage_at_retirement = []

    class GatedAgent(_FakeNixlAgent):
        def __del__(self):
            storage_at_retirement.append(source_ref() is not None)
            entered.set()
            retire.wait(5)
            storage_at_retirement.append(source_ref() is not None)

    monkeypatch.setattr(producer, "_agent_factory", lambda: GatedAgent("blocked"))
    assert producer.put("0", "2", "blocked_0_0", torch.ones(4))[0]
    assert _claim(producer, "blocked_0_0")
    source_ref = weakref.ref(producer._pending["blocked_0_0"].tensors[0])
    monkeypatch.setattr(producer, "_agent_factory", lambda: _FakeNixlAgent("other"))
    assert producer.put("0", "2", "other_0_0", torch.ones(4))[0]
    claim = {"key": "other_0_0", "claim_id": "reader"}
    if operation == "ack":
        metadata = msgspec.msgpack.decode(
            producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(claim))
        )
        claim["generation"] = metadata["generation"]

    def independent_request():
        try:
            if operation == "put":
                assert not producer.put("0", "2", "blocked_0_0", torch.zeros(4))[0]
                assert producer.put("0", "2", "new_0_0", torch.ones(4))[0]
            elif operation == "get":
                metadata = msgspec.msgpack.decode(
                    producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(claim))
                )
                assert metadata["claim_id"] == "reader"
            else:
                producer._handle_handshake_message(_XFER_DONE_MSG + msgspec.msgpack.encode(claim))
                assert "other_0_0" not in producer._pending
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    cleanup = threading.Thread(target=producer.cleanup_prefix, args=("blocked_0_",))
    other = threading.Thread(target=independent_request)
    cleanup.start()
    try:
        assert entered.wait(2), "Native context retirement did not start"
        other.start()
        assert finished.wait(1), f"Context retirement blocked independent {operation}"
        assert not errors
        assert source_ref() is not None
    finally:
        retire.set()
        cleanup.join(3)
        if other.ident is not None:
            other.join(3)
    assert not cleanup.is_alive() and not other.is_alive()
    assert storage_at_retirement == [True, True]
    assert source_ref() is None
    assert producer.put("0", "2", "blocked_0_0", torch.zeros(4))[0]


def test_prefix_retirement_waits_for_sibling_ack_owner_release(producer, monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG, _XFER_DONE_MSG

    detached = threading.Event()
    resume = threading.Event()
    retiring = threading.Event()
    retired = threading.Event()
    storage_at_retirement = []
    errors = []

    class Agent(_FakeNixlAgent):
        def __del__(self):
            storage_at_retirement.append(source_ref() is not None)

    monkeypatch.setattr(producer, "_agent_factory", lambda: Agent("shared"))
    assert producer.put("0", "2", "shared_0_0", torch.ones(4))[0]
    assert producer.put("0", "2", "shared_0_1", torch.ones(4))[0]
    assert _claim(producer, "shared_0_0")
    assert _claim(producer, "shared_0_1")
    completing = producer._pending["shared_0_0"]
    assert completing.agent is producer._pending["shared_0_1"].agent
    source_ref = weakref.ref(producer._pending["shared_0_1"].tensors[0])
    claim = {"key": "shared_0_0", "claim_id": "reader"}
    metadata = msgspec.msgpack.decode(producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(claim)))
    claim["generation"] = metadata["generation"]
    lock = producer._state_lock

    class ReleaseGate:
        def __enter__(self):
            lock.acquire()
            if threading.current_thread() is cleanup:
                retiring.set()

        def __exit__(self, *exc):
            lock.release()
            if threading.current_thread() is ack and completing.agent is None and not detached.is_set():
                detached.set()
                resume.wait(5)

    def complete():
        try:
            producer._handle_handshake_message(_XFER_DONE_MSG + msgspec.msgpack.encode(claim))
        except BaseException as exc:
            errors.append(exc)

    def retire():
        try:
            producer.cleanup_prefix("shared_0_")
        except BaseException as exc:
            errors.append(exc)
        finally:
            retired.set()

    ack = threading.Thread(target=complete)
    cleanup = threading.Thread(target=retire)
    monkeypatch.setattr(producer, "_state_lock", ReleaseGate())
    ack.start()
    try:
        assert detached.wait(2)
        assert producer.put("0", "2", "shared_0_2", torch.ones(4))[0]
        assert _claim(producer, "shared_0_2")
        assert producer._pending["shared_0_2"].agent is producer._pending["shared_0_1"].agent
        cleanup.start()
        assert retiring.wait(2)
        assert not retired.wait(0.1), "Prefix retirement ignored the sibling ACK's native owner"
        assert source_ref() is not None
    finally:
        resume.set()
        ack.join(3)
        if cleanup.ident is not None:
            cleanup.join(3)
    assert not ack.is_alive() and not cleanup.is_alive() and not errors
    assert storage_at_retirement == [True]
    assert source_ref() is None and not producer._pending and not producer._registered_descs


def test_expired_release_failure_does_not_reject_independent_put(producer, monkeypatch):
    observed = threading.Event()
    assert producer.put("0", "2", "expired_0_0", torch.ones(4))[0]
    agent = _FakeNixlAgent("failed-claim")
    monkeypatch.setattr(agent, "get_agent_metadata", _fail_metadata)
    monkeypatch.setattr(producer, "_agent_factory", lambda: agent)
    assert _claim(producer, "expired_0_0") is None
    pending = producer._pending["expired_0_0"]
    pending.deadline = 0

    def fail_release(descs):
        observed.set()
        raise RuntimeError("injected persistent deregistration failure")

    monkeypatch.setattr(pending.agent, "deregister_memory", fail_release)
    producer._lease_wakeup.set()
    assert observed.wait(2)
    assert producer.put("0", "2", "independent_0_0", torch.ones(4))[0]
    assert producer._pending["expired_0_0"] is pending
    assert pending.tensors and pending.registrations
    assert "expired_0_0" not in producer._published


def test_lease_reaper_retries_release_failure_without_exiting(nixl_connector_cls, monkeypatch):
    connector = nixl_connector_cls({"role": "sender", "host": "127.0.0.1"})
    observed = threading.Event()
    retried = threading.Event()
    recover = threading.Event()
    attempts = []
    try:
        assert connector.put("0", "2", "expired_0_0", torch.ones(4))[0]
        agent = _FakeNixlAgent("failed-claim")
        monkeypatch.setattr(agent, "get_agent_metadata", _fail_metadata)
        monkeypatch.setattr(connector, "_agent_factory", lambda: agent)
        assert _claim(connector, "expired_0_0") is None
        pending = connector._pending["expired_0_0"]
        original_release = pending.agent.deregister_memory

        def release(descs):
            attempts.append(time.monotonic())
            if len(attempts) == 3:
                retried.set()
            observed.set()
            if not recover.is_set():
                raise RuntimeError("injected deregistration failure")
            original_release(descs)

        monkeypatch.setattr(pending.agent, "deregister_memory", release)
        pending.deadline = 0
        connector._lease_wakeup.set()
        assert observed.wait(2)
        assert retried.wait(2), "Lease reaper stopped retrying the failed release"
        assert connector._lease_thread.is_alive()
        assert all(b - a >= 0.05 for a, b in zip(attempts[:2], attempts[1:3]))
        assert connector._pending["expired_0_0"] is pending
        assert pending.tensors and pending.registrations
        recover.set()
        deadline = time.monotonic() + 2
        while connector._pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not connector._pending and not connector._registered_descs
        assert connector._lease_thread.is_alive()
    finally:
        recover.set()
        connector.close()


def test_failed_first_claim_registration_still_expires(nixl_connector_cls, monkeypatch):
    connector = nixl_connector_cls({"role": "sender", "host": "127.0.0.1", "lease_seconds": 0.01})
    agent = _FakeNixlAgent("failed-publication")

    def fail_metadata():
        raise RuntimeError("injected metadata publication failure")

    monkeypatch.setattr(agent, "get_agent_metadata", fail_metadata)
    monkeypatch.setattr(connector, "_agent_factory", lambda: agent)
    try:
        assert connector.put("0", "2", "failed_0_0", torch.ones(4))[0]
        assert _claim(connector, "failed_0_0") is None
        deadline = time.monotonic() + 2
        while connector._pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not connector._pending and not connector._registered_descs
        assert not agent.registered
        assert connector._lease_thread.is_alive()
    finally:
        connector.close()


@pytest.mark.parametrize("other_request_id", ["req_1", "req_0_other", "req_0_1_other"])
def test_receiver_cleanup_cannot_acknowledge_active_read(consumer, other_request_id):
    consumer._active_gets["req_0_0"] = 1
    with pytest.raises(TimeoutError, match="ownership still active"):
        consumer.wait_for_cleanup("req_2_", receive_prefix="req_0_", deadline=time.monotonic())
    # A different request with a matching string prefix is unaffected.
    consumer._active_gets[f"{other_request_id}_0_0"] = 1
    del consumer._active_gets["req_0_0"]
    consumer.wait_for_cleanup("req_2_", receive_prefix="req_0_", deadline=time.monotonic())
    consumer._active_gets.clear()


def test_refused_transfer_release_keeps_destination_registration(consumer):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _DeferredTransfer

    tensor = torch.ones(1)
    registration = consumer._agent.get_reg_descs([consumer._tensor_region(tensor)], "DRAM")
    consumer._agent.register_memory(registration)
    transfer = _DeferredTransfer(tensors=[tensor], registrations=[registration], handles=["active"])

    def original(handle):
        pass

    def refuse(handle):
        raise RuntimeError("cannot cancel active transfer")

    consumer._agent.release_xfer_handle = refuse
    consumer._release_deferred_transfer(transfer)
    assert transfer.handles == ["active"]
    assert transfer.registrations == [registration]
    assert consumer._agent.registered == [registration]
    consumer._agent.release_xfer_handle = original
    consumer._agent.release_dlist_handle = original
    consumer._release_deferred_transfer(transfer)
    assert not transfer.handles
    assert not transfer.registrations


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
def test_close_retires_abandoned_source_context(producer, consumer, get_metadata):
    _, _, metadata = producer.put("0", "1", "abandoned", torch.ones(1))
    assert consumer._resolve_metadata("abandoned", metadata)
    producer._pending["abandoned"].deadline = 0
    native = weakref.ref(producer._pending["abandoned"].agent)
    producer.close()
    assert producer._closed
    assert native() is None
    assert not producer._pending
    assert not producer._registered_descs
    assert producer._listener_thread is None
    with pytest.raises(RuntimeError, match="closed"):
        producer.put("0", "1", "new", torch.ones(1))
    metrics = dict(producer._metrics)
    with pytest.raises(RuntimeError, match="Cannot get data: NixlConnector is closed"):
        producer.get("0", "1", "new", get_metadata)
    assert producer._metrics == metrics


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
@pytest.mark.usefixtures("reliable_claim_queries")
def test_zero_byte_leaves_complete_source_claim(
    producer, consumer, ownership_copying_agent, metadata_receive_timeout, case
):
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
    assert not producer._pending and not producer._registered_descs
    assert not consumer._agent.registered
    metadata_receive_timeout.assert_not_called()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("outcome", ["done", "error", "timeout", "unknown"])
@pytest.mark.usefixtures("reliable_claim_queries")
def test_read_ownership_through_terminal_and_deferred_paths(
    producer, consumer, ownership_copying_agent, metadata_receive_timeout, monkeypatch, direct, outcome
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
        assert producer._registered_descs
    finally:
        proceed.set()
        worker.join(5)
    assert not worker.is_alive()
    if outcome in ("timeout", "unknown"):
        assert producer._registered_descs
        assert consumer._deferred_transfers
        consumer._reap_deferred_transfers()
        assert producer._registered_descs
        state[0] = "DONE"
        deadline = time.monotonic() + 5
        while producer._pending and time.monotonic() < deadline:
            time.sleep(0.01)
    assert not producer._pending
    assert not producer._registered_descs
    metadata_receive_timeout.assert_not_called()


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

    def release_xfer_handle(self, handle):
        pass

    def release_dlist_handle(self, handle):
        pass

    def remove_remote_agent(self, agent):
        pass


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
def native_release_engine(producer, consumer, mocker, request):
    from vllm import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    from tests.worker.test_omni_connector_mixin import _make_model_config
    from vllm_omni.core.sched.output import OmniNewRequestData
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
    from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

    stage_id, async_chunk = getattr(request, "param", (0, True))
    suffix = "async_chunk" if async_chunk else "full_payload"
    config = _make_model_config(
        async_chunk=async_chunk,
        custom_func=f"vllm_omni.model_executor.stage_input_processors.qwen3_tts.talker2code2wav_{suffix}",
    )
    config.stage_id = stage_id
    config.model_stage = "talker"
    connector = producer if config.stage_id == 0 else consumer
    producer.config.update(codec_chunk_frames=1, initial_codec_chunk_frames=1, codec_left_context_frames=0)
    mocker.patch.object(OmniRunnerDataPlane, "_create_connector", return_value=connector)
    plane = OmniRunnerDataPlane(mocker.Mock(model_config=config), config)
    request_data = OmniNewRequestData(
        req_id="internal",
        external_req_id="release-target",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=32),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
    )
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner._omni_data_plane = plane
    worker = GPUARWorker.__new__(GPUARWorker)
    worker.model_runner = runner
    wrapper = WorkerWrapperBase()
    wrapper.worker = worker
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = wrapper
    core = StageEngineCoreProc.__new__(StageEngineCoreProc)
    core.model_executor = executor
    core.scheduler = mocker.Mock(spec=Scheduler)
    core.scheduler._native_data_plane = True
    core.scheduler.chunk_transfer_adapter = None
    try:
        yield plane, core, request_data
    finally:
        plane.close()


@pytest.mark.parametrize("worker_finished", [False, True])
def test_engine_release_reclaims_native_terminal_publications(producer, native_release_engine, worker_finished):
    plane, core, request = native_release_engine
    plane.register_request(request)
    assert (
        plane.complete_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes": {"audio": torch.ones((1, 16), dtype=torch.long)}}],
            sampled_token_ids=[[21]],
        )
        == 1
    )
    if worker_finished:
        assert plane.abort_requests({"internal"}) == 1
    assert "release-target_0_0" in producer._pending
    producer.put("0", "1", "survivor_0_0", torch.ones(1))
    survivor = producer._pending["survivor_0_0"]

    core.omni_release_request_resources(["release-target"])
    assert producer._pending == {"survivor_0_0": survivor}
    assert not plane._native_requests
    assert plane.complete_outputs(req_ids=["internal"], inter_stage_outputs=None, sampled_token_ids=[[22]]) == 0
    core.omni_release_request_resources(["release-target"])
    assert producer._pending == {"survivor_0_0": survivor}


@pytest.mark.parametrize("native_release_engine", [(1, True)], indirect=True)
def test_engine_release_cancels_receive_before_model_slot(native_release_engine):
    from vllm_omni.core.sched.output import OmniChunkRecvHandle

    plane, core, _ = native_release_engine
    plane.register_receivers([OmniChunkRecvHandle(request_id="internal", external_req_id="release-target")])
    assert "internal" in plane._pending_load_reqs

    core.omni_release_request_resources(["release-target"])

    assert not plane._pending_load_reqs and not plane._request_ids_mapping


@pytest.mark.parametrize("native_release_engine", [(0, False)], indirect=True)
@pytest.mark.parametrize("worker_finished", [False, True])
def test_engine_release_reclaims_native_full_payload(producer, native_release_engine, worker_finished):
    plane, core, request = native_release_engine
    plane.register_request(request)
    assert (
        plane.complete_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes": {"audio": torch.ones((1, 16), dtype=torch.long)}}],
            sampled_token_ids=[[21]],
        )
        == 0
    )
    if worker_finished:
        assert plane.request_terminal({"internal"}) == 1

    core.omni_release_request_resources(["release-target"])

    assert not producer._pending and not plane._native_requests
    assert not plane._pending_full_payload_send


def test_engine_retirement_allows_queued_native_output(producer, native_release_engine, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    plane, core, request = native_release_engine
    plane.register_request(request)
    entered, resume = threading.Event(), threading.Event()
    cleanup_prefix = producer.cleanup_prefix

    def blocked_cleanup(prefix):
        entered.set()
        assert resume.wait(5)
        return cleanup_prefix(prefix)

    monkeypatch.setattr(producer, "cleanup_prefix", blocked_cleanup)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cleanup = executor.submit(core.omni_release_request_resources, ["release-target"])
        try:
            assert entered.wait(2)
            plane.register_request(replace(request, req_id="survivor", external_req_id="survivor"))
            plane.enqueue_outputs(
                req_ids=["survivor"],
                inter_stage_outputs=[{"codes": {"audio": torch.ones((1, 16), dtype=torch.long)}}],
                sampled_token_ids=[[21]],
            )
            executor.submit(plane.drain_outputs).result(timeout=2)
            assert "survivor_0_0" in producer._pending
        finally:
            resume.set()
            cleanup.result(timeout=2)


@pytest.fixture
def metadata_receive_timeout(consumer, monkeypatch, mocker):
    """Fail socket receives for claim queries, leaving completion ACKs real."""
    original_get_socket = consumer._get_req_socket
    recv = mocker.Mock(side_effect=zmq.Again)

    def get_socket(address, timeout_ms=None):
        socket = original_get_socket(address, timeout_ms)
        if timeout_ms == consumer._metadata_query_timeout_ms:
            monkeypatch.setattr(socket, "recv", recv)
        return socket

    monkeypatch.setattr(consumer, "_get_req_socket", get_socket)
    return recv


def test_metadata_receive_timeout_returns_no_payload(
    producer, consumer, ownership_copying_agent, metadata_receive_timeout
):
    _, _, metadata = producer.put("0", "1", "timed-out-claim", torch.tensor(7))

    assert consumer.get("0", "1", "timed-out-claim", metadata) is None
    metadata_receive_timeout.assert_called_once_with()
    assert not ownership_copying_agent
    assert not consumer._agent.registered


def test_generation_payload_roundtrip_over_zmq(producer, consumer, ownership_copying_agent, monkeypatch):
    """Cover real claim acquisition, CPU copying and completion notification."""
    # This socket integration probe can wait for the producer thread; the
    # bounded 10 ms miss/timeout contract is covered by separate tests.
    monkeypatch.setattr(consumer, "_metadata_query_timeout_ms", consumer._handshake_timeout_ms)
    payload = torch.arange(6, dtype=torch.float32)
    ok, size, metadata = producer.put("0", "1", "zmq-roundtrip", payload)
    assert ok and metadata["generation"]

    actual, received_size = consumer.get("0", "1", "zmq-roundtrip", metadata)

    torch.testing.assert_close(actual, payload, rtol=0, atol=0)
    assert received_size == size
    assert len(ownership_copying_agent) == 1
    assert not producer._pending and not producer._agent.registered
    assert not consumer._agent.registered


@pytest.fixture
def reliable_claim_queries(producer, nixl_connector_cls, monkeypatch):
    """Ownership probes need exact claims, not claims retained after lost replies.

    Exercise the real resolver, wire encoding and producer handler synchronously;
    discovery tests separately cover bounded queries over real ZMQ sockets.
    """
    import uuid

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG, _META_NOT_FOUND

    def query(self, key, host, port, *, generation=None):
        assert (host, port) == (producer.host, producer._zmq_port)
        request = {"key": key, "generation": generation, "claim_id": uuid.uuid4().hex}
        reply = producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(request))
        return None if reply == _META_NOT_FOUND else msgspec.msgpack.decode(reply)

    monkeypatch.setattr(nixl_connector_cls, "_query_metadata_at", query)


def test_put_publishes_its_handshake_endpoint(producer):
    ok, size, metadata = producer.put("0", "1", "req-0", torch.arange(8, dtype=torch.float32))

    assert ok is True
    assert size == 32
    assert metadata["sender_host"] == "127.0.0.1"
    assert metadata["sender_zmq_port"] == PORT


def test_agent_uses_strict_thread_synchronization(producer):
    assert producer._agent.config["sync_mode"] == "strict"
    assert producer._agent.config["num_threads"] == 0


@pytest.mark.parametrize("num_threads", [1, 4])
def test_agent_preserves_explicit_worker_count(nixl_connector_cls, num_threads):
    connector = nixl_connector_cls({"role": "receiver", "num_threads": num_threads})
    try:
        assert connector._agent.config["num_threads"] == num_threads
    finally:
        connector.close()


def test_unclaimed_prefixes_do_not_create_native_contexts(producer, monkeypatch):
    contexts = []

    def create():
        agent = _FakeNixlAgent("claimed")
        contexts.append(agent)
        return agent

    monkeypatch.setattr(producer, "_agent_factory", create)
    for prefix in range(16):
        assert producer.put("0", "2", f"request{prefix}_0_0", torch.ones(4))[0]
    assert contexts == [] and not producer._registered_descs
    assert all(pending.agent is None for pending in producer._pending.values())
    first = _claim(producer, "request0_0_0")
    assert first and len(contexts) == 1
    assert _claim(producer, "request0_0_0") == first
    assert len(contexts) == 1 and len(producer._registered_descs) == 1
    assert _claim(producer, "request1_0_0")
    assert len(contexts) == 2
    assert producer._pending["request0_0_0"].agent is not producer._pending["request1_0_0"].agent


def test_handshake_serves_metadata_when_caller_has_none(producer, consumer):
    _, _, published = producer.put("0", "1", "req-1", torch.arange(4, dtype=torch.float32))

    resolved = _wait_for_metadata(consumer, "req-1")

    # msgpack has no tuple type, so region descriptors arrive as lists; get()
    # re-tuples them before handing them to NIXL.
    assert resolved.pop("claim_id")
    assert resolved.pop("agent_metadata")
    assert "agent_metadata" not in published
    assert resolved == msgspec.msgpack.decode(msgspec.msgpack.encode(published))
    assert [tuple(region) for group in resolved["descriptor_groups"] for region in group["regions"]] == [
        region for group in published["descriptor_groups"] for region in group["regions"]
    ]


@pytest.mark.usefixtures("reliable_claim_queries")
def test_unpublished_payload_poll_is_not_a_metadata_error(producer, consumer, mocker):
    from vllm_omni.distributed.omni_connectors.connectors import nixl_connector

    error = mocker.spy(nixl_connector.logger, "error")
    assert consumer.get("0", "2", "next-chunk") is None
    error.assert_not_called()
    assert consumer.health()["errors"] == 0
    producer.put("0", "2", "next-chunk", torch.ones(1))
    resolved = consumer._resolve_metadata("next-chunk", None)
    assert resolved["generation"] == producer._published["next-chunk"]["generation"]
    consumer._notify_transfer_done("next-chunk", resolved)


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


@pytest.mark.parametrize("force_timeout", [False, True], ids=["real-query", "recv-timeout"])
def test_unknown_key_is_queried_once(producer, consumer, monkeypatch, force_timeout):
    socket = consumer._get_req_socket(f"tcp://127.0.0.1:{PORT}")
    send_count = 0
    original_send = socket.send

    def count_send(message):
        nonlocal send_count
        assert socket.getsockopt(zmq.RCVTIMEO) == 10
        assert socket.getsockopt(zmq.SNDTIMEO) == 10
        send_count += 1
        return original_send(message)

    monkeypatch.setattr(socket, "send", count_send)
    if force_timeout:

        def timeout_recv():
            raise zmq.Again()

        monkeypatch.setattr(socket, "recv", timeout_recv)

    assert consumer._resolve_metadata("never-published", None) is None
    assert send_count == 1
    if force_timeout:
        assert socket.closed


@pytest.mark.usefixtures("reliable_claim_queries")
def test_deadline_get_retains_active_dma_then_releases(producer, consumer, ownership_copying_agent, monkeypatch):
    producer.put("0", "1", "deadline-read", torch.ones(2))
    consumer._stop_event.set()
    consumer._transfer_wakeup.set()
    consumer._transfer_thread.join(timeout=2)
    state = ["PROC"]
    monkeypatch.setattr(consumer._agent, "check_xfer_state", lambda handle: state[0])
    started = time.monotonic()
    assert consumer.get_with_deadline("0", "1", "deadline-read", deadline=started + 0.03) is None
    assert time.monotonic() - started < 0.5
    assert producer._pending["deadline-read"].claims
    assert producer._registered_descs
    assert consumer._agent.registered
    assert consumer._deferred_transfers
    state[0] = "DONE"
    consumer._reap_deferred_transfers()
    assert not producer._pending
    assert not producer._registered_descs
    assert not consumer._agent.registered
    assert not consumer._deferred_transfers


def test_receive_deadline_overrides_dma_timeout(consumer, monkeypatch):
    monkeypatch.setattr(consumer._agent, "check_xfer_state", lambda handle: "PROC", raising=False)
    consumer._transfer_timeout_s = 300
    consumer._req_local.deadline = time.monotonic() + 0.02
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            consumer._wait_for_transfer(1, "bounded")
        assert time.monotonic() - started < 0.5
    finally:
        consumer._req_local.deadline = None


@pytest.mark.parametrize("closing", [False, True])
def test_abandoned_metadata_query_recovers_same_claim_without_read(producer, consumer, monkeypatch, closing):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG

    consumer._stop_event.set()
    consumer._transfer_wakeup.set()
    consumer._transfer_thread.join(timeout=2)
    producer.put("0", "1", "abandoned-query", torch.ones(1))
    messages = []

    def send(message):
        messages.append(message)

    def receive():
        reply = producer._handle_handshake_message(messages[-1])
        if len(messages) == 1:
            raise zmq.Again()
        return reply

    socket = types.SimpleNamespace(send=send, recv=receive)
    monkeypatch.setattr(consumer, "_get_req_socket", lambda *args: socket)
    monkeypatch.setattr(consumer, "_invalidate_req_socket", lambda *args: None)
    assert consumer.get_with_deadline("0", "1", "abandoned-query", deadline=time.monotonic() + 1) is None
    assert not consumer._abandoned_queries
    consumer.abandon_get("abandoned-query")
    claim = next(iter(producer._pending["abandoned-query"].claims))
    if closing:
        producer._closing = True
    consumer._reap_deferred_transfers()
    assert not consumer._abandoned_queries
    assert not producer._pending
    claims = [
        msgspec.msgpack.decode(message[len(_GET_META_MSG) :])["claim_id"]
        for message in messages
        if message.startswith(_GET_META_MSG)
    ]
    assert claims == [claim, claim]


def test_expired_ack_is_deferred_instead_of_raising(consumer):
    consumer._req_local.deadline = time.monotonic() - 1
    try:
        assert consumer._notify_transfer_done("expired", {"source_host": "127.0.0.1", "source_port": PORT}) is False
    finally:
        consumer._req_local.deadline = None


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("lost_replies", [1, 3])
def test_lost_metadata_reply_retry_releases_source(producer, consumer, monkeypatch, direct, lost_replies):
    _, _, metadata = producer.put("0", "1", "lost-reply", torch.ones(1))
    replies = []
    requests = []

    def send(message):
        requests.append(message)
        replies.append(producer._handle_handshake_message(message))

    def recv():
        reply = replies.pop(0)
        if len(requests) <= lost_replies:
            raise zmq.Again()
        return reply

    socket = types.SimpleNamespace(send=send, recv=recv)
    monkeypatch.setattr(consumer, "_get_req_socket", lambda *args: socket)
    forwarded = metadata if direct else None
    for attempt in range(lost_replies):
        assert consumer._resolve_metadata("lost-reply", forwarded) is None
    resolved = consumer._resolve_metadata("lost-reply", forwarded)
    assert resolved is not None
    assert producer._pending["lost-reply"].claims == {resolved["claim_id"]}
    independent = consumer._resolve_metadata("lost-reply", forwarded)
    assert independent["claim_id"] != resolved["claim_id"]
    consumer._notify_transfer_done("lost-reply", resolved)
    consumer._notify_transfer_done("lost-reply", resolved)
    assert producer._pending["lost-reply"].claims == {independent["claim_id"]}
    consumer._notify_transfer_done("lost-reply", independent)
    assert producer._pending == {}
    assert producer._registered_descs == []
    assert consumer._req_local.claims == {}


@pytest.mark.usefixtures("reliable_claim_queries")
def test_transfer_done_releases_the_producer_buffer(producer, consumer):
    _, _, metadata = producer.put("0", "1", "req-3", torch.arange(4, dtype=torch.float32))
    assert producer._pending and not producer._registered_descs

    metadata = consumer._resolve_metadata("req-3", metadata)
    consumer._notify_transfer_done("req-3", metadata)

    assert producer._pending == {}
    assert producer._published == {}
    assert producer._registered_descs == []


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


@pytest.mark.parametrize("role", ["sender", "receiver"])
def test_close_releases_native_agent_while_connector_remains_alive(nixl_connector_cls, role):
    connector = nixl_connector_cls({"role": role, "host": "127.0.0.1"})
    agent = weakref.ref(connector._agent)
    connector.close()
    assert agent() is None
    assert connector._closed and connector._agent is None
    connector.close()


def test_failed_listener_initialization_can_release_partial_resources(producer, nixl_connector_cls):
    connector = nixl_connector_cls.__new__(nixl_connector_cls)
    with pytest.raises(RuntimeError, match="failed to bind handshake socket"):
        connector.__init__({"role": "sender", "host": "127.0.0.1", "zmq_port": PORT})
    connector.close()
    assert connector._closed
    assert connector._listener_thread is None
    assert connector._zmq_ctx is None


@pytest.mark.parametrize("outcome", ["completed-read", "deferred-read", "lost-metadata", "lost-metadata-finalizing"])
def test_close_keeps_active_get_completion_and_cleanup_alive(
    producer, consumer, ownership_copying_agent, monkeypatch, outcome
):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _RETAINED_PRODUCERS

    entered = threading.Event()
    resume = threading.Event()
    state = ["PROC"]
    received = []
    agent = consumer._agent

    def wait(handle, key):
        entered.set()
        assert resume.wait(5), "The test did not release the active get"
        if outcome == "deferred-read":
            raise TimeoutError("READ still owns its destination")
        state[0] = "DONE"

    monkeypatch.setattr(consumer._agent, "check_xfer_state", lambda handle: state[0], raising=False)
    monkeypatch.setattr(consumer, "_wait_for_transfer", wait)
    if outcome == "lost-metadata-finalizing":
        finish = consumer._finish_receive_transfer

        def finish_receive(transfer):
            entered.set()
            assert resume.wait(5)
            return finish(transfer)

        monkeypatch.setattr(consumer, "_finish_receive_transfer", finish_receive)
    messages: list[bytes] = []

    def receive():
        reply = producer._handle_handshake_message(messages[-1])
        if outcome.startswith("lost-metadata") and len(messages) == 1:
            if outcome == "lost-metadata":
                entered.set()
                assert resume.wait(5)
            raise zmq.Again()
        return reply

    socket = types.SimpleNamespace(send=messages.append, recv=receive)
    monkeypatch.setattr(consumer, "_get_req_socket", lambda *args: socket)
    success, _, metadata = producer.put("0", "1", "close-active_0_0", torch.arange(4))
    assert success
    worker = threading.Thread(target=lambda: received.append(consumer.get("0", "1", "close-active_0_0", metadata)))
    worker.start()
    try:
        assert entered.wait(5)
        context = consumer._zmq_ctx
        consumer.close()
        assert consumer._agent is agent
        assert consumer._close_thread is not None
        assert consumer._zmq_ctx is context and not context.closed
        assert consumer in _RETAINED_PRODUCERS
        assert consumer.health()["status"] == "unhealthy"
        with pytest.raises(RuntimeError, match="closed"):
            consumer.get("0", "1", "not-admitted")
        resume.set()
        worker.join(5)
        assert not worker.is_alive()
        if outcome == "deferred-read":
            assert received == [None]
            assert consumer._deferred_transfers
            assert producer._pending
            state[0] = "DONE"
        elif outcome == "completed-read":
            assert received[0] is not None
            torch.testing.assert_close(received[0][0], torch.arange(4), rtol=0, atol=0)
        else:
            assert received == [None]
        consumer._close_thread.join(5)
        assert not consumer._close_thread.is_alive()
        assert consumer._closed and consumer._zmq_ctx is None
        assert not consumer._active_gets and not consumer._deferred_transfers
        assert not agent.registered
        assert consumer._agent is None
        assert consumer not in _RETAINED_PRODUCERS
        assert not producer._pending and not producer._registered_descs
    finally:
        resume.set()
        worker.join(5)
        state[0] = "DONE"
        consumer._reap_deferred_transfers()
        consumer.close()


@pytest.mark.parametrize("defer_second", [False, True], ids=["completed", "deferred"])
@pytest.mark.usefixtures("reliable_claim_queries")
def test_overlapping_gets_keep_shared_remote_agent_until_last_release(
    producer, consumer, ownership_copying_agent, monkeypatch, defer_second
):
    first_ready = threading.Event()
    second_ready = threading.Event()
    resume = threading.Event()
    drained = threading.Event()
    second_handle: list[Any] = []
    received: dict[str, tuple[torch.Tensor, int] | None] = {}
    removed: list[str] = []
    wait = consumer._wait_for_transfer
    transfer = consumer._agent.transfer

    def wait_first(handle, key):
        if key == "overlap_0_1" and defer_second:
            raise TimeoutError("The second READ still owns its remote agent")
        wait(handle, key)
        if key == "overlap_0_0":
            first_ready.set()
            assert second_ready.wait(5)

    def post(handle):
        if threading.current_thread() is second:
            second_handle.append(handle)
            second_ready.set()
            assert resume.wait(5)
        transfer(handle)

    monkeypatch.setattr(consumer, "_wait_for_transfer", wait_first)
    monkeypatch.setattr(consumer._agent, "transfer", post)
    monkeypatch.setattr(consumer._agent, "remove_remote_agent", removed.append)
    monkeypatch.setattr(
        consumer._agent,
        "check_xfer_state",
        lambda handle: (
            "PROC" if defer_second and second_handle and handle is second_handle[0] and not drained.is_set() else "DONE"
        ),
    )
    for index in range(2):
        assert producer.put("0", "1", f"overlap_0_{index}", torch.arange(4) + index)[0]
    assert producer._pending["overlap_0_0"].agent is producer._pending["overlap_0_1"].agent
    first = threading.Thread(target=lambda: received.update(first=consumer.get("0", "1", "overlap_0_0")))
    second = threading.Thread(target=lambda: received.update(second=consumer.get("0", "1", "overlap_0_1")))
    first.start()
    try:
        assert first_ready.wait(5)
        second.start()
        assert second_ready.wait(5)
        first.join(5)
        assert not first.is_alive()
        assert received["first"] is not None
        assert removed == [], "One GET disconnected the remote agent still owned by another GET"
        resume.set()
        second.join(5)
        assert not second.is_alive()
        torch.testing.assert_close(received["first"][0], torch.arange(4), rtol=0, atol=0)
        if defer_second:
            assert received["second"] is None
            assert removed == [] and consumer._deferred_transfers
            assert producer._pending["overlap_0_1"].claims
            drained.set()
            consumer.wait_for_cleanup("overlap_1_", receive_prefix="overlap_0_", deadline=time.monotonic() + 5)
        else:
            assert received["second"] is not None
            torch.testing.assert_close(received["second"][0], torch.arange(4) + 1, rtol=0, atol=0)
        assert removed == ["producer"]
        assert not producer._pending and not consumer._deferred_transfers
    finally:
        resume.set()
        drained.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)


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
    connector._remote_agents.append("producer")
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
    connector._remote_agents.append("producer")
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


def test_poll_failure_requires_handle_release_before_freeing_receive_storage(consumer, monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _DeferredTransfer

    consumer._stop_event.set()
    consumer._transfer_wakeup.set()
    consumer._transfer_thread.join(timeout=2)
    calls = []
    allow_release = False

    def poll(handle):
        raise RuntimeError("remote disconnected")

    def release(handle):
        if not allow_release:
            raise RuntimeError("cannot cancel yet")
        calls.append(("handle", handle))

    monkeypatch.setattr(consumer._agent, "check_xfer_state", poll, raising=False)
    monkeypatch.setattr(consumer._agent, "release_xfer_handle", release, raising=False)
    monkeypatch.setattr(
        consumer._agent, "release_dlist_handle", lambda handle: calls.append(("dlist", handle)), raising=False
    )
    monkeypatch.setattr(consumer._agent, "deregister_memory", lambda desc: calls.append(("registration", desc)))
    transfer = _DeferredTransfer(tensors=[torch.ones(4)], handles=["read"], dlists=["descs"], registrations=["reg"])
    consumer._defer_transfer(transfer)
    try:
        consumer._reap_deferred_transfers()
        assert calls == []
        assert transfer.handles == ["read"]
        assert transfer.tensors and transfer.registrations == ["reg"]
        allow_release = True
        consumer._reap_deferred_transfers()
        assert calls == [("handle", "read"), ("dlist", "descs"), ("registration", "reg")]
        assert not consumer._deferred_transfers and not transfer.tensors
    finally:
        allow_release = True
        monkeypatch.setattr(consumer._agent, "check_xfer_state", lambda handle: "DONE")
        consumer._reap_deferred_transfers()


@pytest.mark.parametrize(
    "operation", ["release_xfer_handle", "release_dlist_handle", "remove_remote_agent", "deregister_memory"]
)
@pytest.mark.usefixtures("reliable_claim_queries")
def test_completed_read_retains_failed_release_for_request_cleanup(
    producer, consumer, ownership_copying_agent, monkeypatch, operation
):
    consumer._stop_event.set()
    consumer._transfer_wakeup.set()
    consumer._transfer_thread.join(timeout=2)
    original = getattr(consumer._agent, operation)

    def fail(*args):
        raise RuntimeError("native release failed")

    monkeypatch.setattr(consumer._agent, operation, fail)
    assert producer.put("0", "2", "req_0_0", torch.arange(4, dtype=torch.int32))[0]
    try:
        result = consumer.get("0", "2", "req_0_0")
        assert result is not None
        torch.testing.assert_close(result[0], torch.arange(4, dtype=torch.int32))
        assert consumer._deferred_transfers
        assert consumer._deferred_transfers[0].tensors
        assert producer._pending["req_0_0"].claims
        with pytest.raises(TimeoutError, match="ownership still active"):
            consumer.wait_for_cleanup("req_2_", receive_prefix="req_0_", deadline=time.monotonic())
        monkeypatch.setattr(consumer._agent, operation, original)
        consumer._reap_deferred_transfers()
        assert not consumer._deferred_transfers
        assert not producer._pending
        assert not consumer._agent.registered
        consumer.wait_for_cleanup("req_2_", receive_prefix="req_0_", deadline=time.monotonic())
    finally:
        monkeypatch.setattr(consumer._agent, operation, original)
        consumer._reap_deferred_transfers()


@pytest.mark.parametrize("poll_raises", [False, True], ids=["stuck", "poll-error"])
def test_close_returns_without_releasing_active_dma(nixl_connector_cls, poll_raises):
    import threading

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
        _RETAINED_PRODUCERS,
        _DeferredTransfer,
    )

    connector = nixl_connector_cls({"role": "receiver"})
    released = []

    agent = connector._agent

    def check_state(handle):
        if poll_raises:
            raise RuntimeError("device unavailable")
        return "PROC"

    connector._agent.check_xfer_state = check_state

    def release(handle):
        if poll_raises:
            raise RuntimeError("cannot cancel active transfer")
        released.append(("handle", handle))

    connector._agent.release_xfer_handle = release
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
    connector._remote_agents.append("producer")
    connector._defer_transfer(transfer)
    closing = threading.Thread(target=connector.close, daemon=True)
    try:
        closing.start()
        closing.join(timeout=2.0)
        assert not closing.is_alive(), "close must not wait forever for DMA completion"
        assert released == []
        assert transfer in connector._deferred_transfers
        assert transfer.tensors
        assert connector._closing
        assert connector in _RETAINED_PRODUCERS
        assert connector.health()["status"] == "unhealthy"
        cleanup_thread = connector._close_thread
        assert cleanup_thread is not None
        connector.close()
        assert connector._close_thread is cleanup_thread
        agent.check_xfer_state = lambda handle: "DONE"
        agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
        cleanup_thread.join(timeout=2.0)
        assert not cleanup_thread.is_alive()
        assert connector._closed
        assert not connector._deferred_transfers
        assert not transfer.tensors
        assert connector not in _RETAINED_PRODUCERS
    finally:
        agent.check_xfer_state = lambda handle: "DONE"
        agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
        closing.join(timeout=2.0)
        connector.close()

    assert released == [
        ("handle", "transfer"),
        ("dlist", "local"),
        ("dlist", "remote"),
        ("agent", "producer"),
        ("registration", "registration"),
    ]
    assert not connector._deferred_transfers
    assert not transfer.tensors
    assert connector not in _RETAINED_PRODUCERS
    connector.close()
    assert len(released) == 5


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
    metadata: dict[str, Any] = {
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
    assert not producer._registered_descs
    assert _claim(producer, "req-mixed")
    assert len(producer._registered_descs) == 2


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


def test_partial_registration_failure_retains_storage_until_context_cleanup(producer, monkeypatch):
    monkeypatch.setattr(
        producer,
        "_resolve_memory_type",
        lambda tensor: "DRAM" if tensor.dtype == torch.uint8 else "VRAM",
    )
    agent = _FakeNixlAgent("registration-fault")
    monkeypatch.setattr(producer, "_agent_factory", lambda: agent)
    original_register = agent.register_memory

    def fail_second_group(descs, backends=None):
        if descs[2] == "VRAM":
            raise RuntimeError("registration failed")
        original_register(descs, backends=backends)

    monkeypatch.setattr(agent, "register_memory", fail_second_group)

    success, _, metadata = producer.put(
        "0",
        "1",
        "req-partial_0_0",
        {"meta": "value", "hidden": torch.ones(4, dtype=torch.float32)},
    )

    assert success and metadata is not None
    assert _claim(producer, "req-partial_0_0") is None
    assert not producer._pending["req-partial_0_0"].claims
    assert producer._registered_descs == agent.registered
    assert len(producer._registered_descs) == 1
    assert producer._pending["req-partial_0_0"].tensors
    assert "req-partial_0_0" not in producer._published
    producer.cleanup_prefix("req-partial_0_")
    assert not producer._pending
    assert not producer._registered_descs


def test_reusing_unclaimed_put_key_releases_previous_storage(producer):
    producer.put("0", "1", "req-reuse", torch.zeros(2))
    previous = weakref.ref(producer._pending["req-reuse"].tensors[0])

    producer.put("0", "1", "req-reuse", torch.ones(2))

    assert previous() is None
    assert not producer._registered_descs
    torch.testing.assert_close(producer._pending["req-reuse"].tensors[0], torch.ones(2))


def test_expired_snapshot_cannot_claim_replacement(producer):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _PendingPayload

    old = _PendingPayload([torch.zeros(1)], ["old"], producer._agent, 0.0)
    replacement = _PendingPayload([torch.ones(1)], ["new"], producer._agent, time.monotonic() + 60)
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
    calls: dict[str, list[Any]] = {"agents": [], "descs": [], "transfers": [], "released": []}

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
@pytest.mark.usefixtures("reliable_claim_queries")
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
    assert producer._registered_descs == []
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
@pytest.mark.usefixtures("reliable_claim_queries")
def test_invalid_empty_tensor_metadata_rejected_before_native_calls(
    producer, consumer, copying_native_agent, defect, monkeypatch
):
    success, _, metadata = producer.put(
        "0", "1", "req-invalid-empty", [torch.empty((0, 2)), torch.tensor(3.0), torch.empty((1, 0))]
    )
    assert success
    # Corrupt the received wire response, not the producer's internal state.
    metadata = msgspec.msgpack.decode(msgspec.msgpack.encode(_claim(producer, "req-invalid-empty")))
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
    monkeypatch.setattr(consumer, "_query_metadata_at", lambda *args, **kwargs: metadata)
    assert consumer.get("0", "1", "req-invalid-empty", metadata) is None
    assert copying_native_agent == {"agents": [], "descs": [], "transfers": [], "released": []}
    assert consumer._agent.registered == []
    # Metadata acquisition now claims the source before validation. With no
    # submitted DMA, failure safely completes that claim instead of leaking it;
    # it must still report no successful payload receive.
    assert producer._pending == producer._published == {}
    assert producer._registered_descs == []
    assert consumer._metrics["gets"] == 0
    assert consumer._metrics["errors"] == 1
