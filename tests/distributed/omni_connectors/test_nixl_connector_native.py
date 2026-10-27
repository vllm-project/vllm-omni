# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import multiprocessing as mp
import os
import socket
import threading
import time
import weakref
from contextlib import closing
from dataclasses import fields
from typing import Any

import msgspec
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.data_entry_keys import EmbeddingsStruct, HiddenStatesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.parallel]


def _native_nixl_available(min_devices: int = 2) -> bool:
    try:
        from vllm.distributed.nixl_utils import NixlWrapper
    except ImportError:
        return False
    return NixlWrapper is not None and torch.cuda.is_available() and torch.accelerator.device_count() >= min_devices


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_payload(receiver, key, metadata):
    """Poll the bounded claim API, including first-claim native setup."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = receiver.get("0", "2", key, metadata)
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail(f"No native payload for {key}")


def _h3_conditioning(payload_kind: str, device: str = "cpu"):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    media = {}
    if payload_kind == "h3_ref2va":
        media = {
            "visual_condition": torch.arange(5 * 96, dtype=torch.float32).reshape(5, 96) / 8,
            "visual_condition_shapes": ((1, 2, 2), (2, 2, 4)),
            "audio_condition": torch.arange(10 * 32, dtype=torch.float32, device=device).reshape(10, 32) / 16,
            "audio_condition_lengths": (2, 3),
            "ref_blocks": (
                {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 2},
                {"kind": "video_audio", "ref_audio_t": 3, "latent_t": 2, "latent_h": 2, "latent_w": 4},
            ),
        }
    elif payload_kind == "h3_fl2va":
        media = {
            "visual_condition": torch.arange(2 * 96, dtype=torch.float32).reshape(2, 96) / 8,
            "visual_condition_shapes": ((1, 2, 2), (1, 2, 2)),
            "keyframe_frame_indices": (0, 16),
        }
    return MiniMaxH3EncoderConditioning(
        hidden_states=(torch.arange(3 * 5120, device=device).reshape(3, 5120) % 127).to(torch.bfloat16),
        token_tags=torch.tensor([1, 0, 1], dtype=torch.int64),
        task=payload_kind.removeprefix("h3_"),
        height=256,
        width=448,
        num_frames=17,
        latent_t=5,
        audio_t=10,
        **media,
    )


def _h3_structured_payload(payload_kind: str) -> OmniPayloadStruct:
    # Only adapt to the existing public Struct; NIXL owns all serialization.
    wire = _h3_conditioning(payload_kind, "cuda:0").to_omni_payload()
    # Exercise both CPU and CUDA empty leaves (T2VA has two empty slots).
    wire["embed"]["speech_feat"] = wire["embed"]["speech_feat"].to("cuda:0")
    return OmniPayloadStruct(
        hidden_states=HiddenStatesStruct(**wire["hidden_states"]),
        meta=MetaStruct(**wire["meta"]),
        embed=EmbeddingsStruct(**wire["embed"]),
        kv_metadata=wire["kv_metadata"],
        request_id="native-smoke",
    )


def _assert_h3_roundtrip(payload: dict[str, Any], payload_kind: str) -> tuple:
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    expected = _h3_conditioning(payload_kind)
    expected_wire = expected.to_omni_payload()
    tensors = []
    for section, values in expected_wire.items():
        for key, reference in values.items():
            actual = payload[section][key]
            assert actual.device == torch.device("cuda:1"), (section, key, actual.device)
            # Exact values, shapes and dtypes, including empty wire slots and layout.
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            tensors.append((f"{section}.{key}", str(actual.dtype), list(actual.shape), str(actual.device)))
    actual = MiniMaxH3EncoderConditioning.from_omni_payload(payload)
    for field in fields(expected):
        reference = getattr(expected, field.name)
        value = getattr(actual, field.name)
        if isinstance(reference, torch.Tensor):
            assert value.device == torch.device("cuda:1"), field.name
            torch.testing.assert_close(value.cpu(), reference, rtol=0, atol=0)
        else:
            assert value == reference, field.name
    assert payload["request_id"] == "native-smoke"
    return (payload_kind, tensors)


def _producer(
    port: int, ready: Any, consumed: Any, result: Any, claimed: Any, proceed: Any, wire: Any, payload_kind: str
) -> None:
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    connector = NixlConnector(
        {
            "role": "sender",
            "host": "127.0.0.1",
            "zmq_port": port,
            "agent_name": "native-smoke-producer",
        }
    )
    try:
        payload = OmniPayloadStruct(
            hidden_states=HiddenStatesStruct(output=torch.arange(8, dtype=torch.float32, device="cuda:0")),
            meta=MetaStruct(token_role_ids=torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")),
            kv_metadata={"empty": torch.empty((2, 0, 3), device="cuda:0")},
            request_id="native-smoke",
        )
        if payload_kind != "generic":
            payload = _h3_structured_payload(payload_kind)
        success, size, metadata = connector.put("0", "1", "native-smoke", payload)
        result.put(("put", success, size, metadata["kind"] if metadata else None))
        wire.put(metadata)
        ready.set()
        if not claimed.wait(timeout=30):
            raise TimeoutError("consumer did not claim source")
        pending = connector._pending["native-smoke"]
        pending.deadline = 0
        connector._cleanup_expired_pending()
        connector.cleanup("native-smoke")
        assert connector._pending["native-smoke"] is pending
        assert pending.claims and connector._registered_descs
        result.put(("source_retained", len(pending.claims)))
        proceed.set()
        if not consumed.wait(timeout=30):
            raise TimeoutError("consumer did not complete the native NIXL transfer")
        deadline = time.monotonic() + 10
        while connector._pending and time.monotonic() < deadline:
            time.sleep(0.01)
        result.put(("cleanup", len(connector._pending), len(connector._registered_descs)))
    except Exception as error:
        result.put(("producer_error", repr(error)))
        raise
    finally:
        connector.close()


def _consumer(
    port: int,
    ready: Any,
    consumed: Any,
    result: Any,
    claimed: Any,
    proceed: Any,
    wire: Any,
    direct: bool,
    payload_kind: str,
) -> None:
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVCacheConfig,
        OmniKVTransferManager,
    )

    if not ready.wait(timeout=30):
        raise TimeoutError("producer did not publish native NIXL metadata")
    current_omni_platform.set_device(1)
    manager = OmniKVTransferManager(
        OmniKVCacheConfig(
            connector_config={
                "type": "NixlConnector",
                "role": "receiver",
                "host": "127.0.0.1",
                "zmq_port": port,  # Shared deployment edge: already bound by producer.
                "backends": ["UCX"],
                "receive_device": "cuda",
                "agent_name": "native-smoke-consumer",
            },
            from_stage="0",
            to_stage="1",
            stage_id=1,
            need_recv_cache=direct,  # Exercise both payload-only and KV receiver setup.
        )
    )
    manager.update_sender_info({"host": "127.0.0.1", "zmq_port": port})
    connector = manager.connector
    assert connector is not None, "Receiver must not bind the producer's occupied port"
    assert connector._zmq_port is None and not connector._serving_handshake
    assert connector._backends == ["UCX"]
    try:
        original_wait = connector._wait_for_transfer
        states = []

        def wait(handle, key):
            states.append(connector._agent.check_xfer_state(handle))
            claimed.set()
            if not proceed.wait(timeout=30):
                raise TimeoutError("producer did not verify active ownership")
            original_wait(handle, key)

        connector._wait_for_transfer = wait
        metadata = wire.get(timeout=5)
        received = None
        deadline = time.monotonic() + 30
        while received is None and time.monotonic() < deadline:
            received = connector.get(
                "0",
                "1",
                "native-smoke",
                metadata if direct else {"source_host": "127.0.0.1", "source_port": port},
            )
        if received is None:
            raise TimeoutError("native NIXL transfer did not complete")
        payload, size = received
        result.put(("native_states", states))
        if payload_kind != "generic":
            result.put(("get", size, *_assert_h3_roundtrip(payload, payload_kind)))
            consumed.set()
            return
        hidden = payload["hidden_states"]["output"]
        token_roles = payload["meta"]["token_role_ids"]
        empty = payload["kv_metadata"]["empty"]
        assert empty.shape == (2, 0, 3) and str(empty.device) == "cuda:1"
        result.put(
            (
                "get",
                size,
                hidden.cpu().tolist(),
                str(hidden.dtype),
                list(hidden.shape),
                str(hidden.device),
                token_roles.cpu().tolist(),
                str(token_roles.dtype),
                list(token_roles.shape),
                str(token_roles.device),
            )
        )
        consumed.set()
    except Exception as error:
        result.put(("consumer_error", repr(error)))
        raise
    finally:
        connector.close()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.skipif(not _native_nixl_available(), reason="requires NIXL and at least two CUDA devices")
@pytest.mark.parametrize("payload_kind", ["generic", "h3_ref2va", "h3_fl2va", "h3_t2va"])
@pytest.mark.parametrize("direct", [False, True])
def test_native_two_process_structured_mixed_device_transfer(direct, payload_kind):
    context = mp.get_context("spawn")
    ready = context.Event()
    consumed = context.Event()
    claimed = context.Event()
    proceed = context.Event()
    wire = context.Queue()
    result = context.Queue()
    port = _free_port()
    producer = context.Process(
        target=_producer, args=(port, ready, consumed, result, claimed, proceed, wire, payload_kind)
    )
    consumer = context.Process(
        target=_consumer, args=(port, ready, consumed, result, claimed, proceed, wire, direct, payload_kind)
    )

    producer.start()
    consumer.start()
    producer.join(timeout=60)
    consumer.join(timeout=60)
    if producer.is_alive():
        producer.terminate()
        producer.join(timeout=10)
    if consumer.is_alive():
        consumer.terminate()
        consumer.join(timeout=10)

    records = [result.get(timeout=5) for _ in range(5)]
    print("Native ownership records:", records)
    assert producer.exitcode == 0, records
    assert consumer.exitcode == 0, records
    assert records[0][0] == "put"
    assert records[0][1] is True
    assert records[0][2] >= 56
    assert records[0][3] == "structured"
    get_record = next(record for record in records if record[0] == "get")
    assert get_record[1] == records[0][2]
    if payload_kind == "generic":
        assert get_record[2:6] == (
            list(range(8)),
            "torch.float32",
            [8],
            "cuda:1",
        )
        assert get_record[6:] == ([1, 2, 3], "torch.int64", [3], "cuda:1")
    else:
        assert get_record[2] == payload_kind
        assert len(get_record[3]) == 5
        print(f"Native H3 exact wire/semantic roundtrip: {get_record}")
    assert next(record for record in records if record[0] == "cleanup") == ("cleanup", 0, 0)
    assert next(record for record in records if record[0] == "source_retained") == ("source_retained", 1)


def _exit_after_native_read(metadata, status) -> None:
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    receiver = NixlConnector({"role": "receiver", "receive_device": "cuda:0"})

    def exit_without_releasing(handle, key):
        status.send(receiver._agent.check_xfer_state(handle))
        # Only this test-owned process exits. Leave its native READ unacknowledged.
        os._exit(88)

    receiver._wait_for_transfer = exit_without_releasing
    _get_payload(receiver, "reader-loss_0_0", metadata)
    raise AssertionError("The receiver did not submit a native READ")


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_reader_exit_reclaims_only_its_prefix_and_allows_key_reuse():
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        payload = torch.ones(8 * 1024 * 1024, device="cuda:0")
        source_ref = weakref.ref(payload)
        success, _, metadata = producer.put("0", "2", "reader-loss_0_0", payload)
        assert success
        del payload
        success, _, other_metadata = producer.put("0", "2", "survivor_0_0", torch.arange(32, device="cuda:0"))
        assert success
        assert producer._pending["survivor_0_0"].agent is None

        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        reader = context.Process(target=_exit_after_native_read, args=(metadata, child))
        reader.start()
        child.close()
        try:
            assert parent.poll(30), "The receiver did not report its native READ"
            assert parent.recv() in {"PROC", "DONE"}
            reader.join(timeout=30)
            assert reader.exitcode == 88
        finally:
            parent.close()
            if reader.is_alive():
                reader.terminate()
                reader.join(timeout=10)
            reader.close()

        assert producer._pending["reader-loss_0_0"].claims
        lost_agent = weakref.ref(producer._pending["reader-loss_0_0"].agent)
        assert producer.cleanup_prefix("reader-loss_0_") == 1
        assert source_ref() is None and lost_agent() is None
        assert producer._pending["survivor_0_0"].agent is None
        success, _, reused_metadata = producer.put("0", "2", "reader-loss_0_0", torch.arange(32, device="cuda:0"))
        assert success

        with closing(NixlConnector({"role": "receiver", "receive_device": "cuda:0"})) as receiver:
            for key, wire in (("survivor_0_0", other_metadata), ("reader-loss_0_0", reused_metadata)):
                received = _get_payload(receiver, key, wire)
                assert received is not None
                torch.testing.assert_close(received[0].cpu(), torch.arange(32), rtol=0, atol=0)
        assert not producer._pending and not producer._registered_descs


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("explicit_event", [False, True])
def test_native_publication_waits_for_cuda_producer_stream(strided, explicit_event):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        payload = torch.zeros(64 if strided else 32, device="cuda:0")
        if strided:
            payload = payload[::2]
            # Cold allocation can otherwise hide a missing writer fence.
            payload.contiguous()
        torch.accelerator.synchronize()
        stream = torch.cuda.Stream(device=0)
        ready = torch.cuda.Event()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            payload.fill_(37)
            ready.record()
        kwargs = {"ready_event": ready} if explicit_event else {}
        success, _, metadata = producer.put("0", "2", "delayed_0_0", payload, **kwargs)
        assert success
        assert stream.query(), "Publication returned before producer work completed"
        with closing(NixlConnector({"role": "receiver", "receive_device": "cuda:0"})) as receiver:
            received = _get_payload(receiver, "delayed_0_0", metadata)
            assert received is not None
            torch.testing.assert_close(received[0].cpu(), torch.full((32,), 37.0), rtol=0, atol=0)
        assert not producer._pending and not producer._registered_descs


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_ready_event_does_not_wait_for_unrelated_stream():
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        payload = torch.arange(32, device="cuda:0")
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()
        unrelated = torch.cuda.Stream()
        with torch.cuda.stream(unrelated):
            torch.cuda._sleep(500_000_000)
        assert producer.put("0", "2", "ready_0_0", payload, ready_event=ready)[0]
        assert not unrelated.query(), "A payload event must not fence independent CUDA work"
        unrelated.synchronize()
        with closing(NixlConnector({"role": "receiver", "receive_device": "cuda:0"})) as receiver:
            result = _get_payload(receiver, "ready_0_0", producer._published["ready_0_0"])
            assert result is not None
            torch.testing.assert_close(result[0].cpu(), torch.arange(32), rtol=0, atol=0)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_close_releases_agents_before_connector_gc():
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    producer = NixlConnector({"role": "sender", "host": "127.0.0.1"})
    receiver = NixlConnector({"role": "receiver", "receive_device": "cuda:0"})
    agents = (weakref.ref(producer._agent), weakref.ref(receiver._agent))
    try:
        expected = torch.arange(32, device="cuda:0")
        assert producer.put("0", "2", "close-owner_0_0", expected)[0]
        received = _get_payload(receiver, "close-owner_0_0", producer._published["close-owner_0_0"])
        assert received is not None
        torch.testing.assert_close(received[0], expected, rtol=0, atol=0)
    finally:
        receiver.close()
        producer.close()
    assert all(agent() is None for agent in agents)
    assert producer._closed and receiver._closed
    producer.close()
    receiver.close()


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_close_keeps_inflight_get_completion_ack(monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    entered = threading.Event()
    resume = threading.Event()
    received = []
    reader_config = {"role": "receiver", "receive_device": "cuda:0", "metadata_query_timeout_ms": 5000}
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        with closing(NixlConnector(reader_config)) as receiver:
            agent = weakref.ref(receiver._agent)
            wait = receiver._wait_for_transfer

            def observe(handle, key):
                wait(handle, key)
                entered.set()
                assert resume.wait(10), "The test did not release the completed native READ"

            monkeypatch.setattr(receiver, "_wait_for_transfer", observe)
            success, _, metadata = producer.put("0", "2", "close-live_0_0", torch.arange(32, device="cuda:0"))
            assert success
            worker = threading.Thread(
                target=lambda: received.append(receiver.get("0", "2", "close-live_0_0", metadata))
            )
            worker.start()
            try:
                assert entered.wait(30)
                assert producer._pending["close-live_0_0"].claims
                receiver.close()
                assert agent() is receiver._agent and agent() is not None
                assert receiver._close_thread is not None
                resume.set()
                worker.join(10)
                assert not worker.is_alive()
                assert received[0] is not None
                torch.testing.assert_close(received[0][0].cpu(), torch.arange(32), rtol=0, atol=0)
                receiver._close_thread.join(10)
                assert not receiver._close_thread.is_alive()
                assert receiver._closed and receiver._zmq_ctx is None
                assert agent() is None
                assert not receiver._active_gets and not receiver._deferred_transfers
                assert not producer._pending and not producer._registered_descs
            finally:
                resume.set()
                worker.join(10)
        success, _, replacement = producer.put("0", "2", "close-live_0_0", torch.arange(32, device="cuda:0") + 7)
        assert success and replacement["generation"] != metadata["generation"]
        with closing(NixlConnector(reader_config)) as reader:
            payload = reader.get("0", "2", "close-live_0_0", replacement)
            assert payload is not None
            torch.testing.assert_close(payload[0].cpu(), torch.arange(32) + 7, rtol=0, atol=0)
        assert not producer._pending and not producer._registered_descs


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_overlapping_gets_preserve_prepared_read(monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    first_ready = threading.Event()
    second_ready = threading.Event()
    resume = threading.Event()
    received: dict[str, tuple[torch.Tensor, int] | None] = {}
    reader_config = {"role": "receiver", "receive_device": "cuda:0", "metadata_query_timeout_ms": 5000}
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        reader_config.update(sender_host=producer.host, sender_zmq_port=producer._zmq_port)
        with closing(NixlConnector(reader_config)) as receiver:
            wait = receiver._wait_for_transfer
            transfer = receiver._agent.transfer

            def wait_first(handle, key):
                wait(handle, key)
                if key == "overlap_0_0":
                    first_ready.set()
                    assert second_ready.wait(10)

            def post(handle):
                if threading.current_thread() is second:
                    second_ready.set()
                    assert resume.wait(10)
                return transfer(handle)

            monkeypatch.setattr(receiver, "_wait_for_transfer", wait_first)
            monkeypatch.setattr(receiver._agent, "transfer", post)
            for index in range(2):
                assert producer.put("0", "2", f"overlap_0_{index}", torch.arange(32, device="cuda:0") + index)[0]
            first = threading.Thread(target=lambda: received.update(first=receiver.get("0", "2", "overlap_0_0")))
            second = threading.Thread(target=lambda: received.update(second=receiver.get("0", "2", "overlap_0_1")))
            first.start()
            try:
                assert first_ready.wait(30)
                second.start()
                assert second_ready.wait(10)
                first.join(10)
                assert not first.is_alive() and received["first"] is not None
                resume.set()
                second.join(10)
                assert not second.is_alive()
                for index, name in enumerate(("first", "second")):
                    result = received[name]
                    assert result is not None, f"The overlapping {name} READ was invalidated"
                    torch.testing.assert_close(result[0].cpu(), torch.arange(32) + index, rtol=0, atol=0)
                assert not producer._pending and not producer._registered_descs
                assert not receiver._deferred_transfers and not receiver._active_gets
            finally:
                resume.set()
                first.join(10)
                if second.ident is not None:
                    second.join(10)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_final_ack_retries_deregistration_failure(monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
        _ACK,
        _GET_META_MSG,
        _XFER_DONE_MSG,
        NixlConnector,
    )

    current_omni_platform.set_device(0)
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        assert producer.put("0", "2", "retry_0_0", torch.ones(32, device="cuda:0"))[0]
        claim = {"key": "retry_0_0", "claim_id": "reader"}
        metadata = msgspec.msgpack.decode(
            producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(claim))
        )
        claim["generation"] = metadata["generation"]
        pending = producer._pending["retry_0_0"]
        original_release = pending.agent.deregister_memory
        attempts = []

        def release(descs):
            attempts.append(descs)
            if len(attempts) == 1:
                raise RuntimeError("injected native deregistration failure")
            original_release(descs)

        monkeypatch.setattr(pending.agent, "deregister_memory", release)
        completion = _XFER_DONE_MSG + msgspec.msgpack.encode(claim)
        with pytest.raises(RuntimeError, match="injected native deregistration failure"):
            producer._handle_handshake_message(completion)
        assert pending.tensors and pending.registrations
        assert "retry_0_0" not in producer._published
        assert producer._handle_handshake_message(completion) == _ACK
        assert len(attempts) == 2
        assert not producer._pending and not producer._registered_descs
        assert producer._handle_handshake_message(completion) == _ACK
        assert len(attempts) == 2


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_context_retirement_allows_other_requests(monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
        _GET_META_MSG,
        _XFER_DONE_MSG,
        NixlConnector,
    )

    current_omni_platform.set_device(0)
    entered = threading.Event()
    retire = threading.Event()
    finished = threading.Event()
    errors = []
    storage_at_retirement = []

    class GatedOwner:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __del__(self):
            storage_at_retirement.append(source_ref() is not None)
            entered.set()
            retire.wait(10)
            storage_at_retirement.append(source_ref() is not None)

    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        factory = producer._agent_factory
        monkeypatch.setattr(producer, "_agent_factory", lambda: GatedOwner(factory()))
        assert producer.put("0", "2", "blocked_0_0", torch.ones(32, device="cuda:0"))[0]
        claim = {"key": "blocked_0_0", "claim_id": "blocked-reader"}
        assert producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(claim))
        source_ref = weakref.ref(producer._pending["blocked_0_0"].tensors[0])
        monkeypatch.setattr(producer, "_agent_factory", factory)
        other_payload = torch.arange(32, device="cuda:0")
        assert producer.put("0", "2", "other_0_0", other_payload)[0]

        def independent():
            try:
                current_omni_platform.set_device(0)
                assert producer.put("0", "2", "other_0_1", other_payload)[0]
                for key in ("other_0_0", "other_0_1"):
                    claim = {"key": key, "claim_id": "other-reader"}
                    metadata = msgspec.msgpack.decode(
                        producer._handle_handshake_message(_GET_META_MSG + msgspec.msgpack.encode(claim))
                    )
                    claim["generation"] = metadata["generation"]
                    producer._handle_handshake_message(_XFER_DONE_MSG + msgspec.msgpack.encode(claim))
                assert "other_0_0" not in producer._pending and "other_0_1" not in producer._pending
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        cleanup = threading.Thread(target=producer.cleanup_prefix, args=("blocked_0_",))
        other = threading.Thread(target=independent)
        cleanup.start()
        try:
            assert entered.wait(5)
            other.start()
            assert finished.wait(2), "Native retirement blocked independent publication, GET or ACK"
            assert not errors
            assert source_ref() is not None
        finally:
            retire.set()
            cleanup.join(10)
            if other.ident is not None:
                other.join(10)
        assert not cleanup.is_alive() and not other.is_alive()
        assert storage_at_retirement == [True, True]
        assert source_ref() is None
        assert not producer._pending and not producer._registered_descs


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_expired_release_failure_is_retried(monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
        _GET_META_MSG,
        _META_NOT_FOUND,
        NixlConnector,
    )

    current_omni_platform.set_device(0)
    observed = threading.Event()
    recover = threading.Event()
    retried = threading.Event()
    attempts = []
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        factory = producer._agent_factory

        def fail_metadata():
            raise RuntimeError("injected native metadata failure")

        def create():
            agent = factory()
            monkeypatch.setattr(agent, "get_agent_metadata", fail_metadata)
            return agent

        monkeypatch.setattr(producer, "_agent_factory", create)
        assert producer.put("0", "2", "expired_0_0", torch.ones(32, device="cuda:0"))[0]
        assert (
            producer._handle_handshake_message(
                _GET_META_MSG + msgspec.msgpack.encode({"key": "expired_0_0", "claim_id": "reader"})
            )
            == _META_NOT_FOUND
        )
        pending = producer._pending["expired_0_0"]
        original_release = pending.agent.deregister_memory

        def release(descs):
            attempts.append(time.monotonic())
            observed.set()
            if len(attempts) >= 3:
                retried.set()
            if not recover.is_set():
                raise RuntimeError("injected native deregistration failure")
            original_release(descs)

        monkeypatch.setattr(pending.agent, "deregister_memory", release)
        pending.deadline = 0
        producer._lease_wakeup.set()
        try:
            assert observed.wait(5)
            assert producer.put("0", "2", "independent_0_0", torch.ones(32, device="cuda:0"))[0]
            assert retried.wait(5)
            assert producer._lease_thread.is_alive()
            assert all(b - a >= 0.05 for a, b in zip(attempts[:2], attempts[1:3]))
            assert pending.tensors and pending.registrations
            recover.set()
            deadline = time.monotonic() + 5
            while "expired_0_0" in producer._pending and time.monotonic() < deadline:
                time.sleep(0.01)
            assert "expired_0_0" not in producer._pending
            assert producer._lease_thread.is_alive()
        finally:
            recover.set()
            producer.cleanup("expired_0_0")
        producer.cleanup("independent_0_0")
        assert not producer._pending and not producer._registered_descs


def _receive_during_prefix_retirement(metadata, status, inspect_read, resume_read):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(1)
    # This probe gates DMA retirement, not the scheduler's bounded polling API.
    config = {"role": "receiver", "receive_device": "cuda:1", "metadata_query_timeout_ms": 5000}
    with closing(NixlConnector(config)) as receiver:
        wait = receiver._wait_for_transfer

        def observe(handle, key):
            status.send(("submitted", receiver._agent.check_xfer_state(handle)))
            assert inspect_read.wait(20)
            status.send(("before_cancel", receiver._agent.check_xfer_state(handle)))
            assert resume_read.wait(20)
            return wait(handle, key)

        receiver._wait_for_transfer = observe
        result = receiver.get("0", "2", "live_0_0", metadata)
        receiver._wait_for_transfer = wait
        if result is not None:
            assert bool((result[0] == 0.25).all()), "A drained READ observed reclaimed source storage"
        status.send(("read_result", "drained" if result is not None else "aborted"))
        for key, wire, offset in status.recv():
            received = _get_payload(receiver, key, wire)
            assert received is not None
            torch.testing.assert_close(received[0].cpu(), torch.arange(32) + offset, rtol=0, atol=0)
        receiver.wait_for_cleanup("live_2_", receive_prefix="live_0_", deadline=time.monotonic() + 10)
        assert not receiver._active_gets and not receiver._deferred_transfers and not receiver._abandoned_queries
        status.send(("finished", True))


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["H100"]}, num_cards=2)
@pytest.mark.skipif(not _native_nixl_available(), reason="requires NIXL and two CUDA devices")
def test_native_prefix_cleanup_while_read_is_proc_preserves_survivor_and_key_reuse():
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    context = mp.get_context("spawn")
    inspect_read = context.Event()
    resume_read = context.Event()
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        # A large actual READ provides a window for observing PROC. A completed
        # transfer is explicitly skipped rather than counted as cancellation.
        tensor = torch.full((128 * 1024 * 1024,), 0.25, device="cuda:0")
        success, _, metadata = producer.put("0", "2", "live_0_0", tensor)
        assert success
        del tensor
        source_ref = weakref.ref(producer._pending["live_0_0"].tensors[0])
        success, _, survivor = producer.put("0", "2", "survivor_0_0", torch.arange(32, device="cuda:0"))
        assert success
        parent, child = context.Pipe()
        reader = context.Process(
            target=_receive_during_prefix_retirement, args=(metadata, child, inspect_read, resume_read)
        )
        reader.start()
        child.close()
        try:
            assert parent.poll(30), "The receiver did not submit its READ"
            phase, state = parent.recv()
            assert phase == "submitted" and state in {"PROC", "DONE"}
            inspect_read.set()
            assert parent.poll(10)
            phase, state = parent.recv()
            assert phase == "before_cancel" and state in {"PROC", "DONE"}
            if state != "PROC":
                resume_read.set()
                assert parent.poll(20) and parent.recv()[0] == "read_result"
                parent.send([("survivor_0_0", survivor, 0)])
                assert parent.poll(20) and parent.recv() == ("finished", True)
                reader.join(20)
                assert reader.exitcode == 0
                pytest.skip("Native READ finished before cancellation; active READ was not covered")
            assert producer._pending["live_0_0"].claims
            assert producer.cleanup_prefix("live_0_") == 1
            assert source_ref() is None
            assert "live_0_0" not in producer._pending
            assert "survivor_0_0" in producer._pending
            success, _, reused = producer.put("0", "2", "live_0_0", torch.arange(32, device="cuda:0") + 7)
            assert success and reused["generation"] != metadata["generation"]
            resume_read.set()
            assert parent.poll(30) and parent.recv()[0] == "read_result"
            parent.send([("survivor_0_0", survivor, 0), ("live_0_0", reused, 7)])
            assert parent.poll(30) and parent.recv() == ("finished", True)
            reader.join(30)
            assert reader.exitcode == 0
            assert not producer._pending and not producer._registered_descs
        finally:
            inspect_read.set()
            resume_read.set()
            parent.close()
            if reader.is_alive():
                reader.terminate()
                reader.join(10)
            reader.close()
