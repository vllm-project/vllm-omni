# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import multiprocessing as mp
import os
import socket
import time
import weakref
from contextlib import closing
from dataclasses import fields
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.data_entry_keys import EmbeddingsStruct, HiddenStatesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.parallel]


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
    receiver.get("0", "2", "reader-loss_0_0", metadata)
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
        lost_agent = weakref.ref(producer._pending["reader-loss_0_0"].agent)
        survivor_agent = producer._pending["survivor_0_0"].agent
        assert lost_agent() is not survivor_agent

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
        assert producer.cleanup_prefix("reader-loss_0_") == 1
        assert source_ref() is None and lost_agent() is None
        assert producer._pending["survivor_0_0"].agent is survivor_agent
        success, _, reused_metadata = producer.put("0", "2", "reader-loss_0_0", torch.arange(32, device="cuda:0"))
        assert success

        with closing(NixlConnector({"role": "receiver", "receive_device": "cuda:0"})) as receiver:
            for key, wire in (("survivor_0_0", other_metadata), ("reader-loss_0_0", reused_metadata)):
                received = receiver.get("0", "2", key, wire)
                assert received is not None
                torch.testing.assert_close(received[0].cpu(), torch.arange(32), rtol=0, atol=0)
        assert not producer._pending and not producer._registered_descs


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not _native_nixl_available(1), reason="requires NIXL and CUDA")
def test_native_publication_waits_for_cuda_producer_stream():
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    current_omni_platform.set_device(0)
    with closing(NixlConnector({"role": "sender", "host": "127.0.0.1"})) as producer:
        payload = torch.zeros(32, device="cuda:0")
        torch.accelerator.synchronize()
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            payload.fill_(37)
        success, _, metadata = producer.put("0", "2", "delayed_0_0", payload)
        assert success
        assert stream.query(), "Publication returned before producer work completed"
        with closing(NixlConnector({"role": "receiver", "receive_device": "cuda:0"})) as receiver:
            received = receiver.get("0", "2", "delayed_0_0", metadata)
            assert received is not None
            torch.testing.assert_close(received[0].cpu(), torch.full((32,), 37.0), rtol=0, atol=0)
        assert not producer._pending and not producer._registered_descs
