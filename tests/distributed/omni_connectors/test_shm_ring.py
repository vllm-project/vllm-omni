# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-process ownership, pressure and compatibility of the default host path."""

import fcntl
import multiprocessing
import os
import time
import uuid
from dataclasses import asdict
from multiprocessing import shared_memory
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch
from PIL import Image
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.distributed.omni_connectors.utils.tensor_frame import prepare_tensor_frame

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def edge():
    scope = uuid.uuid4().hex
    sender = SharedMemoryConnector({"stage_id": 0, "wakeup_scope": scope, "host_ring_bytes": 4096})
    receiver = SharedMemoryConnector({"stage_id": 1, "wakeup_scope": scope, "host_ring_bytes": 4096})
    yield sender, receiver
    receiver.close()
    sender.close()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
    ],
)
def test_owned_tensor_tree_preserves_values_shapes_and_types(edge, dtype):
    sender, receiver = edge
    tensor = torch.arange(24).to(dtype).reshape(4, 6).T
    data = {
        "codes": {"audio": tensor},
        "other": (tensor[0], [torch.empty(0, dtype=dtype), torch.tensor(1)]),
        "__tensor__": False,
        "meta": {"finished": torch.tensor(False)},
    }
    ok, size, metadata = sender.put("0", "1", "tree", data)
    assert ok and "host_ring" in metadata
    output, read_size = receiver.get("0", "1", "tree", metadata)
    assert read_size == size and output["__tensor__"] is False
    assert type(output["other"]) is list and type(output["other"][1]) is list
    assert output["other"][1][0].shape == (0,)
    assert output["other"][1][1].shape == ()
    assert output["codes"]["audio"].dtype == dtype
    assert torch.equal(output["codes"]["audio"], tensor)
    tensor.fill_(0)
    assert torch.count_nonzero(output["codes"]["audio"]) > 0
    for index in range(100):
        sender.put("0", "1", f"wrap{index}", {"codes": torch.full((56,), index)})
        assert receiver.get("0", "1", f"wrap{index}")[0]["codes"][0] == index
    assert torch.count_nonzero(output["codes"]["audio"]) > 0


def test_conjugate_tensor_and_scalar_finish(edge):
    sender, receiver = edge
    tensor = torch.tensor([1 + 2j, 3 - 4j]).conj()
    sender.put("0", "1", "conjugate", {"condition": tensor, "finished": torch.tensor(True)})
    result = receiver.get("0", "1", "conjugate")[0]
    assert torch.equal(result["condition"], tensor)
    assert result["finished"].shape == () and result["finished"].item() is True


def test_shared_typed_payload_keeps_omitted_defaults_and_cpu_tensors(edge):
    sender, receiver = edge
    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.arange(28).reshape(1, 28)),
        meta=MetaStruct(finished=torch.tensor(False), left_context_size=0),
        kv_metadata={"binary_handle": b"\x00\xff\x10"},
    )
    ok, _, metadata = sender.put("0", "1", "typed", payload)
    assert ok and "host_ring" in metadata
    result = receiver.get("0", "1", "typed")[0]
    assert set(result) == {"codes", "meta", "kv_metadata"}
    assert set(result["codes"]) == {"audio"}
    assert torch.equal(result["codes"]["audio"], payload.codes.audio)
    assert result["meta"]["left_context_size"] == 0
    assert result["kv_metadata"]["binary_handle"] == b"\x00\xff\x10"


def test_source_can_be_overwritten_as_soon_as_put_returns(edge):
    sender, receiver = edge
    source = torch.arange(56)
    sender.put("0", "1", "source-reuse", source)
    source.fill_(-1)
    assert receiver.get("0", "1", "source-reuse")[0].tolist() == list(range(56))


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float16, np.float64, np.complex64])
def test_mixed_numpy_image_and_tensor_keep_shared_wire_semantics(edge, dtype):
    sender, receiver = edge
    array = np.arange(24).astype(dtype).reshape(4, 6)[::-1, ::2]
    image = Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    payload = {"codes": torch.arange(28), "array": array, "image": image, "slice": slice(1, 7, 2)}
    _, _, metadata = sender.put("0", "1", "mixed-native", payload)
    assert "host_ring" in metadata
    output = receiver.get("0", "1", "mixed-native", metadata)[0]
    assert np.array_equal(output["array"], array)
    assert output["array"].dtype == array.dtype
    assert not output["array"].flags.writeable
    assert output["image"].mode == image.mode and output["image"].tobytes() == image.tobytes()
    assert output["slice"] == [1, 7, 2]
    assert torch.equal(output["codes"], payload["codes"])
    array.fill(0)
    image.paste(0, (0, 0, 3, 2))
    assert np.count_nonzero(output["array"]) and any(output["image"].tobytes())


def test_large_array_rejects_unused_framing_snapshot(monkeypatch):
    def unexpected_snapshot(value):
        pytest.fail("a too-large array must not be snapshotted while testing ring admission")

    monkeypatch.setattr(OmniSerializer.encoder, "_enc_hook", unexpected_snapshot)
    assert prepare_tensor_frame({"array": np.zeros(1024)}, max_bytes=1024) is None


def test_native_scalar_restore_preserves_nested_markers_and_input_containers():
    array = np.arange(3)
    image = Image.fromarray(np.array([[1, 2]], dtype=np.uint8))
    tensor = torch.arange(2)
    payload = {"ids": list(range(8192)), "mixed": [1.5, None, (array, {"image": image, "tensor": tensor})]}
    decoded = msgspec.msgpack.decode(OmniSerializer.serialize(payload))
    output = OmniSerializer.restore(decoded)
    assert output is not decoded and output["ids"] is not decoded["ids"]
    assert output["ids"] == payload["ids"]
    assert np.array_equal(output["mixed"][2][0], array)
    assert output["mixed"][2][1]["image"].tobytes() == image.tobytes()
    assert torch.equal(output["mixed"][2][1]["tensor"], tensor)
    assert isinstance(decoded["mixed"][2][0], dict)


def test_fallback_miss_does_not_register_an_unclaimed_allocation(edge, monkeypatch, caplog):
    sender, receiver = edge
    key = "fallback-claim-" + uuid.uuid4().hex
    _, _, metadata = sender.put("0", "1", key, torch.arange(4096))
    assert "shm" in metadata
    original = shared_memory.SharedMemory
    calls = []

    def tracked_open(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_memory, "SharedMemory", tracked_open)
    with open(f"/dev/shm/shm_{key}_lockfile.lock", "rb+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert receiver.get("0", "1", key) is None
        assert not calls
    assert torch.equal(receiver.get("0", "1", key)[0], torch.arange(4096))
    assert receiver.get("0", "1", key, metadata) is None
    assert not any(record.levelname == "ERROR" for record in caplog.records)


def test_tensor_frames_preserve_shared_output_reconstruction(edge):
    sender, receiver = edge
    completion = asdict(CompletionOutput(index=0, text="", token_ids=[1], cumulative_logprob=0.0, logprobs=None))
    completion["multimodal_output"] = {"codes": torch.arange(28)}
    payload = {
        "request_id": "output",
        "prompt": None,
        "prompt_token_ids": [1],
        "prompt_logprobs": None,
        "outputs": [completion],
        "finished": False,
    }
    _, _, metadata = sender.put("0", "1", "output", payload)
    assert "host_ring" in metadata
    result = receiver.get("0", "1", "output", metadata)[0]
    assert isinstance(result, RequestOutput)
    assert isinstance(result.outputs[0], CompletionOutput)
    assert torch.equal(result.outputs[0].multimodal_output["codes"], torch.arange(28))


def test_large_tensor_does_not_share_owner_with_small_metadata():
    with SharedMemoryConnector({}) as connector:
        payload = {"hidden": torch.ones(16384), "finished": torch.tensor(False)}
        assert connector.put("0", "1", "large-owner", payload)[0]
        output = connector.get("0", "1", "large-owner")[0]
        assert output["finished"].untyped_storage().nbytes() == 1
        assert output["hidden"].untyped_storage().nbytes() == 65536


def test_full_ring_and_large_frame_use_existing_shm(edge):
    sender, receiver = edge
    payload = {"codes": torch.arange(56)}
    metadata = {}
    for index in range(12):
        ok, _size, metadata[index] = sender.put("0", "1", f"full{index}", payload)
        assert ok
    assert any("host_ring" in item for item in metadata.values())
    assert any("shm" in item for item in metadata.values())
    # Unclaimed earlier frames pin the ring; later requests still arrive.
    for index in reversed(range(12)):
        assert torch.equal(receiver.get("0", "1", f"full{index}")[0]["codes"], payload["codes"])
    ok, _size, large_metadata = sender.put("0", "1", "large", torch.arange(4096))
    assert ok and "shm" in large_metadata
    assert torch.equal(receiver.get("0", "1", "large", large_metadata)[0], torch.arange(4096))
    assert "host_ring" in sender.put("0", "1", "recovered", payload)[2]
    assert receiver.get("0", "1", "recovered") is not None


def test_reput_across_ring_and_fallback_and_stale_descriptor(edge):
    sender, receiver = edge
    _, _, old = sender.put("0", "1", "reuse", torch.tensor([11]))
    _, _, current = sender.put("0", "1", "reuse", torch.tensor([22]))
    assert receiver.get("0", "1", "reuse", old) is None
    assert receiver.get("0", "1", "reuse", current)[0].item() == 22
    sender.put("0", "1", "reuse", torch.tensor([33]))
    sender.put("0", "1", "reuse", torch.full((4096,), 44))
    assert receiver.get("0", "1", "reuse")[0][0].item() == 44
    sender.put("0", "1", "reuse", torch.full((4096,), 55))
    sender.put("0", "1", "reuse", torch.tensor([66]))
    assert receiver.get("0", "1", "reuse")[0].item() == 66
    assert not os.path.exists("/dev/shm/reuse")


def test_cancel_without_receiver_scan_returns_credit_and_prefix_is_exact(edge):
    sender, receiver = edge
    for key in ("req_0_0", "req_0_1", "req_0_10other", "request_0_0"):
        assert sender.put("0", "1", key, torch.arange(56))[0]
    assert sender.cleanup_prefix("req_0_") == 2
    assert receiver.get("0", "1", "req_0_0") is None
    assert receiver.get("0", "1", "req_0_10other") is not None
    assert receiver.get("0", "1", "request_0_0") is not None
    assert "host_ring" in sender.put("0", "1", "next", torch.arange(56))[2]


def test_flat_scope_and_metadata_work_across_different_parents(monkeypatch):
    monkeypatch.setattr(os, "getppid", lambda: 100)
    sender = SharedMemoryConnector({"stage_id": 0, "wakeup_scope": "flat-scope"})
    monkeypatch.setattr(os, "getppid", lambda: 200)
    receiver = SharedMemoryConnector({"stage_id": 1, "wakeup_scope": "flat-scope"})
    try:
        assert sender._wake_directory == receiver._wake_directory
        assert sender.put("0", "1", "scope", torch.tensor([7]))[0]
        assert receiver.get("0", "1", "scope")[0].item() == 7
    finally:
        sender.close()
        receiver.close()


def test_empty_poll_reuses_discovery_and_never_waits_for_a_ring_lock(edge, monkeypatch):
    sender, receiver = edge
    sender.put("0", "1", "prime-poll", torch.arange(3))
    assert receiver.get("0", "1", "prime-poll") is not None

    def unexpected_work(*args, **kwargs):
        pytest.fail("an empty, unchanged channel must not rescan discovery or acquire its frame lock")

    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_ring.glob.iglob", unexpected_work)
    monkeypatch.setattr(fcntl, "flock", unexpected_work)
    for index in range(128):
        assert receiver.get("0", "1", f"missing-poll-{index}") is None
    monkeypatch.undo()
    sender.put("0", "1", "new-poll", torch.arange(5))
    assert receiver.get("0", "1", "new-poll")[0].tolist() == list(range(5))


def test_new_producer_is_discovered_after_a_miss_even_when_the_clock_does_not_advance(edge, monkeypatch):
    sender, receiver = edge
    sender.put("0", "1", "first-producer", torch.arange(3))
    assert receiver.get("0", "1", "first-producer") is not None
    assert receiver.get("0", "1", "second-producer") is None
    old_version = receiver._host_ring._discovery_versions["0", "1"]
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_ring.time.time_ns", lambda: 0)
    with SharedMemoryConnector(sender.config) as second:
        second.put("0", "1", "second-producer", torch.arange(5))
        assert receiver.get("0", "1", "second-producer")[0].tolist() == list(range(5))
        new_version = receiver._host_ring._discovery_versions["0", "1"]
        assert new_version[2] == old_version[2] + 1


def test_one_receiver_discovers_each_input_edge_at_the_same_registry_version(edge):
    sender, receiver = edge
    with SharedMemoryConnector({**sender.config, "stage_id": 2}) as second:
        sender.put("0", "1", "fanin-first", torch.arange(3))
        second.put("2", "1", "fanin-second", torch.arange(5))
        assert receiver.get("0", "1", "fanin-first")[0].tolist() == list(range(3))
        assert receiver.get("2", "1", "fanin-second")[0].tolist() == list(range(5))


def test_discovery_attaches_all_producers_even_when_the_first_one_has_the_requested_key(edge, monkeypatch):
    sender, receiver = edge
    with SharedMemoryConnector(sender.config) as second:
        first_metadata = sender.put("0", "1", "multi-first", torch.arange(3))[2]
        second_metadata = second.put("0", "1", "multi-second", torch.arange(5))[2]
        paths = [
            f"/dev/shm/{receiver._host_ring.directory}/{metadata['host_ring']['name']}"
            for metadata in (first_metadata, second_metadata)
        ]
        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.connectors.shm_ring.glob.iglob", lambda _: iter(paths)
        )
        assert receiver.get("0", "1", "multi-first")[0].tolist() == list(range(3))
        assert receiver.get("0", "1", "multi-second")[0].tolist() == list(range(5))


def test_closed_channel_releases_reader_cache_while_the_producer_process_is_still_alive(edge):
    sender, receiver = edge
    sender.put("0", "1", "close-cache", torch.arange(3))
    assert receiver.get("0", "1", "close-cache") is not None
    channel = next(iter(receiver._host_ring.readers.values()))
    fd = channel.fd
    sender.close()
    assert receiver.get("0", "1", "not-published-after-close") is None
    assert not receiver._host_ring.readers
    with pytest.raises(OSError):
        os.fstat(fd)


def test_disabled_ring_and_unsupported_tree_keep_legacy_serializer(edge):
    sender, receiver = edge
    with SharedMemoryConnector({"host_ring_bytes": 0}) as legacy:
        _, _, metadata = legacy.put("0", "1", "legacy", {"codes": torch.arange(3)})
        assert "shm" in metadata
        assert receiver.get("0", "1", "legacy", metadata)[0]["codes"].tolist() == [0, 1, 2]
    assert prepare_tensor_frame({"codes": torch.arange(3), "opaque": object()}) is None
    assert sender.put("0", "1", "falsey", False)[0]
    assert receiver.get("0", "1", "falsey")[0] is False


def test_tensor_free_lists_reuse_native_serialization():
    payload = {"prompt_token_ids": list(range(8192)), "condition": [0.25] * 8192}
    with SharedMemoryConnector({}) as connector:
        frame = prepare_tensor_frame(payload)
        assert isinstance(frame, bytes)
        assert frame == connector.serialize_obj(payload)
        assert connector.put("0", "1", "native-lists", payload)[0]
        assert connector.get("0", "1", "native-lists")[0] == payload


def test_scalar_lists_and_literal_extensions_preserve_wire_semantics():
    literal = msgspec.msgpack.Ext(42, bytes(20))
    payload = {"condition": [0.25] * 8192, "codes": torch.arange(28), "extension": literal}
    with SharedMemoryConnector({}) as connector:
        _, _, metadata = connector.put("0", "1", "mixed-lists", payload)
        assert "host_ring" in metadata
        result = connector.get("0", "1", "mixed-lists")[0]
        assert result["condition"] == payload["condition"]
        assert torch.equal(result["codes"], payload["codes"])
        assert result["extension"].code == literal.code
        assert result["extension"].data == literal.data


def _locked_writer(name, ready, release):
    fd = os.open(f"/dev/shm/{name}", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        ready.set()
        assert release.wait(10)
    finally:
        os.close(fd)


@pytest.mark.parametrize("use_metadata", [False, True])
def test_deadline_receive_never_waits_for_another_process_ring_lock(edge, use_metadata):
    sender, receiver = edge
    _, _, metadata = sender.put("0", "1", "locked", torch.arange(3))
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    writer = ctx.Process(target=_locked_writer, args=(metadata["host_ring"]["name"], ready, release))
    writer.start()
    try:
        assert ready.wait(15)
        start = time.monotonic()
        assert (
            receiver.get_with_deadline("0", "1", "locked", metadata if use_metadata else None, deadline=start + 1)
            is None
        )
        assert time.monotonic() - start < 1
    finally:
        release.set()
        writer.join(15)
        if writer.is_alive():
            writer.kill()
            writer.join()
    assert writer.exitcode == 0
    assert receiver.get("0", "1", "locked", metadata if use_metadata else None)[0].tolist() == [0, 1, 2]


def _competing_reader(scope, barrier, results):
    with SharedMemoryConnector({"stage_id": 1, "wakeup_scope": scope}) as receiver:
        barrier.wait(timeout=20)
        result = receiver.get("0", "1", "shared")
        results.put(None if result is None else result[0]["codes"].tolist())


def test_two_reader_processes_claim_exactly_once_and_close_does_not_unlink_owner():
    scope = uuid.uuid4().hex
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(3), ctx.Queue()
    with SharedMemoryConnector({"stage_id": 0, "wakeup_scope": scope}) as sender:
        _, _, metadata = sender.put("0", "1", "shared", {"codes": torch.arange(56)})
        readers = [ctx.Process(target=_competing_reader, args=(scope, barrier, results)) for _ in range(2)]
        for reader in readers:
            reader.start()
        try:
            barrier.wait(timeout=30)
            outputs = [results.get(timeout=20) for _ in readers]
            assert outputs.count(None) == 1
            assert next(output for output in outputs if output is not None) == list(range(56))
            assert os.path.exists(f"/dev/shm/{metadata['host_ring']['name']}")
        finally:
            for reader in readers:
                reader.join(15)
                if reader.is_alive():
                    reader.kill()
                    reader.join()
            results.close()
        assert all(reader.exitcode == 0 for reader in readers)


def test_ring_setup_failure_keeps_connector_usable(edge, monkeypatch):
    sender, receiver = edge

    def no_space(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "posix_fallocate", no_space)
    ok, _, metadata = sender.put("0", "1", "no-ring-space", torch.arange(3))
    assert ok and "shm" in metadata
    assert receiver.get("0", "1", "no-ring-space")[0].tolist() == [0, 1, 2]


def test_discovery_stamp_failure_keeps_fallback_and_cleanup_usable(edge, monkeypatch):
    sender, receiver = edge

    def stamp_denied(*args, **kwargs):
        raise PermissionError("directory timestamp update denied")

    monkeypatch.setattr(os, "utime", stamp_denied)
    ok, _, metadata = sender.put("0", "1", "no-ring-stamp", torch.arange(3))
    assert ok and "shm" in metadata
    assert not sender._host_ring.producers
    assert not list(Path(f"/dev/shm/{sender._host_ring.directory}").iterdir())
    assert receiver.get("0", "1", "no-ring-stamp")[0].tolist() == [0, 1, 2]
    sender.close()


def test_close_is_idempotent_and_removes_owned_ring(edge):
    sender, receiver = edge
    _, _, metadata = sender.put("0", "1", "close", torch.arange(3))
    receiver.get("0", "1", "close")
    sender.close()
    sender.close()
    receiver.close()
    receiver.close()
    assert not os.path.exists(f"/dev/shm/{metadata['host_ring']['name']}")


def _orphan_producer(scope, pipe):
    producer = SharedMemoryConnector({"stage_id": 0, "wakeup_scope": scope})
    metadata = producer.put("0", "1", "restart", torch.tensor([11]))[2]
    pipe.send(metadata)
    pipe.recv()
    pipe.close()
    os._exit(0)


def test_dead_producer_payload_cannot_reach_a_reused_key():
    scope = uuid.uuid4().hex
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_orphan_producer, args=(scope, child))
    process.start()
    child.close()
    old = parent.recv()
    try:
        with SharedMemoryConnector({"stage_id": 1, "wakeup_scope": scope}) as receiver:
            # Cache the live channel without claiming the pending frame.
            assert receiver.get("0", "1", "not-published") is None
            channel = next(iter(receiver._host_ring.readers.values()))
            fd = channel.fd
            parent.send("exit")
            process.join(15)
            assert process.exitcode == 0
            assert receiver.get("0", "1", "restart", old) is None
            assert not receiver._host_ring.readers
            with pytest.raises(OSError):
                os.fstat(fd)
            with SharedMemoryConnector({"stage_id": 0, "wakeup_scope": scope}) as replacement:
                replacement.put("0", "1", "restart", torch.tensor([22]))
                assert receiver.get("0", "1", "restart")[0].item() == 22
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        # The multiprocessing parent still owns this test's resource tracker.
        # Retire only the intentionally orphaned allocation and its exact marker.
        segment = shared_memory.SharedMemory(name=old["host_ring"]["name"])
        segment.close()
        segment.unlink()
        directory = old["host_ring"]["name"].rsplit("_", 4)[0]
        try:
            os.unlink(f"/dev/shm/{directory}/{old['host_ring']['name']}")
        except FileNotFoundError:
            pass
        try:
            os.rmdir(f"/dev/shm/{directory}")
        except FileNotFoundError:
            pass
        parent.close()
