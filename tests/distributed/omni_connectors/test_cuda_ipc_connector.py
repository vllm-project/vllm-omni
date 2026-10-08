# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CudaIpcConnector: handle ownership on CPU through fake exports, real IPC on GPU."""

import glob
import multiprocessing as mp
import os
import signal
import time
import uuid
from multiprocessing import shared_memory as shm_pkg

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.data_entry_keys import CodesStruct, HiddenStatesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.connectors import cuda_ipc_connector as ipc_mod
from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
    CudaIpcConnector,
    extract_tensors,
    restore_tensors,
    tensor_desc,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

pytestmark = [pytest.mark.core_model]

_FAKE_GPU = "GPU-fake"


def _key(tag: str) -> str:
    return f"ipc_{tag}_{uuid.uuid4().hex}"


@pytest.fixture()
def released(monkeypatch):
    """Counter handles released through ``_release_counter``, in order."""
    calls: list[str] = []
    monkeypatch.setattr(ipc_mod, "_release_counter", lambda args: calls.append(args[ipc_mod._ARG_REF_COUNTER_HANDLE]))
    return calls


class _FakeExports:
    """Export every framed tensor as a fake IPC handle named ``counter-<n>``."""

    def __init__(self, connector: CudaIpcConnector):
        self.names: list[str] = []
        connector._export_tensor = self
        connector._maybe_collect = lambda force=False: None

    def __call__(self, tensor: torch.Tensor):
        name = f"counter-{len(self.names)}"
        self.names.append(name)
        args = (None,) * ipc_mod._ARG_REF_COUNTER_HANDLE + (name, 0, None, False)
        desc = {**tensor_desc(tensor), "ipc": list(args), "gpu": _FAKE_GPU}
        return desc, (args, tensor.nbytes)


@pytest.fixture()
def producer():
    connector = CudaIpcConnector({"inline_tensor_bytes": 1024})
    fake = _FakeExports(connector)
    yield connector, fake
    connector.close()


def _payload() -> dict:
    return {"hidden": torch.randn(64, 64), "pair": (torch.ones(512), "tag"), "small": torch.arange(4)}


@pytest.mark.cpu
class TestHandleOwnership:
    def test_host_tensors_use_framed_shm(self):
        connector = CudaIpcConnector({"use_ipc": False, "inline_tensor_bytes": 1024})
        try:
            key = _key("host")
            payload = _payload()
            connector.put("0", "1", key, payload)
            restored, _ = connector.get("0", "1", key)
            assert torch.equal(restored["hidden"], payload["hidden"])
            assert isinstance(restored["pair"], tuple) and torch.equal(restored["pair"][0], torch.ones(512))
            assert connector.health()["ipc_exports"] == 0
        finally:
            connector.close()

    def test_unread_cleanup_releases_each_export_once(self, producer, released):
        connector, fake = producer
        key = _key("unread")
        connector.put("0", "1", key, _payload())
        health = connector.health()
        assert health["ipc_exports"] == 2 and health["ipc_export_bytes"] == 64 * 64 * 4 + 512 * 4

        connector.cleanup(key)
        connector.cleanup(key)

        assert released == fake.names
        assert connector.health()["ipc_exports"] == 0

    def test_reput_releases_the_replaced_exports(self, producer, released):
        connector, fake = producer
        key = _key("reput")
        connector.put("0", "1", key, {"x": torch.ones(1024)})
        connector.put("0", "1", key, {"x": torch.zeros(1024)})

        assert released == fake.names[:1]
        connector.close()
        assert released == fake.names

    def test_reader_that_cannot_see_the_gpu_owns_the_release(self, producer, released, monkeypatch):
        connector, fake = producer
        consumer = CudaIpcConnector({})
        monkeypatch.setattr(ipc_mod, "_visible_gpus", lambda: {})
        try:
            key = _key("invisible")
            connector.put("0", "1", key, _payload())

            assert consumer.get("0", "1", key) is None
            assert released == fake.names

            connector.reap_consumed()  # the claim retires producer state without a second release
            assert released == fake.names
            assert connector.health()["ipc_exports"] == 0
        finally:
            consumer.close()

    def test_failed_import_releases_unmapped_handles(self, producer, released, monkeypatch):
        connector, fake = producer
        consumer = CudaIpcConnector({})
        monkeypatch.setattr(ipc_mod, "_visible_gpus", lambda: {_FAKE_GPU: 0})

        def no_stream(index):
            raise RuntimeError("no stream")

        monkeypatch.setattr(ipc_mod, "_copy_stream", no_stream)
        try:
            key = _key("failed")
            connector.put("0", "1", key, _payload())
            assert consumer.get("0", "1", key) is None
            assert released == fake.names
        finally:
            consumer.close()

    def test_missing_gpu_error_names_the_fix(self, monkeypatch):
        monkeypatch.setattr(ipc_mod, "_visible_gpus", lambda: {})
        with pytest.raises(RuntimeError, match="use_ipc: false"):
            CudaIpcConnector._local_device_index(_FAKE_GPU)


# ── Framed payloads (host staging, use_ipc: false) ───────────────────


def _segment_bytes(key: str) -> bytes:
    seg = shm_pkg.SharedMemory(name=key)
    try:
        return bytes(seg.buf)
    finally:
        seg.close()


@pytest.fixture()
def framed():
    """CudaIpcConnector with the IPC data plane off: framed SHM host staging."""
    c = CudaIpcConnector({"use_ipc": False, "inline_tensor_bytes": 64 * 1024})
    yield c
    c.close()


class TestFramedPayload:
    def test_large_tensors_round_trip_out_of_band(self, framed):
        connector = framed
        key = _key("framed")
        hidden = torch.randn(256, 128)
        payload = OmniPayloadStruct(
            hidden_states=HiddenStatesStruct(output=hidden),
            codes=CodesStruct(audio=torch.arange(28)),
            meta=MetaStruct(request_id="r", override_keys=[("a", "b")]),
        )

        ok, size, _ = connector.put("0", "1", key, payload)
        assert ok and _segment_bytes(key)[:4] == b"\xc1OMF"
        assert size >= hidden.nbytes
        restored, _ = connector.get("0", "1", key)

        assert torch.equal(restored["hidden_states"]["output"], hidden)
        assert torch.equal(restored["codes"]["audio"], torch.arange(28))
        assert restored["meta"]["request_id"] == "r"
        # The frame header serializes the tensor-free skeleton as plain
        # msgpack (Structs decoded to dicts), so nested tuples come back as
        # lists — same shape the legacy dict path produces.
        assert restored["meta"]["override_keys"] == [["a", "b"]]

    def test_mixed_dtypes_and_non_contiguous(self, framed):
        connector = framed
        key = _key("framed_dtypes")
        base = torch.randn(512, 64)
        payload = {
            "bf16": base.to(torch.bfloat16),
            "t": base.t(),
            "flags": torch.ones(70000, dtype=torch.bool),
            "pair": (base[:300], "tag"),
        }

        connector.put("0", "1", key, payload)
        restored, _ = connector.get("0", "1", key)

        assert restored["bf16"].dtype == torch.bfloat16 and torch.equal(restored["bf16"], payload["bf16"])
        assert torch.equal(restored["t"], base.t())
        assert restored["flags"].dtype == torch.bool and bool(restored["flags"].all())
        assert isinstance(restored["pair"], tuple) and torch.equal(restored["pair"][0], base[:300])

    def test_small_payload_keeps_legacy_bytes(self, framed):
        connector = framed
        key = _key("legacy")
        payload = {"codes": torch.arange(28), "finished": torch.tensor(True)}

        connector.put("0", "1", key, payload)

        assert _segment_bytes(key) == connector.serialize_obj(payload)
        assert torch.equal(connector.get("0", "1", key)[0]["codes"], torch.arange(28))

    def test_threshold_is_configurable(self):
        small_frames = CudaIpcConnector({"use_ipc": False, "inline_tensor_bytes": 16})
        try:
            key = _key("threshold")
            small_frames.put("0", "1", key, {"x": torch.arange(8)})
            assert _segment_bytes(key)[:4] == b"\xc1OMF"
            assert torch.equal(small_frames.get("0", "1", key)[0]["x"], torch.arange(8))
        finally:
            small_frames.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_tensor_is_copied_into_segment(self, framed):
        connector = framed
        key = _key("framed_cuda")
        hidden = torch.randn(256, 128, device="cuda")

        connector.put("0", "1", key, {"h": hidden})
        restored, _ = connector.get("0", "1", key)

        assert restored["h"].device.type == "cpu"
        assert torch.equal(restored["h"], hidden.cpu())


# ── Real CUDA IPC ────────────────────────────────────────────────────


def _reclaim() -> None:
    """Give CUDA a few beats to return IPC clones after their consumer is gone."""
    for _ in range(5):
        torch.cuda.ipc_collect()
        time.sleep(0.2)


def _soak_consumer(count: int, conn) -> None:
    import gc

    torch.cuda.init()
    connector = CudaIpcConnector({})
    baseline = torch.accelerator.memory_allocated()
    got = 0
    try:
        for i in range(count):
            key, metadata, marker = conn.recv()
            payload, _ = connector.get("0", "1", key, metadata)
            if payload is not None and payload["x"].is_cuda and float(payload["x"][0]) == marker:
                got += 1
            del payload
        gc.collect()
        torch.accelerator.synchronize()
        conn.send({"got": got, "leaked_bytes": torch.accelerator.memory_allocated() - baseline})
    finally:
        connector.close()
        conn.close()


def _doomed_consumer(conn) -> None:
    """Claim a payload, map its IPC handle, then die without releasing."""
    import multiprocessing.shared_memory as shm_pkg

    import torch.multiprocessing.reductions as reductions

    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import _FRAME_PREFIX, _rebuild_args
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

    torch.cuda.init()
    key, metadata = conn.recv()
    seg = shm_pkg.SharedMemory(name=metadata["shm"]["name"])
    seg.unlink()  # claim like the connector does
    buf = seg.buf
    header_len = int.from_bytes(buf[4:_FRAME_PREFIX], "little")
    header = SharedMemoryConnector({}).deserialize_obj(bytes(buf[_FRAME_PREFIX : _FRAME_PREFIX + header_len]))
    mapped = reductions.rebuild_cuda_tensor(*_rebuild_args(header["tensors"][0]))
    conn.send(float(mapped[0]))
    time.sleep(0.2)
    os.kill(os.getpid(), signal.SIGKILL)


def _doomed_producer(conn) -> None:
    """Export a payload, hand over its metadata, then die before it is read."""
    torch.cuda.init()
    connector = CudaIpcConnector({})
    key = f"ipc_doomp_{os.getpid()}"
    _, _, metadata = connector.put("0", "1", key, {"x": torch.full((1 << 20,), 7.0, device="cuda")})
    conn.send((key, metadata))
    time.sleep(0.2)
    os.kill(os.getpid(), signal.SIGKILL)


# ── Real CUDA IPC ────────────────────────────────────────────────────


def _expected(n: int, device) -> torch.Tensor:
    return torch.arange(n * n, dtype=torch.float32, device=device).reshape(n, n)


def _consume(key: str, metadata: dict, conn) -> None:
    torch.cuda.init()
    consumer = CudaIpcConnector({})
    try:
        payload, _ = consumer.get("0", "1", key, metadata)
        hidden = payload["hidden"]
        conn.send(
            {
                "device": hidden.device.type,
                "hidden_ok": torch.equal(hidden.cpu(), _expected(1024, "cpu")),
                "bf16_ok": torch.equal(payload["bf16"].cpu(), _expected(512, "cpu").to(torch.bfloat16)),
                "small_ok": torch.equal(payload["small"].cpu(), torch.arange(8)),
                "ids": payload["ids"],
            }
        )
    finally:
        consumer.close()
        conn.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestCudaIpc:
    @pytest.fixture()
    def gpu_producer(self):
        connector = CudaIpcConnector({})
        yield connector
        connector.close()

    def test_cross_process_round_trip_returns_the_clone(self, gpu_producer):
        torch.accelerator.synchronize()
        torch.cuda.ipc_collect()
        baseline = torch.accelerator.memory_allocated()
        key = _key("xproc")
        hidden = _expected(1024, "cuda")
        payload = {
            "hidden": hidden,
            "bf16": _expected(512, "cuda").to(torch.bfloat16),
            "small": torch.arange(8, device="cuda"),
            "ids": [1, 2, 3],
        }
        _, _, metadata = gpu_producer.put("0", "1", key, payload)
        hidden.zero_()  # the reader must see the clone, not the reused source
        del payload, hidden
        assert gpu_producer.health()["ipc_exports"] == 2

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_consume, args=(key, metadata, child))
        process.start()
        child.close()
        try:
            result = parent.recv()
        finally:
            process.join(timeout=120)
        assert process.exitcode == 0
        assert result == {
            "device": "cuda",
            "hidden_ok": True,
            "bf16_ok": True,
            "small_ok": True,
            "ids": [1, 2, 3],
        }

        gpu_producer.reap_consumed()
        # The consumer's exit-time frees reach the producer's allocator
        # asynchronously and ipc_collect only hints; give it a few beats.
        for _ in range(5):
            time.sleep(0.5)
            torch.cuda.ipc_collect()
        assert gpu_producer.health()["ipc_exports"] == 0
        # Torch reclaims shared clones lazily: identical runs strand between
        # none and all of the 4.5 MiB of clones for seconds after a clean
        # consumer exit. What must hold is bounded-by-payload residual.
        clone_bytes = (1 << 22) + (1 << 19)  # the hidden + bf16 clones
        assert torch.accelerator.memory_allocated() - baseline <= 2 * clone_bytes

    def test_unread_cleanup_frees_the_clone(self, gpu_producer):
        torch.accelerator.synchronize()
        torch.cuda.ipc_collect()
        source = torch.ones(1 << 20, device="cuda")
        baseline = torch.accelerator.memory_allocated()
        key = _key("gpu_unread")

        gpu_producer.put("0", "1", key, {"x": source})
        # The export keeps a private clone of the payload alive.
        after_put = torch.accelerator.memory_allocated()
        assert after_put >= source.nbytes
        gpu_producer.cleanup(key)
        torch.cuda.ipc_collect()

        # ipc_collect reclaims lazily (earlier tests in this process can
        # still be returning clones); bound the residual instead of
        # demanding the exact baseline.
        assert torch.accelerator.memory_allocated() <= baseline + (2 << 20)

    def test_export_failure_stages_through_host(self, gpu_producer, monkeypatch):
        import torch.multiprocessing.reductions as reductions

        def refuse(tensor):
            raise RuntimeError("export refused")

        monkeypatch.setattr(reductions, "reduce_tensor", refuse)
        key = _key("fallback")
        source = torch.randn(256, 256, device="cuda")

        gpu_producer.put("0", "1", key, {"x": source})
        assert gpu_producer.health()["ipc_exports"] == 0
        restored, _ = gpu_producer.get("0", "1", key)

        assert torch.equal(restored["x"], source.cpu())

    def test_soak_10k_transfers_leak_nothing(self, gpu_producer):
        count = 10_000
        prefix = f"ipc_soak_{os.getpid()}"
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_soak_consumer, args=(count, child))
        process.start()
        child.close()

        torch.accelerator.synchronize()
        torch.cuda.ipc_collect()
        baseline = torch.accelerator.memory_allocated()
        try:
            for i in range(count):
                marker = float(i % 97)
                payload = {"i": i, "x": torch.full((32_768,), marker, device="cuda")}
                key = f"{prefix}_{i}"
                _, _, metadata = gpu_producer.put("0", "1", key, payload)
                parent.send((key, metadata, marker))
                gpu_producer.reap_consumed()
            summary = parent.recv()
        finally:
            parent.close()
            process.join(timeout=600)
        assert process.exitcode == 0, "soak consumer died"
        assert summary["got"] == count
        # A fixed allocator residual (copy-stream slack, the last payload) is
        # fine; the soak asserts no per-transfer growth.
        assert summary["leaked_bytes"] <= 2 << 20, "consumer leaked device memory per transfer"

        # reap_consumed sweeps a bounded 64 keys per call (production calls
        # it per put); drain the backlog left by the burst.
        deadline = time.monotonic() + 30
        while gpu_producer.health()["ipc_exports"] > 0 and time.monotonic() < deadline:
            gpu_producer.reap_consumed()
            time.sleep(0.1)
        _reclaim()
        # ipc_collect timing varies run to run (0.4-8.2 MiB observed after
        # 10k x 128 KiB with the export-time clone; a per-transfer leak would
        # be ~1.25 GiB). Bound the residual at a small multiple of the payload.
        assert torch.accelerator.memory_allocated() - baseline <= 16 << 20, "producer clones leaked"
        assert gpu_producer.health()["ipc_exports"] == 0, "producer bookkeeping did not drain"
        assert not glob.glob(f"/dev/shm/{prefix}*"), "segments outlived the soak"

    def test_killed_consumer_returns_the_clone(self, gpu_producer):
        torch.accelerator.synchronize()
        torch.cuda.ipc_collect()
        baseline = torch.accelerator.memory_allocated()
        payload_bytes = (1 << 22) * 4  # 16 MiB of float32
        source = torch.full((1 << 22,), 3.0, device="cuda")
        key = _key("doomc")
        _, _, metadata = gpu_producer.put("0", "1", key, {"x": source})
        del source

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_doomed_consumer, args=(child,))
        process.start()
        child.close()
        try:
            parent.send((key, metadata))  # the consumer claims and maps before dying
            seen = parent.recv()
        finally:
            parent.close()
            process.join(timeout=120)
        assert process.exitcode == -signal.SIGKILL, "consumer was not SIGKILLed"
        assert seen == 3.0, "consumer mapped something else"

        _reclaim()
        # SIGKILL skips torch's IPC counter teardown, so a mapping the
        # consumer held can strand the clone until the producer closes —
        # torch's documented limitation ("Note [Sharing CUDA tensors]"),
        # observed at up to 2x the payload. Bound it there and require the
        # producer context to stay usable.
        residual = torch.accelerator.memory_allocated() - baseline
        assert residual <= 3 * payload_bytes, f"clone leak beyond the payload: {residual} bytes"
        assert float(torch.ones(8, device="cuda").sum()) == 8.0, "producer context unusable"

    def test_killed_producer_leaves_consumer_context_intact(self):
        torch.cuda.init()
        torch.zeros(1, device="cuda")

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_doomed_producer, args=(child,))
        process.start()
        child.close()
        try:
            key, metadata = parent.recv()
        finally:
            parent.close()
            process.join(timeout=120)
        assert process.exitcode == -signal.SIGKILL, "producer was not SIGKILLed"

        consumer = CudaIpcConnector({})
        try:
            # Either the mapping still resolves or get() reports the payload
            # gone; both are acceptable answers to a dead producer.
            consumer.get("0", "1", key, metadata)
        except Exception as exc:  # noqa: BLE001 -- the contract under test
            pytest.fail(f"get() must not raise on a dead producer: {exc!r}")
        assert float(torch.ones(8, device="cuda").sum()) == 8.0, "consumer context unusable"
        consumer.close()


# ── Tensor-tree codec (framing's skeleton/tensor split) ──────────────


def _codec_round_trip(payload, select=None):
    skeleton, tensors = extract_tensors(payload, select)
    wire = OmniSerializer.deserialize(OmniSerializer.serialize(skeleton))
    return restore_tensors(wire, tensors), tensors


class TestTensorTreeCodec:
    def test_nested_containers_keep_types_and_identity(self):
        hidden = torch.randn(3, 4, dtype=torch.bfloat16)
        ids = torch.tensor(7)
        payload = {"a": [hidden, {"ids": ids}], "pair": (1, hidden[:, :2]), "text": "x", "n": None}

        restored, tensors = _codec_round_trip(payload)

        assert len(tensors) == 3
        assert restored["a"][0] is hidden
        assert restored["a"][1]["ids"] is ids
        assert isinstance(restored["pair"], tuple) and restored["pair"][0] == 1
        assert torch.equal(restored["pair"][1], hidden[:, :2])
        assert restored["text"] == "x" and restored["n"] is None

    def test_struct_becomes_dict_like_serializer_decode(self):
        payload = OmniPayloadStruct(
            hidden_states=HiddenStatesStruct(output=torch.ones(4)),
            meta=MetaStruct(left_context_size=2),
        )

        restored, tensors = _codec_round_trip(payload)

        assert len(tensors) == 1
        assert restored["meta"]["left_context_size"] == 2
        assert restored["hidden_states"]["output"] is tensors[0]

    def test_ndarray_round_trips_as_ndarray(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        restored, tensors = _codec_round_trip({"wave": arr, "names": np.array(["a", "b"])})

        assert len(tensors) == 1
        assert isinstance(restored["wave"], np.ndarray)
        np.testing.assert_array_equal(restored["wave"], arr)
        assert list(restored["names"]) == ["a", "b"]

    def test_selector_keeps_rejected_tensors_inline(self):
        small = torch.ones(2)
        large = torch.ones(1024)
        skeleton, tensors = extract_tensors({"s": small, "l": large}, select=lambda t: t.nbytes >= 1024)

        assert tensors == [large]
        assert skeleton["s"] is small
        assert restore_tensors(skeleton, tensors)["l"] is large

    def test_tensor_free_subtrees_are_not_copied(self):
        meta = MetaStruct(left_context_size=1)
        ids = [1, 2, 3]
        payload = {"meta": meta, "ids": ids, "x": [(torch.ones(1),)]}

        skeleton, tensors = extract_tensors(payload)

        assert len(tensors) == 1
        assert skeleton is not payload and payload["x"][0][0] is tensors[0]
        assert skeleton["meta"] is meta and skeleton["ids"] is ids
        assert extract_tensors({"meta": meta, "ids": ids}) == ({"meta": meta, "ids": ids}, [])
        assert extract_tensors(payload["meta"])[0] is meta


# ── Transport snapshot (model-thread isolation before the save loop) ─


class TestSnapshot:
    def test_host_payloads_pass_through_without_event(self):
        connector = CudaIpcConnector({"use_ipc": False, "inline_tensor_bytes": 1 << 20})
        try:
            payload = {"hidden": torch.randn(8), "ids": [1, 2], "pair": (torch.ones(3), "tag")}
            snapshot, event = connector.snapshot_for_transport(payload)
            assert event is None
            assert snapshot["hidden"] is payload["hidden"]
            assert snapshot["ids"] == [1, 2]
            assert isinstance(snapshot["pair"], tuple) and snapshot["pair"][0] is payload["pair"][0]
        finally:
            connector.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_tensors_are_cloned_and_gated_by_event(self):
        connector = CudaIpcConnector({})
        try:
            source = torch.ones(1024, device="cuda")
            payload = {"h": source, "frames": [source, source[:4]], "n": 3}
            snapshot, event = connector.snapshot_for_transport(payload)
            assert event is not None
            source.zero_()  # the producer reuses its buffer
            event.synchronize()
            assert snapshot["h"] is not source and bool(snapshot["h"].all())
            assert all(bool(item.all()) for item in snapshot["frames"])
            assert snapshot["n"] == 3
        finally:
            connector.close()


# ── Default connector never frames ────────────────────────────────────


def test_default_connector_never_frames():
    """SharedMemoryConnector has no framing: payload bytes are plain msgpack."""
    connector = SharedMemoryConnector({})
    try:
        key = _key("default")
        payload = {"hidden": torch.randn(256, 128)}

        connector.put("0", "1", key, payload)

        assert _segment_bytes(key) == connector.serialize_obj(payload)
        assert torch.equal(connector.get("0", "1", key)[0]["hidden"], payload["hidden"])
    finally:
        connector.close()
