# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for BufferAllocator and ManagedBuffer.
These tests do NOT require Mooncake or RDMA environment.
"""

import threading

import pytest
import torch

import vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector as mooncake_module
from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    BufferAllocator,
    ManagedBuffer,
    MooncakeTransferEngineConnector,
    _WarmPool,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import KVCacheTransferData

# Most tests in this file are pure-CPU unit tests; CUDA smoke tests are skip-guarded.
pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


@pytest.mark.core_model
class TestBufferAllocator:
    """Unit tests for BufferAllocator."""

    def test_basic_alloc_free(self):
        """Verify alloc, free, and reuse of freed space."""
        allocator = BufferAllocator(total_size=4096, alignment=64)

        offset1 = allocator.alloc(512)
        assert offset1 == 0

        offset2 = allocator.alloc(512)
        assert offset2 > 0

        # Free first block, should be reusable
        allocator.free(offset1, 512)
        offset3 = allocator.alloc(512)
        assert offset3 == 0

    def test_alignment(self):
        """Verify allocation respects alignment."""
        allocator = BufferAllocator(total_size=4096, alignment=128)

        _offset1 = allocator.alloc(100)
        offset2 = allocator.alloc(100)

        assert offset2 % 128 == 0
        assert offset2 == 128

    def test_exhaustion_and_recovery(self):
        """Test that full allocation fails, then succeeds after free."""
        allocator = BufferAllocator(total_size=1024, alignment=64)

        offset = allocator.alloc(1024)
        assert offset == 0

        with pytest.raises(MemoryError):
            allocator.alloc(64)

        allocator.free(offset, 1024)
        offset2 = allocator.alloc(1024)
        assert offset2 == 0

    def test_thread_safety(self):
        """Verify allocator is thread-safe under concurrent access."""
        allocator = BufferAllocator(total_size=1024 * 1024, alignment=64)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    size = 1024 + (i % 10) * 64
                    offset = allocator.alloc(size)
                    allocator.free(offset, size)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestAllocatorInvariants:
    """
    Defensive invariant tests for BufferAllocator: double-free, partial
    overlap corruption, and adjacent-block merging.

    Marked @pytest.mark.slow so they are skipped in quick CI but retained
    as regression safety-net.
    """

    @pytest.mark.slow
    def test_double_free_exact_is_safe(self):
        """Double-free of exact same block should warn but NOT crash."""
        allocator = BufferAllocator(total_size=4096, alignment=64)
        offset = allocator.alloc(256)
        allocator.free(offset, 256)
        # Second free of the same block — should be silently ignored
        allocator.free(offset, 256)
        # Pool should still be consistent: allocate full size back
        offset2 = allocator.alloc(4096)
        assert offset2 == 0

    @pytest.mark.slow
    def test_double_free_after_merge_is_safe(self):
        """
        Free A then B (adjacent → merged), then free A again.
        The allocator must detect A is already within the merged block.
        """
        allocator = BufferAllocator(total_size=4096, alignment=64)
        a = allocator.alloc(64)
        b = allocator.alloc(64)
        allocator.free(a, 64)
        allocator.free(b, 64)  # triggers merge with A
        # Now free A again — contained within the merged block
        allocator.free(a, 64)  # should not raise
        # Pool should still be fully usable
        offset = allocator.alloc(4096)
        assert offset == 0

    @pytest.mark.slow
    def test_partial_overlap_raises_corruption(self):
        """Freeing a region that partially overlaps a free block must raise RuntimeError."""
        allocator = BufferAllocator(total_size=4096, alignment=64)
        a = allocator.alloc(128)
        b = allocator.alloc(128)
        allocator.free(a, 128)  # [0, 128) is now free
        # Try to free a region that starts inside [0,128) but extends beyond
        with pytest.raises(RuntimeError):
            allocator.free(64, 128)  # [64, 192) overlaps with free [0, 128)

        # b is still allocated; freeing b should be fine
        allocator.free(b, 128)

    @pytest.mark.slow
    def test_merge_adjacent_blocks(self):
        """Free three adjacent blocks; they should merge into one contiguous region."""
        allocator = BufferAllocator(total_size=4096, alignment=64)
        a = allocator.alloc(64)
        b = allocator.alloc(64)
        c = allocator.alloc(64)
        # Free in non-sequential order to exercise sorting + merging
        allocator.free(b, 64)
        allocator.free(a, 64)
        allocator.free(c, 64)
        # After merge, free_blocks should contain one block starting at 0
        # covering at least 192 bytes (3 * 64).
        # Verify by allocating a contiguous block of 192 bytes.
        offset = allocator.alloc(192)
        assert offset == 0, "Adjacent blocks were not merged properly"

    @pytest.mark.slow
    def test_fragmentation_and_defrag(self):
        """
        Allocate A B C D that exactly fill the pool, free B and D to
        create fragmentation, verify a large contiguous alloc fails,
        then free A and C — should result in full defrag.
        """
        # Total pool = 4 * 64 = 256 bytes, so 4 allocs exhaust it completely
        allocator = BufferAllocator(total_size=256, alignment=64)
        a = allocator.alloc(64)
        b = allocator.alloc(64)
        c = allocator.alloc(64)
        d = allocator.alloc(64)

        allocator.free(b, 64)  # free blocks: [64, 128)
        allocator.free(d, 64)  # free blocks: [64, 128) and [192, 256)

        # Pool has two 64-byte holes; contiguous 128 is unavailable
        with pytest.raises(MemoryError):
            allocator.alloc(128)

        allocator.free(a, 64)
        allocator.free(c, 64)

        # After freeing everything, full pool should be available
        offset = allocator.alloc(256)
        assert offset == 0


@pytest.mark.core_model
class TestManagedBuffer:
    """Unit tests for ManagedBuffer."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        # automatically invoked for every test method in the class
        self.allocator = BufferAllocator(total_size=4096, alignment=64)
        self.pool = torch.zeros(4096, dtype=torch.uint8)

    def test_tensor_view(self):
        """Verify tensor property and as_tensor return correct views."""
        offset = self.allocator.alloc(64)
        buf = ManagedBuffer(self.allocator, offset, 64, self.pool)

        # Write float32 data via pool
        src = torch.arange(16, dtype=torch.float32)
        self.pool[offset : offset + 64] = src.view(torch.uint8)

        # Raw uint8 view
        assert buf.tensor.shape[0] == 64

        # Typed view
        typed = buf.as_tensor(dtype=torch.float32, shape=(4, 4))
        assert typed.shape == (4, 4)
        assert torch.equal(typed.flatten(), src)

        buf.release()

    def test_context_manager_releases_buffer(self):
        """Verify context manager releases buffer and space is reusable."""
        offset = self.allocator.alloc(128)

        with ManagedBuffer(self.allocator, offset, 128, self.pool) as buf:
            assert not buf._released

        assert buf._released

        # Space should be reusable
        new_offset = self.allocator.alloc(128)
        assert new_offset == offset

    def test_to_bytes_cpu_pool(self):
        """Smoke-test Mooncake's ManagedBuffer byte extraction on CPU pools."""
        expected = bytes(range(64))
        offset = self.allocator.alloc(len(expected))
        buf = ManagedBuffer(self.allocator, offset, len(expected), self.pool)
        self.pool[offset : offset + len(expected)] = torch.tensor(list(expected), dtype=torch.uint8)

        try:
            assert buf.to_bytes() == expected
        finally:
            buf.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU pool smoke test")
    def test_to_bytes_cuda_pool(self):
        """Smoke-test Mooncake's existing CUDA pool D2H byte path."""
        expected = bytes(range(64))
        allocator = BufferAllocator(total_size=4096, alignment=64)
        pool = torch.zeros(4096, dtype=torch.uint8, device="cuda:0")
        offset = allocator.alloc(len(expected))
        buf = ManagedBuffer(allocator, offset, len(expected), pool)
        pool[offset : offset + len(expected)].copy_(torch.tensor(list(expected), dtype=torch.uint8, device="cuda:0"))

        try:
            assert buf.to_bytes() == expected
        finally:
            buf.release()


def _make_kv_payload(device: str | torch.device = "cpu") -> tuple[KVCacheTransferData, torch.Tensor, torch.Tensor]:
    key_tensor = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    value_tensor = key_tensor + 100
    payload = KVCacheTransferData(
        request_id="moon-smoke",
        layer_blocks={"key_cache": [key_tensor], "value_cache": [value_tensor]},
        block_ids=[1],
        metadata={"seq_len": 3},
    )
    return payload, key_tensor, value_tensor


class TestMooncakePackedPayloadSmoke:
    """Smoke tests for Mooncake raw payload helpers without real Mooncake/RDMA."""

    def test_load_header_from_cpu_tensor(self):
        payload, _, _ = _make_kv_payload()
        raw = payload.to_bytes()
        packed = torch.tensor(list(raw), dtype=torch.uint8)
        header, data_start = KVCacheTransferData._load_header_from_tensor(packed)

        assert header["rid"] == "moon-smoke"
        assert data_start == 4 + int.from_bytes(raw[:4], "big")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for packed CUDA smoke test")
    def test_from_bytes_device_cuda_round_trip(self):
        payload, key_tensor, value_tensor = _make_kv_payload(device="cuda:0")
        packed = payload.to_gpu_tensor()

        header, data_start = KVCacheTransferData._load_header_from_tensor(packed)
        decoded = KVCacheTransferData.from_bytes_device(packed)

        assert header["rid"] == "moon-smoke"
        assert data_start == 4 + int.from_bytes(payload.to_bytes()[:4], "big")
        assert decoded["layer_blocks"]["key_cache"][0].is_cuda
        assert decoded["layer_blocks"]["value_cache"][0].is_cuda
        assert torch.equal(decoded["layer_blocks"]["key_cache"][0].cpu(), key_tensor.cpu())
        assert torch.equal(decoded["layer_blocks"]["value_cache"][0].cpu(), value_tensor.cpu())


@pytest.mark.core_model
class TestManagedBufferReleaseCallback:
    """Unit tests for ManagedBuffer release callbacks."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.allocator = BufferAllocator(total_size=4096, alignment=64)
        self.pool = torch.zeros(4096, dtype=torch.uint8)

    def test_release_callback_can_retain_allocation(self):
        """Verify warm-pool style release callbacks can retain an allocation."""
        retained = []
        offset = self.allocator.alloc(128)

        def retain(buf):
            retained.append(buf)
            return True

        buf = ManagedBuffer(
            self.allocator,
            offset,
            64,
            self.pool,
            release_callback=retain,
            allocation_size=128,
        )

        buf.release()

        assert buf._released
        assert retained == [buf]
        with pytest.raises(MemoryError):
            self.allocator.alloc(4096)

        buf._release_callback = None
        buf._released = False
        buf.release()
        assert self.allocator.alloc(4096) == 0


class TestWarmPool:
    """Unit tests for pre-warmed ManagedBuffer leasing."""

    def _connector(self, slots=2, slot_size=128):
        connector = MooncakeTransferEngineConnector.__new__(MooncakeTransferEngineConnector)
        connector._closed = False
        connector.allocator = BufferAllocator(total_size=1024, alignment=64)
        connector.pool = torch.zeros(1024, dtype=torch.uint8)
        connector.pool_prewarm_slots = slots
        connector.pool_prewarm_size = slot_size
        connector._warm_pool = _WarmPool()
        connector._warm_pool_lock = threading.Lock()
        connector._metrics = {"warm_pool_hits": 0, "warm_pool_misses": 0}
        return connector

    def test_warm_pool_no_stall_for_prewarmed_slots(self):
        connector = self._connector(slots=2, slot_size=128)

        MooncakeTransferEngineConnector._prewarm_pool(connector)
        buffers = [MooncakeTransferEngineConnector._alloc_managed_buffer(connector, 64) for _ in range(2)]

        assert connector._metrics["warm_pool_hits"] == 2
        assert connector._metrics["warm_pool_misses"] == 0
        assert len(connector._warm_pool.buffers) == 0

        for buf in buffers:
            buf.release()
        assert len(connector._warm_pool.buffers) == 2


class TestCachedSocket:
    """Unit tests for idle/error-triggered socket recreation."""

    class _FakeSocket:
        def __init__(self):
            self.closed = False
            self.connected_to = None
            self.options = {}

        def connect(self, addr):
            self.connected_to = addr

        def setsockopt(self, key, value):
            self.options[key] = value

        def close(self, linger=0):
            del linger
            self.closed = True

    class _FakeContext:
        def __init__(self):
            self.sockets = []

        def socket(self, socket_type):
            del socket_type
            sock = TestCachedSocket._FakeSocket()
            self.sockets.append(sock)
            return sock

    def _connector(self):
        connector = MooncakeTransferEngineConnector.__new__(MooncakeTransferEngineConnector)
        connector._req_local = threading.local()
        connector.zmq_ctx = self._FakeContext()
        connector.socket_health_interval_s = 30.0
        connector._metrics = {"socket_recreates": 0}
        return connector

    def test_socket_recovery_recreates_failed_cached_socket(self):
        connector = self._connector()
        addr = "tcp://127.0.0.1:5555"

        first = MooncakeTransferEngineConnector._get_req_socket(connector, addr, timeout_ms=100)
        connector._req_local.cache[addr].had_error = True
        second = MooncakeTransferEngineConnector._get_req_socket(connector, addr, timeout_ms=100)

        assert first is not second
        assert first.closed
        assert second.connected_to == addr
        assert connector._metrics["socket_recreates"] == 2


def test_profile_disabled_does_not_define_span():
    """Use _Span non-existence as the CI-safe proxy for zero profiling overhead.

    Full put/get profiling overhead needs Mooncake, but when profiling is off
    this verifies the module does not even define the span context manager that
    would allocate and enter/exit on hot paths.
    """
    if mooncake_module._PROFILE:
        pytest.skip("VLLM_OMNI_CONNECTOR_PROFILE is enabled for this test process")
    assert not hasattr(mooncake_module, "_Span")
