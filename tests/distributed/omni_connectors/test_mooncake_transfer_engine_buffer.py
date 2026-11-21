# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for BufferAllocator and ManagedBuffer.
These tests do NOT require Mooncake or RDMA environment.
"""

import threading

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    BufferAllocator,
    ManagedBuffer,
    MooncakeTransferEngineConnector,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import KVCacheTransferData
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerde

# CPU cases opt in below. The CUDA pool smokes use hardware_test instead.
pytestmark = [pytest.mark.parallel, pytest.mark.core_model]


@pytest.mark.core_model
@pytest.mark.cpu
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


@pytest.mark.cpu
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

    @pytest.mark.cpu
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

    @pytest.mark.cpu
    def test_context_manager_releases_buffer(self):
        """Verify context manager releases buffer and space is reusable."""
        offset = self.allocator.alloc(128)

        with ManagedBuffer(self.allocator, offset, 128, self.pool) as buf:
            assert not buf._released

        assert buf._released

        # Space should be reusable
        new_offset = self.allocator.alloc(128)
        assert new_offset == offset

    @pytest.mark.cpu
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

    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
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

    @pytest.mark.cpu
    def test_load_header_from_cpu_tensor(self):
        payload, _, _ = _make_kv_payload()
        raw = payload.to_bytes()
        packed = torch.tensor(list(raw), dtype=torch.uint8)
        header, data_start = KVCacheTransferData._load_header_from_tensor(packed)

        assert header["rid"] == "moon-smoke"
        assert data_start == 4 + int.from_bytes(raw[:4], "big")

    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is required for packed CUDA smoke test",
    )
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


@pytest.mark.cpu
def test_put_non_raw_payload_copies_reusable_encode_buffer():
    connector = object.__new__(MooncakeTransferEngineConnector)
    connector._closed = False
    connector.can_put = True
    connector._put_lock = threading.Lock()
    connector._encode_buffer = threading.local()
    connector._serde = OmniSerde()
    connector.allocator = BufferAllocator(total_size=4096, alignment=64)
    connector.pool = torch.zeros(4096, dtype=torch.uint8)
    connector.base_ptr = connector.pool.data_ptr()
    connector._local_buffers = {}
    connector._local_buffers_lock = threading.Lock()
    connector._metrics = {"puts": 0, "bytes_transferred": 0, "errors": 0}
    connector.host = "127.0.0.1"
    connector.zmq_port = 1

    payload = {"text": "hello" * 100}
    ok, size, metadata = connector._put_impl("key-1", payload)
    assert ok
    assert size > 0
    assert metadata is not None
    assert metadata["is_fast_path"] is False

    src_addrs, lengths, holder, _should_release, _is_fast_path, _created = connector._local_buffers["key-1"]
    assert src_addrs[0] == connector.base_ptr
    assert lengths[0] == size
    assert isinstance(holder, ManagedBuffer)
    assert holder.size == size
    copied = bytes(holder.tensor.cpu().numpy())
    assert OmniSerde().deserialize(copied) == payload

    ok, _size, _metadata = connector._put_impl("key-2", {"text": "world" * 200})
    assert ok
    copied_after = bytes(holder.tensor.cpu().numpy())
    assert OmniSerde().deserialize(copied_after) == payload
    connector._closed = True
