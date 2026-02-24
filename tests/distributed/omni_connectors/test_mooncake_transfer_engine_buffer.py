# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for BufferAllocator and ManagedBuffer.
These tests do NOT require Mooncake or RDMA environment.
"""

import threading
import unittest

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    BufferAllocator,
    ManagedBuffer,
)

# All tests in this file are pure-CPU unit tests for the memory allocator.
pytestmark = [pytest.mark.cpu, pytest.mark.parallel]


@pytest.mark.core_model
class TestBufferAllocator(unittest.TestCase):
    """Unit tests for BufferAllocator."""

    def test_basic_alloc_free(self):
        """Verify alloc, free, and reuse of freed space."""
        allocator = BufferAllocator(total_size=4096, alignment=64)

        offset1 = allocator.alloc(512)
        self.assertEqual(offset1, 0)

        offset2 = allocator.alloc(512)
        self.assertGreater(offset2, 0)

        # Free first block, should be reusable
        allocator.free(offset1, 512)
        offset3 = allocator.alloc(512)
        self.assertEqual(offset3, 0)

    def test_alignment(self):
        """Verify allocation respects alignment."""
        allocator = BufferAllocator(total_size=4096, alignment=128)

        _offset1 = allocator.alloc(100)
        offset2 = allocator.alloc(100)

        self.assertEqual(offset2 % 128, 0)
        self.assertEqual(offset2, 128)

    def test_exhaustion_and_recovery(self):
        """Test that full allocation fails, then succeeds after free."""
        allocator = BufferAllocator(total_size=1024, alignment=64)

        offset = allocator.alloc(1024)
        self.assertEqual(offset, 0)

        with self.assertRaises(MemoryError):
            allocator.alloc(64)

        allocator.free(offset, 1024)
        offset2 = allocator.alloc(1024)
        self.assertEqual(offset2, 0)

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

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


class TestAllocatorInvariants(unittest.TestCase):
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
        self.assertEqual(offset2, 0)

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
        self.assertEqual(offset, 0)

    @pytest.mark.slow
    def test_partial_overlap_raises_corruption(self):
        """Freeing a region that partially overlaps a free block must raise RuntimeError."""
        allocator = BufferAllocator(total_size=4096, alignment=64)
        a = allocator.alloc(128)
        b = allocator.alloc(128)
        allocator.free(a, 128)  # [0, 128) is now free
        # Try to free a region that starts inside [0,128) but extends beyond
        with self.assertRaises(RuntimeError):
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
        self.assertEqual(offset, 0, "Adjacent blocks were not merged properly")

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
        with self.assertRaises(MemoryError):
            allocator.alloc(128)

        allocator.free(a, 64)
        allocator.free(c, 64)

        # After freeing everything, full pool should be available
        offset = allocator.alloc(256)
        self.assertEqual(offset, 0)


@pytest.mark.core_model
class TestManagedBuffer(unittest.TestCase):
    """Unit tests for ManagedBuffer."""

    def setUp(self):
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
        self.assertEqual(buf.tensor.shape[0], 64)

        # Typed view
        typed = buf.as_tensor(dtype=torch.float32, shape=(4, 4))
        self.assertEqual(typed.shape, (4, 4))
        self.assertTrue(torch.equal(typed.flatten(), src))

        buf.release()

    def test_context_manager_releases_buffer(self):
        """Verify context manager releases buffer and space is reusable."""
        offset = self.allocator.alloc(128)

        with ManagedBuffer(self.allocator, offset, 128, self.pool) as buf:
            self.assertFalse(buf._released)

        self.assertTrue(buf._released)

        # Space should be reusable
        new_offset = self.allocator.alloc(128)
        self.assertEqual(new_offset, offset)


if __name__ == "__main__":
    unittest.main()
