# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the CuMemAllocator free-path patch (#8016).

Releasing an inline diffusion engine's MemPool makes torch free every cached
block through ``_python_free_callback``. Blocks already swept by
``use_memory_pool()`` or unmapped by ``sleep()`` must come back as handles the
C extension can release, instead of raising and poisoning later frees.
"""

import pytest
from vllm.device_allocator import AllocationData
from vllm.device_allocator import cumem as cumem_module
from vllm.device_allocator.cumem import CuMemAllocator

import vllm_omni.patch  # noqa: F401

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HANDLE = (0, 2 << 20, 0x7F0000000000, 1234)
PTR = HANDLE[2]


@pytest.fixture
def allocator(monkeypatch):
    allocator = CuMemAllocator.__new__(CuMemAllocator)
    allocator.pointer_to_data = {}
    monkeypatch.setattr(CuMemAllocator, "instance", allocator)
    calls = []
    monkeypatch.setattr(cumem_module, "python_unmap_and_release", lambda *h: calls.append(("unmap", h)))
    monkeypatch.setattr(cumem_module, "python_create_and_map", lambda *h: calls.append(("map", h)))
    allocator.calls = calls
    return allocator


def test_swept_block_is_rebacked_when_torch_frees_it(allocator):
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights")
    # use_memory_pool() exit sweep: untrack, then unmap the idle segment.
    cumem_module.unmap_and_release(allocator._python_free_callback(PTR))
    assert PTR not in allocator.pointer_to_data

    # The MemPool still caches the block and frees it again on release.
    assert allocator._python_free_callback(PTR) == HANDLE
    assert allocator.calls == [("unmap", HANDLE), ("map", HANDLE)]
    with pytest.raises(KeyError):
        allocator._python_free_callback(PTR)


def test_asleep_block_is_rebacked_on_free(allocator):
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights", is_asleep=True)

    assert allocator._python_free_callback(PTR) == HANDLE
    assert allocator.calls == [("map", HANDLE)]
    assert PTR not in allocator.pointer_to_data


def test_sleep_unmap_of_tracked_block_is_not_recorded_as_swept(allocator):
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights")
    cumem_module.unmap_and_release(HANDLE)

    assert not allocator.__dict__.get("_omni_swept_handles")


def test_unknown_pointer_still_raises(allocator):
    with pytest.raises(KeyError):
        allocator._python_free_callback(PTR)
