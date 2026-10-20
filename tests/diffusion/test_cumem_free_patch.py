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

import vllm_omni.patch as omni_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HANDLE = (0, 2 << 20, 0x7F0000000000, 1234)
EMPTY_HANDLE: tuple[int, int, int, list[int]] = (*HANDLE[:3], [])
PTR = HANDLE[2]


@pytest.fixture(params=[pytest.param(False, id="cuda"), pytest.param(True, id="rocm")])
def allocator(request, monkeypatch):
    allocator = CuMemAllocator.__new__(CuMemAllocator)
    allocator.pointer_to_data = {}
    monkeypatch.setattr(CuMemAllocator, "instance", allocator)
    monkeypatch.setattr(omni_patch, "_cumem_platform_is_rocm", lambda: request.param)
    calls = []
    monkeypatch.setattr(cumem_module, "python_unmap_and_release", lambda *h: calls.append(("unmap", h)))
    monkeypatch.setattr(cumem_module, "python_create_and_map", lambda *h: calls.append(("map", h)))
    allocator.calls = calls
    return allocator, request.param


def test_swept_block_is_releasable_when_torch_frees_it(allocator):
    allocator, is_rocm = allocator
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights")
    # use_memory_pool() exit sweep: untrack, then unmap the idle segment.
    cumem_module.unmap_and_release(allocator._python_free_callback(PTR))
    assert PTR not in allocator.pointer_to_data

    # The MemPool still caches the block and frees it again on release.
    assert allocator._python_free_callback(PTR) == (EMPTY_HANDLE if is_rocm else HANDLE)
    expected_calls = [("unmap", HANDLE)]
    if not is_rocm:
        expected_calls.append(("map", HANDLE))
    assert allocator.calls == expected_calls
    with pytest.raises(KeyError):
        allocator._python_free_callback(PTR)


def test_asleep_block_is_releasable_on_free(allocator):
    allocator, is_rocm = allocator
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights", is_asleep=True)

    assert allocator._python_free_callback(PTR) == (EMPTY_HANDLE if is_rocm else HANDLE)
    assert allocator.calls == ([] if is_rocm else [("map", HANDLE)])
    assert PTR not in allocator.pointer_to_data


def test_sleep_unmap_of_tracked_block_is_not_recorded_as_swept(allocator):
    allocator, _ = allocator
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights")
    cumem_module.unmap_and_release(HANDLE)

    assert not allocator.__dict__.get("_omni_swept_handles")


def test_shutdown_retirement_drains_mapped_and_asleep_blocks(allocator):
    allocator, is_rocm = allocator
    asleep_handle = (HANDLE[0], HANDLE[1], HANDLE[2] + HANDLE[1], HANDLE[3])
    asleep_ptr = asleep_handle[2]
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights")
    allocator.pointer_to_data[asleep_ptr] = AllocationData(
        asleep_handle,
        "weights",
        is_asleep=True,
    )

    retired_bytes = omni_patch.release_cumem_allocations_for_shutdown(allocator)

    assert retired_bytes == HANDLE[1] + asleep_handle[1]
    assert allocator.get_current_usage() == 0
    assert allocator.__dict__["_omni_swept_handles"] == {
        PTR: HANDLE,
        asleep_ptr: asleep_handle,
    }
    assert allocator.calls == [("unmap", HANDLE)]

    expected_handle = EMPTY_HANDLE if is_rocm else HANDLE
    assert allocator._python_free_callback(PTR) == expected_handle
    expected_asleep_handle = (*asleep_handle[:3], []) if is_rocm else asleep_handle
    assert allocator._python_free_callback(asleep_ptr) == expected_asleep_handle
    expected_calls = [("unmap", HANDLE)]
    if not is_rocm:
        expected_calls.extend([("map", HANDLE), ("map", asleep_handle)])
    assert allocator.calls == expected_calls


def test_shutdown_retirement_keeps_failed_unmap_tracked(allocator, monkeypatch):
    allocator, _ = allocator
    allocator.pointer_to_data[PTR] = AllocationData(HANDLE, "weights")

    def fail_unmap(*_handle):
        raise RuntimeError("unmap failed")

    monkeypatch.setattr(cumem_module, "unmap_and_release", fail_unmap)

    with pytest.raises(RuntimeError, match="unmap failed"):
        omni_patch.release_cumem_allocations_for_shutdown(allocator)

    assert allocator.pointer_to_data[PTR].handle == HANDLE
    assert not allocator.__dict__.get("_omni_swept_handles")


def test_unknown_pointer_still_raises(allocator):
    allocator, _ = allocator
    with pytest.raises(KeyError):
        allocator._python_free_callback(PTR)
