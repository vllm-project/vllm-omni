# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm_omni.model_executor.models.easymagpie.buffer_leases import BufferLeasePool, storage_owned_host_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Event:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


def test_host_reuse_waits_for_detached_tensor_and_numpy_storage_aliases():
    pool = BufferLeasePool(2)
    lease = pool.acquire(8, dtype=torch.float32, device="cpu")
    pointer = lease.tensor.data_ptr()
    lease.tensor.copy_(torch.arange(8, dtype=torch.float32))
    tensor = storage_owned_host_tensor(lease)
    detached = tensor[1:5].detach()
    numpy_view = detached.numpy()[1:]
    del lease, tensor
    gc.collect()
    assert pool.status()["active"] == 1
    del detached
    gc.collect()
    assert pool.status()["active"] == 1
    assert numpy_view.tolist() == [2, 3, 4]
    del numpy_view
    gc.collect()
    assert pool.status()["active"] == 0
    reused = pool.acquire(8, dtype=torch.float32, device="cpu")
    assert reused.tensor.data_ptr() == pointer
    assert pool.status()["reuses"] == 1


def test_event_fence_survives_early_consumer_disposal():
    pool = BufferLeasePool(1)
    event = Event()
    lease = pool.acquire(8, dtype=torch.float32, device="cpu")
    pointer = lease.tensor.data_ptr()
    lease.ready_after(event)
    tensor = storage_owned_host_tensor(lease)
    del lease, tensor
    gc.collect()
    assert pool.status()["states"] == {"pending": 1}
    other = pool.acquire(8, dtype=torch.float32, device="cpu")
    assert other.tensor.data_ptr() != pointer
    other.release()
    event.ready = True
    recovered = pool.acquire(8, dtype=torch.float32, device="cpu")
    assert recovered.tensor.data_ptr() == pointer


def test_unpooled_pending_dma_retains_source_until_completion():
    pool = BufferLeasePool(1)
    occupied = pool.acquire(1, dtype=torch.float32, device="cpu")
    event = Event()
    transient = pool.acquire(1, dtype=torch.float32, device="cpu")
    transient.ready_after(event)
    transient.release()
    assert pool.status()["pending_unpooled"] == 1
    event.ready = True
    occupied.release()
    next_lease = pool.acquire(1, dtype=torch.float32, device="cpu")
    assert pool.status()["pending_unpooled"] == 0
    next_lease.release()


def test_double_release_cannot_release_a_new_generation():
    pool = BufferLeasePool(1)
    old = pool.acquire(1, dtype=torch.float32, device="cpu")
    old.release()
    new = pool.acquire(1, dtype=torch.float32, device="cpu")
    old.release()
    assert pool.status()["active"] == 1
    new.release()
    assert pool.status()["active"] == 0


def test_repeated_zero_and_mixed_sizes_keep_capacity_bounded():
    pool = BufferLeasePool(2)
    for size in [0, 8, 4, 7, 2, 8] * 10:
        lease = pool.acquire(size, dtype=torch.float32, device="cpu")
        assert lease.tensor.numel() == size
        lease.release()
    assert pool.status()["retained_entries"] <= 2
    assert pool.status()["active"] == 0
    assert pool.status()["reuses"] > 50


def test_concurrent_leases_never_alias_live_owners():
    pool = BufferLeasePool(2)
    barrier = threading.Barrier(8)

    def worker(index):
        lease = pool.acquire(8, dtype=torch.float32, device="cpu")
        lease.tensor.fill_(index)
        barrier.wait(timeout=5)
        assert lease.tensor.tolist() == [index] * 8
        pointer = lease.tensor.data_ptr()
        barrier.wait(timeout=5)
        lease.release()
        return pointer

    with ThreadPoolExecutor(max_workers=8) as executor:
        pointers = list(executor.map(worker, range(8)))
    assert len(set(pointers)) == 8
    assert pool.status()["active"] == 0


def test_host_export_rejects_wrong_dtype():
    pool = BufferLeasePool(1)
    lease = pool.acquire(8, dtype=torch.int64, device="cpu")
    with pytest.raises(ValueError):
        storage_owned_host_tensor(lease)


def test_group_metadata_pool_preserves_routing_initial_flags_and_source():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.easymagpie.codec.uniform_groups import CodecUniformGroups

    original = SimpleNamespace(
        num_decodes=2,
        num_prefills=3,
        num_decode_tokens=2,
        num_prefill_tokens=6,
        num_reqs=5,
        state_indices_tensor_d=torch.tensor([[5], [9]]),
        state_indices_tensor_p=torch.tensor([2, 8, 11]),
        has_initial_states_p=torch.tensor([False, True, False]),
    )
    pool = BufferLeasePool(24)
    expected = CodecUniformGroups._group_metadata(original, [4, 2, 3], 2, torch.device("cpu"))
    for _ in range(4):
        leases: list = []
        actual = CodecUniformGroups._group_metadata(original, [4, 2, 3], 2, torch.device("cpu"), pool, leases)
        for name in ("state_indices_tensor_p", "has_initial_states_p", "query_start_loc_p"):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
        assert actual.num_prefills == actual.num_reqs == 3
        assert original.state_indices_tensor_p.tolist() == [2, 8, 11]
        assert original.has_initial_states_p.tolist() == [False, True, False]
        for lease in leases:
            lease.release()
    assert pool.status()["reuses"] > 0
    assert pool.status()["active"] == 0


def test_startup_small_buffers_grow_to_c512_without_permanent_transient_allocations():
    pool = BufferLeasePool(2)
    startup = [pool.acquire(32, dtype=torch.float32, device="cpu") for _ in range(2)]
    for lease in startup:
        lease.release()
    size = 512 * 8 * 1764
    for iteration in range(4):
        leases = [pool.acquire(size, dtype=torch.float32, device="cpu") for _ in range(2)]
        assert all(lease.tensor.numel() == size for lease in leases)
        for lease in leases:
            lease.release()
    status = pool.status()
    assert status["retained_entries"] == 2
    assert status["free_slot_grows"] == 2
    assert status.get("unpooled_allocations", 0) == 0
    assert status["allocations"] == 4
    assert status["reuses"] == 6
    assert status["active"] == 0


def test_free_slot_growth_never_evicts_live_numpy_alias_or_pending_dma():
    pool = BufferLeasePool(2)
    live = pool.acquire(8, dtype=torch.float32, device="cpu")
    live.tensor.fill_(7)
    alias = storage_owned_host_tensor(live).numpy()
    del live
    pending = pool.acquire(8, dtype=torch.float32, device="cpu")
    event = Event()
    pending.ready_after(event)
    pending.release()
    transient = pool.acquire(32, dtype=torch.float32, device="cpu")
    assert pool.status().get("free_slot_replacements", 0) == 0
    assert pool.status()["unpooled_allocations"] == 1
    assert alias.tolist() == [7] * 8
    transient.release()
    event.ready = True
    grown = pool.acquire(32, dtype=torch.float32, device="cpu")
    assert pool.status()["free_slot_grows"] == 1
    assert alias.tolist() == [7] * 8
    grown.release()
    del alias
    gc.collect()
    assert pool.status()["active"] == 0
