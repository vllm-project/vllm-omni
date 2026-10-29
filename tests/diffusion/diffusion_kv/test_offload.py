# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id

from vllm_omni.diffusion.diffusion_kv.offload import (
    DiffusionKVOffloadCoordinator,
    DiffusionKVOffloadIndex,
    KVPageOffloadStore,
    KVPageRestoreRequest,
    LocalKVPageOffloadTransport,
    OffloadedKVPage,
    make_page_key,
)
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK_SIZE = 4


def _hash(seed: int) -> bytes:
    return f"block-{seed}".encode()


def _page(
    seed: int,
    *,
    group_index: int = 0,
    source_block_id: int = 0,
    num_tokens: int = BLOCK_SIZE,
) -> OffloadedKVPage:
    return OffloadedKVPage.create(
        group_index=group_index,
        block_hash=_hash(seed),
        source_block_id=source_block_id,
        num_tokens=num_tokens,
    )


def _payload(seed: int, *, numel: int = 8) -> torch.Tensor:
    return torch.full((numel,), float(seed), dtype=torch.bfloat16)


class _FakeDevice:
    """Device-side KV blocks addressed by (group_index, block_id)."""

    def __init__(self) -> None:
        self.blocks: dict[tuple[int, int], torch.Tensor] = {}

    def reader(self, page: OffloadedKVPage) -> torch.Tensor:
        return self.blocks[(page.group_index, page.source_block_id)]

    def writer(self, request: KVPageRestoreRequest, payload: torch.Tensor) -> None:
        group_index = get_group_id(request.key)
        self.blocks[(group_index, request.target_block_id)] = payload.clone()


def _transport(device: _FakeDevice, store: KVPageOffloadStore | None = None) -> LocalKVPageOffloadTransport:
    return LocalKVPageOffloadTransport(
        store if store is not None else KVPageOffloadStore(),
        device_reader=device.reader,
        device_writer=device.writer,
    )


def test_page_key_matches_native_group_scoped_encoding() -> None:
    page = _page(1, group_index=3)

    assert page.key == make_page_key(_hash(1), 3)
    assert get_block_hash(page.key) == _hash(1)
    assert get_group_id(page.key) == 3


def test_same_block_hash_in_two_groups_yields_distinct_pages() -> None:
    group_zero = _page(1, group_index=0)
    group_one = _page(1, group_index=1)

    assert group_zero.key != group_one.key
    assert group_zero.block_hash == group_one.block_hash


def test_page_identity_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="group_index must be non-negative"):
        _page(1, group_index=-1)
    with pytest.raises(ValueError, match="source_block_id must be non-negative"):
        _page(1, source_block_id=-1)
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        _page(1, num_tokens=0)


def test_store_round_trips_payload_and_tracks_bytes() -> None:
    store = KVPageOffloadStore()
    page = _page(1)

    size = store.save(page, _payload(7))

    assert size == _payload(7).numel() * _payload(7).element_size()
    assert store.bytes_used == size
    assert len(store) == 1
    assert torch.equal(store.restore(page.key), _payload(7))
    assert store.page(page.key) is page


def test_store_save_copies_the_payload_instead_of_aliasing_it() -> None:
    store = KVPageOffloadStore()
    page = _page(1)
    payload = _payload(7)

    store.save(page, payload)
    payload.fill_(99.0)

    assert torch.equal(store.restore(page.key), _payload(7))


def test_store_resave_replaces_the_payload_and_keeps_one_copy() -> None:
    store = KVPageOffloadStore()
    page = _page(1)
    store.save(page, _payload(1))

    store.save(page, _payload(2))

    assert len(store) == 1
    assert torch.equal(store.restore(page.key), _payload(2))
    assert store.bytes_used == _payload(2).numel() * _payload(2).element_size()


def test_store_discard_releases_bytes() -> None:
    store = KVPageOffloadStore()
    page = _page(1)
    store.save(page, _payload(7))

    assert store.discard(page.key) is True
    assert store.discard(page.key) is False
    assert store.bytes_used == 0
    assert len(store) == 0
    with pytest.raises(KeyError):
        store.restore(page.key)


def test_store_rejects_non_tensor_payloads() -> None:
    store = KVPageOffloadStore()

    with pytest.raises(TypeError, match="payload must be a torch.Tensor"):
        store.save(_page(1), [1.0, 2.0])  # type: ignore[arg-type]


def test_index_release_below_zero_is_rejected() -> None:
    index = DiffusionKVOffloadIndex()
    page = _page(1)
    index.track(page)

    index.acquire(page.key)
    index.release(page.key)
    with pytest.raises(RuntimeError, match="no live reference to release"):
        index.release(page.key)


def test_index_remove_requires_no_live_references() -> None:
    index = DiffusionKVOffloadIndex()
    page = _page(1)
    index.track(page)
    index.acquire(page.key)

    with pytest.raises(RuntimeError, match="still has live references"):
        index.remove(page.key)

    index.release(page.key)
    assert index.remove(page.key) is page
    assert len(index) == 0


def test_index_track_is_idempotent_and_rejects_conflicting_identity() -> None:
    index = DiffusionKVOffloadIndex()
    page = _page(1)
    index.track(page, num_bytes=128)
    index.acquire(page.key)
    index.track(page, num_bytes=256)

    assert len(index) == 1
    assert index.refs(page.key) == 1
    assert index.bytes_used == 256

    # The key is derived from the block hash, so a conflicting identity can
    # only be produced by a hand-built page: same key, different content hash.
    conflicting = OffloadedKVPage(
        key=page.key,
        group_index=page.group_index,
        block_hash=b"other",
        source_block_id=page.source_block_id,
        num_tokens=page.num_tokens,
    )
    with pytest.raises(ValueError, match="Conflicting offloaded KV page identity"):
        index.track(conflicting)


def test_index_find_prefix_requires_a_contiguous_chain() -> None:
    index = DiffusionKVOffloadIndex()
    index.track(_page(1))
    index.track(_page(3))

    assert len(index.find_prefix([_hash(1), _hash(3)], 0)) == 2
    # Page 2 is missing, so page 3 is unreachable.
    assert len(index.find_prefix([_hash(1), _hash(2), _hash(3)], 0)) == 1
    assert index.find_prefix([_hash(2)], 0) == ()


def test_index_find_prefix_is_group_scoped() -> None:
    index = DiffusionKVOffloadIndex()
    index.track(_page(1, group_index=0))

    assert len(index.find_prefix([_hash(1)], 0)) == 1
    assert index.find_prefix([_hash(1)], 1) == ()


def test_coordinator_offload_tracks_pages_once() -> None:
    device = _FakeDevice()
    device.blocks[(0, 0)] = _payload(7)
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device))
    page = _page(1)

    offloaded = coordinator.offload_pages([page, page])

    assert offloaded == (page,)
    assert len(coordinator.index) == 1
    assert coordinator.bytes_used == _payload(7).numel() * _payload(7).element_size()


def test_coordinator_restores_into_a_fresh_device_block() -> None:
    device = _FakeDevice()
    device.blocks[(0, 0)] = _payload(7)
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device))
    page = _page(1, source_block_id=0)
    coordinator.offload_pages([page])

    # The source block is reused for unrelated content before the restore.
    device.blocks[(0, 0)] = _payload(123)
    coordinator.restore_pages([KVPageRestoreRequest(key=page.key, target_block_id=5)])

    assert torch.equal(device.blocks[(0, 5)], _payload(7))
    assert torch.equal(device.blocks[(0, 0)], _payload(123))
    assert coordinator.index.refs(page.key) == 1


def test_coordinator_restore_requires_a_tracked_page() -> None:
    device = _FakeDevice()
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device))
    page = _page(1)

    with pytest.raises(KeyError):
        coordinator.restore_pages([KVPageRestoreRequest(key=page.key, target_block_id=5)])


def test_coordinator_only_restores_a_prefix_shared_by_every_group() -> None:
    device = _FakeDevice()
    for group_index in range(2):
        device.blocks[(group_index, 0)] = _payload(group_index + 1)
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device))
    coordinator.offload_pages([_page(1, group_index=0), _page(1, group_index=1)])
    coordinator.offload_pages([_page(2, group_index=0)])

    candidates = coordinator.find_restore_candidates([[_hash(1), _hash(2)], [_hash(1)]])

    assert len(candidates) == 2
    assert all(len(prefix) == 1 for prefix in candidates)


def test_coordinator_budget_drops_only_unreferenced_pages() -> None:
    device = _FakeDevice()
    device.blocks[(0, 0)] = _payload(7)
    device.blocks[(0, 1)] = _payload(8)
    payload_bytes = _payload(7).numel() * _payload(7).element_size()
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device), max_bytes=payload_bytes)
    first = _page(1, source_block_id=0)
    second = _page(2, source_block_id=1)
    coordinator.offload_pages([first, second])

    # The first page is referenced by a restore, so only the second is dropped.
    coordinator.restore_pages([KVPageRestoreRequest(key=first.key, target_block_id=7)])
    dropped = coordinator.enforce_budget()

    assert dropped == (second.key,)
    assert len(coordinator.index) == 1
    assert first.key in coordinator.index
    assert coordinator.bytes_used == payload_bytes


def test_coordinator_keeps_referenced_pages_over_budget() -> None:
    device = _FakeDevice()
    device.blocks[(0, 0)] = _payload(7)
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device), max_bytes=1)
    page = _page(1)
    coordinator.offload_pages([page])
    coordinator.restore_pages([KVPageRestoreRequest(key=page.key, target_block_id=7)])

    assert coordinator.enforce_budget() == ()
    assert len(coordinator.index) == 1


def test_coordinator_rejects_an_invalid_budget() -> None:
    device = _FakeDevice()

    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        DiffusionKVOffloadCoordinator(transport=_transport(device), max_bytes=0)


def test_store_and_index_agree_on_which_pages_are_available() -> None:
    device = _FakeDevice()
    device.blocks[(0, 0)] = _payload(7)
    store = KVPageOffloadStore()
    coordinator = DiffusionKVOffloadCoordinator(transport=_transport(device, store))
    page = _page(1)
    coordinator.offload_pages([page])

    assert page.key in store
    assert page.key in coordinator.index
    assert store.bytes_used == coordinator.bytes_used


def _make_backend(num_blocks: int = 3) -> Any:
    from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
    from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend

    backend = object.__new__(DiffusionKVModelRunnerBackend)
    backend.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    backend.device = torch.device("cpu")
    backend.block_tables = SimpleNamespace(blocks_per_kv_block=[2])
    backend.kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(layer_names=["l0", "l1"])],
    )
    base = torch.arange(6 * 2 * 2 * 8, dtype=torch.float32).reshape(6, 2, 2, 8).to(torch.bfloat16)
    backend._kv_cache_layer_adapters = {
        "l0": SimpleNamespace(kv_cache=base.clone()),
        "l1": SimpleNamespace(kv_cache=base.add(1).to(torch.bfloat16)),
    }
    backend._kv_page_offload_store = None
    return backend


def test_backend_capture_covers_all_layers_and_kernel_blocks() -> None:
    backend = _make_backend()
    page = _page(1, source_block_id=1)

    sizes = backend.save_diffusion_kv_pages([page])

    expected = (
        torch.cat(
            [
                backend._kv_cache_layer_adapters["l0"].kv_cache[2:4],
                backend._kv_cache_layer_adapters["l1"].kv_cache[2:4],
            ]
        )
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
    )
    assert sizes == {page.key: expected.numel()}
    store = backend._kv_page_offload_store
    assert store is not None and len(store) == 1
    assert torch.equal(store.restore(page.key), expected)
    assert store.bytes_used == expected.numel()


def test_backend_capture_uses_each_group_kernel_block_count() -> None:
    backend = _make_backend()
    backend.block_tables = SimpleNamespace(blocks_per_kv_block=[2, 1])
    backend.kv_cache_config.kv_cache_groups.append(SimpleNamespace(layer_names=["l2"]))
    backend._kv_cache_layer_adapters["l2"] = SimpleNamespace(
        kv_cache=backend._kv_cache_layer_adapters["l0"].kv_cache.add(2).to(torch.bfloat16)
    )

    group_zero = _page(1, group_index=0, source_block_id=1)
    group_one = _page(1, group_index=1, source_block_id=2)
    sizes = backend.save_diffusion_kv_pages([group_zero, group_one])

    assert group_zero.key != group_one.key
    store = backend._kv_page_offload_store
    assert torch.equal(
        store.restore(group_one.key),
        backend._kv_cache_layer_adapters["l2"].kv_cache[2:3].contiguous().view(torch.uint8).reshape(-1),
    )
    # Group zero spans two layers x two kernel blocks; group one one layer x one block.
    assert sizes[group_zero.key] == 4 * sizes[group_one.key]


def test_backend_capture_rejects_unknown_group_and_block() -> None:
    backend = _make_backend(num_blocks=2)

    with pytest.raises(ValueError, match="unknown cache group"):
        backend.save_diffusion_kv_pages([_page(1, group_index=5)])
    with pytest.raises(ValueError, match="outside"):
        backend.save_diffusion_kv_pages([_page(1, source_block_id=9)])


def test_backend_capture_requires_initialized_paged_kv() -> None:
    from types import SimpleNamespace

    from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
    from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend

    backend = object.__new__(DiffusionKVModelRunnerBackend)
    backend.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)

    with pytest.raises(RuntimeError, match="not initialized"):
        backend.save_diffusion_kv_pages([])


class _FakeExecutor(DiffusionExecutor):
    def __init__(self, replies: list) -> None:
        self.replies = replies
        self.calls: list[tuple[str, tuple]] = []

    def _init_executor(self) -> None:
        return None

    def is_dead(self) -> bool:
        return False

    def execute_request(self, scheduler_output):
        raise NotImplementedError

    def execute_batch(self, scheduler_output):
        raise NotImplementedError

    def execute_step(self, scheduler_output):
        raise NotImplementedError

    def shutdown(self) -> None:
        return None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None, exec_all_ranks=False):
        self.calls.append((method, args))
        return self.replies

    def check_health(self) -> None:
        return None


def test_executor_save_dedupes_pages_and_returns_sizes() -> None:
    page = _page(1)
    executor = _FakeExecutor(replies=[{page.key: 128}])

    sizes = executor.save_diffusion_kv_pages([page, page])

    assert sizes == {page.key: 128}
    method, args = executor.calls[0]
    assert method == "save_diffusion_kv_pages"
    assert args == ([page],)


def test_executor_save_rejects_disagreeing_ranks() -> None:
    page = _page(1)
    executor = _FakeExecutor(replies=[{page.key: 128}, {page.key: 256}])

    with pytest.raises(RuntimeError, match="disagrees on saved Diffusion KV page sizes"):
        executor.save_diffusion_kv_pages([page])


def test_executor_save_with_no_pages_skips_the_rpc() -> None:
    executor = _FakeExecutor(replies=[])

    assert executor.save_diffusion_kv_pages([]) == {}
    assert executor.calls == []
