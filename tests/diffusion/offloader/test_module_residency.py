# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.module_residency as residency_module
from vllm_omni.diffusion.offloader.module_residency import (
    BoundedAllocatorCache,
    PinnedModuleStager,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _DummyEvent:
    def __init__(self) -> None:
        self.record_count = 0

    def record(self, _stream) -> None:
        self.record_count += 1


@contextmanager
def _dummy_stream(_stream):
    yield


@pytest.fixture
def patched_runtime(monkeypatch, mocker):
    empty_cache = mocker.Mock()
    synchronize = mocker.Mock()
    monkeypatch.setattr(residency_module.current_omni_platform, "Stream", _DummyStream)
    monkeypatch.setattr(residency_module.current_omni_platform, "Event", _DummyEvent)
    monkeypatch.setattr(residency_module.current_omni_platform, "current_stream", _DummyStream)
    monkeypatch.setattr(residency_module.current_omni_platform, "stream", _dummy_stream)
    monkeypatch.setattr(residency_module.current_omni_platform, "empty_cache", empty_cache)
    monkeypatch.setattr(residency_module.current_omni_platform, "synchronize", synchronize)
    return empty_cache, synchronize


def _aliased_modules() -> tuple[nn.Module, nn.Module]:
    parameter_storage = torch.arange(12, dtype=torch.float32)
    buffer_storage = torch.arange(8, dtype=torch.int64)
    left = nn.Module()
    right = nn.Module()
    left.weight = nn.Parameter(parameter_storage[:8].view(2, 4))
    right.weight = nn.Parameter(parameter_storage[4:].view(2, 4))
    left.register_buffer("state", buffer_storage[:6])
    right.register_buffer("state", buffer_storage[2:])
    return left, right


def test_module_group_preserves_parameter_and_buffer_storage_aliases(patched_runtime, mocker):
    left, right = _aliased_modules()
    expected_left_weight = left.weight.detach().clone()
    expected_right_weight = right.weight.detach().clone()
    expected_left_state = left.state.clone()
    expected_right_state = right.state.clone()
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    stager = PinnedModuleStager(
        [left, right],
        torch.device("cpu"),
        pin_memory=False,
        cache_retention=retention,
    )

    master_weight_ptr = left.weight.untyped_storage().data_ptr()
    master_state_ptr = left.state.untyped_storage().data_ptr()
    assert right.weight.untyped_storage().data_ptr() == master_weight_ptr
    assert right.state.untyped_storage().data_ptr() == master_state_ptr
    assert (left.weight.storage_offset(), right.weight.storage_offset()) == (0, 4)
    assert (left.state.storage_offset(), right.state.storage_offset()) == (0, 2)

    stager.load()

    assert left.weight.untyped_storage().data_ptr() != master_weight_ptr
    assert right.weight.untyped_storage().data_ptr() == left.weight.untyped_storage().data_ptr()
    assert right.state.untyped_storage().data_ptr() == left.state.untyped_storage().data_ptr()
    left.weight.data[1, 0] = -1
    left.state[2] = -1
    assert right.weight[0, 0].item() == -1
    assert right.state[0].item() == -1

    stager.offload()

    assert left.weight.untyped_storage().data_ptr() == master_weight_ptr
    assert right.weight.untyped_storage().data_ptr() == master_weight_ptr
    assert left.state.untyped_storage().data_ptr() == master_state_ptr
    assert right.state.untyped_storage().data_ptr() == master_state_ptr
    assert torch.equal(left.weight, expected_left_weight)
    assert torch.equal(right.weight, expected_right_weight)
    assert torch.equal(left.state, expected_left_state)
    assert torch.equal(right.state, expected_right_state)
    retention.release_if_needed.assert_called_once_with(force=False)


def test_staging_transitions_are_idempotent_and_reuse_one_event(patched_runtime, mocker):
    module = nn.Linear(2, 2)
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    stager = PinnedModuleStager(
        module,
        torch.device("cpu"),
        pin_memory=False,
        cache_retention=retention,
    )
    ready_event = stager._ready_event

    stager.load()
    first_storage = module.weight.untyped_storage().data_ptr()
    stager.load()
    assert module.weight.untyped_storage().data_ptr() == first_storage
    assert stager._ready_event is ready_event
    assert ready_event.record_count == 1

    stager.offload()
    stager.offload()
    assert not stager.loaded
    retention.release_if_needed.assert_called_once_with(force=False)


def test_failed_load_restores_master_and_forces_cache_release(monkeypatch, patched_runtime, mocker):
    module = nn.Linear(2, 2)
    expected = module.weight.detach().clone()
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    stager = PinnedModuleStager(
        module,
        torch.device("cpu"),
        pin_memory=False,
        cache_retention=retention,
    )
    master_ptr = module.weight.untyped_storage().data_ptr()

    def fail_after_rebind() -> None:
        device_storages = [torch.empty_like(group.master) for group in stager._groups]
        stager._bind(device_storages)
        module.weight.data.zero_()
        raise RuntimeError("copy failed")

    monkeypatch.setattr(stager, "_load_once", fail_after_rebind)
    with pytest.raises(RuntimeError, match="copy failed"):
        stager.load()

    assert not stager.loaded
    assert module.weight.untyped_storage().data_ptr() == master_ptr
    assert torch.equal(module.weight, expected)
    retention.release_if_needed.assert_called_once_with(force=True)


def test_out_of_memory_releases_cache_and_retries_once(monkeypatch, patched_runtime, mocker):
    module = nn.Linear(2, 2)
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    stager = PinnedModuleStager(
        module,
        torch.device("cpu"),
        pin_memory=False,
        cache_retention=retention,
    )
    load_once = stager._load_once
    attempts = 0

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.OutOfMemoryError("memory pressure")
        load_once()

    monkeypatch.setattr(stager, "_load_once", fail_once)
    stager.load()

    assert stager.loaded
    assert attempts == 2
    retention.release_if_needed.assert_called_once_with(force=True)


@pytest.mark.parametrize(
    ("reserved", "allocated", "free", "expected_release"),
    [
        (20, 10, 80, False),
        (40, 10, 80, True),
        (20, 10, 4, True),
    ],
)
def test_allocator_cache_policy_enforces_cache_and_free_memory_bounds(
    monkeypatch,
    patched_runtime,
    reserved,
    allocated,
    free,
    expected_release,
):
    empty_cache, _ = patched_runtime
    telemetry = {"reserved": 20, "allocated": 10, "free": 80}
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: telemetry["reserved"])
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: telemetry["allocated"])
    monkeypatch.setattr(
        residency_module.current_omni_platform,
        "get_device_memory",
        lambda _device: (telemetry["free"], 400),
    )
    retention = BoundedAllocatorCache(torch.device("cpu"))

    # Prime the learned budget: the cold-start check releases (budget 0) and
    # records cached = 10, so later checks retain at budget 12.5.
    assert retention.release_if_needed()
    assert retention.observed_cached_peak == 10
    empty_cache.reset_mock()

    telemetry.update(reserved=reserved, allocated=allocated, free=free)

    released = retention.release_if_needed()

    assert released is expected_release
    assert empty_cache.call_count == int(expected_release)


def test_attached_stagers_form_a_detach_ledger_that_resets_the_observed_peak(patched_runtime):
    left, right = _aliased_modules()
    retention = BoundedAllocatorCache(torch.device("cpu"))
    assert retention.budget_bytes == 0

    stager = PinnedModuleStager(
        [left, right],
        torch.device("cpu"),
        pin_memory=False,
        cache_retention=retention,
    )
    expected_bytes = 12 * 4 + 8 * 8  # float32 parameter + int64 buffer storages
    assert stager.staged_bytes == expected_bytes
    assert retention._staged_bytes == expected_bytes

    # Re-attaching the same cache does not double count.
    stager.set_cache_retention(retention)
    assert retention._staged_bytes == expected_bytes

    # A second stager adds its own staged bytes to the ledger.
    extra = PinnedModuleStager(torch.nn.Linear(2, 2), torch.device("cpu"), pin_memory=False)
    extra.set_cache_retention(retention)
    assert retention._staged_bytes == expected_bytes + extra.staged_bytes

    # A learned peak survives a partial detach...
    retention._observed_cached_peak = 30
    extra.set_cache_retention(None)
    assert retention.observed_cached_peak == 30
    assert retention.budget_bytes == 37

    # ...but the last detach resets it so one workload cannot influence the next.
    stager.set_cache_retention(None)
    assert retention.observed_cached_peak == 0
    assert retention.budget_bytes == 0


def test_allocator_cache_budget_is_capped_by_device_capacity(monkeypatch, patched_runtime):
    empty_cache, _ = patched_runtime
    telemetry = {"reserved": 40, "allocated": 10, "free": 80}
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: telemetry["reserved"])
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: telemetry["allocated"])
    monkeypatch.setattr(
        residency_module.current_omni_platform,
        # Capacity 100 keeps the 25% cap at 25, below the learned budget of
        # 30 * 1.25 = 37.5, so the cap must govern.
        "get_device_memory",
        lambda _device: (telemetry["free"], 100),
    )
    retention = BoundedAllocatorCache(torch.device("cpu"))

    # Cold start: budget 0 releases and records the peak of 30.
    assert retention.release_if_needed()
    assert retention.observed_cached_peak == 30
    empty_cache.reset_mock()

    # The learned budget 37.5 is capped to 25, so cached 30 keeps releasing.
    assert retention.release_if_needed()
    assert retention.release_if_needed()
    empty_cache.reset_mock()

    # Within the cap (cached 20 <= 25) the policy retains.
    telemetry.update(reserved=30, allocated=10)
    assert not retention.release_if_needed()
    assert empty_cache.call_count == 0


def test_allocator_cache_releases_when_telemetry_is_unavailable(monkeypatch, patched_runtime):
    empty_cache, _ = patched_runtime

    def unavailable(_device):
        raise NotImplementedError

    monkeypatch.setattr(torch.accelerator, "memory_reserved", unavailable)
    retention = BoundedAllocatorCache(torch.device("cpu"))

    assert retention.release_if_needed()
    empty_cache.assert_called_once_with()


def test_allocator_cache_budget_adapts_to_observed_peak(monkeypatch, patched_runtime):
    empty_cache, _ = patched_runtime
    telemetry = {"reserved": 40, "allocated": 10, "free": 80}
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: telemetry["reserved"])
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: telemetry["allocated"])
    monkeypatch.setattr(
        residency_module.current_omni_platform,
        "get_device_memory",
        # Capacity 400 keeps the 25% cap (100) above the learned budget so
        # this test isolates the peak-adaptation behavior from the cap.
        lambda _device: (telemetry["free"], 400),
    )
    retention = BoundedAllocatorCache(torch.device("cpu"))

    # Cold-start boundary: budget 0, so the policy releases and records the
    # observed peak.
    assert retention.release_if_needed()
    assert retention.observed_cached_peak == 30

    # The observation raises the budget to 30 * 1.25 = 37, so the same cached
    # level is retained on every later boundary without another release.
    assert not retention.release_if_needed()
    assert not retention.release_if_needed()
    assert empty_cache.call_count == 1
