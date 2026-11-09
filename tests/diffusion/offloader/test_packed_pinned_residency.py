# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Allocator and residency coverage for packed ROCm pinned-host backing."""

import gc
from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.module_residency as residency_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "sizes",
    [
        [1, 17, 257],
        [513, 1025, 2049],
        [270412],
        [],
        [8193, 3, 1025],
    ],
)
def test_pinned_plan_never_increases_allocator_backing(sizes: list[int]) -> None:
    plan = residency_module._plan_pinned_slabs(sizes, slab_bytes=4096)

    assert plan.packed_backing_bytes <= plan.separate_backing_bytes
    assert len(plan.placements) == len(sizes)
    for size, (slab_index, offset) in zip(sizes, plan.placements, strict=True):
        assert offset % 256 == 0
        assert offset + size <= plan.allocations[slab_index]
    for left, (slab_index, start) in enumerate(plan.placements):
        for right, (other_slab, other_start) in enumerate(plan.placements):
            if right > left and slab_index == other_slab:
                assert start + sizes[left] <= other_start or other_start + sizes[right] <= start


@pytest.mark.cpu
def test_pinned_plan_reduces_representative_power_of_two_rounding() -> None:
    gib = 1024 * 1024 * 1024
    sizes = [1100 * 1024 * 1024] * 3

    plan = residency_module._plan_pinned_slabs(sizes, slab_bytes=4 * gib)

    assert plan.allocations == (4 * gib,)
    assert plan.packed_backing_bytes == 4 * gib
    assert plan.separate_backing_bytes == 6 * gib


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("sizes", "capacity"),
    [
        ([0], 4096),
        ([-1], 4096),
        ([1], 255),
        ([1], 768),
    ],
)
def test_pinned_plan_rejects_invalid_storage(sizes: list[int], capacity: int) -> None:
    with pytest.raises(ValueError):
        residency_module._plan_pinned_slabs(sizes, slab_bytes=capacity)


@pytest.mark.cpu
def test_source_synchronization_targets_each_device(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class TrackingStream:
        def synchronize(self) -> None:
            events.append("sync")

    monkeypatch.setattr(residency_module.current_omni_platform, "current_device", staticmethod(lambda: 7))
    monkeypatch.setattr(
        residency_module.current_omni_platform,
        "set_device",
        staticmethod(lambda device: events.append(f"set:{device}")),
    )
    monkeypatch.setattr(
        residency_module.current_omni_platform,
        "current_stream",
        staticmethod(TrackingStream),
    )

    residency_module._synchronize_source_devices({torch.device("cuda:1"), torch.device("cuda:0")})

    assert events == ["set:cuda:0", "sync", "set:cuda:1", "sync", "set:cuda:7"]


def _aliased_module(device: torch.device) -> nn.Module:
    module = nn.Module()
    raw = torch.arange(65536, dtype=torch.float32, device=device)
    module.weight = nn.Parameter(raw[:32768].view(128, 256), requires_grad=False)
    module.alias = nn.Parameter(raw[1024:33792].view(128, 256).t(), requires_grad=False)
    module.register_buffer("shared", raw[2048:4096])
    module.register_buffer("counter", torch.arange(1025, dtype=torch.int64, device=device))
    return module


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_packed_offload_preserves_values_aliases_and_layouts() -> None:
    device = current_omni_platform.get_torch_device(0)
    module = _aliased_module(device)
    expected = {
        name: tensor.detach().cpu().clone() for name, tensor in (*module.named_parameters(), *module.named_buffers())
    }
    layouts = {
        name: (tensor.shape, tensor.stride()) for name, tensor in (*module.named_parameters(), *module.named_buffers())
    }

    assert SequentialOffloadHook._move_params(
        module,
        torch.device("cpu"),
        pin_memory=True,
        non_blocking=True,
    )

    assert module.weight.untyped_storage().data_ptr() == module.alias.untyped_storage().data_ptr()
    assert module.weight.untyped_storage().data_ptr() == module.shared.untyped_storage().data_ptr()
    assert module.alias.storage_offset() - module.weight.storage_offset() == 1024
    assert module.shared.storage_offset() - module.weight.storage_offset() == 2048
    for name, tensor in (*module.named_parameters(), *module.named_buffers()):
        assert tensor.is_pinned()
        assert (tensor.shape, tensor.stride()) == layouts[name]
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_packed_offload_uses_less_live_allocator_backing() -> None:
    device = current_omni_platform.get_torch_device(0)
    host_memory_stats = getattr(torch.get_device_module(device), "host_memory_stats", None)
    if not callable(host_memory_stats):
        pytest.skip("Torch does not expose pinned-host allocation counters")
    stats = cast(Callable[[], dict[str, int]], host_memory_stats)

    module = nn.Module()
    size = 1_100_000
    for index in range(3):
        module.register_buffer(f"state_{index}", torch.full((size,), index, dtype=torch.uint8, device=device))
    source_sizes = [buffer.untyped_storage().nbytes() for buffer in module.buffers()]
    plan = residency_module._plan_pinned_slabs(source_sizes)
    assert plan.packed_backing_bytes < plan.separate_backing_bytes

    initial = stats()
    assert SequentialOffloadHook._move_params(
        module,
        torch.device("cpu"),
        pin_memory=True,
        non_blocking=True,
    )
    current = stats()
    live_delta = current["allocated_bytes.current"] - initial["allocated_bytes.current"]
    assert 0 < live_delta <= plan.packed_backing_bytes
    assert live_delta < plan.separate_backing_bytes
    for index, buffer in enumerate(module.buffers()):
        assert buffer.is_pinned()
        assert buffer[0].item() == index

    del buffer, module
    gc.collect()
    final = stats()
    assert final["allocated_bytes.current"] == initial["allocated_bytes.current"]
    assert final["allocations.current"] == initial["allocations.current"]


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_packed_copy_failure_synchronizes_without_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    device = current_omni_platform.get_torch_device(0)
    module = _aliased_module(device)
    expected = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}
    original_copy = torch.Tensor.copy_
    original_current_stream = current_omni_platform.current_stream
    events: list[str] = []

    class TrackingStream:
        def __init__(self):
            self._stream = original_current_stream()

        def synchronize(self) -> None:
            self._stream.synchronize()
            events.append("synchronized")

    def fail_second_copy(target, source, *args, **kwargs):
        if target.device.type == "cpu" and target.is_pinned() and source.device.type == "cuda":
            events.append("copy")
            if events.count("copy") == 2:
                raise RuntimeError("injected second pinned copy failure")
        return original_copy(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second_copy)
    monkeypatch.setattr(current_omni_platform, "current_stream", staticmethod(TrackingStream))
    with pytest.raises(RuntimeError, match="injected second pinned copy failure"):
        residency_module._move_module_to_packed_pinned_cpu(module, non_blocking=True)

    assert events == ["copy", "copy", "synchronized"]
    for name, tensor in module.named_parameters():
        assert tensor.device == device
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_packed_allocation_failure_falls_back_without_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    device = current_omni_platform.get_torch_device(0)
    module = _aliased_module(device)
    expected = {name: tensor.detach().clone() for name, tensor in (*module.named_parameters(), *module.named_buffers())}
    original_empty = torch.empty

    def fail_pinned_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("injected pinned slab allocation failure")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_pinned_empty)

    assert residency_module._move_module_to_packed_pinned_cpu(module, non_blocking=True) is None
    for name, tensor in (*module.named_parameters(), *module.named_buffers()):
        assert tensor.device == device
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_stager_allocation_failure_uses_separate_pinned_masters(monkeypatch: pytest.MonkeyPatch) -> None:
    device = current_omni_platform.get_torch_device(0)
    module = nn.Module()
    for index in range(3):
        module.register_buffer(
            f"state_{index}",
            torch.full((1_100_000,), index, dtype=torch.uint8, device=device),
        )
    original_empty = torch.empty
    failed = False

    def fail_first_pinned_empty(*args, **kwargs):
        nonlocal failed
        if kwargs.get("pin_memory") and not failed:
            failed = True
            raise RuntimeError("injected pinned slab allocation failure")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_first_pinned_empty)

    stager = PinnedModuleStager(module, device, pin_memory=True)

    assert failed
    assert len({group.master.untyped_storage().data_ptr() for group in stager._groups}) == 3
    for index, buffer in enumerate(module.buffers()):
        assert buffer.is_pinned()
        assert buffer[0].item() == index


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_packed_rebind_failure_rolls_back_committed_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    device = current_omni_platform.get_torch_device(0)
    module = _aliased_module(device)
    expected = {name: tensor.detach().clone() for name, tensor in (*module.named_parameters(), *module.named_buffers())}
    original_set_storage = residency_module._set_storage_preserving_inference
    calls = 0

    def fail_second_rebind(target: torch.Tensor, value: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second pinned rebind failure")
        original_set_storage(target, value)

    monkeypatch.setattr(residency_module, "_set_storage_preserving_inference", fail_second_rebind)
    with pytest.raises(RuntimeError, match="injected second pinned rebind failure"):
        residency_module._move_module_to_packed_pinned_cpu(module, non_blocking=True)

    for name, tensor in (*module.named_parameters(), *module.named_buffers()):
        assert tensor.device == device
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
def test_rocm_module_stager_packs_pinned_masters() -> None:
    device = current_omni_platform.get_torch_device(0)
    module = nn.Module()
    for index in range(3):
        module.register_buffer(
            f"state_{index}",
            torch.full((1_100_000,), index, dtype=torch.uint8, device=device),
        )

    stager = PinnedModuleStager(module, device, pin_memory=True)

    assert len(stager._groups) == 3
    assert len({group.master.untyped_storage().data_ptr() for group in stager._groups}) == 1
    assert all(buffer.is_pinned() for buffer in module.buffers())
    stager.load()
    assert all(buffer.device == device for buffer in module.buffers())
    stager.offload()
    assert all(buffer.device.type == "cpu" and buffer.is_pinned() for buffer in module.buffers())
    for index, buffer in enumerate(module.buffers()):
        assert buffer[0].item() == index
