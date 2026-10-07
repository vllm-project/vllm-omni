# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned offload contracts: allocation bounds, current data and transfer failure."""

import gc

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.offloader import pinned_host
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize("sizes", [[1, 17, 257], [513, 1025, 2049], [270412], [], [8193, 3, 1025]])
def test_pinned_plan_never_increases_allocator_backing(sizes):
    allocations, placements = pinned_host.plan_pinned_slabs(sizes, slab_bytes=4096)

    assert sum(allocations) <= sum(1 << (size - 1).bit_length() for size in sizes)
    for size, (slab, offset) in zip(sizes, placements, strict=True):
        assert offset % 256 == 0
        assert offset + size <= allocations[slab]
    for left, (slab, start) in enumerate(placements):
        for right, (other_slab, other_start) in enumerate(placements):
            if right > left and slab == other_slab:
                assert start + sizes[left] <= other_start or other_start + sizes[right] <= start


@pytest.mark.cpu
@pytest.mark.parametrize("sizes,capacity", [([0], 4096), ([-1], 4096), ([1], 255), ([1], 768)])
def test_pinned_plan_rejects_invalid_storage(sizes, capacity):
    with pytest.raises(ValueError):
        pinned_host.plan_pinned_slabs(sizes, slab_bytes=capacity)


def _aliased_module() -> nn.Module:
    module = nn.Module()
    raw = torch.arange(65536, dtype=torch.float32, device="cuda")
    module.weight = nn.Parameter(raw[:32768].view(128, 256), requires_grad=False)
    module.alias = nn.Parameter(raw[1024:33792].view(128, 256).t(), requires_grad=False)
    module.register_buffer("shared", raw[2048:4096])
    module.register_buffer("counter", torch.arange(1025, dtype=torch.int64, device="cuda"))
    return module


@hardware_test(res={"cuda": ["H100", "B200"], "rocm": "MI325"}, num_cards=1)
def test_pinned_offload_preserves_data_aliases_and_releases_backing():
    if not torch.version.hip:
        pytest.skip("Requires the ROCm pinned-host allocator")
    stats = getattr(torch.cuda, "host_memory_stats", None)
    if not callable(stats):
        pytest.skip("Torch does not expose pinned allocation counters")
    assert callable(stats)
    pinned_host.release_cached_pinned_memory()
    initial = stats()
    module = _aliased_module()
    tensors = [*module.named_parameters(), *module.named_buffers()]
    expected = {name: tensor.detach().cpu().clone() for name, tensor in tensors}
    layouts = {name: (tensor.shape, tensor.stride()) for name, tensor in tensors}
    del tensors

    assert SequentialOffloadHook._move_params(module, torch.device("cpu"), pin_memory=True, non_blocking=True)
    assert module.weight.untyped_storage().data_ptr() == module.alias.untyped_storage().data_ptr()
    assert module.weight.untyped_storage().data_ptr() == module.shared.untyped_storage().data_ptr()
    assert module.alias.storage_offset() - module.weight.storage_offset() == 1024
    assert module.shared.storage_offset() - module.weight.storage_offset() == 2048
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        assert tensor.is_pinned()
        assert (tensor.shape, tensor.stride()) == layouts[name]
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert not SequentialOffloadHook._move_params(module, torch.device("cpu"), pin_memory=True)

    # GPU loading keeps its existing semantics. Copy the next live values,
    # rather than retaining a stale original CPU master across offload cycles.
    assert SequentialOffloadHook._move_params(module, torch.device("cuda"))
    module.weight.add_(7)
    expected["weight"].add_(7)
    assert SequentialOffloadHook._move_params(module, torch.device("cpu"), pin_memory=True)
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        assert tensor.is_pinned()
        assert (tensor.shape, tensor.stride()) == layouts[name]
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    del tensor, module
    gc.collect()
    pinned_host.release_cached_pinned_memory()
    final = stats()
    assert final["allocated_bytes.current"] == initial["allocated_bytes.current"]
    assert final["allocations.current"] == initial["allocations.current"]


@hardware_test(res={"cuda": ["H100", "B200"], "rocm": "MI325"}, num_cards=1)
def test_failed_pinned_copy_synchronizes_before_releasing_storage(monkeypatch):
    if not torch.version.hip:
        pytest.skip("Requires ROCm pinned-host transfer")
    module = _aliased_module()
    originals = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}
    stream = torch.cuda.current_stream()
    original_copy = torch.Tensor.copy_
    events = []

    class TrackingStream:
        def synchronize(self):
            stream.synchronize()
            events.append("synchronized")

    def fail_second_copy(target, source, *args, **kwargs):
        if target.device.type == "cpu" and target.is_pinned() and source.device.type == "cuda":
            events.append("copy")
            if events.count("copy") == 2:
                raise RuntimeError("injected second pinned copy failure")
        return original_copy(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second_copy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: TrackingStream())
    with pytest.raises(RuntimeError, match="injected second pinned copy failure"):
        pinned_host.move_module_to_pinned_cpu(module, non_blocking=True)

    assert events == ["copy", "copy", "synchronized"]
    for name, tensor in module.named_parameters():
        assert tensor.device.type == "cuda"
        torch.testing.assert_close(tensor, originals[name], rtol=0, atol=0)
