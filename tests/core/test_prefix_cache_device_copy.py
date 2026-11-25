# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-copy contracts without allocating accelerator memory."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController
from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import _snapshot_leftover_mm_cpu

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _NPUTensor(torch.Tensor):
    @staticmethod
    def __new__(cls):
        return torch.Tensor._make_subclass(cls, torch.zeros(1, 1, 8))

    @property
    def device(self):
        return SimpleNamespace(type="npu", index=0)

    def detach(self):
        return self

    def cpu(self):
        raise AssertionError("NPU leftover copy must not synchronously wait on the producer")


def test_npu_leftover_constant_uses_pinned_copy_and_records_completion(monkeypatch):
    calls = []
    tensor = _NPUTensor()
    completion = SimpleNamespace(record=lambda: calls.append("record"))
    backend = SimpleNamespace(device=lambda _device: nullcontext(), Event=lambda: completion)
    monkeypatch.setattr(torch, "get_device_module", lambda device: backend)

    def empty(shape, *, dtype, device, pin_memory):
        assert tuple(shape) == (1, 1, 8) and device == "cpu" and pin_memory

        def copy(source, *, non_blocking):
            assert source is tensor and non_blocking
            calls.append("copy")

        return SimpleNamespace(copy_=copy)

    monkeypatch.setattr(torch, "empty", empty)
    output, event = _snapshot_leftover_mm_cpu({"embed.tts_bos": [tensor]}, set(), 32)
    assert output["embed.tts_bos"][0] is not tensor
    assert event is completion
    assert calls == ["copy", "record"]


def test_cpu_leftover_snapshot_owns_values_without_device_queries(monkeypatch):
    tensor = torch.arange(8).reshape(1, 1, 8)
    monkeypatch.setattr(torch, "get_device_module", Mock(side_effect=AssertionError("CPU needs no backend")))
    output, event = _snapshot_leftover_mm_cpu({"embed.tts_bos": [tensor]}, set(), 32)
    tensor.fill_(99)
    assert event is None
    assert torch.equal(output["embed.tts_bos"][0], torch.arange(8).reshape(1, 1, 8))


def test_explicit_npu_controller_uses_npu_streams_when_cuda_unavailable(monkeypatch):
    backend = SimpleNamespace(is_available=lambda: True, Stream=Mock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "get_device_module", lambda device: backend if device == "npu" else torch.cuda)
    # Do not start a real worker; only test backend selection/stream creation.
    monkeypatch.setattr("vllm_omni.core.prefix_cache.controller.threading.Thread", Mock())
    controller = OmniPrefixCacheController(Mock(), PrefixCacheConfig(num_blocks=16, block_size=4), device_type="npu")
    try:
        assert not controller._eager
        assert backend.Stream.call_count == 2
    finally:
        controller.shutdown()
