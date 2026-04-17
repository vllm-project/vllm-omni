# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for SequentialOffloadBackend."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def accelerator_device() -> torch.device:
    """Fixture that provides accelerator device or skips test if unavailable."""
    if current_omni_platform.get_device_count() == 0:
        pytest.skip("Accelerator required for this test")
    return current_omni_platform.get_torch_device(0)


def _create_simple_module() -> nn.Module:
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)

    return SimpleModule()


def _track_pin_memory_calls():
    tracker = {"called": False}
    original = torch.Tensor.pin_memory

    def mock(self):
        tracker["called"] = True
        return original(self)

    return tracker, mock


class TestMoveParamsPinMemory:
    def test_dtensor_skips_pin_memory(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """DTensor should skip pin_memory to avoid RuntimeError."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        original_isinstance = isinstance

        def fake_isinstance(obj, cls):
            if cls.__name__ == "DTensor":
                return True
            return original_isinstance(obj, cls)

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        monkeypatch.setattr("builtins.isinstance", fake_isinstance)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=accelerator_device,
            pin_memory=True,
            use_hsdp=False,
        )
        hook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=True,
        )
        assert not tracker["called"], "pin_memory should not be called for DTensor"

    def test_regular_tensor_calls_pin_memory(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """Regular tensor should call pin_memory when moving to CPU."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=accelerator_device,
            pin_memory=True,
            use_hsdp=False,
        )
        hook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=True,
        )
        assert tracker["called"], "pin_memory should be called for regular tensors"

    def test_pin_memory_skipped_when_disabled(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """pin_memory should not be called when pin_memory=False."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=accelerator_device,
            pin_memory=False,
            use_hsdp=False,
        )
        hook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=False,
        )
        assert not tracker["called"], "pin_memory should not be called when disabled"

    def test_pin_memory_skipped_for_non_cpu_target(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """pin_memory should not be called for non-CPU targets."""
        module = _create_simple_module().to("cpu")
        tracker, mock_pin = _track_pin_memory_calls()

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=torch.device("cpu"),
            pin_memory=True,
            use_hsdp=False,
        )
        hook._move_params(module, accelerator_device, non_blocking=False, pin_memory=True)
        assert not tracker["called"], "pin_memory should not be called for non-CPU target"
