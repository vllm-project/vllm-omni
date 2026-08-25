# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for SequentialOffloadBackend."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.sequential_backend import (
    ModelLevelOffloadBackend,
    SequentialOffloadHook,
    apply_sequential_offload,
    remove_sequential_offload,
    sequential_offload_component,
)
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


def test_model_level_backend_delegates_to_custom_pipeline_offload() -> None:
    class CustomPipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enable_args = None
            self.disable_called = False

        def enable_omni_model_cpu_offload(self, **kwargs) -> None:
            self.enable_args = kwargs

        def disable_omni_model_cpu_offload(self) -> None:
            self.disable_called = True

    pipeline = CustomPipeline()
    backend = ModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
        torch.device("cpu"),
    )

    backend.enable(pipeline)

    assert backend.enabled is True
    assert pipeline.enable_args == {
        "device": torch.device("cpu"),
        "pin_memory": False,
        "use_hsdp": False,
    }

    backend.disable()

    assert backend.enabled is False
    assert pipeline.disable_called is True


def test_sequential_offload_can_begin_with_dit_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    dit = _create_simple_module()
    encoder = _create_simple_module()
    offloaded: list[nn.Module] = []
    monkeypatch.setattr(
        SequentialOffloadHook,
        "_to_cpu",
        lambda self, module: offloaded.append(module),
    )

    apply_sequential_offload(
        dit_modules=[dit],
        encoder_modules=[encoder],
        device=torch.device("cpu"),
        offload_initial_dits=True,
    )

    assert offloaded == [dit]
    remove_sequential_offload([dit, encoder])


def test_direct_component_activation_failure_still_offloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = _create_simple_module()
    apply_sequential_offload(
        dit_modules=[_create_simple_module()],
        encoder_modules=[component],
        device=torch.device("cpu"),
    )
    hook = component._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME)
    offloaded: list[nn.Module] = []

    def fail_activation(module: nn.Module) -> None:
        raise RuntimeError("activation failed")

    monkeypatch.setattr(hook, "pre_forward", fail_activation)
    monkeypatch.setattr(hook, "_to_cpu", lambda module: offloaded.append(module))

    with pytest.raises(RuntimeError, match="activation failed"):
        with sequential_offload_component(component):
            pass

    assert offloaded == [component]


class TestMoveParamsPinMemory:
    def test_dtensor_skips_pin_memory(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """DTensor should skip pin_memory to avoid RuntimeError."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        original_isinstance = isinstance

        def fake_isinstance(obj, cls):
            # torchada's Tensor.to() checks tuples such as
            # ``(str, torch.device)``.  Keep the DTensor probe compatible with
            # the normal isinstance contract instead of breaking those calls.
            if type(cls) is tuple:
                return any(fake_isinstance(obj, candidate) for candidate in cls)
            if getattr(cls, "__name__", None) == "DTensor":
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
