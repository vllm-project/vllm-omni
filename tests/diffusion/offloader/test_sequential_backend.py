# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for SequentialOffloadBackend."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
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


class _CustomPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enable_args: dict[str, object] | None = None
        self.disable_called = False

    def enable_omni_model_cpu_offload(self, **kwargs) -> None:
        self.enable_args = kwargs

    def disable_omni_model_cpu_offload(self) -> None:
        self.disable_called = True


def test_model_level_backend_delegates_to_custom_pipeline_offload() -> None:
    pipeline = _CustomPipeline()
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


def test_model_level_backend_passes_explicit_component_selection() -> None:
    pipeline = _CustomPipeline()
    backend = ModelLevelOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.MODEL_LEVEL,
            components=frozenset({"dit", "text_encoder"}),
        ),
        torch.device("cpu"),
    )

    backend.enable(pipeline)

    assert pipeline.enable_args is not None
    assert pipeline.enable_args["offload_components"] == frozenset({"dit", "text_encoder"})


def test_model_level_backend_rolls_back_partial_custom_enable() -> None:
    class CustomPipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.disable_called = False

        def enable_omni_model_cpu_offload(self, **kwargs) -> None:
            del kwargs
            raise RuntimeError("injected custom enable failure")

        def disable_omni_model_cpu_offload(self) -> None:
            self.disable_called = True

    pipeline = CustomPipeline()
    backend = ModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL),
        torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="injected custom enable failure"):
        backend.enable(pipeline)

    assert pipeline.disable_called
    assert not backend.enabled


def test_sequential_offload_rolls_back_partial_hook_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    dit = _create_simple_module()
    encoder = _create_simple_module()
    original_register = HookRegistry.register_hook
    calls = 0

    def fail_second_registration(self, name, hook):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected registration failure")
        return original_register(self, name, hook)

    monkeypatch.setattr(HookRegistry, "register_hook", fail_second_registration)

    with pytest.raises(RuntimeError, match="injected registration failure"):
        apply_sequential_offload(
            dit_modules=[dit],
            encoder_modules=[encoder],
            device=torch.device("cpu"),
        )

    for module in (dit, encoder):
        registry = getattr(module, "_hook_registry", None)
        assert registry is None or registry.get_hook(SequentialOffloadHook._HOOK_NAME) is None


def test_model_level_disable_cleans_remaining_hooks_and_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = nn.Module()
    pipeline.transformer = _create_simple_module()
    pipeline.text_encoder = _create_simple_module()
    backend = ModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
        torch.device("cpu"),
    )
    backend.enable(pipeline)

    original_remove = HookRegistry.remove_hook
    calls = 0

    def fail_first_removal(self, name):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected hook removal failure")
        original_remove(self, name)

    monkeypatch.setattr(HookRegistry, "remove_hook", fail_first_removal)

    with pytest.raises(RuntimeError, match="Failed to remove one or more sequential offload hooks"):
        backend.disable()

    assert backend.enabled
    assert backend._offload_modules == [pipeline.transformer, pipeline.text_encoder]
    assert pipeline.transformer._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME) is not None
    assert pipeline.text_encoder._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME) is None

    backend.disable()

    assert not backend.enabled
    assert not backend._offload_modules
    for module in (pipeline.transformer, pipeline.text_encoder):
        assert module._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME) is None


def test_sequential_offload_filters_cpu_eligible_components() -> None:
    dit = _create_simple_module()
    encoder = _create_simple_module()
    resident_stage = _create_simple_module()

    apply_sequential_offload(
        dit_modules=[dit],
        encoder_modules=[encoder, resident_stage],
        device=torch.device("cpu"),
        offload_dit_modules=[dit],
        offload_encoder_modules=[encoder],
    )

    dit_hook = dit._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME)
    encoder_hook = encoder._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME)
    resident_hook = resident_stage._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME)
    assert dit_hook.offload_targets == [encoder]
    assert encoder_hook.offload_targets == [dit]
    assert encoder_hook.offload_after_context is True
    assert resident_hook.offload_targets == [dit]
    assert resident_hook.offload_after_context is False

    remove_sequential_offload([dit, encoder, resident_stage])


def test_move_params_handles_buffer_only_modules() -> None:
    module = nn.Module()
    module.register_buffer("state", torch.ones(2))

    # An indexed CPU device compares differently while remaining usable on
    # CPU-only test hosts, so the helper must inspect the buffer itself.
    moved = SequentialOffloadHook._move_params(module, torch.device("cpu:1"))

    assert moved
    assert module.state.device.type == "cpu"


def test_model_level_backend_keeps_declared_image_encoder_resident() -> None:
    class MixedEncoderPipeline(nn.Module):
        _offload_plan = OffloadPlan(
            encoder_component_types={
                "text_encoder": "text_encoder",
            }
        )

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _create_simple_module()
            self.text_encoder = _create_simple_module()
            self.image_encoder = _create_simple_module()

    pipeline = MixedEncoderPipeline()
    backend = ModelLevelOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.MODEL_LEVEL,
            components=frozenset({"dit", "text_encoder"}),
        ),
        torch.device("cpu"),
    )

    backend.enable(pipeline)

    text_hook = pipeline.text_encoder._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME)
    image_hook = pipeline.image_encoder._hook_registry.get_hook(SequentialOffloadHook._HOOK_NAME)
    assert text_hook.offload_after_context is True
    assert image_hook.offload_after_context is False

    backend.disable()


@pytest.mark.parametrize(
    ("component", "attribute", "message"),
    [
        ("text_encoder", "text_encoder", "requires a DiT/transformer"),
        ("dit", "transformer", "requires an encoder execution stage"),
    ],
)
def test_component_selective_model_offload_requires_swap_counterpart(component, attribute, message) -> None:
    pipeline = nn.Module()
    setattr(pipeline, attribute, _create_simple_module())
    backend = ModelLevelOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.MODEL_LEVEL,
            components=frozenset({component}),
        ),
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match=message):
        backend.enable(pipeline)


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
