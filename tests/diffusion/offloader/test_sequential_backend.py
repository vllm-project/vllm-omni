# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for SequentialOffloadBackend."""

from typing import ClassVar

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.hooks.base import _WrappedMethod
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.base import (
    OffloadConfig,
    OffloadGranularity,
    OffloadStrategy,
)
from vllm_omni.diffusion.offloader.sequential_backend import (
    ModelLevelOffloadBackend,
    SequentialOffloadHook,
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


# ---------------------------------------------------------------------------
# ModelLevelOffloadBackend: VAE .decode/.encode method wrapping
# ---------------------------------------------------------------------------


class _StubVae(nn.Module):
    """Minimal VAE-like module invoked via .decode/.encode."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _StubVaeDecodeOnly(nn.Module):
    """VAE-like module that exposes only .decode (no .encode)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _StubStrictPipeline(nn.Module, SupportsComponentDiscovery):
    """Minimal STRICT-mode pipeline (every component evicts every other)."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _offload_granularity: ClassVar[OffloadGranularity] = OffloadGranularity.STRICT

    def __init__(self, decode_only_vae: bool = False):
        super().__init__()
        self.transformer = nn.Linear(8, 8)
        self.text_encoder = nn.Linear(8, 8)
        self.vae = _StubVaeDecodeOnly() if decode_only_vae else _StubVae()


class _StubGroupedPipeline(nn.Module, SupportsComponentDiscovery):
    """Minimal GROUPED-mode pipeline (legacy: VAE stays resident on GPU)."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    # _offload_granularity left at the protocol default = GROUPED.

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(8, 8)
        self.text_encoder = nn.Linear(8, 8)
        self.vae = _StubVae()


def _params_on(module: nn.Module, device: torch.device) -> bool:
    """All parameters and buffers of `module` live on `device`."""
    for p in module.parameters():
        if p.device.type != device.type:
            return False
        if device.type != "cpu" and p.device.index != device.index:
            return False
    for b in module.buffers():
        if b.device.type != device.type:
            return False
        if device.type != "cpu" and b.device.index != device.index:
            return False
    return True


class TestModelLevelOffloadStrict:
    """Verify STRICT granularity: full N-way exclusion + VAE method wrap."""

    def test_vae_decode_swaps_to_gpu_and_evicts_others(self, accelerator_device):
        """Calling vae.decode after enable() must swap VAE to GPU and evict others."""
        pipeline = _StubStrictPipeline()
        cpu = torch.device("cpu")

        backend = ModelLevelOffloadBackend(
            OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
            accelerator_device,
        )
        backend.enable(pipeline)

        # After enable, every offload-participating module is on CPU.
        assert _params_on(pipeline.transformer, cpu)
        assert _params_on(pipeline.text_encoder, cpu)
        assert _params_on(pipeline.vae, cpu)

        # Run the transformer first to put it on GPU.
        x = torch.randn(2, 8, device=accelerator_device)
        _ = pipeline.transformer(x)
        assert _params_on(pipeline.transformer, accelerator_device)
        assert _params_on(pipeline.vae, cpu)

        # Now call vae.decode — the wrapper must move VAE to GPU and
        # evict the transformer back to CPU.
        out = pipeline.vae.decode(x)
        assert out.device.type == accelerator_device.type
        assert _params_on(pipeline.vae, accelerator_device)
        assert _params_on(pipeline.transformer, cpu)

        backend.disable()

    def test_disable_restores_original_decode(self, accelerator_device):
        """After disable(), vae.decode must be the original unwrapped method."""
        pipeline = _StubStrictPipeline()

        backend = ModelLevelOffloadBackend(
            OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
            accelerator_device,
        )
        backend.enable(pipeline)

        # While enabled, decode is wrapped.
        assert isinstance(pipeline.vae.decode, _WrappedMethod)
        assert hasattr(pipeline.vae, "_omni_original_decode")

        backend.disable()

        # After disable, the wrapper is gone and the original method is restored.
        assert not isinstance(pipeline.vae.decode, _WrappedMethod)
        assert not hasattr(pipeline.vae, "_omni_original_decode")
        # And calling it still works (runs on whatever device the params are on).
        pipeline.vae.to(accelerator_device)
        x = torch.randn(2, 8, device=accelerator_device)
        out = pipeline.vae.decode(x)
        assert out.shape == (2, 8)

    def test_vae_without_encode_does_not_error(self, accelerator_device):
        """A VAE that only exposes .decode (no .encode) must not break enable()."""
        pipeline = _StubStrictPipeline(decode_only_vae=True)
        assert not hasattr(pipeline.vae, "encode")

        backend = ModelLevelOffloadBackend(
            OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
            accelerator_device,
        )
        backend.enable(pipeline)

        # decode is still wrapped; encode is left alone (it doesn't exist).
        assert isinstance(pipeline.vae.decode, _WrappedMethod)
        assert not hasattr(pipeline.vae, "_omni_original_encode")

        backend.disable()


class TestModelLevelOffloadGrouped:
    """Verify GROUPED (default) granularity preserves legacy behavior."""

    def test_vae_stays_resident_on_gpu(self, accelerator_device):
        """Under GROUPED, VAE is preloaded to GPU and stays there across DiT calls."""
        pipeline = _StubGroupedPipeline()
        cpu = torch.device("cpu")

        backend = ModelLevelOffloadBackend(
            OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
            accelerator_device,
        )
        backend.enable(pipeline)

        # GROUPED preloads encoders and VAEs on GPU.
        assert _params_on(pipeline.text_encoder, accelerator_device)
        assert _params_on(pipeline.vae, accelerator_device)

        # Running the transformer must evict the text_encoder but leave VAE alone.
        x = torch.randn(2, 8, device=accelerator_device)
        _ = pipeline.transformer(x)
        assert _params_on(pipeline.transformer, accelerator_device)
        assert _params_on(pipeline.text_encoder, cpu)
        assert _params_on(pipeline.vae, accelerator_device)

        # VAE methods are NOT wrapped under GROUPED (no method-wrap plumbing).
        assert not isinstance(pipeline.vae.decode, _WrappedMethod)
        assert not hasattr(pipeline.vae, "_omni_original_decode")

        backend.disable()
