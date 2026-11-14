# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Module-mode offload of selected VAE stages."""

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend, SequentialOffloadHook

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# An indexed CPU device never matches the index-less CPU parameters, so every
# stage call looks non-resident without needing an accelerator.
OTHER_DEVICE = torch.device("cpu", 1)


class _VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)


class _Pipeline(nn.Module):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.vae = _VAE()


def _backend(components: set[str], device: torch.device = OTHER_DEVICE) -> ModelLevelOffloadBackend:
    return ModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False, components=frozenset(components)),
        device,
    )


def _hook(module: nn.Module) -> SequentialOffloadHook | None:
    registry = getattr(module, "_hook_registry", None)
    return None if registry is None else registry.get_hook(SequentialOffloadHook._HOOK_NAME)


def _required_hook(module: nn.Module) -> SequentialOffloadHook:
    hook = _hook(module)
    assert hook is not None
    return hook


@pytest.fixture
def swaps(monkeypatch):
    """Record stage activations instead of moving weights to OTHER_DEVICE."""
    calls: list[nn.Module] = []

    def pre_forward(self, module, *args, **kwargs):
        calls.append(module)
        return args, kwargs

    monkeypatch.setattr(SequentialOffloadHook, "pre_forward", pre_forward)
    monkeypatch.setattr(SequentialOffloadHook, "_to_cpu", lambda self, module: None)
    return calls


def _od_config(public):
    return SimpleNamespace(
        diffusion_offload_config=public,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        dlo_chunk_size_mb=64,
        dlo_resident_layers=0,
        pin_cpu_memory=True,
        parallel_config=None,
        model="/fake/model",
    )


class TestSchema:
    def test_module_mode_accepts_vae(self):
        config = OffloadConfig.from_od_config(_od_config({"mode": "module", "components": ["dit", "vae"]}))

        assert config.strategy is OffloadStrategy.MODEL_LEVEL
        assert config.offloads("vae")

    def test_layer_mode_rejects_vae(self):
        with pytest.raises(ValueError, match="'vae' supports only mode='module'"):
            OffloadConfig.from_od_config(_od_config({"mode": "layer", "components": ["dit", "vae"]}))


class TestModelLevelVAE:
    def test_unselected_vae_stays_out_of_the_swap(self, swaps):
        pipeline = _Pipeline()
        backend = _backend({"dit", "text_encoder"}, torch.device("cpu"))

        backend.enable(pipeline)

        assert _hook(pipeline.vae) is None
        assert _required_hook(pipeline.transformer).offload_targets == [pipeline.text_encoder]
        backend.disable()

    def test_selected_vae_swaps_with_the_dit(self, swaps):
        pipeline = _Pipeline()
        backend = _backend({"dit", "vae"})

        backend.enable(pipeline)

        vae_hook = _required_hook(pipeline.vae)
        assert vae_hook.offload_targets == [pipeline.transformer]
        assert vae_hook.offload_after_context is True
        assert pipeline.vae in _required_hook(pipeline.transformer).offload_targets
        backend.disable()

    def test_decode_and_direct_submodule_calls_activate_the_vae(self, swaps):
        pipeline = _Pipeline()
        backend = _backend({"dit", "vae"})
        backend.enable(pipeline)
        swaps.clear()

        pipeline.vae.decode(torch.zeros(1, 2))
        pipeline.vae.encoder(torch.zeros(1, 2))

        assert swaps == [pipeline.vae, pipeline.vae]

        backend.disable()
        pipeline.vae.decode(torch.zeros(1, 2))
        assert swaps == [pipeline.vae, pipeline.vae]
        assert _hook(pipeline.vae) is None

    def test_resident_vae_does_not_swap_again(self, swaps):
        pipeline = _Pipeline()
        backend = _backend({"dit", "vae"}, torch.device("cpu"))
        backend.enable(pipeline)
        swaps.clear()

        pipeline.vae.decode(torch.zeros(1, 2))

        assert swaps == []
        backend.disable()

    def test_selected_vae_requires_a_discovered_vae(self):
        pipeline = _Pipeline()
        del pipeline.vae

        with pytest.raises(ValueError, match="No VAE modules found"):
            _backend({"dit", "vae"}).enable(pipeline)
