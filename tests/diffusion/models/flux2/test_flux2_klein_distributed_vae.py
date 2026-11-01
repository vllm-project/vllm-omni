# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.configuration_utils import FrozenDict

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import (
    DistributedAutoencoderKLFlux2,
)
from vllm_omni.diffusion.models.flux2_klein import pipeline_flux2_klein

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Component:
    def __init__(self, config) -> None:
        self.config = config

    def to(self, _device):
        return self


@pytest.mark.parametrize("vae_parallel_mode", ["tile", "batch"])
@pytest.mark.parametrize("vae_patch_parallel_size", [1, 2])
def test_klein_pipeline_loads_distributed_vae(monkeypatch, vae_parallel_mode, vae_patch_parallel_size) -> None:
    loaded_factories = {}

    def fake_load(factory, _model, *, subfolder, **_kwargs):
        loaded_factories[subfolder] = factory
        if subfolder == "vae":
            return _Component(FrozenDict(block_out_channels=[1, 1, 1, 1], latent_channels=32))
        return _Component(FrozenDict())

    monkeypatch.setattr(pipeline_flux2_klein, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_flux2_klein, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_flux2_klein, "from_pretrained_with_prefetch", fake_load)
    monkeypatch.setattr(
        pipeline_flux2_klein.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        pipeline_flux2_klein.Qwen2TokenizerFast,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(pipeline_flux2_klein, "get_transformer_config_kwargs", lambda *_args: {})
    monkeypatch.setattr(pipeline_flux2_klein, "Flux2Transformer2DModel", lambda **_kwargs: object())
    monkeypatch.setattr(
        pipeline_flux2_klein.Flux2KleinPipeline,
        "setup_diffusion_pipeline_profiler",
        lambda *_args, **_kwargs: None,
    )
    od_config = OmniDiffusionConfig(
        model="test-model",
        num_gpus=vae_patch_parallel_size,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=vae_patch_parallel_size,
            vae_parallel_mode=vae_parallel_mode,
            vae_patch_parallel_size=vae_patch_parallel_size,
        ),
    )

    pipeline_flux2_klein.Flux2KleinPipeline(od_config=od_config)

    assert loaded_factories["vae"].__self__ is DistributedAutoencoderKLFlux2
