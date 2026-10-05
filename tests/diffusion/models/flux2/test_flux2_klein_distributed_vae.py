# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import (
    DistributedAutoencoderKLFlux2,
)
from vllm_omni.diffusion.models.flux2_klein import pipeline_flux2_klein


class _Component:
    def __init__(self, config) -> None:
        self.config = config

    def to(self, _device):
        return self


def test_klein_pipeline_loads_distributed_vae(monkeypatch) -> None:
    loaded_factories = {}

    def fake_load(factory, _model, *, subfolder, **_kwargs):
        loaded_factories[subfolder] = factory
        if subfolder == "vae":
            return _Component(SimpleNamespace(block_out_channels=[1, 1, 1, 1], latent_channels=32))
        return _Component(SimpleNamespace())

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
    od_config = SimpleNamespace(
        model="test-model",
        tf_model_config={},
        quantization_config=None,
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline_flux2_klein.Flux2KleinPipeline(od_config=od_config)

    assert loaded_factories["vae"].__self__ is DistributedAutoencoderKLFlux2
