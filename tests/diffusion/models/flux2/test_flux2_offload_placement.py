# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-level CPU offload must not move the FLUX.2 text encoder or VAE onto the accelerator at construction."""

from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.flux2 import pipeline_flux2
from vllm_omni.diffusion.models.flux2_klein import pipeline_flux2_klein
from vllm_omni.diffusion.offloader.config import OffloadStrategy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# The meta device stands in for the accelerator, so a CPU-only run can tell placements apart.
ACCELERATOR = torch.device("meta")


@dataclass
class _ParallelConfig:
    vae_parallel_mode: str = "batch"


@dataclass
class _OdConfig:
    model: str = "unused/flux2-checkpoint"
    quantization_config: object = None
    tf_model_config: dict = field(default_factory=dict)
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)
    enable_diffusion_pipeline_profiler: bool = False


@dataclass
class _VaeConfig:
    block_out_channels: tuple[int, ...] = (1, 2)
    latent_channels: int = 16


@dataclass
class _EncoderConfig:
    pass


class _TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.config = _EncoderConfig()

    def set_processor(self, *args, **kwargs):
        pass


class _Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(2, 2)
        self.config = _VaeConfig()


class _Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class _Pretrained:
    """Stands in for a class whose ``from_pretrained`` would read a checkpoint."""

    def __init__(self, factory=object):
        self._factory = factory

    def from_pretrained(self, *args, **kwargs):
        return self._factory()


def _from_pretrained_with_prefetch(load, model, *, subfolder, **kwargs):
    return _TextEncoder() if subfolder == "text_encoder" else _Vae()


@pytest.fixture
def build_flux2(monkeypatch):
    for name, value in {
        "get_local_device": lambda: ACCELERATOR,
        "FlowMatchEulerDiscreteScheduler": _Pretrained(),
        "AutoConfig": _Pretrained(),
        "MistralEncoderModel": _TextEncoder,
        "PixtralProcessor": _Pretrained(),
        "DistributedAutoencoderKLFlux2": _Pretrained(_Vae),
        "get_transformer_config_kwargs": lambda *args, **kwargs: {},
        "Flux2Transformer2DModel": _Transformer,
        "Flux2ImageProcessor": lambda **kwargs: object(),
    }.items():
        monkeypatch.setattr(pipeline_flux2, name, value)

    def build(strategy: OffloadStrategy) -> pipeline_flux2.Flux2Pipeline:
        monkeypatch.setattr(pipeline_flux2, "resolve_offload_strategy", lambda config: strategy)
        return pipeline_flux2.Flux2Pipeline(od_config=_OdConfig())

    return build


@pytest.fixture
def build_flux2_klein(monkeypatch):
    for name, value in {
        "get_local_device": lambda: ACCELERATOR,
        "prefetch_subfolders": lambda *args, **kwargs: None,
        "FlowMatchEulerDiscreteScheduler": _Pretrained(),
        "from_pretrained_with_prefetch": _from_pretrained_with_prefetch,
        "Qwen2TokenizerFast": _Pretrained(),
        "get_transformer_config_kwargs": lambda *args, **kwargs: {},
        "Flux2Transformer2DModel": _Transformer,
        "_use_local_single_stream_tp": lambda od_config: False,
    }.items():
        monkeypatch.setattr(pipeline_flux2_klein, name, value)

    def build(strategy: OffloadStrategy) -> pipeline_flux2_klein.Flux2KleinPipeline:
        monkeypatch.setattr(pipeline_flux2_klein, "resolve_offload_strategy", lambda config: strategy)
        return pipeline_flux2_klein.Flux2KleinPipeline(od_config=_OdConfig())

    return build


def _device_types(pipeline: nn.Module) -> tuple[str, str]:
    return pipeline.text_encoder.proj.weight.device.type, pipeline.vae.conv.weight.device.type


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (OffloadStrategy.MODEL_LEVEL, ("cpu", "cpu")),
        (OffloadStrategy.NONE, ("meta", "meta")),
        (OffloadStrategy.LAYER_WISE, ("meta", "meta")),
    ],
    ids=["model-level-offload", "no-offload", "layer-wise-offload"],
)
@pytest.mark.parametrize("builder", ["build_flux2", "build_flux2_klein"])
def test_text_encoder_and_vae_placement(request, builder, strategy, expected):
    pipeline = request.getfixturevalue(builder)(strategy)
    assert _device_types(pipeline) == expected


def test_deferred_flux2_text_encoder_is_left_on_meta(build_flux2, monkeypatch):
    class _DeferredTextEncoder(_TextEncoder):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.to("meta")
            self.moved = False

        def to(self, *args, **kwargs):
            self.moved = True
            return super().to(*args, **kwargs)

    monkeypatch.setattr(pipeline_flux2, "MistralEncoderModel", _DeferredTextEncoder)
    pipeline = build_flux2(OffloadStrategy.NONE)
    assert pipeline.text_encoder.moved is False
