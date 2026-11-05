# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MXFP8 prefix requirements use the resolved diffusion offload policy."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch
from torch import nn
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.model_loader import diffusers_loader as loader_module
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _prefix_config(*, steps: int = 5, mode: str = "dlo") -> DiffusionMXFP8Config:
    return DiffusionMXFP8Config(prefix_bf16_steps=steps, prefix_bf16_master_mode=mode)


@pytest.mark.parametrize(
    "offload",
    [
        {},
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"diffusion_offload_config": {"mode": "module", "components": ["dit"]}},
        {"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}},
    ],
)
@pytest.mark.parametrize("component_routing", [False, True])
def test_active_dlo_prefix_rejects_non_distributed_offload(offload, component_routing):
    quant = ComponentQuantizationConfig({"transformer": _prefix_config()}) if component_routing else _prefix_config()

    with pytest.raises(ValueError, match="prefix_bf16_master_mode='dlo'.*distributed layerwise offload"):
        OmniDiffusionConfig(model="", num_gpus=1, quantization_config=quant, **offload)


def test_dlo_prefix_in_component_default_requires_distributed_offload():
    quant = ComponentQuantizationConfig({"text_encoder": None}, default_config=_prefix_config())
    with pytest.raises(ValueError, match="prefix_bf16_master_mode='dlo'.*distributed layerwise offload"):
        OmniDiffusionConfig(model="", num_gpus=1, quantization_config=quant)


@pytest.mark.parametrize(
    "offload",
    [
        {"enable_distributed_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True, "dlo_use_allgather": False},
        {
            "diffusion_offload_config": {
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"weight_transfer": "allgather"}},
            }
        },
    ],
)
def test_dlo_prefix_accepts_legacy_and_compact_distributed_policies(offload):
    config = OmniDiffusionConfig(model="", num_gpus=1, quantization_config=_prefix_config(), **offload)
    assert resolve_offload(config).strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE


@pytest.mark.parametrize("steps,mode", [(5, "cpu"), (5, "resident"), (0, "dlo")])
def test_other_prefix_modes_preserve_no_offload(steps, mode):
    config = OmniDiffusionConfig(model="", num_gpus=1, quantization_config=_prefix_config(steps=steps, mode=mode))
    assert resolve_offload(config).strategy is OffloadStrategy.NONE


def test_dlo_prefix_from_raw_mapping_is_validated():
    with pytest.raises(ValueError, match="prefix_bf16_master_mode='dlo'.*distributed layerwise offload"):
        OmniDiffusionConfig(
            model="",
            num_gpus=1,
            quantization_config={
                "transformer": {"method": "mxfp8", "prefix_bf16_steps": 5, "prefix_bf16_master_mode": "dlo"}
            },
        )


def test_dlo_prefix_from_late_checkpoint_config_is_validated():
    config = OmniDiffusionConfig(model="", num_gpus=1)
    checkpoint = TransformerConfig(quant_method="mxfp8", quant_config=_prefix_config())
    with pytest.raises(ValueError, match="prefix_bf16_master_mode='dlo'.*distributed layerwise offload"):
        config.set_tf_model_config(checkpoint)


@dataclass
class _ConfiguredMethod:
    quant_config: DiffusionMXFP8Config


class _PrefixStack(nn.Module):
    def __init__(self, quant: DiffusionMXFP8Config | None):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.proj = nn.Linear(2, 2)
        if quant is not None:
            for layer in (*self.blocks, self.proj):
                layer.quant_method = _ConfiguredMethod(quant)


class _PrefixPipeline(nn.Module):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vision"]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = []
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        encoder_block_attrs={"text_encoder": ("blocks",), "vision": ("blocks",)},
        encoder_component_types={"text_encoder": "text_encoder", "vision": "text_encoder"},
    )

    def __init__(self, quantized_component: str, quant: DiffusionMXFP8Config):
        super().__init__()
        for path in ("transformer", "text_encoder", "vision"):
            setattr(self, path, _PrefixStack(quant if path == quantized_component else None))


def _initialize_pipeline(monkeypatch, config, pipeline):
    monkeypatch.setattr(loader_module, "initialize_model", lambda _: pipeline)
    monkeypatch.setattr(loader_module, "prepare_model_for_weight_reload", lambda *_: None)
    loader = DiffusersPipelineLoader(LoadConfig(), config)
    return loader._init_from_load_format("default", torch.device("cpu"))


@pytest.mark.parametrize(
    "quantized_component,selected_component",
    [("transformer", "text_encoder"), ("text_encoder", "dit"), ("vision", "dit")],
)
def test_dlo_prefix_rejects_component_excluded_by_resolved_plan(monkeypatch, quantized_component, selected_component):
    quant = _prefix_config()
    config = OmniDiffusionConfig(
        model="",
        num_gpus=1,
        quantization_config=ComponentQuantizationConfig({quantized_component: quant}),
        diffusion_offload_config={
            "mode": "layer",
            "components": [selected_component],
            "layer_options": {selected_component: {"weight_transfer": "allgather"}},
        },
    )
    pipeline = _PrefixPipeline(quantized_component, quant)
    with pytest.raises(ValueError, match=f"for component '{quantized_component}'"):
        _initialize_pipeline(monkeypatch, config, pipeline)


@pytest.mark.parametrize(
    "quantized_component,selected_component",
    [("transformer", "dit"), ("text_encoder", "text_encoder"), ("vision", "text_encoder")],
)
def test_dlo_prefix_accepts_selected_components_and_non_block_auxiliary_linears(
    monkeypatch, quantized_component, selected_component
):
    quant = _prefix_config()
    config = OmniDiffusionConfig(
        model="",
        num_gpus=1,
        quantization_config=ComponentQuantizationConfig({quantized_component: quant}),
        diffusion_offload_config={
            "mode": "layer",
            "components": [selected_component],
            "layer_options": {selected_component: {"weight_transfer": "allgather"}},
        },
    )
    pipeline = _PrefixPipeline(quantized_component, quant)
    assert _initialize_pipeline(monkeypatch, config, pipeline) is pipeline


def test_dlo_prefix_accepts_legacy_auxiliary_component_topology(monkeypatch):
    quant = _prefix_config()
    config = OmniDiffusionConfig(
        model="", num_gpus=1, quantization_config=quant, enable_distributed_layerwise_offload=True
    )
    pipeline = _PrefixPipeline("vision", quant)
    assert _initialize_pipeline(monkeypatch, config, pipeline) is pipeline


def test_dlo_prefix_rejects_undeclared_top_level_quantized_layer(monkeypatch):
    quant = _prefix_config()
    config = OmniDiffusionConfig(
        model="",
        num_gpus=1,
        quantization_config=ComponentQuantizationConfig({"extra_proj": quant}),
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": "allgather"}},
        },
    )
    pipeline = _PrefixPipeline("extra_proj", quant)
    pipeline.extra_proj = nn.Linear(2, 2)
    pipeline.extra_proj.quant_method = _ConfiguredMethod(quant)

    with pytest.raises(ValueError, match="layer 'extra_proj'.*selected"):
        _initialize_pipeline(monkeypatch, config, pipeline)
