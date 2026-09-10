# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side module-level (component) offload contracts.

CPU-only: these tests assert discovery, strategy resolution, and admission
validation, not transfer behavior. The transfer path is exercised by GPU
lifecycle tests.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _validate_module_offload_runtime,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.config import OffloadStrategy
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PREVIEW_LLM_TYPE = "mammothmoda2_qwen2_5_vl"
DEV_LLM_TYPE = "mammothmoda2_qwen3_vl"


def _raw_config() -> dict:
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": PREVIEW_LLM_TYPE,
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 8,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [8, 8]},
        "gen_dit_config": {"hidden_size": 8, "in_channels": 4},
    }


def _od_config(**overrides) -> OmniDiffusionConfig:
    kwargs = dict(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_raw_config()),
    )
    kwargs.update(overrides)
    return OmniDiffusionConfig(**kwargs)


def _module_config(**overrides) -> OmniDiffusionConfig:
    kwargs = dict(diffusion_offload_config={"mode": "module", "components": ["dit", "text_encoder"]})
    kwargs.update(overrides)
    return _od_config(**kwargs)


def _pipeline_shell() -> MammothModa2DiTPipeline:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipeline)
    pipeline.gen_transformer = nn.Module()
    pipeline.gen_image_condition_refiner = nn.Module()
    pipeline.gen_vae = nn.Module()
    return pipeline


def test_module_mode_resolves_to_model_level_strategy() -> None:
    resolved = OffloadConfig.from_od_config(_module_config())
    assert resolved.strategy is OffloadStrategy.MODEL_LEVEL


def test_module_mode_discovers_dit_encoder_and_vae() -> None:
    pipeline = _pipeline_shell()
    discovered = ModuleDiscovery.discover(pipeline)
    assert discovered.dit_names == ["gen_transformer"]
    assert discovered.dits == [pipeline.gen_transformer]
    assert discovered.encoder_names == ["gen_image_condition_refiner"]
    assert discovered.encoders == [pipeline.gen_image_condition_refiner]
    assert discovered.vae_names == ["gen_vae"]
    assert discovered.vaes == [pipeline.gen_vae]


def test_module_mode_honors_component_selection() -> None:
    dit_only = OffloadConfig.from_od_config(
        _module_config(diffusion_offload_config={"mode": "module", "components": ["dit"]})
    )
    assert dit_only.offloads("dit") is True
    assert dit_only.offloads("text_encoder") is False


def test_module_mode_admission_passes_for_preview_single_request() -> None:
    od_config = _module_config()
    config = _build_mammoth_config(od_config)
    assert _validate_module_offload_runtime(od_config, config) is True


def test_module_mode_admission_returns_false_when_no_offload() -> None:
    od_config = _od_config()  # no diffusion_offload_config
    config = _build_mammoth_config(od_config)
    assert _validate_module_offload_runtime(od_config, config) is False


def test_module_mode_admission_rejects_dev() -> None:
    od_config = _module_config()
    config = SimpleNamespace(llm_config=SimpleNamespace(model_type=DEV_LLM_TYPE))
    with pytest.raises(ValueError, match="Preview"):
        _validate_module_offload_runtime(od_config, config)


def test_module_mode_admission_rejects_multi_request() -> None:
    od_config = _module_config(max_num_seqs=2)
    config = _build_mammoth_config(od_config)
    with pytest.raises(ValueError, match="max_num_seqs"):
        _validate_module_offload_runtime(od_config, config)


def test_enable_model_offload_stages_dit_encoder_and_vae(monkeypatch) -> None:
    pipeline = _pipeline_shell()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.apply_sequential_offload",
        lambda **kwargs: captured.update(kwargs),
    )
    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cpu"), pin_memory=True, use_hsdp=False
    )
    assert captured["dit_modules"] == [pipeline.gen_transformer]
    assert captured["encoder_modules"] == [pipeline.gen_image_condition_refiner, pipeline.gen_vae]
    assert pipeline._model_cpu_offload_modules == [
        pipeline.gen_transformer,
        pipeline.gen_image_condition_refiner,
        pipeline.gen_vae,
    ]


def test_disable_model_offload_removes_hooks(monkeypatch) -> None:
    pipeline = _pipeline_shell()
    removed: list[object] = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.apply_sequential_offload",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.remove_sequential_offload",
        lambda modules: removed.append(modules),
    )
    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cpu"), pin_memory=True, use_hsdp=False
    )
    expected = pipeline._model_cpu_offload_modules
    pipeline.disable_omni_model_cpu_offload()
    assert removed == [expected]
    assert pipeline._model_cpu_offload_modules == []


def test_component_on_device_is_noop_without_offload() -> None:
    pipeline = _pipeline_shell()
    with pipeline._component_on_device(pipeline.gen_vae):
        pass
