# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preset defaults must survive CLI/deploy merging and runtime projection."""

from pathlib import Path

import pytest

from vllm_omni.config.omni_config import (
    VllmOmniConfig,
    extract_diffusion_stage_config_kwargs,
    normalize_and_validate_diffusion_engine_ingress_kwargs,
)
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig, normalize_omni_diffusion_kwargs
from vllm_omni.engine.stage_init_utils import _project_omni_stage_engine_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROFILE = Path(__file__).parents[2] / "vllm_omni/deploy/diffusion_low_vram.yaml"


def _pipeline():
    return PipelineConfig(
        model_type="low-vram-test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="diffusion",
                model_arch="ZImagePipeline",
                execution_type=StageExecutionType.DIFFUSION,
                final_output=True,
            ),
        ),
    )


def test_ingress_does_not_expand_defaults():
    raw = {"low_vram_preset": "fp8"}
    assert normalize_and_validate_diffusion_engine_ingress_kwargs(raw, stage_id=0) == raw


def test_preset_cache_default_overrides_environment(monkeypatch):
    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "cache_dit")
    assert normalize_omni_diffusion_kwargs({"low_vram_preset": "fp8"})["cache_backend"] == "none"
    assert normalize_omni_diffusion_kwargs({})["cache_backend"] == "cache_dit"


def test_explicit_engine_quantization_is_validated_before_preset_defaults():
    with pytest.raises(ValueError, match="requires FP8"):
        extract_diffusion_stage_config_kwargs({"low_vram_preset": "fp8", "quantization": "int8"}, stage_id=0)
    resolved = extract_diffusion_stage_config_kwargs(
        {"low_vram_preset": "fp8", "quantization": "fp8"},
        stage_id=0,
    )
    assert resolved["quantization_config"] == "fp8"


def test_profile_and_cli_cache_override_survive_runtime_projection():
    config = VllmOmniConfig.from_pipeline_config(
        _pipeline(),
        user_deploy_config=load_deploy_config(_PROFILE),
        cli_overrides={"cache_backend": "tea_cache"},
    )
    stage = config.stage_by_id(0)
    assert stage.diffusion_config.cache_backend == "tea_cache"
    assert stage.diffusion_config.vae_use_tiling is True
    assert stage.quantization_config.get_name() == "fp8"
    runtime = _project_omni_stage_engine_args(stage)
    assert runtime["low_vram_preset"] == "fp8"
    assert runtime["vae_use_tiling"] is True
    # Match build_diffusion_config: projected stage args also contain engine
    # adapter metadata, which the terminal diffusion config does not own.
    terminal = OmniDiffusionConfig.from_kwargs(
        **extract_diffusion_stage_config_kwargs(runtime, stage_id=0, include_engine_adapter_metadata=True)
    )
    assert terminal.quantization_config.get_name() == "fp8"
    assert terminal.enable_layerwise_offload is True
    assert terminal.cache_backend == "tea_cache"


def test_deploy_offload_selection_is_not_overwritten_by_cli_preset():
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                diffusion_offload_config={"mode": "module", "components": ["dit", "text_encoder"]},
            )
        ]
    )
    # The explicit deploy setting must be validated rather than silently
    # replaced by the preset's layer default.
    with pytest.raises(ValueError, match="produced invalid images"):
        VllmOmniConfig.from_pipeline_config(
            _pipeline(),
            user_deploy_config=deploy,
            cli_overrides={"low_vram_preset": "fp8"},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_sleep_mode": True},
        {"stage_0_tensor_parallel_size": 2},
        {"stage_0_text_encoder_tp_size": 2},
        {"stage_0_vae_patch_parallel_size": 2},
        {"stage_0_num_replicas": 2},
    ],
)
def test_unsupported_runtime_combinations_fail_before_launch(overrides):
    with pytest.raises(ValueError, match="low_vram_preset"):
        VllmOmniConfig.from_pipeline_config(
            _pipeline(),
            user_deploy_config=load_deploy_config(_PROFILE),
            cli_overrides=overrides,
        )
