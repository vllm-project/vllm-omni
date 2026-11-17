# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preset defaults must survive CLI/deploy merging and runtime projection."""

from pathlib import Path

import pytest
import yaml

from vllm_omni.config.config_factory import StageConfigFactory
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
    # Normalization leaves the preset's settings unset; they are filled after
    # sources merge, so a deploy value can still apply.
    assert "cache_backend" not in normalize_omni_diffusion_kwargs({"low_vram_preset": "fp8"})
    assert extract_diffusion_stage_config_kwargs({"low_vram_preset": "fp8"}, stage_id=0)["cache_backend"] == "none"
    assert normalize_omni_diffusion_kwargs({})["cache_backend"] == "cache_dit"


def test_normalization_does_not_expand_preset():
    normalized = normalize_omni_diffusion_kwargs({"low_vram_preset": "fp8"})
    for name in ("quantization_config", "diffusion_offload_config", "vae_use_tiling"):
        assert normalized.get(name) is None


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


def _write_deploy(tmp_path, **stage):
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump({"stages": [{"stage_id": 0, **stage}]}))
    return str(path)


def _generic(tmp_path, cli, **stage):
    config = StageConfigFactory.create_typed_default_diffusion(
        "dummy/z-image",
        {"model_class_name": "ZImagePipeline", **cli},
        deploy_config_path=_write_deploy(tmp_path, **stage),
    )
    return config.stage_configs[0].diffusion_config


def test_generic_factory_cli_preset_keeps_deploy_cache_backend(tmp_path):
    diffusion = _generic(tmp_path, {"low_vram_preset": "fp8"}, cache_backend="tea_cache")
    assert diffusion.cache_backend == "tea_cache"
    assert diffusion.diffusion_offload_config == {"mode": "layer", "components": ["dit", "text_encoder"]}
    assert diffusion.vae_use_tiling is True


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (
            {"diffusion_offload_config": {"mode": "module", "components": ["dit", "text_encoder"]}},
            "produced invalid images",
        ),
        ({"quantization": "int8"}, "requires FP8"),
        ({"vae_use_tiling": False}, "vae_use_tiling"),
    ],
)
def test_generic_factory_cli_preset_validates_deploy_settings(tmp_path, stage, message):
    # The CLI preset must not hide incompatible deploy settings behind defaults.
    with pytest.raises(ValueError, match=message):
        _generic(tmp_path, {"low_vram_preset": "fp8"}, **stage)


@pytest.mark.parametrize(
    ("cli", "stage"),
    [
        ({"low_vram_preset": "fp8", "cache_backend": "tea_cache", "cache_config": "{bad"}, {}),
        ({"cache_backend": "tea_cache", "cache_config": "{bad"}, {"low_vram_preset": "fp8"}),
    ],
)
def test_generic_factory_rejects_malformed_cache_json_under_preset(tmp_path, cli, stage):
    with pytest.raises(ValueError, match="valid cache_config JSON"):
        _generic(tmp_path, cli, **stage)


@pytest.mark.parametrize(
    "cli_overrides",
    [
        {"cache_backend": "tea_cache", "cache_config": "{bad"},
        {"stage_0_cache_config": "{bad"},
        {"low_vram_preset": "fp8", "cache_config": "{bad"},
    ],
)
def test_registered_path_rejects_malformed_cache_json_under_preset(cli_overrides):
    with pytest.raises(ValueError, match="valid cache_config JSON"):
        VllmOmniConfig.from_pipeline_config(
            _pipeline(),
            user_deploy_config=load_deploy_config(_PROFILE),
            cli_overrides=cli_overrides,
        )


def test_malformed_cache_json_without_preset_keeps_existing_fallback(tmp_path):
    # Unchanged behavior without the preset: the malformed CLI value is dropped
    # with a warning, so the deploy value still applies.
    stage = {"cache_backend": "tea_cache", "cache_config": {"rel_l1_thresh": 0.3}}
    assert _generic(tmp_path, {"cache_config": "{bad"}, **stage).cache_config.rel_l1_thresh == 0.3
    config = VllmOmniConfig.from_pipeline_config(
        _pipeline(),
        user_deploy_config=load_deploy_config(_write_deploy(tmp_path, **stage)),
        cli_overrides={"cache_config": "{bad"},
    )
    assert config.stage_by_id(0).diffusion_config.cache_config.rel_l1_thresh == 0.3


def test_direct_construction_retains_all_preset_values():
    config = OmniDiffusionConfig(model="dummy", low_vram_preset="fp8")
    assert config.vae_use_tiling is True
    assert config.cache_backend == "none"
    assert config.quantization_config.get_name() == "fp8"
    assert config.diffusion_offload_config == {"mode": "layer", "components": ["dit", "text_encoder"]}
    assert config.enable_layerwise_offload is True
    assert OmniDiffusionConfig(model="dummy", low_vram_preset="fp8", vae_use_tiling=None).vae_use_tiling is True
    with pytest.raises(ValueError, match="vae_use_tiling"):
        OmniDiffusionConfig(model="dummy", low_vram_preset="fp8", vae_use_tiling=False)
    with pytest.raises(ValueError, match="valid cache_config JSON"):
        OmniDiffusionConfig.from_kwargs(model="dummy", low_vram_preset="fp8", cache_config="{bad")


def test_direct_construction_without_preset_keeps_tiling_off():
    assert OmniDiffusionConfig(model="dummy").vae_use_tiling is False
    assert OmniDiffusionConfig(model="dummy", vae_use_tiling=None).vae_use_tiling is False


@pytest.mark.parametrize(("dp_size_local", "rejected"), [(1, False), (2, True)])
def test_headless_local_replicas_respect_single_gpu_preset(dp_size_local, rejected):
    from vllm_omni.engine.stage_runtime import DistStageRuntime

    stage = VllmOmniConfig.from_pipeline_config(
        _pipeline(), user_deploy_config=load_deploy_config(_PROFILE), cli_overrides={}
    ).stage_by_id(0)
    runtime = object.__new__(DistStageRuntime)
    runtime._single_stage_id_filter = 0
    runtime._stage_configs = [stage]
    runtime._omni_dp_size_local = dp_size_local
    if rejected:
        with pytest.raises(ValueError, match="omni-dp-size-local=2"):
            runtime._validate_single_stage_mode_replica_constraints()
        assert stage.runtime_config.num_replicas == 1
    else:
        runtime._validate_single_stage_mode_replica_constraints()
        assert stage.runtime_config.num_replicas == 1
