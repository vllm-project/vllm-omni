# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import msgspec
import pytest

from vllm_omni.config.omni_config import (
    VllmOmniConfig,
    _DiffusionConfigProjection,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    build_stage_runtime_overrides,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.stage_engine_startup import _serialize_stage_config
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("option", ["vae_tile_size", "vae_tile_overlap"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "384"])
def test_invalid_tile_geometry_config(option, value):
    with pytest.raises(ValueError):
        _DiffusionConfigProjection(**{option: value})
    with pytest.raises(ValueError, match=option):
        OmniDiffusionConfig(**{option: value})


def test_zero_overlap_is_valid_but_zero_tile_size_is_not():
    assert _DiffusionConfigProjection(vae_tile_overlap=0).vae_tile_overlap == 0
    assert OmniDiffusionConfig(vae_tile_overlap=0).vae_tile_overlap == 0
    with pytest.raises(ValueError):
        _DiffusionConfigProjection(vae_tile_size=0)
    with pytest.raises(ValueError, match="vae_tile_size"):
        OmniDiffusionConfig(vae_tile_size=0)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "deploy_values,cli_values,expected",
    [
        ({}, {}, {"vae_tile_size": None, "vae_tile_overlap": None}),
        ({"vae_tile_size": 256, "vae_tile_overlap": 64}, {}, {"vae_tile_size": 256, "vae_tile_overlap": 64}),
        (
            {"vae_tile_size": 256, "vae_tile_overlap": 64},
            {"vae_tile_size": 64, "vae_tile_overlap": 0},
            {"vae_tile_size": 64, "vae_tile_overlap": 0},
        ),
        ({"vae_tile_size": 384}, {"vae_tile_overlap": 32}, {"vae_tile_size": 384, "vae_tile_overlap": 32}),
    ],
)
def test_tile_geometry_reaches_worker_through_stage_config(typed, deploy_values, cli_values, expected):
    pipeline = PipelineConfig(
        model_type="tile-size-test",
        stages=(
            StagePipelineConfig(
                stage_id=0, model_stage="model", execution_type=StageExecutionType.DIFFUSION, final_output=True
            ),
        ),
    )
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, engine_extras=deploy_values)])
    if typed:
        stage = VllmOmniConfig.from_pipeline_config(
            pipeline, user_deploy_config=deploy, cli_overrides=cli_values
        ).stage_configs[0]
        serialized = msgspec.msgpack.decode(msgspec.msgpack.encode(_serialize_stage_config(stage)))
        for option, value in expected.items():
            assert serialized["diffusion_config"][option] == value
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0]
        stage.runtime_overrides = build_stage_runtime_overrides(0, cli_values)
        args = build_legacy_engine_args_dict(stage.to_omegaconf(), model="unused")
    kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
    kwargs.pop("model")
    config = OmniDiffusionConfig.from_kwargs(**kwargs)
    for option, value in expected.items():
        assert getattr(config, option) == value
    assert config.vae_use_tiling is False
