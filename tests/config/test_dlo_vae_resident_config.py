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


@pytest.fixture(params=["dlo_vae_resident", "dlo_text_encoder_resident"])
def resident_option(request):
    return request.param


def test_resident_defaults(resident_option):
    assert getattr(_DiffusionConfigProjection(), resident_option) is False
    assert getattr(OmniDiffusionConfig(), resident_option) is False
    assert getattr(StageDeployConfig(stage_id=0), resident_option) is None


@pytest.mark.parametrize("value", [0, 1, "false", "true", 1.5])
def test_resident_requires_boolean(value, resident_option):
    for config_type in (_DiffusionConfigProjection, OmniDiffusionConfig, StageDeployConfig):
        with pytest.raises(ValueError, match=resident_option):
            kwargs = {"stage_id": 0} if config_type is StageDeployConfig else {}
            config_type(**{resident_option: value}, **kwargs)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_resident_stage_override_preserves_false(value, reverse, resident_option):
    items = [(resident_option, not value), (f"stage_1_{resident_option}", value)]
    overrides = dict(reversed(items) if reverse else items)
    assert build_stage_runtime_overrides(1, overrides)[resident_option] is value
    assert build_stage_runtime_overrides(0, overrides)[resident_option] is not value


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("cli_value,expected", [(None, True), (False, False), (True, True)])
def test_resident_reaches_worker_and_ipc(typed, cli_value, expected, resident_option):
    pipeline = PipelineConfig(
        model_type="dlo-resident-test",
        stages=(
            StagePipelineConfig(
                stage_id=0, model_stage="model", execution_type=StageExecutionType.DIFFUSION, final_output=True
            ),
        ),
    )
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, **{resident_option: True})])
    cli = {resident_option: cli_value}
    if typed:
        stage = VllmOmniConfig.from_pipeline_config(
            pipeline, user_deploy_config=deploy, cli_overrides=cli
        ).stage_configs[0]
        serialized = msgspec.msgpack.decode(msgspec.msgpack.encode(_serialize_stage_config(stage)))
        assert serialized["diffusion_config"][resident_option] is expected
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0]
        stage.runtime_overrides = build_stage_runtime_overrides(0, cli)
        args = build_legacy_engine_args_dict(stage.to_omegaconf(), model="unused")
    kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
    kwargs.pop("model")
    assert getattr(OmniDiffusionConfig.from_kwargs(**kwargs), resident_option) is expected
