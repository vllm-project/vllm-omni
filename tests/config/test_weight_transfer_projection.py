# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.config.weight_transfer import WeightTransferConfig

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.engine.stage_init_utils import _project_omni_stage_engine_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pipeline(execution_type):
    return PipelineConfig(
        model_type="weight_transfer_projection_test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="AR" if execution_type != StageExecutionType.DIFFUSION else "diffusion",
                execution_type=execution_type,
                final_output=True,
            ),
        ),
    )


@pytest.mark.parametrize("execution_type", list(StageExecutionType))
@pytest.mark.parametrize("source", ["global_cli", "stage_cli", "stage_deploy", "global_deploy"])
@pytest.mark.parametrize("typed", [False, True])
def test_weight_transfer_reaches_backend_through_structured_config(execution_type, source, typed):
    value = WeightTransferConfig(backend="ipc") if typed else {"backend": "ipc"}
    cli = {}
    deploy = DeployConfig()
    if source == "global_cli":
        cli["weight_transfer_config"] = value
    elif source == "stage_cli":
        cli["stage_0_weight_transfer_config"] = value
    elif source == "stage_deploy":
        deploy.stages = [StageDeployConfig(stage_id=0, engine_extras={"weight_transfer_config": value})]
    else:
        deploy.weight_transfer_config = value
    resolved = VllmOmniConfig.from_pipeline_config(
        _pipeline(execution_type), user_deploy_config=deploy, cli_overrides=cli
    )
    projected = _project_omni_stage_engine_args(resolved.stage_by_id(0))
    actual = projected["weight_transfer_config"]
    assert (actual["backend"] if isinstance(actual, dict) else actual.backend) == "ipc"


def test_stage_scoped_weight_transfer_does_not_enable_other_stages():
    pipeline = PipelineConfig(
        model_type="mixed_weight_transfer_projection_test",
        stages=(
            StagePipelineConfig(stage_id=0, model_stage="AR", execution_type=StageExecutionType.LLM_AR),
            StagePipelineConfig(
                stage_id=1,
                model_stage="diffusion",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(0,),
                final_output=True,
            ),
        ),
    )
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=DeployConfig(),
        cli_overrides={"stage_1_weight_transfer_config": {"backend": "ipc"}},
    )
    assert _project_omni_stage_engine_args(config.stage_by_id(0)).get("weight_transfer_config") is None
    assert _project_omni_stage_engine_args(config.stage_by_id(1))["weight_transfer_config"] == {"backend": "ipc"}
