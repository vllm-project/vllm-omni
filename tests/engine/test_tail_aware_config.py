# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise admission configuration through serve parsing, resolution and startup."""

import pytest
import yaml

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.engine import omni_engine_base as engine_module
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def admission_settings():
    return {"enabled": True, "max_pending_requests": 7}


@pytest.mark.parametrize(
    "yaml_enabled,cli,enabled,limit",
    [
        (True, [], True, 7),
        (False, ["--enable-tail-aware-scheduling"], True, 7),
        (True, ["--no-enable-tail-aware-scheduling"], False, 7),
        (True, ["--tail-aware-scheduling-config", '{"max_pending_requests":3}'], True, 3),
        (True, ["--tail-aware-scheduling-config", '{"enabled":false}'], False, 7),
    ],
    ids=["yaml", "cli-enable", "cli-disable", "cli-limit", "cli-json-disable"],
)
def test_serve_admission_config_reaches_resolver(
    tmp_path, mocker, admission_settings, yaml_enabled, cli, enabled, limit
):
    pipeline = PipelineConfig(
        model_type="admission-test",
        stages=(
            StagePipelineConfig(
                stage_id=0, model_stage="diffusion", execution_type=StageExecutionType.DIFFUSION, final_output=True
            ),
        ),
    )
    # Model discovery is external; YAML loading, stage construction and merging stay real.
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=pipeline)
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "async_chunk": False,
                "enable_tail_aware_scheduling": yaml_enabled,
                "tail_aware_scheduling_config": admission_settings,
            }
        )
    )
    parser = TrackingArgumentParser()
    serve = OmniServeCommand().subparser_init(parser.add_subparsers())
    overrides = serve.parse_args(cli).get_explicit_kwargs_dict()
    result = resolve_omni_config(
        "local-test-model",
        trust_remote_code=False,
        deploy_config_path=str(deploy),
        cli_overrides=overrides,
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert isinstance(result, OmniConfigResolution)
    assert result.tail_aware_scheduling_config == admission_settings | {
        "enabled": enabled,
        "max_pending_requests": limit,
    }
    assert result.stage_configs[0].stage_type == "diffusion"
    assert "tail_aware_scheduling_config" not in result.stage_configs[0].diffusion_config.__dict__


@pytest.mark.parametrize(
    "layout,error",
    [
        ("local", None),
        ("distributed", "one local head"),
        ("multi-client", "one local head"),
        ("multi-stage", "one non-streaming diffusion stage"),
        ("async-chunk", "one non-streaming diffusion stage"),
        ("streaming", "streaming diffusion output"),
        ("duplex", "one non-streaming diffusion stage"),
    ],
)
def test_admission_gates_run_before_engine_launch(mocker, admission_settings, layout, error):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
        ),
    )
    stage.diffusion_config.model_class_name = "QwenImagePipeline"
    stage.connector_config.async_chunk = layout == "async-chunk"
    stage.diffusion_config.streaming_output = layout == "streaming"
    stages = (stage, stage) if layout == "multi-stage" else (stage,)
    resolution = OmniConfigResolution(None, stages, tail_aware_scheduling_config=admission_settings)
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=None)
    resolve = mocker.patch.object(engine_module, "resolve_omni_config", return_value=resolution)
    # A resolved stage fixture isolates engine gating from the projection matrix above.
    if layout == "duplex":
        from vllm_omni.config.stage_config import DeployConfig

        mocker.patch.object(engine_module, "load_deploy_config", return_value=DeployConfig(session_mode="duplex"))
    queues = mocker.patch.object(engine_module.janus, "Queue", side_effect=RuntimeError("startup boundary"))
    launch = mocker.patch.object(engine_module.threading.Thread, "start")
    with pytest.raises(ValueError if error else RuntimeError, match=error or "startup boundary"):
        engine_module.OmniEngineBase(
            "local-test-model",
            single_stage_mode=layout == "distributed",
            client_config={"client_count": 2 if layout == "multi-client" else 1},
            **({"deploy_config": "duplex.yaml"} if layout == "duplex" else {}),
        )
    resolve.assert_called_once()
    assert queues.call_count == (0 if error else 1)
    launch.assert_not_called()
