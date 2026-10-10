# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise admission configuration through serve parsing, resolution and startup."""

from contextlib import nullcontext

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
    return {"enabled": True, "max_pending_requests": 7, "hardware_profile": "910B2"}


@pytest.mark.parametrize(
    "yaml_enabled,cli,enabled,limit,error",
    [
        (True, [], True, 7, None),
        (False, ["--enable-tail-aware-scheduling"], True, 7, None),
        (True, ["--no-enable-tail-aware-scheduling"], False, 7, None),
        (True, ["--tail-aware-scheduling-config", '{"max_pending_requests":3}'], True, 3, None),
        (True, ["--tail-aware-scheduling-config", '{"enabled":false}'], False, 7, None),
        (True, ["--tail-aware-scheduling-config", '{"hardware_profile":null}'], True, 7, "hardware_profile"),
        (True, ["--tail-aware-scheduling-config", '{"hardware_profile":"unknown"}'], True, 7, "hardware_profile"),
    ],
    ids=["yaml", "cli-enable", "cli-disable", "cli-limit", "cli-json-disable", "missing-profile", "invalid-profile"],
)
def test_serve_admission_config_reaches_resolver(
    tmp_path, mocker, admission_settings, yaml_enabled, cli, enabled, limit, error
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
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        result = resolve_omni_config(
            "local-test-model",
            trust_remote_code=False,
            deploy_config_path=str(deploy),
            cli_overrides=overrides,
            stage_overrides=None,
            strategy_config_path=None,
        )
        assert isinstance(result, OmniConfigResolution)
        expected = admission_settings | {"enabled": enabled, "max_pending_requests": limit}
        assert all(result.tail_aware_scheduling_config[key] == value for key, value in expected.items())
        assert result.stage_configs[0].stage_type == "diffusion"
        assert "tail_aware_scheduling_config" not in result.stage_configs[0].diffusion_config.__dict__


@pytest.mark.parametrize(
    "layout,error",
    [
        ("local", None),
        ("wan", None),
        ("wan22", None),
        ("qwen-b3", None),
        ("wan-b2", None),
        ("wan22-b2", None),
        ("unsupported-model", "not supported for model class"),
        ("custom-pipeline", "default native diffusion engine"),
        ("diffusers", "default native diffusion engine"),
        ("custom-backend", "default native diffusion engine"),
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
    execution_overrides = {
        "wan": {"model_class_name": "WanPipeline"},
        "wan22": {"model_class_name": "Wan22Pipeline"},
        "wan-b2": {"model_class_name": "WanPipeline"},
        "wan22-b2": {"model_class_name": "Wan22Pipeline"},
        "unsupported-model": {"model_class_name": "FluxPipeline"},
        "custom-pipeline": {"custom_pipeline_args": {"pipeline_class": "custom.Pipeline"}},
        "diffusers": {"diffusion_load_format": "diffusers"},
        "custom-backend": {"engine_backend": "custom"},
    }
    for key, value in execution_overrides.get(layout, {}).items():
        setattr(stage.diffusion_config, key, value)
    stage.connector_config.async_chunk = layout == "async-chunk"
    stage.diffusion_config.streaming_output = layout == "streaming"
    if layout in {"wan", "wan22", "qwen-b3"}:
        admission_settings["hardware_profile"] = "910B3"
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
