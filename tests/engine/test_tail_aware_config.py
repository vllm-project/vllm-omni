# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise admission configuration through serve parsing, resolution and startup."""

import pytest
import yaml

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniConfig, VllmOmniDiffusionStageConfig
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.engine import omni_engine_base as engine_module
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.scheduling.config import TailAwareSchedulingConfig
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def admission_settings():
    return {"enabled": True, "max_pending_requests": 7}


@pytest.mark.parametrize("generic", [False, True], ids=["registered", "generic"])
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
    tmp_path, mocker, admission_settings, generic, yaml_enabled, cli, enabled, limit
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
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=None if generic else pipeline)
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
        return_value=(True, "QwenImagePipeline"),
    )
    mocker.patch("vllm_omni.config.config_factory.get_diffusion_output_type", return_value="image")
    build = mocker.spy(VllmOmniConfig, "from_pipeline_config")
    validate = mocker.spy(TailAwareSchedulingConfig, "from_dict")
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
    config = result.tail_aware_scheduling_config
    assert isinstance(config, TailAwareSchedulingConfig)
    assert config.enabled is enabled and config.max_pending_requests == limit
    assert config is build.spy_return.orchestrator_config.tail_aware_scheduling_config
    validate.assert_called_once()
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
    config = TailAwareSchedulingConfig.from_dict(admission_settings)
    resolution = OmniConfigResolution(None, stages, tail_aware_scheduling_config=config)
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=None)
    resolve = mocker.patch.object(engine_module, "resolve_omni_config", return_value=resolution)
    # A resolved stage fixture isolates engine gating from the projection matrix above.
    if layout == "duplex":
        from vllm_omni.config.stage_config import DeployConfig

        mocker.patch.object(engine_module, "load_deploy_config", return_value=DeployConfig(session_mode="duplex"))
    queues = mocker.patch.object(engine_module.janus, "Queue", side_effect=RuntimeError("startup boundary"))
    launch = mocker.patch.object(engine_module.threading.Thread, "start")
    engine = object.__new__(engine_module.OmniEngineBase)
    with pytest.raises(ValueError if error else RuntimeError, match=error or "startup boundary"):
        engine.__init__(
            "local-test-model",
            single_stage_mode=layout == "distributed",
            client_config={"client_count": 2 if layout == "multi-client" else 1},
            **({"deploy_config": "duplex.yaml"} if layout == "duplex" else {}),
        )
    resolve.assert_called_once()
    assert engine._tail_aware_scheduling_config is config
    assert queues.call_count == (0 if error else 1)
    launch.assert_not_called()


@pytest.mark.parametrize("with_yaml", [False, True], ids=["without-yaml", "with-yaml"])
@pytest.mark.parametrize("world,tp,enabled", [(2, 1, False), (4, 1, True), (4, 2, True)])
def test_generic_preserves_explicit_inputs(tmp_path, mocker, admission_settings, world, tp, enabled, with_yaml):
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=None)
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
        return_value=(True, "QwenImagePipeline"),
    )
    mocker.patch("vllm_omni.config.config_factory.get_diffusion_output_type", return_value="image")
    devices = ";".join(",".join(str(i) for i in range(n * world, (n + 1) * world)) for n in range(2))
    deploy = tmp_path / "generic.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "async_chunk": False,
                "enable_tail_aware_scheduling": enabled,
                "tail_aware_scheduling_config": admission_settings,
                "stages": [
                    {
                        "stage_id": 0,
                        "num_replicas": 2,
                        "devices": devices,
                        "engine_args": {"tensor_parallel_size": tp, "vae_use_tiling": True, "max_num_seqs": 7},
                    }
                ],
            }
        )
    )
    parser = TrackingArgumentParser()
    serve = OmniServeCommand().subparser_init(parser.add_subparsers())
    cli = ["--num-gpus", str(world), "--tail-aware-scheduling-config", '{"max_pending_requests":3}']
    overrides = serve.parse_args(cli).get_explicit_kwargs_dict()
    overrides["max_batch_size"] = 3
    # This is only valid after the YAML TP axis has been merged.
    if tp > 1:
        overrides["data_parallel_size"] = world // tp
    if not with_yaml:
        overrides.update(
            tensor_parallel_size=tp,
            vae_use_tiling=True,
            devices=devices,
            num_replicas=2,
            enable_tail_aware_scheduling=enabled,
            tail_aware_scheduling_config=admission_settings | {"max_pending_requests": 3},
        )
    result = resolve_omni_config(
        "local-test-model",
        trust_remote_code=False,
        deploy_config_path=str(deploy) if with_yaml else None,
        cli_overrides=overrides,
        stage_overrides=None,
        strategy_config_path=None,
    )
    stage = result.stage_configs[0]
    assert stage.parallel_config.tensor_parallel_size == tp
    assert stage.parallel_config.data_parallel_size == world // tp
    assert stage.runtime_config.num_gpus == world
    assert stage.runtime_config.num_replicas == 2
    assert stage.runtime_config.devices == devices
    assert stage.diffusion_config.vae_use_tiling
    assert stage.scheduler_config.max_num_seqs == 3
    assert result.tail_aware_scheduling_config.enabled is enabled
    assert result.tail_aware_scheduling_config.max_pending_requests == 3


@pytest.mark.parametrize("with_yaml", [False, True])
def test_generic_rejects_removed_promoted_field(tmp_path, mocker, with_yaml):
    # Upstream removed auxiliary_text_encoder with HiDream; do not restore it
    # as a generic override merely to accommodate an older review reproducer.
    mocker.patch.object(StageConfigFactory, "get_pipeline_config", return_value=None)
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
        return_value=(True, "QwenImagePipeline"),
    )
    deploy = tmp_path / "generic.yaml"
    deploy.write_text("async_chunk: false\n")
    with pytest.raises(ValueError, match="auxiliary_text_encoder"):
        resolve_omni_config(
            "local-test-model",
            trust_remote_code=False,
            deploy_config_path=str(deploy) if with_yaml else None,
            cli_overrides={
                "auxiliary_text_encoder": "/models/new",
                "extras": {"auxiliary_text_encoder": "/models/old"},
            },
            stage_overrides=None,
            strategy_config_path=None,
        )
