# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise admission configuration through serve parsing, resolution and startup."""

from contextlib import nullcontext

import pytest
import yaml

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniConfig, VllmOmniDiffusionStageConfig
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.diffusion.sched.dispatch.config import CentralSchedulingConfig
from vllm_omni.engine import omni_engine_base as engine_module
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def admission_settings():
    return {"enabled": True, "max_pending_requests": 7, "hardware_profile": "910B2"}


@pytest.mark.parametrize("generic", [False, True], ids=["registered", "generic"])
@pytest.mark.parametrize(
    "yaml_enabled,cli,enabled,limit,error",
    [
        (True, [], True, 7, None),
        (False, ["--enable-central-scheduling"], True, 7, None),
        (True, ["--no-enable-central-scheduling"], False, 7, None),
        (True, ["--central-scheduling-config", '{"max_pending_requests":3}'], True, 3, None),
        (True, ["--central-scheduling-config", '{"enabled":false}'], False, 7, None),
        (True, ["--central-scheduling-config", '{"hardware_profile":null}'], True, 7, "hardware_profile"),
        (True, ["--central-scheduling-config", '{"hardware_profile":"unknown"}'], True, 7, "hardware_profile"),
    ],
    ids=["yaml", "cli-enable", "cli-disable", "cli-limit", "cli-json-disable", "missing-profile", "invalid-profile"],
)
def test_serve_admission_config_reaches_resolver(
    tmp_path, mocker, admission_settings, generic, yaml_enabled, cli, enabled, limit, error
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
    validate = mocker.spy(CentralSchedulingConfig, "from_dict")
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "async_chunk": False,
                "enable_central_scheduling": yaml_enabled,
                "central_scheduling_config": admission_settings,
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
        config = result.central_scheduling_config
        assert isinstance(config, CentralSchedulingConfig)
        assert config.enabled is enabled and config.max_pending_requests == limit
        assert config.hardware_profile == admission_settings["hardware_profile"]
        assert config is build.spy_return.orchestrator_config.central_scheduling_config
        validate.assert_called_once()
        assert result.stage_configs[0].stage_type == "diffusion"
        assert "central_scheduling_config" not in result.stage_configs[0].diffusion_config.__dict__


@pytest.mark.parametrize(
    "layout,error",
    [
        ("local", None),
        ("wan", None),
        ("wan22", None),
        ("qwen-b3", None),
        ("wan-b2", None),
        ("wan22-b2", None),
        ("wan-ti2v", "standard T2V only"),
        ("wan-dmd", "standard T2V only"),
        ("wan-i2v", "standard T2V only"),
        ("wan-image-encoder", "standard T2V only"),
        ("wan-missing-index", "standard T2V only"),
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
    stage.diffusion_config.model_class_name = "WanPipeline" if layout.startswith("wan-") else "QwenImagePipeline"
    model_index: dict[str, object] = {"_class_name": "WanPipeline"}
    if layout == "wan-ti2v":
        model_index["expand_timesteps"] = True
    elif layout == "wan-dmd":
        model_index["_class_name"] = "WanDMDPipeline"
    elif layout == "wan-i2v":
        model_index["_class_name"] = "WanImageToVideoPipeline"
    elif layout == "wan-image-encoder":
        model_index["image_encoder"] = ["transformers", "CLIPVisionModel"]
    read_index = mocker.patch(
        "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
        return_value=None if layout == "wan-missing-index" else model_index,
    )
    execution_overrides: dict[str, dict[str, object]] = {
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
    if layout.startswith("wan"):
        stage.model_config.model = "stage-selected-wan"
        stage.model_config.revision = "stage-revision"
    stage.connector_config.async_chunk = layout == "async-chunk"
    stage.diffusion_config.streaming_output = layout == "streaming"
    if layout in {"wan", "wan22", "qwen-b3"}:
        admission_settings["hardware_profile"] = "910B3"
    stages = (stage, stage) if layout == "multi-stage" else (stage,)
    config = CentralSchedulingConfig.from_dict(admission_settings)
    resolution = OmniConfigResolution(None, stages, central_scheduling_config=config)
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
    assert engine._central_scheduling_config is config
    assert queues.call_count == (0 if error else 1)
    launch.assert_not_called()
    if layout.startswith("wan"):
        read_index.assert_called_once_with("stage-selected-wan", revision="stage-revision")


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
                "enable_central_scheduling": enabled,
                "central_scheduling_config": admission_settings,
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
    cli = ["--num-gpus", str(world), "--central-scheduling-config", '{"max_pending_requests":3}']
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
            enable_central_scheduling=enabled,
            central_scheduling_config=admission_settings | {"max_pending_requests": 3},
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
    assert result.central_scheduling_config.enabled is enabled
    assert result.central_scheduling_config.max_pending_requests == 3


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
