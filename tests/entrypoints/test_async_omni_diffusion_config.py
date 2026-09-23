# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _terminal_config(stage_cfg: dict) -> OmniDiffusionConfig:
    return OmniDiffusionConfig.from_kwargs(**stage_cfg["engine_args"])










@pytest.mark.parametrize(
    "stage_overrides",
    [
        {"0": {"extras": {"ltx2_use_conv_vae": True}}},
        '{"0":{"extras":{"ltx2_use_conv_vae":true}}}',
    ],
)
def test_stage_override_preserves_model_extras_for_default_diffusion_stage(mocker, stage_overrides):
    """Local/unregistered Diffusers checkpoints still honor stage-0 extras."""
    mocker.patch(
        "vllm_omni.config.resolver.StageConfigFactory.create_from_model",
        return_value=None,
    )
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
        return_value=(True, "LTX2Pipeline"),
    )
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)

    _, stage_configs = engine._resolve_stage_configs(
        "/models/LTX-2.5-Diffusers",
        {"stage_overrides": stage_overrides},
        trust_remote_code=False,
    )

    assert stage_configs[0].diffusion_config.extras["ltx2_use_conv_vae"] is True


















@pytest.mark.parametrize("sampling_defaults", [{"0": {"guidance_scale": 7.5}}, '{"0":{"guidance_scale":7.5}}'])
def test_generic_diffusion_sampling_defaults_remain_overridable(sampling_defaults):
    from vllm_omni.entrypoints.omni_base import OmniBase
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    kwargs = {"model_class_name": "QwenImagePipeline", "default_sampling_params": sampling_defaults}
    stage = StageConfigFactory.create_typed_default_diffusion("generic-diffusion", kwargs).stage_configs[0]
    metadata = stage_init_utils.extract_stage_metadata(stage)
    base = OmniBase.__new__(OmniBase)
    base.engine = SimpleNamespace(num_stages=1, stage_configs=[stage])
    base.default_sampling_params_list = [metadata.default_sampling_params]
    base.sampling_constraints_list = base._get_sampling_constraints_list([stage])

    assert base.resolve_sampling_params_list(None)[0].guidance_scale == 7.5
    requested = OmniDiffusionSamplingParams(guidance_scale=2.0)
    assert base.resolve_sampling_params_list(requested)[0].guidance_scale == 2.0
    assert requested.guidance_scale == 2.0
    assert base.default_sampling_params_list[0].guidance_scale == 7.5
    assert base.sampling_constraints_list == [{}]
























def test_invalid_diffusion_offload_config_fails_before_model_loading(monkeypatch, mocker):
    load_model = mocker.patch("vllm_omni.diffusion.model_loader.diffusers_loader.DiffusersPipelineLoader.load_model")
    create_client = mocker.patch("vllm_omni.diffusion.stage_diffusion_client.create_diffusion_client")
    monkeypatch.setattr(
        stage_init_utils,
        "project_engine_args",
        lambda *_args, **_kwargs: {
            "model": "test",
            "diffusion_offload_config": {
                "mode": "layerwise",
                "components": ["dit"],
            },
        },
    )

    with pytest.raises(ValueError, match="Unknown diffusion offload mode"):
        stage_init_utils.initialize_diffusion_stage(
            stage_id=0,
            model="test",
            stage_cfg=object(),
            metadata=mocker.Mock(),
            stage_init_timeout=30,
        )

    create_client.assert_not_called()
    load_model.assert_not_called()










@pytest.mark.parametrize("bad_wait", ["nan", "inf", "-inf", "-1"])
def test_serve_cli_rejects_invalid_request_batch_max_wait_ms(bad_wait: str):
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "serve",
                "Qwen/Qwen-Image",
                "--omni",
                "--request-batch-max-wait-ms",
                bad_wait,
            ]
        )




def test_resolve_stage_configs_delegates_overrides_to_resolver(mocker):
    """The engine consumes resolver output without a second merge pass."""
    additional_config = {"torchair_graph_config": {"enabled": True}}
    fake_diffusion_stage = SimpleNamespace(
        stage_type="diffusion",
        engine_args=SimpleNamespace(additional_config=additional_config),
    )
    resolve_config = mocker.patch(
        "vllm_omni.engine.omni_engine_base.resolve_omni_config",
        return_value=OmniConfigResolution(
            config_path="dummy.yaml",
            stage_configs=(fake_diffusion_stage,),
        ),
    )

    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)

    _, stage_configs = engine._resolve_stage_configs(
        "dummy-model",
        {
            "deploy_config": "dummy.yaml",
            "additional_config": additional_config,
        },
        trust_remote_code=False,
    )

    assert stage_configs == [fake_diffusion_stage]
    assert resolve_config.call_args.args == ("dummy-model",)
    assert resolve_config.call_args.kwargs["deploy_config_path"] == "dummy.yaml"
    assert resolve_config.call_args.kwargs["cli_overrides"]["additional_config"] is additional_config


@pytest.mark.parametrize(
    ("legacy_arg", "value"),
    [
        ("stage_configs_path", "legacy.yaml"),
        ("stage_configs", [{"stage_id": 0}]),
    ],
)
def test_resolve_stage_configs_rejects_legacy_config_arguments(legacy_arg, value):
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)

    with pytest.raises(ValueError, match=rf"`{legacy_arg}`.*`deploy_config`"):
        engine._resolve_stage_configs(
            "dummy-model",
            {legacy_arg: value},
            trust_remote_code=False,
        )




@pytest.mark.parametrize("model_class_name", ["HeliosPipeline", "HunyuanVideo15Pipeline"])
def test_generic_diffusion_uses_canonical_video_output_type(model_class_name):
    config = StageConfigFactory.create_typed_default_diffusion(
        "generic-video",
        {"model_class_name": model_class_name},
    )

    assert config.stage_configs[0].final_output_type == "video"


def test_generic_diffusion_resolves_structured_stage_without_legacy_conversion(mocker):
    """Generic diffusion reaches runtime as the structured stage itself."""
    mocker.patch("vllm_omni.config.resolver.StageConfigFactory.create_from_model", return_value=None)
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
        return_value=(True, "FakeDiffusionPipeline"),
    )

    resolved = resolve_omni_config(
        "generic-diffusion",
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides={
            "num_gpus": 4,
            "tensor_parallel_size": 2,
            "default_sampling_params": '{"0": {"guidance_scale": 7.5}}',
        },
        stage_overrides=None,
        strategy_config_path=None,
    )

    assert resolved.pipeline_config is not None
    assert resolved.pipeline_config.model_type == "generic_diffusion"
    assert len(resolved.stage_configs) == 1
    stage = resolved.stage_configs[0]
    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.model_config.model == "generic-diffusion"
    assert stage.model_config.default_sampling_params == {"guidance_scale": 7.5}
    assert stage.parallel_config.tensor_parallel_size == 2
    assert stage.parallel_config.data_parallel_size == 2
    assert stage.parallel_config.world_size == 4
    assert stage.runtime_config.devices == "0,1,2,3"


def test_generic_diffusion_structured_stage_reaches_standard_startup(mocker):
    """Standard runtime resolves and starts the typed stage through the real launcher."""
    from vllm_omni.engine import stage_engine_startup as startup_module
    from vllm_omni.engine import stage_runtime as runtime_module
    from vllm_omni.engine.stage_runtime import StageRuntime

    mocker.patch("vllm_omni.config.resolver.StageConfigFactory.create_from_model", return_value=None)
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
        return_value=(True, "FakeDiffusionPipeline"),
    )
    resolved = resolve_omni_config(
        "generic-diffusion",
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides={"num_gpus": 1},
        stage_overrides=None,
        strategy_config_path=None,
    )
    stage = resolved.stage_configs[0]
    launched: dict[str, Any] = {}
    client = SimpleNamespace(input_address=None, shutdown=mocker.Mock())

    mocker.patch.object(runtime_module, "prepare_engine_environment")
    mocker.patch.object(runtime_module, "load_omni_transfer_config_for_model", return_value=None)
    mocker.patch.object(runtime_module, "get_stage_connector_spec", return_value={})
    mocker.patch.object(runtime_module, "resolve_omni_kv_config_for_stage", return_value=(None, None, None))

    def _initialize_typed_stage(stage_id, model, stage_config, metadata, **kwargs):
        launched.update(
            stage_id=stage_id,
            model=model,
            stage_config=stage_config,
            metadata=metadata,
            **kwargs,
        )
        return client

    mocker.patch.object(startup_module, "initialize_diffusion_stage", side_effect=_initialize_typed_stage)

    runtime = StageRuntime(
        stage_configs=list(resolved.stage_configs),
        model="generic-diffusion",
        config_path="",
        stage_init_timeout=10,
        async_chunk=False,
    )
    runtime.initialize()

    assert launched["stage_config"] is stage
    assert isinstance(launched["stage_config"], VllmOmniDiffusionStageConfig)
    assert launched["metadata"].stage_type == "diffusion"
    assert launched["metadata"].model_stage == "diffusion"
    assert launched["use_inline"] is True
    assert launched["model"] == "generic-diffusion"
    assert launched["stage_id"] == 0
    assert runtime.stage_pools[0].clients == [client]
