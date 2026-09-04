# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _terminal_config(stage_cfg: dict) -> OmniDiffusionConfig:
    return OmniDiffusionConfig.from_kwargs(**stage_cfg["engine_args"])


def test_default_stage_config_includes_cache_backend():
    """Ensure cache knobs survive the default diffusion-stage builder."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "cache_backend": "cache_dit",
            "cache_config": '{"Fn_compute_blocks": 2}',
            "vae_use_slicing": True,
            "ulysses_degree": 2,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]
    assert stage_cfg["stage_type"] == "diffusion"
    assert engine_args["cache_backend"] == "cache_dit"
    assert engine_args["cache_config"]["Fn_compute_blocks"] == 2
    assert engine_args["vae_use_slicing"] is True
    assert engine_args["parallel_config"]["ulysses_degree"] == 2
    assert engine_args["model_stage"] == "diffusion"


def test_default_cache_config_used_when_missing():
    """Ensure default cache_config is synthesized when only backend is given."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "cache_backend": "cache_dit",
        }
    )[0]

    cache_config = _terminal_config(stage_cfg).cache_config
    assert cache_config.Fn_compute_blocks == 1


def test_default_stage_devices_from_sequence_parallel():
    """Ensure runtime devices reflect computed diffusion world size."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "ulysses_degree": 2,
            "ring_degree": 2,
        }
    )[0]

    assert stage_cfg["runtime"]["devices"] == "0,1,2,3"


def test_default_stage_devices_and_dp_from_num_gpus():
    """Resolve omitted DP before deriving devices from an explicit GPU count."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "num_gpus": 8,
            "tensor_parallel_size": 2,
            "ulysses_degree": 2,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config["data_parallel_size"] == 2
    assert stage_cfg["engine_args"]["num_gpus"] == 8
    assert stage_cfg["runtime"]["devices"] == "0,1,2,3,4,5,6,7"


def test_default_stage_config_uses_parallel_size_kwargs():
    """Ensure default diffusion parallel_config uses CLI/API parallel sizes."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "pipeline_parallel_size": 2,
            "data_parallel_size": 3,
            "tensor_parallel_size": 4,
            "enable_expert_parallel": True,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config["pipeline_parallel_size"] == 2
    assert parallel_config["data_parallel_size"] == 3
    assert parallel_config["tensor_parallel_size"] == 4
    assert parallel_config["enable_expert_parallel"] is True


def test_default_stage_config_preserves_omitted_dp_for_runtime_inference():
    """Keep omitted DP unresolved until the runtime WORLD size is known."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "pipeline_parallel_size": None,
            "data_parallel_size": None,
            "tensor_parallel_size": None,
            "enable_expert_parallel": None,
            "enforce_eager": None,
            "diffusion_compile_granularity": None,
            "diffusion_compile_dynamic": None,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config["pipeline_parallel_size"] == 1
    assert parallel_config["data_parallel_size"] is None
    assert parallel_config["tensor_parallel_size"] == 1
    assert parallel_config["enable_expert_parallel"] is False
    terminal_config = _terminal_config(stage_cfg)
    assert terminal_config.enforce_eager is False
    assert terminal_config.diffusion_compile_granularity == "regional"
    assert terminal_config.diffusion_compile_dynamic is True


def test_default_stage_config_propagates_ulysses_mode():
    """Ensure UAA mode survives default diffusion-stage creation."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "ulysses_degree": 4,
            "ulysses_mode": "advanced_uaa",
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config["ulysses_degree"] == 4
    assert parallel_config["ulysses_mode"] == "advanced_uaa"


def test_default_stage_config_includes_default_sampling_params():
    """Ensure default sampling params survive the default diffusion-stage builder."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "default_sampling_params": '{"0": {"generator_device":"cpu", "guidance_scale":7.5}}',
        }
    )[0]

    assert stage_cfg["default_sampling_params"] == {
        "generator_device": "cpu",
        "guidance_scale": 7.5,
    }


def test_default_stage_config_includes_diffusion_attention_backend():
    """Ensure diffusion attention shorthand lands in engine_args.diffusion_attention_config."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "diffusion_attention_backend": "FLASH_ATTN",
        }
    )[0]

    diffusion_attention_config = _terminal_config(stage_cfg).diffusion_attention_config
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FLASH_ATTN"


def test_default_stage_config_includes_diffusion_attention_config():
    """Ensure structured diffusion attention config survives default stage creation."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "diffusion_attention_config": {
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"cross": {"backend": "TORCH_SDPA"}},
            },
        }
    )[0]

    diffusion_attention_config = _terminal_config(stage_cfg).diffusion_attention_config
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FLASH_ATTN"
    assert diffusion_attention_config.per_role["cross"].backend == "TORCH_SDPA"


def test_default_stage_config_rejects_conflicting_diffusion_attention_inputs():
    """Ensure shorthand and default.backend stay mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        StageConfigFactory.create_default_diffusion(
            {
                "diffusion_attention_backend": "FLASH_ATTN",
                "diffusion_attention_config": {
                    "default": {"backend": "TORCH_SDPA"},
                },
            }
        )


def test_default_stage_config_engine_args():
    """Ensure default diffusion-stage builder sets and propagates engine_args."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            "distributed_executor_backend": "ray",
            "boundary_ratio": 0.875,
            "flow_shift": 5.0,
            "trust_remote_code": True,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]
    assert engine_args["distributed_executor_backend"] == "ray"
    assert engine_args["boundary_ratio"] == 0.875
    assert engine_args["flow_shift"] == 5.0
    assert engine_args["trust_remote_code"] is True


def test_default_stage_config_whitelist_none_fallback():
    """DeployConfig / StageDeployConfig whitelist fields with value None
    fall back to OmniDiffusionConfig dataclass defaults."""
    stage_cfg = StageConfigFactory.create_default_diffusion(
        {
            # DeployConfig pipeline-wide
            "trust_remote_code": None,
            "distributed_executor_backend": None,
            "dtype": None,
            # StageDeployConfig
            "enforce_eager": None,
        }
    )[0]

    terminal = _terminal_config(stage_cfg)
    assert terminal.trust_remote_code is False
    assert terminal.distributed_executor_backend is None
    assert terminal.dtype == torch.bfloat16
    assert terminal.enforce_eager is False


def test_serve_cli_accepts_ulysses_mode():
    """Ensure diffusion serve CLI exposes ulysses_mode and wires it to parallel_config."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--usp",
            "4",
            "--ulysses-mode",
            "advanced_uaa",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]
    parallel_config = stage_cfg["engine_args"]["parallel_config"]

    assert args.ulysses_mode == "advanced_uaa"
    assert parallel_config["ulysses_degree"] == 4
    assert parallel_config["ulysses_mode"] == "advanced_uaa"


def test_serve_cli_forwards_model_defined_task_type_to_diffusion_stage():
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "MiniMaxAI/MiniMax-H3",
            "--omni",
            "--task-type",
            "fl2va",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]

    assert args.task_type == "fl2va"
    assert stage_cfg["engine_args"]["task_type"] == "fl2va"


def test_serve_cli_accepts_diffusion_pipeline_profiler_flag():
    """Ensure diffusion serve CLI exposes the profiler switch."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--omni",
            "--enable-diffusion-pipeline-profiler",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]

    assert args.enable_diffusion_pipeline_profiler is True
    assert stage_cfg["engine_args"]["enable_diffusion_pipeline_profiler"] is True


def test_serve_cli_forwards_distilled_lora_to_diffusion_stage():
    """Ensure startup distilled LoRA options reach the online diffusion stage."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--omni",
            "--lora-backend",
            "distill",
            "--lora-path",
            "/models/high.safetensors",
            "/models/low.safetensors",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]
    engine_args = stage_cfg["engine_args"]

    assert explicit_kwargs["lora_backend"] == "distill"
    assert engine_args["lora_backend"] == "distill"
    assert engine_args["lora_path"] == [
        "/models/high.safetensors",
        "/models/low.safetensors",
    ]


def test_serve_cli_forwards_distributed_offload_residency():
    """Ensure the two-GPU DLO placement controls reach the diffusion stage."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "MiniMaxAI/MiniMax-H3",
            "--omni",
            "--enable-distributed-layerwise-offload",
            "--dlo-no-use-allgather",
            "--dlo-resident-layers",
            "20",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]
    engine_args = stage_cfg["engine_args"]

    assert args.enable_distributed_layerwise_offload is True
    assert args.dlo_use_allgather is False
    assert args.dlo_resident_layers == 20
    assert engine_args["enable_distributed_layerwise_offload"] is True
    assert engine_args["dlo_use_allgather"] is False
    assert engine_args["dlo_resident_layers"] == 20


def test_serve_cli_forwards_hwr_policy_for_no_allgather_dlo():
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "MiniMaxAI/MiniMax-H3",
            "--omni",
            "--enable-distributed-layerwise-offload",
            "--dlo-no-use-allgather",
            "--host-weight-runtime-mode",
            "preferred",
            "--host-weight-runtime-root",
            "/var/cache/vllm-omni/hwr",
            "--dlo-host-registration-limit-gib",
            "80",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]
    engine_args = stage_cfg["engine_args"]

    assert explicit_kwargs["host_weight_runtime_mode"] == "preferred"
    assert explicit_kwargs["host_weight_runtime_root"] == "/var/cache/vllm-omni/hwr"
    assert explicit_kwargs["dlo_host_registration_limit_gib"] == 80
    assert engine_args["host_weight_runtime_mode"] == "preferred"
    assert engine_args["host_weight_runtime_root"] == "/var/cache/vllm-omni/hwr"
    assert engine_args["dlo_host_registration_limit_gib"] == 80


def test_serve_cli_accepts_diffusion_compile_controls():
    """Ensure both compile controls reach the diffusion stage."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Lightricks/LTX-Video-0.9.8-13B-distilled",
            "--omni",
            "--diffusion-compile-granularity",
            "full",
            "--no-diffusion-compile-dynamic",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]

    assert args.diffusion_compile_granularity == "full"
    assert args.diffusion_compile_dynamic is False
    assert stage_cfg["engine_args"]["diffusion_compile_granularity"] == "full"
    assert stage_cfg["engine_args"]["diffusion_compile_dynamic"] is False


def test_serve_cli_accepts_diffusion_attention_backend():
    """Ensure diffusion serve CLI exposes the shorthand backend flag."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--diffusion-attention-backend",
            "FASTVIDEO_VSA",
            "--fastvideo-vsa-topk",
            "96",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]
    diffusion_attention_config = stage_cfg["engine_args"]["diffusion_attention_config"]

    assert args.diffusion_attention_backend == "FASTVIDEO_VSA"
    assert args.fastvideo_vsa_topk == 96
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FASTVIDEO_VSA"
    assert diffusion_attention_config.default.backend_kwargs() == {"topk": 96}


def test_serve_cli_accepts_request_batch_max_wait_ms():
    """Ensure diffusion serve CLI forwards request-batch admission wait to stage config."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--request-batch-max-wait-ms",
            "250",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = StageConfigFactory.create_default_diffusion(explicit_kwargs)[0]

    assert args.request_batch_max_wait_ms == 250.0
    assert stage_cfg["engine_args"]["request_batch_max_wait_ms"] == 250.0


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


def test_serve_cli_accepts_additional_config():
    """Ensure diffusion serve CLI exposes additional_config and forwards it to stage config."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--additional-config",
            '{"torchair_graph_config":{"enabled":true}}',
        ]
    )

    stage_cfg = StageConfigFactory.create_default_diffusion(vars(args))[0]

    engine_args = stage_cfg["engine_args"]

    assert args.additional_config == {"torchair_graph_config": {"enabled": True}}
    assert engine_args["additional_config"] == {"torchair_graph_config": {"enabled": True}}


def test_resolve_stage_configs_delegates_overrides_to_resolver(mocker):
    """The engine consumes resolver output without a second merge pass."""
    additional_config = {"torchair_graph_config": {"enabled": True}}
    fake_diffusion_stage = SimpleNamespace(
        stage_type="diffusion",
        engine_args=SimpleNamespace(additional_config=additional_config),
    )
    resolve_config = mocker.patch(
        "vllm_omni.engine.async_omni_engine.resolve_omni_config",
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


def test_default_stage_config_includes_quantization_config():
    """Ensure structured quantization_config survives default diffusion-stage creation."""
    quantization_config = {
        "method": "example_quant",
        "weights": "weights.bin",
    }

    stage_cfg = StageConfigFactory.create_default_diffusion({"quantization_config": quantization_config})[0]

    assert stage_cfg["engine_args"]["quantization_config"] == quantization_config


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
    launched = {}
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
        typed_stage_configs=list(resolved.stage_configs),
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
