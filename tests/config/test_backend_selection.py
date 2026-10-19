# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for Item 1 backend fields and their final projections."""

import json
import warnings
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from vllm.config import KernelConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import (
    OmniStageModelConfig,
    VllmOmniConfig,
    VllmOmniDiffusionStageConfig,
    _DiffusionConfigProjection,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    _apply_platform_overrides,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pipeline(execution_type: StageExecutionType) -> PipelineConfig:
    return PipelineConfig(
        model_type="backend-test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="model",
                execution_type=execution_type,
                final_output=True,
            ),
        ),
    )


@pytest.fixture
def tiny_ar_model(tmp_path):
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.fixture
def generic_flux_model(tmp_path):
    model = tmp_path / "flux"
    model.mkdir()
    (model / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    return str(model)


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
@pytest.mark.parametrize("typed", [False, True])
def test_deploy_backends_reach_final_engine_config(tmp_path, tiny_ar_model, execution_type, typed):
    attention = (
        "diffusion_attention_backend: torch_sdpa"
        if execution_type == StageExecutionType.DIFFUSION
        else "attention_backend: triton_attn"
    )
    path = tmp_path / "deploy.yaml"
    path.write_text(f"stages:\n  - stage_id: 0\n    linear_backend: TORCH\n    moe_backend: TRITON\n    {attention}\n")
    deploy = load_deploy_config(str(path))
    pipeline = _pipeline(execution_type)

    if typed:
        stage = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
        args = build_legacy_engine_args_dict(stage, model="unused")

    assert args["linear_backend"] == "torch"
    assert args["moe_backend"] == "triton"
    if execution_type == StageExecutionType.DIFFUSION:
        kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
        kwargs.pop("model")
        diffusion_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        final = create_diffusion_vllm_config(torch.device("cpu"), diffusion_config)
        assert final.kernel_config.linear_backend == "torch"
        assert final.kernel_config.moe_backend == "triton"
        assert diffusion_config.diffusion_attention_config.default.backend == "torch_sdpa"
    else:
        names = {field.name for field in fields(OmniEngineArgs)}
        kwargs = {name: value for name, value in args.items() if name in names}
        kwargs.update(model=tiny_ar_model, skip_tokenizer_init=True, max_model_len=64, enforce_eager=True)
        final = OmniEngineArgs(**kwargs).create_engine_config()
        assert final.kernel_config.linear_backend == "torch"
        assert final.kernel_config.moe_backend == "triton"
        assert final.attention_config.backend is AttentionBackendEnum.TRITON_ATTN


@pytest.mark.parametrize("use_cli_overrides", [False, True], ids=["deploy-yaml", "explicit-kwargs-control"])
def test_generic_diffusion_resolver_preserves_backend_selection(tmp_path, generic_flux_model, use_cli_overrides):
    """Resolve an unregistered diffusion pipeline without weights or hub access."""
    model = generic_flux_model
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        "distributed_executor_backend: uni\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    linear_backend: torch\n"
        "    diffusion_attention_backend: torch_sdpa\n"
        "    enforce_eager: true\n"
    )
    expected = {
        "linear_backend": "torch",
        "diffusion_attention_backend": "torch_sdpa",
        "enforce_eager": True,
        "distributed_executor_backend": "uni",
    }
    resolution = resolve_omni_config(
        str(model),
        trust_remote_code=False,
        deploy_config_path=str(deploy),
        cli_overrides=expected if use_cli_overrides else {},
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert len(resolution.stage_configs) == 1
    args = build_engine_args_dict_from_omni_stage_config(resolution.stage_configs[0], model=str(model))
    config = OmniDiffusionConfig.from_kwargs(
        **extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
    )
    final = create_diffusion_vllm_config(torch.device("cpu"), config)
    attention = config.diffusion_attention_config.default
    assert {
        "linear_backend": final.kernel_config.linear_backend,
        "diffusion_attention_backend": None if attention is None else attention.backend,
        "enforce_eager": config.enforce_eager,
        "distributed_executor_backend": config.distributed_executor_backend,
    } == expected


@pytest.mark.parametrize("with_none_overrides", [False, True])
def test_generic_diffusion_preserves_complete_deploy(tmp_path, generic_flux_model, with_none_overrides):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        "dtype: float32\n"
        "distributed_executor_backend: uni\n"
        "data_parallel_size: 2\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    devices: '4,5,6,7'\n"
        "    tensor_parallel_size: 2\n"
        "    num_replicas: 2\n"
        "    env: {BACKEND_TEST: configured}\n"
        "    linear_backend: torch\n"
        "    load_format: dummy\n"
        "    cache_backend: tea_cache\n"
        "    default_sampling_params: {guidance_scale: 3.0, width: 128}\n"
        "    engine_extras:\n"
        "      extras: {keep: yaml, default_llama_model_id: custom/llama}\n"
    )
    overrides = {"dtype": None, "linear_backend": None, "cache_backend": None} if with_none_overrides else {}
    resolution = resolve_omni_config(
        generic_flux_model,
        trust_remote_code=None,
        deploy_config_path=str(deploy),
        cli_overrides=overrides,
        stage_overrides=None,
        strategy_config_path=None,
    )
    stage = resolution.stage_configs[0]
    assert resolution.config_path == str(deploy)
    assert stage.diffusion_config.dtype is torch.float32
    assert stage.diffusion_config.cache_backend == "tea_cache"
    assert stage.diffusion_config.linear_backend == "torch"
    assert stage.load_config.load_format == "dummy"
    assert stage.parallel_config.tensor_parallel_size == 2
    assert stage.parallel_config.data_parallel_size == 2
    assert stage.parallel_config.world_size == 4
    assert stage.runtime_config.devices == "4,5,6,7"
    assert stage.runtime_config.num_gpus == 4
    assert stage.runtime_config.num_replicas == 2
    assert stage.runtime_config.env == {"BACKEND_TEST": "configured"}
    assert stage.model_config.default_sampling_params == {"guidance_scale": 3.0, "width": 128}
    assert stage.diffusion_config.extras == {
        "keep": "yaml",
        "default_llama_model_id": "custom/llama",
        "auxiliary_text_encoder": None,
    }


@pytest.mark.parametrize("cli_dtype", [None, torch.float32])
def test_generic_diffusion_cli_overrides_platform_deploy(tmp_path, generic_flux_model, monkeypatch, cli_dtype):
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform", SimpleNamespace(device_name="cuda"))
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        "dtype: float32\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    devices: '0,1'\n"
        "    tensor_parallel_size: 2\n"
        "    linear_backend: cutlass\n"
        "    default_sampling_params: {guidance_scale: 3.0, height: 128}\n"
        "    engine_extras:\n"
        "      extras: {keep: yaml, default_llama_model_id: yaml/llama}\n"
        "platforms:\n"
        "  cuda:\n"
        "    stages:\n"
        "      - stage_id: 0\n"
        "        devices: '2,3'\n"
        "        dtype: float16\n"
        "        enforce_eager: true\n"
        "        default_sampling_params: {guidance_scale: 4.0, width: 256}\n"
        "        extras: {keep: platform, added: platform}\n"
    )
    resolution = resolve_omni_config(
        generic_flux_model,
        trust_remote_code=None,
        deploy_config_path=str(deploy),
        cli_overrides={
            "dtype": cli_dtype,
            "linear_backend": "torch",
            "default_sampling_params": {"0": {"guidance_scale": 7.5}},
            "extras": {"added": "cli"},
        },
        stage_overrides={"0": {"tensor_parallel_size": 1, "devices": "5"}},
        strategy_config_path=None,
    )
    stage = resolution.stage_configs[0]
    assert stage.diffusion_config.dtype is (torch.float16 if cli_dtype is None else cli_dtype)
    assert build_engine_args_dict_from_omni_stage_config(stage, model=generic_flux_model)["enforce_eager"] is True
    assert stage.diffusion_config.linear_backend == "torch"
    assert stage.parallel_config.world_size == 1
    assert stage.runtime_config.devices == "5"
    assert stage.model_config.default_sampling_params == {"guidance_scale": 7.5, "height": 128, "width": 256}
    assert stage.diffusion_config.extras["keep"] == "platform"
    assert stage.diffusion_config.extras["added"] == "cli"


@pytest.mark.parametrize("data_parallel_size", [None, 2, 3])
def test_generic_diffusion_num_gpus_uses_deploy_parallelism(tmp_path, generic_flux_model, data_parallel_size):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        ("" if data_parallel_size is None else f"data_parallel_size: {data_parallel_size}\n")
        + "stages:\n  - stage_id: 0\n    tensor_parallel_size: 2\n"
    )
    kwargs = dict(
        model=generic_flux_model,
        trust_remote_code=None,
        deploy_config_path=str(deploy),
        cli_overrides={"num_gpus": 4},
        stage_overrides=None,
        strategy_config_path=None,
    )
    if data_parallel_size == 3:
        with pytest.raises(ValueError, match="does not match WORLD-derived"):
            resolve_omni_config(**kwargs)
    else:
        stage = resolve_omni_config(**kwargs).stage_configs[0]
        assert stage.parallel_config.tensor_parallel_size == 2
        assert stage.parallel_config.data_parallel_size == 2
        assert stage.parallel_config.world_size == stage.runtime_config.num_gpus == 4
        assert stage.runtime_config.devices == "0,1,2,3"


def test_generic_diffusion_deploy_model_class_sets_output_metadata(tmp_path, generic_flux_model):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("stages:\n  - stage_id: 0\n    model_class_name: LTX2Pipeline\n")
    resolution = resolve_omni_config(
        generic_flux_model,
        trust_remote_code=None,
        deploy_config_path=str(deploy),
        cli_overrides={},
        stage_overrides=None,
        strategy_config_path=None,
    )
    stage = resolution.stage_configs[0]
    assert stage.diffusion_config.model_class_name == "LTX2Pipeline"
    assert stage.model_config.model_arch == stage.diffusion_config.model_arch == "LTX2Pipeline"
    assert stage.final_output_type == "video"
    assert resolution.pipeline_config.stages[0] is stage.stage_pipeline_config


@pytest.mark.parametrize("with_deploy", [False, True], ids=["kwargs-only", "deploy-and-kwargs"])
def test_generic_diffusion_top_level_auxiliary_encoder_overrides_extras(tmp_path, generic_flux_model, with_deploy):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("stages:\n  - stage_id: 0\n")
    original_extras = {"auxiliary_text_encoder": "cli/old", "keep": "cli"}
    overrides = {
        "auxiliary_text_encoder": "cli/new",
        "extras": original_extras,
    }
    resolution = resolve_omni_config(
        generic_flux_model,
        trust_remote_code=None,
        deploy_config_path=str(deploy) if with_deploy else None,
        cli_overrides=overrides,
        stage_overrides=None,
        strategy_config_path=None,
    )
    extras = resolution.stage_configs[0].diffusion_config.extras
    assert extras["auxiliary_text_encoder"] == "cli/new"
    assert extras["keep"] == "cli"
    assert original_extras["auxiliary_text_encoder"] == "cli/old"


@pytest.mark.parametrize("with_deploy", [False, True], ids=["kwargs-only", "deploy-and-kwargs"])
def test_generic_diffusion_quantization_alias_reaches_stage(tmp_path, generic_flux_model, with_deploy):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("stages:\n  - stage_id: 0\n")
    resolution = resolve_omni_config(
        generic_flux_model,
        trust_remote_code=None,
        deploy_config_path=str(deploy) if with_deploy else None,
        cli_overrides={"quantization": "fp8"},
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert resolution.stage_configs[0].quantization_config == "fp8"


def test_generic_diffusion_deploy_preserves_orchestrator_overrides(tmp_path, generic_flux_model):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("stages:\n  - stage_id: 0\n")
    config = StageConfigFactory.create_typed_default_diffusion(
        generic_flux_model,
        {"stage_init_timeout": 42, "omni_lb_policy": "round_robin"},
        deploy_config_path=str(deploy),
        default_model_class_name="FluxPipeline",
    )
    assert config.orchestrator_config.stage_init_timeout == 42
    assert config.orchestrator_config.omni_lb_policy == "round_robin"


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("stage_scoped", [False, True])
def test_backend_cli_overrides_deploy_yaml(tmp_path, execution_type, typed, stage_scoped):
    attention_field = (
        "diffusion_attention_backend" if execution_type == StageExecutionType.DIFFUSION else "attention_backend"
    )
    yaml_attention = "flash_attn" if execution_type == StageExecutionType.DIFFUSION else "FLASHINFER"
    cli_attention = "torch_sdpa" if execution_type == StageExecutionType.DIFFUSION else "TRITON_ATTN"
    path = tmp_path / "deploy.yaml"
    path.write_text(
        f"stages:\n  - stage_id: 0\n    linear_backend: cutlass\n    moe_backend: auto\n"
        f"    {attention_field}: {yaml_attention}\n"
    )
    deploy = load_deploy_config(str(path))
    pipeline = _pipeline(execution_type)
    prefix = "stage_0_" if stage_scoped else ""
    cli = {
        f"{prefix}linear_backend": "torch",
        f"{prefix}moe_backend": "triton",
        f"{prefix}{attention_field}": cli_attention,
    }

    if typed:
        stage = VllmOmniConfig.from_pipeline_config(
            pipeline, user_deploy_config=deploy, cli_overrides=cli
        ).stage_configs[0]
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0]
        stage.runtime_overrides = build_stage_runtime_overrides(stage.stage_id, cli)
        args = build_legacy_engine_args_dict(stage.to_omegaconf(), model="unused")

    assert args["linear_backend"] == "torch"
    assert args["moe_backend"] == "triton"
    if execution_type == StageExecutionType.DIFFUSION:
        kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
        kwargs.pop("model")
        config = OmniDiffusionConfig.from_kwargs(**kwargs)
        assert config.diffusion_attention_config.default.backend == cli_attention
        assert config.linear_backend == "torch"
        assert config.moe_backend == "triton"
    else:
        assert args[attention_field] == cli_attention


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
@pytest.mark.parametrize("typed", [False, True])
def test_deploy_backends_are_independent_per_stage(tmp_path, execution_type, typed):
    attention_field = (
        "diffusion_attention_backend" if execution_type == StageExecutionType.DIFFUSION else "attention_backend"
    )
    attentions = (
        ("torch_sdpa", "flash_attn")
        if execution_type == StageExecutionType.DIFFUSION
        else ("TRITON_ATTN", "FLASHINFER")
    )
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "stages:\n"
        f"  - stage_id: 0\n    linear_backend: torch\n    moe_backend: triton\n"
        f"    {attention_field}: {attentions[0]}\n"
        f"  - stage_id: 1\n    linear_backend: cutlass\n    moe_backend: auto\n"
        f"    {attention_field}: {attentions[1]}\n"
    )
    deploy = load_deploy_config(str(path))
    pipeline = PipelineConfig(
        model_type="backend-test",
        stages=tuple(
            StagePipelineConfig(
                stage_id=stage_id,
                model_stage=f"model_{stage_id}",
                execution_type=execution_type,
                final_output=stage_id == 1,
            )
            for stage_id in range(2)
        ),
    )
    if typed:
        stages = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs
        args = [build_engine_args_dict_from_omni_stage_config(stage, model="unused") for stage in stages]
    else:
        stages = merge_pipeline_deploy(pipeline, deploy)
        args = [build_legacy_engine_args_dict(stage.to_omegaconf(), model="unused") for stage in stages]

    for stage_id, (linear, moe) in enumerate((("torch", "triton"), ("cutlass", "auto"))):
        assert args[stage_id]["linear_backend"] == linear
        assert args[stage_id]["moe_backend"] == moe
        if execution_type == StageExecutionType.DIFFUSION:
            kwargs = extract_diffusion_stage_config_kwargs(
                args[stage_id], stage_id=stage_id, include_engine_adapter_metadata=True
            )
            kwargs.pop("model")
            config = OmniDiffusionConfig.from_kwargs(**kwargs)
            assert config.diffusion_attention_config.default.backend == attentions[stage_id]
        else:
            assert args[stage_id][attention_field] == attentions[stage_id]


@pytest.mark.parametrize(
    ("backend_field", "first_class", "extra", "expected"),
    [
        ("linear_backend", "TORCH", "TORCH", "torch"),
        ("linear_backend", "torch", "TORCH", "torch"),
        ("moe_backend", "TRITON", "triton", "triton"),
        ("moe_backend", "FLASHINFER-CUTLASS", "flashinfer_cutlass", "flashinfer_cutlass"),
        ("linear_backend", "FLASHINFER-CUTLASS", "FLASHINFER-CUTLASS", "flashinfer_cutlass"),
    ],
)
@pytest.mark.parametrize("platform_overlay", [False, True])
def test_equivalent_stage_and_extra_backends_do_not_warn(backend_field, first_class, extra, expected, platform_overlay):
    extras = {backend_field: extra, "unrelated": "keep"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stage = StageDeployConfig(
            stage_id=0,
            **{backend_field: first_class},
            engine_extras={} if platform_overlay else extras,
        )
        if platform_overlay:
            deploy = DeployConfig(
                stages=[stage],
                platforms={"rocm": {"stages": [{"stage_id": 0, "engine_extras": extras}]}},
            )
            stage = _apply_platform_overrides(deploy, platform="rocm").stages[0]
    assert not [warning for warning in caught if "engine_extras" in str(warning.message)]
    assert getattr(stage, backend_field) == expected
    assert backend_field not in stage.engine_extras
    assert stage.engine_extras["unrelated"] == "keep"
    assert extras == {backend_field: extra, "unrelated": "keep"}


@pytest.mark.parametrize(("backend_field", "backend"), [("linear_backend", "torch"), ("moe_backend", "triton")])
def test_first_class_backend_still_overrides_invalid_extra(backend_field, backend):
    with pytest.warns(UserWarning, match=f"{backend_field}.*engine_extras"):
        stage = StageDeployConfig(
            stage_id=0,
            **{backend_field: backend},
            engine_extras={backend_field: "not-a-backend"},
        )
    assert getattr(stage, backend_field) == backend
    assert backend_field not in stage.engine_extras


@pytest.mark.parametrize("config_cls", [OmniStageModelConfig, OmniDiffusionConfig])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"linear_backend": "AUTO", "moe_backend": "AUTO"},
        {"linear_backend": "FLASHINFER-CUTLASS", "moe_backend": "FLASHINFER-CUTLASS"},
    ],
)
def test_backend_defaults_and_upstream_normalization(config_cls, kwargs):
    config = config_cls(**kwargs)
    upstream = KernelConfig(**kwargs)
    assert config.linear_backend == upstream.linear_backend
    assert config.moe_backend == upstream.moe_backend


@pytest.mark.parametrize("config_cls", [OmniStageModelConfig, OmniDiffusionConfig])
@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_invalid_backend_rejected_by_upstream(config_cls, field):
    with pytest.raises(ValueError, match=field):
        config_cls(**{field: "not-a-backend"})


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_first_class_backend_wins_over_engine_extras(execution_type, backend):
    with pytest.warns(UserWarning, match="linear_backend.*engine_extras"):
        deploy = DeployConfig(
            stages=[
                StageDeployConfig(
                    stage_id=0,
                    linear_backend=backend,
                    engine_extras={"linear_backend": "cutlass"},
                )
            ]
        )
    pipeline = _pipeline(execution_type)
    legacy = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    typed = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
    assert build_legacy_engine_args_dict(legacy, model="unused")["linear_backend"] == backend
    assert build_engine_args_dict_from_omni_stage_config(typed, model="unused")["linear_backend"] == backend


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
def test_extras_only_backend_is_preserved(execution_type):
    pipeline = _pipeline(execution_type)
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, engine_extras={"linear_backend": "torch"})])
    legacy = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    typed = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
    assert build_legacy_engine_args_dict(legacy, model="unused")["linear_backend"] == "torch"
    assert build_engine_args_dict_from_omni_stage_config(typed, model="unused")["linear_backend"] == "torch"


def test_platform_backend_override_replaces_legacy_extra():
    deploy = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, engine_extras={"attention_backend": "FLASHINFER"})],
        platforms={"rocm": {"stages": [{"stage_id": 0, "attention_backend": "TRITON_ATTN"}]}},
    )
    deploy = _apply_platform_overrides(deploy, platform="rocm")
    pipeline = _pipeline(StageExecutionType.LLM_AR)
    legacy = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    typed = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
    assert build_legacy_engine_args_dict(legacy, model="unused")["attention_backend"] == "TRITON_ATTN"
    assert build_engine_args_dict_from_omni_stage_config(typed, model="unused")["attention_backend"] == "TRITON_ATTN"


@pytest.mark.parametrize(
    ("field", "first_class", "legacy"),
    [
        ("attention_backend", "TRITON_ATTN", "FLASHINFER"),
        ("moe_backend", "triton", "torch"),
        ("linear_backend", "torch", "cutlass"),
    ],
)
def test_nested_platform_engine_extras_cannot_override_first_class_backend(field, first_class, legacy):
    deploy = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, **{field: first_class})],
        platforms={"rocm": {"stages": [{"stage_id": 0, "engine_extras": {field: legacy}}]}},
    )
    with pytest.warns(UserWarning, match=f"{field}.*engine_extras"):
        deploy = _apply_platform_overrides(deploy, platform="rocm")

    pipeline = _pipeline(StageExecutionType.LLM_AR)
    legacy = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    typed = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
    assert deploy.stages[0].engine_extras.get(field) is None
    expected = first_class if field == "attention_backend" else first_class.lower()
    assert build_legacy_engine_args_dict(legacy, model="unused")[field] == expected
    assert build_engine_args_dict_from_omni_stage_config(typed, model="unused")[field] == expected


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_diffusion_model_config_conflict_warns_and_model_wins(field):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        model_config=OmniStageModelConfig(**{field: "torch" if field == "linear_backend" else "triton"}),
        diffusion_config=_DiffusionConfigProjection.from_kwargs(
            **{field: "cutlass" if field == "linear_backend" else "auto"}
        ),
    )
    with pytest.warns(UserWarning, match=f"model_config.*{field}.*diffusion_config"):
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    assert args[field] == ("torch" if field == "linear_backend" else "triton")


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_diffusion_equal_normalized_explicit_backends_do_not_warn(field):
    value = "torch" if field == "linear_backend" else "triton"
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        model_config=OmniStageModelConfig(**{field: value.upper()}),
        diffusion_config=_DiffusionConfigProjection.from_kwargs(**{field: value}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    assert not [warning for warning in caught if "model_config" in str(warning.message)]
    assert args[field] == value


@pytest.mark.parametrize("config_field", ["model_config", "diffusion_config"])
def test_diffusion_explicit_backend_is_not_overwritten_by_defaults(config_field):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        **{config_field: {"linear_backend": "torch", "moe_backend": "triton"}},
    )
    args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    assert args["linear_backend"] == "torch"
    assert args["moe_backend"] == "triton"
