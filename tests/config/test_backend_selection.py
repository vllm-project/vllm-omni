# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for Item 1 backend fields and their final projections."""

from dataclasses import fields

import pytest
import torch
from transformers import LlamaConfig
from vllm.config import KernelConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_omni.config.omni_config import (
    OmniStageModelConfig,
    VllmOmniConfig,
    VllmOmniDiffusionStageConfig,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    _apply_platform_overrides,
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
    else:
        names = {field.name for field in fields(OmniEngineArgs)}
        kwargs = {name: value for name, value in args.items() if name in names}
        kwargs.update(model=tiny_ar_model, skip_tokenizer_init=True, max_model_len=64, enforce_eager=True)
        final = OmniEngineArgs(**kwargs).create_engine_config()
        assert final.kernel_config.linear_backend == "torch"
        assert final.kernel_config.moe_backend == "triton"
        assert final.attention_config.backend is AttentionBackendEnum.TRITON_ATTN


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


@pytest.mark.parametrize("config_field", ["model_config", "diffusion_config"])
def test_diffusion_explicit_backend_is_not_overwritten_by_defaults(config_field):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        **{config_field: {"linear_backend": "torch", "moe_backend": "triton"}},
    )
    args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    assert args["linear_backend"] == "torch"
    assert args["moe_backend"] == "triton"
