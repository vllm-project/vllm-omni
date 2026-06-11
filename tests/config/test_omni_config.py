# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the additive structured Omni config."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_omni.config.omni_config import (
    DiffusionConfig,
    ParallelConfig,
    RuntimeConfig,
    VllmOmniConfig,
    VllmOmniStageConfig,
)
from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _PIPELINE_REGISTRY,
    DeployConfig,
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)

_DEPLOY_DIR = Path(__file__).parents[2] / "vllm_omni" / "deploy"


def _load_default_deploy(model_type: str) -> DeployConfig:
    deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
    if deploy_path.exists():
        return load_deploy_config(deploy_path)
    return DeployConfig()


@pytest.mark.parametrize("model_type", sorted(_OMNI_PIPELINES))
def test_vllm_omni_config_from_registry_matches_merge_pipeline_deploy(model_type: str):
    pipeline = _PIPELINE_REGISTRY[model_type]
    legacy_deploy = _load_default_deploy(model_type)
    structured_deploy = _load_default_deploy(model_type)

    legacy_stages = merge_pipeline_deploy(pipeline, legacy_deploy)
    omni_config = VllmOmniConfig.from_registry(pipeline, structured_deploy)

    assert omni_config.pipeline_config is pipeline
    assert omni_config.pipeline is pipeline
    assert len(omni_config.stage_configs) == len(legacy_stages)

    for legacy_stage, omni_stage in zip(legacy_stages, omni_config.stage_configs, strict=True):
        assert omni_config.stage_by_id(legacy_stage.stage_id) is omni_stage
        assert omni_stage.to_legacy_stage_config() == legacy_stage

        assert omni_stage.stage_pipeline_config is pipeline.get_stage(legacy_stage.stage_id)
        assert omni_stage.model_config.default_sampling_params == legacy_stage.yaml_extras.get(
            "default_sampling_params"
        )
        assert omni_stage.connector_config.output_connectors == legacy_stage.yaml_extras.get("output_connectors")
        assert omni_stage.connector_config.input_connectors == legacy_stage.yaml_extras.get("input_connectors")
        assert omni_stage.runtime_config.devices == legacy_stage.yaml_runtime.get("devices")
        assert omni_stage.runtime_config.num_replicas == legacy_stage.yaml_runtime.get("num_replicas", 1)

        engine_args = legacy_stage.yaml_engine_args
        assert omni_stage.model_config.enforce_eager == engine_args.get("enforce_eager", False)
        assert omni_stage.load_config.load_format == engine_args.get("load_format", "auto")
        assert omni_stage.load_config.tokenizer_mode == engine_args.get("tokenizer_mode", "auto")
        assert omni_stage.cache_config.gpu_memory_utilization == engine_args.get("gpu_memory_utilization", 0.90)
        assert omni_stage.scheduler_config.max_num_seqs == engine_args.get("max_num_seqs", 128)
        assert omni_stage.scheduler_config.max_num_batched_tokens == engine_args.get("max_num_batched_tokens")
        assert omni_stage.scheduler_config.async_scheduling == engine_args.get("async_scheduling", True)
        assert omni_stage.parallel_config.tensor_parallel_size == engine_args.get("tensor_parallel_size", 1)
        assert omni_stage.parallel_config.data_parallel_size == engine_args.get("data_parallel_size", 1)
        assert omni_stage.parallel_config.pipeline_parallel_size == engine_args.get("pipeline_parallel_size", 1)

        if omni_stage.stage_pipeline_config.execution_type == StageExecutionType.DIFFUSION:
            assert omni_stage.diffusion_config is not None
            assert omni_stage.diffusion_config.stage_id == legacy_stage.stage_id
            assert omni_stage.diffusion_config.model_arch == engine_args.get("model_arch")
        else:
            assert omni_stage.diffusion_config is None


def test_stage_by_id_raises_for_unknown_stage():
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    deploy = _load_default_deploy("qwen3_tts")
    omni_config = VllmOmniConfig.from_registry(pipeline, deploy)

    with pytest.raises(KeyError, match="no stage 99"):
        omni_config.stage_by_id(99)


def test_vllm_omni_stage_config_public_fields_match_rfc_container_shape():
    public_fields = {f.name for f in fields(VllmOmniStageConfig)}

    assert public_fields == {
        "stage_pipeline_config",
        "model_config",
        "load_config",
        "cache_config",
        "scheduler_config",
        "connector_config",
        "runtime_config",
        "parallel_config",
        "diffusion_config",
        "quantization_config",
    }


def test_runtime_config_fields_match_rfc_runtime_scope():
    assert {f.name for f in fields(RuntimeConfig)} == {
        "devices",
        "num_replicas",
        "env",
        "num_gpus",
        "log_level",
        "log_stats",
        "profiler_config",
    }


def test_parallel_config_rejects_cfg_parallel_size_outside_rfc_bound():
    with pytest.raises(ValidationError):
        ParallelConfig(cfg_parallel_size=3)


def test_diffusion_config_preserves_existing_coercion_hooks():
    import torch

    from vllm_omni.diffusion.data import AttentionConfig, DiffusionCacheConfig

    cfg = DiffusionConfig(
        dtype="float32",
        cache_config={"rel_l1_thresh": 0.3},
        diffusion_attention_config={"default": "flash_attn"},
        diffusion_kv_cache_skip_steps="0-2,4",
        diffusion_kv_cache_skip_layers=[1, 3],
    )

    assert cfg.dtype is torch.float32
    assert isinstance(cfg.cache_config, DiffusionCacheConfig)
    assert isinstance(cfg.diffusion_attention_config, AttentionConfig)
    assert cfg.diffusion_attention_config.default.backend == "flash_attn"
    assert cfg.diffusion_kv_cache_skip_step_indices == {0, 1, 2, 4}
    assert cfg.diffusion_kv_cache_skip_layer_indices == {1, 3}
    assert cfg.max_cpu_loras == 1
