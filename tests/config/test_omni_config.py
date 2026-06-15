# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the additive structured Omni config."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_omni.config import omni_config as omni_config_module
from vllm_omni.config.omni_config import (
    CacheConfig,
    ConnectorConfig,
    DiffusionConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    RuntimeConfig,
    SchedulerConfig,
    VllmOmniConfig,
    VllmOmniStageConfig,
)
from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _PIPELINE_REGISTRY,
    PIPELINE_WIDE_ENGINE_FIELDS,
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.engine.stage_init_utils import build_engine_args_dict

_DEPLOY_DIR = Path(__file__).parents[2] / "vllm_omni" / "deploy"


@pytest.fixture(autouse=True)
def _stable_test_platform(monkeypatch):
    from vllm_omni import platforms

    platform = platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(platform, "device_type", "cpu", raising=False)


def _load_default_deploy(model_type: str) -> DeployConfig:
    deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
    if deploy_path.exists():
        return load_deploy_config(deploy_path)
    return DeployConfig()


@pytest.mark.parametrize("model_type", sorted(_OMNI_PIPELINES))
def test_vllm_omni_config_from_registry_matches_merge_pipeline_deploy(model_type: str):
    pipeline = _PIPELINE_REGISTRY[model_type]
    legacy_deploy = _load_default_deploy(model_type)

    legacy_stages = merge_pipeline_deploy(pipeline, legacy_deploy)
    omni_config = VllmOmniConfig.from_registry(model_type)

    assert omni_config.pipeline_config is pipeline
    assert len(omni_config.stage_configs) == len(legacy_stages)

    for legacy_stage, omni_stage in zip(legacy_stages, omni_config.stage_configs, strict=True):
        assert omni_config.stage_by_id(legacy_stage.stage_id) is omni_stage

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
        legacy_parallel_config = engine_args.get("parallel_config") or {}
        assert omni_stage.parallel_config.tensor_parallel_size == legacy_parallel_config.get(
            "tensor_parallel_size",
            engine_args.get("tensor_parallel_size", 1),
        )

        if omni_stage.stage_pipeline_config.execution_type == StageExecutionType.DIFFUSION:
            assert omni_stage.diffusion_config is not None
            assert omni_stage.diffusion_config.stage_id == legacy_stage.stage_id
            assert omni_stage.diffusion_config.model_arch == engine_args.get("model_arch")
        else:
            assert omni_stage.diffusion_config is None


def test_stage_by_id_raises_for_unknown_stage():
    omni_config = VllmOmniConfig.from_registry("qwen3_tts")

    with pytest.raises(KeyError, match="no stage 99"):
        omni_config.stage_by_id(99)


def test_from_registry_preserves_current_pipeline_config_object():
    omni_config = VllmOmniConfig.from_registry("minicpmo_4_5")

    assert omni_config.pipeline_config is _PIPELINE_REGISTRY["minicpmo_4_5"]
    assert not hasattr(omni_config, "pipeline")
    assert "hf_config_predicate" in {f.name for f in fields(PipelineConfig)}
    assert omni_config.pipeline_config.hf_config_predicate is _PIPELINE_REGISTRY["minicpmo_4_5"].hf_config_predicate


def test_from_registry_normalizes_stage_engine_extras_without_expanding_stage_deploy_config():
    assert not hasattr(StageDeployConfig, "model_config")
    assert not hasattr(StageDeployConfig, "parallel_config")

    stage = VllmOmniConfig.from_registry("dreamzero", deploy_config_path="dreamzero_tp1_cfg2").stage_by_id(0)

    assert stage.parallel_config.tensor_parallel_size == 1
    assert stage.parallel_config.cfg_parallel_size == 2
    assert stage.diffusion_config is not None
    assert stage.diffusion_config.model_config["default_robot_embodiment"] == "roboarena"


def test_from_registry_applies_cli_overrides_without_stage_config_runtime_bridge():
    omni_config = VllmOmniConfig.from_registry(
        "qwen3_tts",
        cli_overrides={
            "stage_0_max_num_seqs": 7,
            "stage_1_tensor_parallel_size": 2,
        },
    )

    stage0 = omni_config.stage_by_id(0)
    stage1 = omni_config.stage_by_id(1)

    assert stage0.scheduler_config.max_num_seqs == 7
    assert stage1.parallel_config.tensor_parallel_size == 2


def test_pipeline_deploy_cli_fields_reuse_legacy_pipeline_wide_engine_fields():
    assert omni_config_module._PIPELINE_DEPLOY_CLI_FIELDS is PIPELINE_WIDE_ENGINE_FIELDS
    assert "active_stream_window" in omni_config_module._PIPELINE_DEPLOY_CLI_FIELDS
    assert "custom_voice_dir" in omni_config_module._PIPELINE_DEPLOY_CLI_FIELDS


def test_from_registry_keeps_worker_backend_separate_from_distributed_executor_backend():
    omni_config = VllmOmniConfig.from_registry("dreamzero", deploy_config_path="dreamzero_tp1_cfg2")

    assert omni_config.stage_by_id(0).parallel_config.distributed_executor_backend == "mp"
    assert omni_config.orchestrator_config.worker_backend == "multi_process"


def test_from_registry_maps_orchestrator_cli_overrides():
    omni_config = VllmOmniConfig.from_registry(
        "qwen3_tts",
        cli_overrides={
            "stage_init_timeout": 1200,
            "init_timeout": 1800,
            "worker_backend": "ray",
            "ray_address": "ray://127.0.0.1:10001",
            "omni_master_address": "127.0.0.1",
            "omni_master_port": 12345,
            "omni_dp_size_local": 2,
            "omni_lb_policy": "round_robin",
            "omni_heartbeat_timeout": 9.5,
            "shm_threshold_bytes": 4096,
            "batch_timeout": 3,
        },
    )

    orchestrator_config = omni_config.orchestrator_config
    assert orchestrator_config.stage_init_timeout == 1200
    assert orchestrator_config.init_timeout == 1800
    assert orchestrator_config.worker_backend == "ray"
    assert orchestrator_config.ray_address == "ray://127.0.0.1:10001"
    assert orchestrator_config.omni_master_address == "127.0.0.1"
    assert orchestrator_config.omni_master_port == 12345
    assert orchestrator_config.omni_dp_size_local == 2
    assert orchestrator_config.omni_lb_policy == "round_robin"
    assert orchestrator_config.omni_heartbeat_timeout == 9.5
    assert orchestrator_config.shm_threshold_bytes == 4096
    assert orchestrator_config.batch_timeout == 3


def test_from_registry_records_loaded_deploy_path_on_orchestrator_config():
    omni_config = VllmOmniConfig.from_registry("dreamzero", deploy_config_path="dreamzero_tp1_cfg2")

    assert omni_config.pipeline_config.model_type == "dreamzero"
    assert omni_config.orchestrator_config.deploy_config_path == str(_DEPLOY_DIR / "dreamzero_tp1_cfg2.yaml")


def test_from_registry_dispatches_async_chunk_processors_without_mutating_topology():
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]

    async_config = VllmOmniConfig.from_registry("qwen3_tts")
    assert async_config.stage_by_id(0).custom_process_next_stage_input_func.endswith("talker2code2wav_async_chunk")
    assert async_config.stage_by_id(1).custom_process_input_func is None

    sync_config = VllmOmniConfig.from_registry("qwen3_tts", cli_overrides={"async_chunk": False})
    assert sync_config.stage_by_id(0).custom_process_next_stage_input_func.endswith("talker2code2wav_full_payload")
    assert sync_config.stage_by_id(1).custom_process_input_func.endswith("talker2code2wav_token_only")

    assert pipeline.get_stage(0).custom_process_next_stage_input_func.endswith("talker2code2wav_full_payload")
    assert pipeline.get_stage(1).custom_process_input_func is None


def test_vllm_omni_stage_config_public_fields_match_rfc_container_shape():
    assert not hasattr(VllmOmniStageConfig, "from_stage_config")
    assert not hasattr(VllmOmniStageConfig, "to_legacy_stage_config")

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


def test_sub_config_fields_match_rfc_scopes():
    assert {f.name for f in fields(ModelConfig)} == {
        "enable_sleep_mode",
        "default_sampling_params",
        "subtalker_sampling_params",
        "has_sampling_extra_args",
        "task_type",
        "codec_frame_rate_hz",
        "enforce_eager",
        "enable_flashinfer_autotune",
        "compilation_config",
        "enable_multithread_weight_load",
        "num_weight_load_threads",
        "disable_autocast",
    }
    assert {f.name for f in fields(LoadConfig)} == {
        "load_format",
        "tokenizer_mode",
        "config_format",
        "skip_mm_profiling",
    }
    assert {f.name for f in fields(CacheConfig)} == {
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "disable_hybrid_kv_cache_manager",
        "mm_processor_cache_gb",
    }
    assert {f.name for f in fields(SchedulerConfig)} == {
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "enable_chunked_prefill",
        "async_scheduling",
    }
    assert {f.name for f in fields(ConnectorConfig)} == {
        "stage_connector",
        "output_connectors",
        "input_connectors",
    }
    assert {f.name for f in fields(ParallelConfig)} == {
        "pipeline_parallel_size",
        "data_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "ulysses_degree",
        "ring_degree",
        "ulysses_mode",
        "cfg_parallel_size",
        "vae_patch_parallel_size",
        "use_hsdp",
        "hsdp_shard_size",
        "hsdp_replicate_size",
        "enable_expert_parallel",
        "world_size",
    }


def test_parallel_config_keeps_current_diffusion_parallel_surface():
    cfg = ParallelConfig(
        pipeline_parallel_size=2,
        data_parallel_size=3,
        tensor_parallel_size=4,
        cfg_parallel_size=3,
    )

    assert cfg.pipeline_parallel_size == 2
    assert cfg.data_parallel_size == 3
    assert cfg.cfg_parallel_size == 3
    assert cfg.world_size == 72


def test_from_registry_preserves_legacy_pp_dp_for_world_size():
    cfg = VllmOmniConfig.from_registry("hunyuan_image3_dit").stage_by_id(0).parallel_config

    assert cfg.pipeline_parallel_size == 1
    assert cfg.data_parallel_size == 1
    assert cfg.tensor_parallel_size == 4
    assert cfg.world_size == 4


def test_parallel_config_rejects_cfg_parallel_size_outside_current_bound():
    with pytest.raises(ValidationError):
        ParallelConfig(cfg_parallel_size=4)


def test_from_registry_matches_stage_config_to_omegaconf_behavior_for_representative_stage():
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    legacy_stage = merge_pipeline_deploy(pipeline, _load_default_deploy("qwen3_tts"))[0]
    omega_stage = legacy_stage.to_omegaconf()
    omni_stage = VllmOmniConfig.from_registry("qwen3_tts").stage_by_id(legacy_stage.stage_id)

    assert omega_stage.stage_id == omni_stage.stage_id
    assert omega_stage.stage_type == omni_stage.stage_type.value
    assert omega_stage.engine_input_source == omni_stage.input_sources
    assert omega_stage.final_output == omni_stage.final_output
    assert omega_stage.final_output_type == omni_stage.final_output_type
    assert omega_stage.is_comprehension == omni_stage.is_comprehension
    assert omega_stage.engine_args.model_stage == omni_stage.model_stage
    assert omega_stage.engine_args.worker_type == omni_stage.worker_type
    assert omega_stage.engine_args.scheduler_cls == omni_stage.scheduler_cls
    assert omega_stage.runtime.process is True
    assert omega_stage.runtime.requires_multimodal_data == omni_stage.requires_multimodal_data


def test_from_registry_matches_to_omegaconf_diffusion_parallel_config():
    pipeline = _PIPELINE_REGISTRY["hunyuan_image3_dit"]
    legacy_stage = merge_pipeline_deploy(pipeline, _load_default_deploy("hunyuan_image3_dit"))[0]
    omega_stage = legacy_stage.to_omegaconf()
    omni_stage = VllmOmniConfig.from_registry("hunyuan_image3_dit").stage_by_id(legacy_stage.stage_id)

    assert (
        omega_stage.engine_args.parallel_config.pipeline_parallel_size
        == omni_stage.parallel_config.pipeline_parallel_size
    )
    assert omega_stage.engine_args.parallel_config.data_parallel_size == omni_stage.parallel_config.data_parallel_size
    assert (
        omega_stage.engine_args.parallel_config.tensor_parallel_size == omni_stage.parallel_config.tensor_parallel_size
    )
    assert (
        omega_stage.engine_args.parallel_config.sequence_parallel_size
        == omni_stage.parallel_config.sequence_parallel_size
    )
    assert omega_stage.engine_args.parallel_config.cfg_parallel_size == omni_stage.parallel_config.cfg_parallel_size
    assert (
        omega_stage.engine_args.parallel_config.vae_patch_parallel_size
        == omni_stage.parallel_config.vae_patch_parallel_size
    )


def test_from_registry_matches_build_engine_args_dict_behavior_for_representative_stage(monkeypatch):
    from vllm_omni.engine import stage_init_utils

    monkeypatch.setattr(stage_init_utils, "resolve_worker_cls", lambda engine_args: None)
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    legacy_stage = merge_pipeline_deploy(pipeline, _load_default_deploy("qwen3_tts"))[0]
    omega_stage = legacy_stage.to_omegaconf()
    legacy_engine_args = build_engine_args_dict(
        omega_stage,
        model="/tmp/qwen3-tts",
        stage_connector_spec={"name": "SharedMemoryConnector", "extra": {}},
    )
    omni_stage = VllmOmniConfig.from_registry("qwen3_tts").stage_by_id(legacy_stage.stage_id)

    assert legacy_engine_args["model"] == "/tmp/qwen3-tts"
    assert legacy_engine_args["stage_id"] == omni_stage.stage_id
    assert legacy_engine_args["model_stage"] == omni_stage.model_stage
    assert legacy_engine_args["worker_type"] == omni_stage.worker_type
    assert legacy_engine_args["scheduler_cls"] == omni_stage.scheduler_cls
    assert legacy_engine_args["stage_connector_spec"] == {"name": "SharedMemoryConnector", "extra": {}}
    assert legacy_engine_args["has_sampling_extra_args"] == bool(
        (omni_stage.model_config.default_sampling_params or {}).get("extra_args")
    )


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


def test_diffusion_config_field_classification_covers_current_fields():
    classified_fields = (
        omni_config_module._DIFFUSION_SHARED_CONFIG_FIELDS
        | omni_config_module._DIFFUSION_RUNTIME_CONFIG_FIELDS
        | omni_config_module._DIFFUSION_PARALLEL_CONFIG_FIELDS
        | omni_config_module._DIFFUSION_ONLY_CONFIG_FIELDS
    )

    assert classified_fields == {f.name for f in fields(DiffusionConfig)}
    assert {
        "enable_prompt_embed_cache",
        "prompt_embed_cache_size",
        "diffusion_kv_cache_dtype",
    } <= omni_config_module._DIFFUSION_ONLY_CONFIG_FIELDS
    assert {
        "revision",
        "trust_remote_code",
        "distributed_executor_backend",
    } <= omni_config_module._DIFFUSION_SHARED_CONFIG_FIELDS
    assert "prompt_file_path" in omni_config_module._DIFFUSION_RUNTIME_CONFIG_FIELDS
