# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured vLLM-Omni configuration classes.

This module is additive for Phase 2 of RFC #4021.  ``VllmOmniConfig.from_registry``
builds the structured view directly from the pipeline registry and deploy config
so parity can be proven before later PRs cut consumers over to these classes.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import InitVar, field, fields
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field
from vllm.config.utils import config

from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    _DEPLOY_DIR,
    _PIPELINE_REGISTRY,
    load_deploy_config,
)


_STAGE_OVERRIDE_PATTERN = re.compile(r"^stage_(\d+)_(.+)$")

_EXECUTION_TYPE_TO_STAGE_WORKER: dict[StageExecutionType, tuple[StageType, str | None]] = {
    StageExecutionType.LLM_AR: (StageType.LLM, "ar"),
    StageExecutionType.LLM_GENERATION: (StageType.LLM, "generation"),
    StageExecutionType.DIFFUSION: (StageType.DIFFUSION, None),
}

_ASYNC_AR_SCHEDULER = "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler"
_SYNC_AR_SCHEDULER = "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"
_GENERATION_SCHEDULER = "vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler"

_ORCHESTRATOR_ONLY_CLI_KEYS = frozenset(
    {
        "api_key",
        "allowed_local_media_path",
        "allowed_media_domains",
        "chat_template",
        "chat_template_content_format",
        "deploy_config_path",
        "disable_frontend_multiprocessing",
        "enable_auto_tool_choice",
        "enable_prompt_tokens_details",
        "enable_request_id_headers",
        "enable_server_load_tracking",
        "enable_ssl_refresh",
        "enable_tokenizer_info_endpoint",
        "enable_tool_server",
        "enable_tool_template",
        "enable_log_outputs",
        "host",
        "port",
        "root_path",
        "served_model_name",
        "ssl_ca_certs",
        "ssl_cert_reqs",
        "ssl_certfile",
        "ssl_keyfile",
        "tool_call_parser",
        "tool_parser_plugin",
        "uvicorn_log_level",
    }
)

_PIPELINE_DEPLOY_CLI_FIELDS = (
    "trust_remote_code",
    "distributed_executor_backend",
    "dtype",
    "quantization",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "data_parallel_size",
    "pipeline_parallel_size",
)


def _copy_value(value: Any) -> Any:
    """Copy nested config values so the structured view owns its data."""
    return copy.deepcopy(value)


def _copy_if_not_none(value: Any) -> Any:
    return None if value is None else _copy_value(value)


def _first_defined(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return _copy_value(value)
    return default


def _validate_async_chunk_support(pipeline: PipelineConfig, deploy: DeployConfig) -> None:
    if deploy.async_chunk and not any(
        stage.async_chunk_process_next_stage_input_func or stage.custom_process_next_stage_input_func
        for stage in pipeline.stages
    ):
        raise ValueError(
            f"Pipeline {pipeline.model_type!r} has async_chunk=True in deploy but no stage "
            "declares a next-stage input processor "
            "(``async_chunk_process_next_stage_input_func`` or ``custom_process_next_stage_input_func``). "
            "Either set async_chunk=False or implement an async-chunk processor on the pipeline."
        )


def _resolve_execution_mode(execution_type: StageExecutionType) -> tuple[StageType, str | None]:
    return _EXECUTION_TYPE_TO_STAGE_WORKER.get(execution_type, (StageType.LLM, None))


def _resolve_scheduler_path(execution_type: StageExecutionType, async_scheduling: bool = True) -> str | None:
    if execution_type == StageExecutionType.LLM_AR:
        return _ASYNC_AR_SCHEDULER if async_scheduling else _SYNC_AR_SCHEDULER
    if execution_type == StageExecutionType.LLM_GENERATION:
        return _GENERATION_SCHEDULER
    return None


def _select_processor_funcs(
    topology: StagePipelineConfig,
    async_chunk: bool,
) -> tuple[str | None, str | None]:
    input_proc = topology.custom_process_input_func
    next_stage_proc = topology.custom_process_next_stage_input_func
    if async_chunk and topology.async_chunk_process_next_stage_input_func:
        next_stage_proc = topology.async_chunk_process_next_stage_input_func
    elif not async_chunk and topology.sync_process_input_func:
        input_proc = topology.sync_process_input_func
    return input_proc, next_stage_proc


def _get_recursively_merged_dict(original: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(original)
    for key, update_value in update.items():
        original_value = merged.get(key)
        if isinstance(original_value, Mapping) and isinstance(update_value, Mapping):
            merged[key] = _get_recursively_merged_dict(original_value, update_value)
        else:
            merged[key] = _copy_value(update_value)
    return merged


_PLATFORM_DEEP_MERGE_KEYS = frozenset(
    {
        "default_sampling_params",
        "subtalker_sampling_params",
        "engine_extras",
        "engine_args",
    }
)


def _platform_stage_overrides(
    stage_data: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    runtime_cfg = _mapping_or_empty(stage_data.get("runtime"))
    if "engine_args" in stage_data:
        overrides = dict(_mapping_or_empty(stage_data.get("engine_args")))
        if "num_replicas" in runtime_cfg:
            overrides["num_replicas"] = runtime_cfg["num_replicas"]
        return overrides, runtime_cfg.get("devices"), runtime_cfg.get("env")

    overrides = {
        key: _copy_value(value)
        for key, value in stage_data.items()
        if key not in ("stage_id", "devices", "env")
    }
    return overrides, stage_data.get("devices"), stage_data.get("env")


def _apply_platform_overrides_to_deploy(
    deploy: DeployConfig,
    platform: str | None = None,
) -> DeployConfig:
    if platform is None:
        from vllm_omni.platforms import current_omni_platform

        device_name = getattr(current_omni_platform, "device_name", None)
        platform = device_name.lower() if isinstance(device_name, str) else None
    if platform is None or deploy.platforms is None:
        return deploy

    platform_section = deploy.platforms.get(platform)
    if platform_section is None:
        return deploy

    stage_by_id = {stage.stage_id: stage for stage in deploy.stages}
    for stage_data in platform_section.get("stages", []):
        stage_id = stage_data.get("stage_id")
        stage_deploy = stage_by_id.get(stage_id)
        if stage_deploy is None:
            continue

        overrides, devices, env = _platform_stage_overrides(stage_data)
        if devices is not None:
            stage_deploy.devices = devices
        if env is not None:
            if isinstance(stage_deploy.env, Mapping) and isinstance(env, Mapping):
                stage_deploy.env = {**stage_deploy.env, **env}
            else:
                stage_deploy.env = env

        for key, value in overrides.items():
            if hasattr(stage_deploy, key):
                if key in _PLATFORM_DEEP_MERGE_KEYS and isinstance(value, Mapping):
                    base_value = getattr(stage_deploy, key, None)
                    if isinstance(base_value, Mapping):
                        setattr(stage_deploy, key, _get_recursively_merged_dict(base_value, value))
                        continue
                setattr(stage_deploy, key, value)
            else:
                stage_deploy.engine_extras[key] = _copy_value(value)

    return deploy


def _stage_cli_overrides(stage_id: int, cli_overrides: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in cli_overrides.items():
        if value is None or key in _ORCHESTRATOR_ONLY_CLI_KEYS:
            continue

        match = _STAGE_OVERRIDE_PATTERN.match(key)
        if match is not None:
            override_stage_id = int(match.group(1))
            if override_stage_id == stage_id:
                result[match.group(2)] = _copy_value(value)
            continue

        if key in {
            "model",
            "stage_id",
            "stage_configs_path",
            "async_chunk",
        }:
            continue
        result[key] = _copy_value(value)
    return result


def _resolve_deploy_path(model_type: str, deploy_config_path: str | None = None) -> Path:
    if deploy_config_path is None:
        return _DEPLOY_DIR / f"{model_type}.yaml"

    deploy_path = Path(deploy_config_path)
    if not deploy_path.exists() and deploy_path.parent == Path("."):
        bare_name = deploy_path.name
        if not bare_name.endswith(".yaml"):
            bare_name = f"{bare_name}.yaml"
        candidate = _DEPLOY_DIR / bare_name
        if candidate.exists():
            return candidate
    return deploy_path


@config
class ModelConfig:
    """Per-stage model behavior."""

    enable_sleep_mode: bool = False
    default_sampling_params: dict[str, Any] | None = None
    subtalker_sampling_params: dict[str, Any] | None = None
    has_sampling_extra_args: bool = False
    task_type: str | None = None
    codec_frame_rate_hz: float | None = None
    enforce_eager: bool = False
    enable_flashinfer_autotune: bool | None = None
    compilation_config: dict[str, Any] | None = None
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = Field(default=4, ge=1)
    disable_autocast: bool = False


@config
class LoadConfig:
    """Per-stage loading behavior."""

    load_format: str = "auto"
    tokenizer_mode: str = "auto"
    config_format: str | None = None
    skip_mm_profiling: bool | None = None


@config
class CacheConfig:
    """Per-stage cache and memory behavior."""

    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    enable_prefix_caching: bool = False
    disable_hybrid_kv_cache_manager: bool = False
    mm_processor_cache_gb: float | None = Field(default=None, ge=0.0)


@config
class SchedulerConfig:
    """Per-stage request scheduling behavior."""

    max_num_seqs: int = Field(default=128, ge=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    max_model_len: int | None = Field(default=None, ge=-1)
    enable_chunked_prefill: bool = False
    async_scheduling: bool = True

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= max_num_seqs ({self.max_num_seqs})"
            )


@config
class ConnectorConfig:
    """Per-stage inter-stage connector wiring."""

    stage_connector: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "SharedMemoryConnector",
            "extra": {},
        }
    )
    output_connectors: dict[str, Any] | None = None
    input_connectors: dict[str, Any] | None = None


@config
class RuntimeConfig:
    """Per-stage process placement and runtime behavior."""

    devices: str | None = None
    num_replicas: int = Field(default=1, ge=1)
    env: dict[str, Any] | None = None
    num_gpus: int = Field(default=1, ge=1)
    log_level: str = "info"
    log_stats: bool = False
    profiler_config: dict[str, Any] | None = None


@config
class ParallelConfig:
    """Per-stage distributed parallelism behavior."""

    _pipeline_parallel_size: InitVar[int | None] = None
    _data_parallel_size: InitVar[int | None] = None
    tensor_parallel_size: int = Field(default=1, ge=1)
    sequence_parallel_size: int = Field(default=1, ge=1)
    ulysses_degree: int = Field(default=1, ge=1)
    ring_degree: int = Field(default=1, ge=1)
    ulysses_mode: str = "strict"
    cfg_parallel_size: int = Field(default=1, ge=1, le=2)
    vae_patch_parallel_size: int = Field(default=1, ge=1)
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = Field(default=1, ge=1)
    enable_expert_parallel: bool = False
    world_size: int = Field(default=1, ge=1)

    def __post_init__(
        self,
        _pipeline_parallel_size: int | None,
        _data_parallel_size: int | None,
    ) -> None:
        if self.sequence_parallel_size != self.ulysses_degree * self.ring_degree:
            raise ValueError(
                f"sequence_parallel_size ({self.sequence_parallel_size}) must equal "
                f"ulysses_degree * ring_degree ({self.ulysses_degree * self.ring_degree})"
            )
        if self.ulysses_mode not in {"strict", "advanced_uaa"}:
            raise ValueError("ulysses_mode must be 'strict' or 'advanced_uaa'")

        base_world_size = (
            (_pipeline_parallel_size or 1)
            * (_data_parallel_size or 1)
            * self.tensor_parallel_size
            * self.sequence_parallel_size
            * self.cfg_parallel_size
            * self.vae_patch_parallel_size
        )
        if self.use_hsdp:
            if self.hsdp_shard_size <= 0:
                raise ValueError("hsdp_shard_size must be set when use_hsdp=True")
            self.world_size = self.hsdp_replicate_size * self.hsdp_shard_size
        else:
            self.world_size = base_world_size


@config(config=ConfigDict(arbitrary_types_allowed=True))
class DiffusionConfig:
    """Diffusion-specific per-stage settings.

    Shared AR/diffusion fields are projected into the other sub-configs.  This
    class keeps the diffusion-only knobs from ``OmniDiffusionConfig`` without
    running its startup-time side effects such as port probing or HF metadata
    loading.
    """

    stage_id: int = 0
    model: str | None = None
    model_class_name: str | None = None
    model_arch: str | None = None
    dtype: Any = "auto"
    trust_remote_code: bool = False
    revision: str | None = None
    distributed_executor_backend: str = "mp"
    dist_timeout: int | None = None
    nccl_port: int | None = None
    master_port: int | None = None
    host: str | None = None
    port: int | None = None
    scheduler_port: int = 5555
    model_config: dict[str, Any] = field(default_factory=dict)
    tf_model_config: Any = None
    diffusion_attention_config: Any = None
    cache_strategy: str = "none"
    cache_backend: str = "none"
    cache_config: Any = field(default_factory=dict)
    enable_cache_dit_summary: bool = False
    enable_prompt_embed_cache: bool = False
    prompt_embed_cache_size: int = Field(default=32, ge=1)
    diffusion_load_format: str = "default"
    diffusers_load_kwargs: dict[str, Any] = field(default_factory=dict)
    diffusers_call_kwargs: dict[str, Any] = field(default_factory=dict)
    diffusers_pipeline_cls: Any = None
    lora_path: str | None = None
    lora_scale: float = 1.0
    max_cpu_loras: int | None = None
    output_type: str = "pil"
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    pin_cpu_memory: bool = True
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False
    mask_strategy_file_path: str | None = None
    skip_time_steps: int = 15
    VSA_sparsity: float = 0.0
    moba_config_path: str | None = None
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    diffusion_kv_cache_dtype: str | None = None
    diffusion_kv_cache_skip_steps: str | list[int] | tuple[int, ...] | set[int] | None = None
    diffusion_kv_cache_skip_layers: str | list[int] | tuple[int, ...] | set[int] | None = None
    diffusion_kv_cache_skip_step_indices: set[int] | None = None
    diffusion_kv_cache_skip_layer_indices: set[int] | None = None
    force_cutlass_fp8: bool = False
    enable_diffusion_pipeline_profiler: bool = False
    step_execution: bool = False
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
        }
    )
    override_transformer_cls_name: str | None = None
    worker_extension_cls: str | None = None
    custom_pipeline_args: dict[str, Any] | None = None
    additional_config: dict[str, Any] = field(default_factory=dict)
    enable_stage_verification: bool = True
    prompt_file_path: str | None = None
    quantization_config: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

        from vllm_omni.diffusion.data import (
            AttentionConfig,
            DiffusionCacheConfig,
            TransformerConfig,
            build_attention_config,
            parse_kv_cache_skip_selector,
        )
        from vllm_omni.quantization import build_quant_config

        if self.tf_model_config is None:
            self.tf_model_config = TransformerConfig()
        elif isinstance(self.tf_model_config, Mapping):
            self.tf_model_config = TransformerConfig.from_dict(dict(self.tf_model_config))

        if self.additional_config is None:
            self.additional_config = {}
        elif isinstance(self.additional_config, Mapping):
            self.additional_config = dict(self.additional_config)
        else:
            raise TypeError(f"additional_config must be a mapping or None, got {type(self.additional_config)!r}")

        if isinstance(self.dtype, str):
            import torch

            dtype_map = {
                "auto": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
            }
            self.dtype = dtype_map.get(self.dtype.lower(), torch.bfloat16)

        if isinstance(self.cache_config, Mapping):
            self.cache_config = DiffusionCacheConfig.from_dict(dict(self.cache_config))
        elif not isinstance(self.cache_config, DiffusionCacheConfig):
            self.cache_config = DiffusionCacheConfig()

        self._propagate_quantization_from_tf_config(self.tf_model_config)
        if self.quantization_config is not None:
            if isinstance(self.quantization_config, QuantizationConfig):
                pass
            elif isinstance(self.quantization_config, str):
                self.quantization_config = build_quant_config(self.quantization_config)
            elif isinstance(self.quantization_config, Mapping):
                self.quantization_config = build_quant_config(dict(self.quantization_config))
            else:
                raise TypeError(
                    "quantization_config must be str, dict, QuantizationConfig, or None, "
                    f"got {type(self.quantization_config)!r}"
                )

        if self.diffusion_attention_config is None or isinstance(
            self.diffusion_attention_config,
            (AttentionConfig, Mapping),
        ):
            self.diffusion_attention_config = build_attention_config(self.diffusion_attention_config)
        else:
            raise TypeError(
                "diffusion_attention_config must be an AttentionConfig, mapping, or None, "
                f"got {type(self.diffusion_attention_config)!r}"
            )

        self.diffusion_kv_cache_skip_step_indices = parse_kv_cache_skip_selector(self.diffusion_kv_cache_skip_steps)
        self.diffusion_kv_cache_skip_layer_indices = parse_kv_cache_skip_selector(self.diffusion_kv_cache_skip_layers)

        if self.max_cpu_loras is None:
            self.max_cpu_loras = 1
        elif self.max_cpu_loras < 1:
            raise ValueError("max_cpu_loras must be >= 1 for diffusion LoRA")

        if self.diffusion_load_format != "diffusers" and (self.diffusers_load_kwargs or self.diffusers_call_kwargs):
            raise ValueError(
                "diffusers_load_kwargs and diffusers_call_kwargs are only "
                "valid together with diffusion_load_format=diffusers"
            )

    def _propagate_quantization_from_tf_config(self, tf_config: Any) -> None:
        quant_config = getattr(tf_config, "quant_config", None)
        if quant_config is None:
            return
        quant_method = getattr(tf_config, "quant_method", None)
        is_checkpoint_fp8 = bool(getattr(quant_config, "is_checkpoint_fp8_serialized", False))
        is_checkpoint_nvfp4 = bool(getattr(quant_config, "is_checkpoint_nvfp4_serialized", False))
        should_use_checkpoint_config = (
            self.quantization_config is None
            or (is_checkpoint_fp8 and self._is_generic_fp8_quant_config(self.quantization_config))
            or (is_checkpoint_nvfp4 and self._is_generic_nvfp4_quant_config(self.quantization_config))
        )
        if should_use_checkpoint_config:
            self.quantization_config = quant_config
            if quant_method is not None:
                self.additional_config.setdefault("auto_detected_quant_method", quant_method)

    @staticmethod
    def _is_generic_fp8_quant_config(quant_config: object) -> bool:
        if isinstance(quant_config, str):
            return quant_config.lower() == "fp8"
        if isinstance(quant_config, Mapping):
            method = quant_config.get("method", quant_config.get("quant_method"))
            return isinstance(method, str) and method.lower() == "fp8"
        if hasattr(quant_config, "get_name"):
            return quant_config.get_name() == "fp8"
        return False

    @staticmethod
    def _is_generic_nvfp4_quant_config(quant_config: object) -> bool:
        if isinstance(quant_config, str):
            return quant_config.lower() in {"fp4", "nvfp4", "modelopt_fp4"}
        if isinstance(quant_config, Mapping):
            method = quant_config.get("method", quant_config.get("quant_method"))
            return isinstance(method, str) and method.lower() in {"fp4", "nvfp4", "modelopt_fp4"}
        if hasattr(quant_config, "get_name"):
            return quant_config.get_name() == "modelopt_fp4"
        return False

    def set_tf_model_config(self, tf_config: Any) -> None:
        self.tf_model_config = tf_config
        self._propagate_quantization_from_tf_config(tf_config)

    def enrich_config(self) -> None:
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        omni_diffusion_fields = frozenset(f.name for f in fields(OmniDiffusionConfig))
        kwargs = {
            name: _copy_value(getattr(self, name)) for name in _DIFFUSION_CONFIG_FIELDS if name in omni_diffusion_fields
        }
        omni_diffusion_config = OmniDiffusionConfig(**kwargs)
        omni_diffusion_config.enrich_config()
        for name in _DIFFUSION_CONFIG_FIELDS:
            if hasattr(omni_diffusion_config, name):
                setattr(self, name, _copy_value(getattr(omni_diffusion_config, name)))


_DIFFUSION_CONFIG_FIELDS = frozenset(f.name for f in fields(DiffusionConfig))


_STAGE_DEPLOY_TYPED_ENGINE_FIELDS: tuple[str, ...] = (
    "subtalker_sampling_params",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "enforce_eager",
    "async_scheduling",
    "disable_hybrid_kv_cache_manager",
    "mm_processor_cache_gb",
    "enable_expert_parallel",
    "ulysses_degree",
    "ulysses_mode",
    "ring_degree",
    "sequence_parallel_size",
    "cfg_parallel_size",
    "vae_patch_parallel_size",
    "use_hsdp",
    "hsdp_shard_size",
    "hsdp_replicate_size",
    "compilation_config",
    "profiler_config",
    "skip_mm_profiling",
    "enable_flashinfer_autotune",
    "config_format",
    "load_format",
    "tokenizer_mode",
)

_DIFFUSION_STAGE_ENGINE_FIELDS = _DIFFUSION_CONFIG_FIELDS - {"model", "stage_id"}


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stage_engine_overrides(stage_deploy: StageDeployConfig | None) -> dict[str, Any]:
    if stage_deploy is None:
        return {}

    overrides: dict[str, Any] = {}
    for name in _STAGE_DEPLOY_TYPED_ENGINE_FIELDS:
        value = getattr(stage_deploy, name)
        if value is not None:
            overrides[name] = _copy_value(value)
    overrides.update(_copy_value(stage_deploy.engine_extras))
    return overrides


def _stage_engine_values(
    stage_deploy: StageDeployConfig | None,
    stage_cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    engine = _stage_engine_overrides(stage_deploy)
    if stage_cli_overrides:
        engine.update(_copy_value(stage_cli_overrides))
    return engine


def _stage_sampling_params(
    stage_deploy: StageDeployConfig | None,
    topology: StagePipelineConfig,
) -> dict[str, Any] | None:
    sampling: dict[str, Any] = {}
    if stage_deploy is not None and stage_deploy.default_sampling_params:
        sampling.update(_copy_value(stage_deploy.default_sampling_params))
    sampling.update(_copy_value(topology.sampling_constraints))
    return sampling or None


@config
class OrchestratorConfig:
    """Configuration consumed by the orchestrator process only."""

    stage_init_timeout: int = Field(default=300, ge=1)
    init_timeout: int = Field(default=600, ge=1)
    worker_backend: str = "multi_process"
    ray_address: str | None = None
    deploy_config_path: str | None = None
    omni_master_address: str | None = None
    omni_master_port: int | None = None
    omni_dp_size_local: int = Field(default=1, ge=1)
    omni_lb_policy: str = "random"
    omni_heartbeat_timeout: float = Field(default=30.0, gt=0.0)
    shm_threshold_bytes: int = Field(default=65536, ge=0)
    batch_timeout: int = Field(default=10, ge=0)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniStageConfig:
    """Structured config for one Omni stage."""

    stage_pipeline_config: StagePipelineConfig
    model_config: ModelConfig = field(default_factory=ModelConfig)
    load_config: LoadConfig = field(default_factory=LoadConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    connector_config: ConnectorConfig = field(default_factory=ConnectorConfig)
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    diffusion_config: DiffusionConfig | None = None
    quantization_config: Any = None

    @property
    def stage_id(self) -> int:
        return self.stage_pipeline_config.stage_id

    @property
    def model_stage(self) -> str:
        return self.stage_pipeline_config.model_stage

    @property
    def input_sources(self) -> list[int]:
        return list(self.stage_pipeline_config.input_sources)

    @property
    def final_output(self) -> bool:
        return self.stage_pipeline_config.final_output

    @property
    def final_output_type(self) -> str | None:
        return self.stage_pipeline_config.final_output_type

    @property
    def hf_config_name(self) -> str | None:
        return self.stage_pipeline_config.hf_config_name

    @property
    def stage_type(self) -> StageType:
        stage_type, _ = _resolve_execution_mode(self.stage_pipeline_config.execution_type)
        return stage_type

    @property
    def worker_type(self) -> str | None:
        _, worker_type = _resolve_execution_mode(self.stage_pipeline_config.execution_type)
        return worker_type

    @property
    def scheduler_cls(self) -> str | None:
        return _resolve_scheduler_path(
            self.stage_pipeline_config.execution_type,
            self.scheduler_config.async_scheduling,
        )

    @property
    def custom_process_input_func(self) -> str | None:
        return getattr(
            self,
            "_resolved_custom_process_input_func",
            self.stage_pipeline_config.custom_process_input_func,
        )

    @property
    def custom_process_next_stage_input_func(self) -> str | None:
        return getattr(
            self,
            "_resolved_custom_process_next_stage_input_func",
            self.stage_pipeline_config.custom_process_next_stage_input_func,
        )

    @property
    def is_comprehension(self) -> bool:
        return self.stage_pipeline_config.owns_tokenizer

    @property
    def engine_output_type(self) -> str | None:
        return self.stage_pipeline_config.engine_output_type

    @property
    def requires_multimodal_data(self) -> bool:
        return self.stage_pipeline_config.requires_multimodal_data

    @property
    def prompt_expand_func(self) -> str | None:
        return self.stage_pipeline_config.prompt_expand_func

    @property
    def cfg_kv_collect_func(self) -> str | None:
        return self.stage_pipeline_config.cfg_kv_collect_func


def _build_stage_config(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: Mapping[str, Any],
    *,
    model: str | None,
) -> VllmOmniStageConfig:
    input_proc, next_stage_proc = _select_processor_funcs(topology, bool(deploy.async_chunk))
    quantization_config = _build_quantization_config(deploy, engine)
    parallel_config = _build_parallel_config(deploy, engine)

    stage_config = VllmOmniStageConfig(
        stage_pipeline_config=topology,
        model_config=_build_model_config(topology, stage_deploy, engine),
        load_config=_build_load_config(engine),
        cache_config=_build_cache_config(deploy, engine),
        scheduler_config=_build_scheduler_config(deploy, engine),
        connector_config=_build_connector_config(stage_deploy),
        runtime_config=_build_runtime_config(stage_deploy, engine),
        parallel_config=parallel_config,
        diffusion_config=_build_diffusion_config(
            pipeline,
            deploy,
            topology,
            engine,
            model=model,
            quantization_config=quantization_config,
        ),
        quantization_config=_copy_value(quantization_config),
    )
    stage_config._resolved_custom_process_input_func = input_proc
    stage_config._resolved_custom_process_next_stage_input_func = next_stage_proc
    return stage_config


def _build_quantization_config(deploy: DeployConfig, engine: Mapping[str, Any]) -> Any:
    return _first_defined(
        engine.get("quantization_config"),
        engine.get("quantization"),
        deploy.quantization,
    )


def _build_model_config(
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: Mapping[str, Any],
) -> ModelConfig:
    return ModelConfig(
        enable_sleep_mode=_first_defined(engine.get("enable_sleep_mode"), default=False),
        default_sampling_params=_stage_sampling_params(stage_deploy, topology),
        subtalker_sampling_params=_copy_if_not_none(engine.get("subtalker_sampling_params")),
        has_sampling_extra_args=_first_defined(engine.get("has_sampling_extra_args"), default=False),
        task_type=_copy_if_not_none(engine.get("task_type")),
        codec_frame_rate_hz=_copy_if_not_none(engine.get("codec_frame_rate_hz")),
        enforce_eager=_first_defined(engine.get("enforce_eager"), default=False),
        enable_flashinfer_autotune=_copy_if_not_none(engine.get("enable_flashinfer_autotune")),
        compilation_config=_copy_if_not_none(engine.get("compilation_config")),
        enable_multithread_weight_load=_first_defined(engine.get("enable_multithread_weight_load"), default=True),
        num_weight_load_threads=_first_defined(engine.get("num_weight_load_threads"), default=4),
        disable_autocast=_first_defined(engine.get("disable_autocast"), default=False),
    )


def _build_load_config(engine: Mapping[str, Any]) -> LoadConfig:
    return LoadConfig(
        load_format=_first_defined(engine.get("load_format"), default="auto"),
        tokenizer_mode=_first_defined(engine.get("tokenizer_mode"), default="auto"),
        config_format=_copy_if_not_none(engine.get("config_format")),
        skip_mm_profiling=_copy_if_not_none(engine.get("skip_mm_profiling")),
    )


def _build_cache_config(
    deploy: DeployConfig,
    engine: Mapping[str, Any],
) -> CacheConfig:
    return CacheConfig(
        gpu_memory_utilization=_first_defined(
            engine.get("gpu_memory_utilization"),
            default=0.90,
        ),
        enable_prefix_caching=_first_defined(
            engine.get("enable_prefix_caching"),
            deploy.enable_prefix_caching,
            default=False,
        ),
        disable_hybrid_kv_cache_manager=_first_defined(
            engine.get("disable_hybrid_kv_cache_manager"),
            default=False,
        ),
        mm_processor_cache_gb=_copy_if_not_none(engine.get("mm_processor_cache_gb")),
    )


def _build_scheduler_config(
    deploy: DeployConfig,
    engine: Mapping[str, Any],
) -> SchedulerConfig:
    return SchedulerConfig(
        max_num_seqs=_first_defined(engine.get("max_num_seqs"), default=128),
        max_num_batched_tokens=_copy_if_not_none(engine.get("max_num_batched_tokens")),
        max_model_len=_copy_if_not_none(engine.get("max_model_len")),
        enable_chunked_prefill=_first_defined(
            engine.get("enable_chunked_prefill"),
            deploy.enable_chunked_prefill,
            default=False,
        ),
        async_scheduling=_first_defined(engine.get("async_scheduling"), default=True),
    )


def _build_connector_config(stage_deploy: StageDeployConfig | None) -> ConnectorConfig:
    output_connectors = stage_deploy.output_connectors if stage_deploy is not None else None
    input_connectors = stage_deploy.input_connectors if stage_deploy is not None else None
    return ConnectorConfig(
        output_connectors=_copy_value(output_connectors) if output_connectors else None,
        input_connectors=_copy_value(input_connectors) if input_connectors else None,
    )


def _build_runtime_config(stage_deploy: StageDeployConfig | None, engine: Mapping[str, Any]) -> RuntimeConfig:
    devices = _first_defined(engine.get("devices"), stage_deploy.devices if stage_deploy is not None else None)
    num_replicas = _first_defined(
        engine.get("num_replicas"),
        stage_deploy.num_replicas if stage_deploy is not None else 1,
    )
    env = _first_defined(engine.get("env"), stage_deploy.env if stage_deploy is not None else None)
    return RuntimeConfig(
        devices=_copy_if_not_none(devices),
        num_replicas=int(num_replicas),
        env=_copy_if_not_none(env),
        num_gpus=_first_defined(engine.get("num_gpus"), default=1),
        log_level=_first_defined(engine.get("log_level"), default="info"),
        log_stats=_first_defined(engine.get("log_stats"), default=False),
        profiler_config=_copy_if_not_none(engine.get("profiler_config")),
    )


def _build_parallel_config(
    deploy: DeployConfig,
    engine: Mapping[str, Any],
) -> ParallelConfig:
    parallel_config = _mapping_or_empty(engine.get("parallel_config"))
    ulysses_degree = _first_defined(parallel_config.get("ulysses_degree"), engine.get("ulysses_degree"), default=1)
    ring_degree = _first_defined(parallel_config.get("ring_degree"), engine.get("ring_degree"), default=1)
    sequence_parallel_size = _first_defined(
        parallel_config.get("sequence_parallel_size"),
        engine.get("sequence_parallel_size"),
        default=ulysses_degree * ring_degree,
    )
    return ParallelConfig(
        _pipeline_parallel_size=_first_defined(
            parallel_config.get("pipeline_parallel_size"),
            engine.get("pipeline_parallel_size"),
            deploy.pipeline_parallel_size,
            default=1,
        ),
        _data_parallel_size=_first_defined(
            parallel_config.get("data_parallel_size"),
            engine.get("data_parallel_size"),
            deploy.data_parallel_size,
            default=1,
        ),
        tensor_parallel_size=_first_defined(
            parallel_config.get("tensor_parallel_size"),
            engine.get("tensor_parallel_size"),
            default=1,
        ),
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_mode=_first_defined(
            parallel_config.get("ulysses_mode"),
            engine.get("ulysses_mode"),
            default="strict",
        ),
        cfg_parallel_size=_first_defined(
            parallel_config.get("cfg_parallel_size"),
            engine.get("cfg_parallel_size"),
            default=1,
        ),
        vae_patch_parallel_size=_first_defined(
            parallel_config.get("vae_patch_parallel_size"),
            engine.get("vae_patch_parallel_size"),
            default=1,
        ),
        use_hsdp=_first_defined(
            parallel_config.get("use_hsdp"),
            engine.get("use_hsdp"),
            default=False,
        ),
        hsdp_shard_size=_first_defined(
            parallel_config.get("hsdp_shard_size"),
            engine.get("hsdp_shard_size"),
            default=-1,
        ),
        hsdp_replicate_size=_first_defined(
            parallel_config.get("hsdp_replicate_size"),
            engine.get("hsdp_replicate_size"),
            default=1,
        ),
        enable_expert_parallel=_first_defined(
            parallel_config.get("enable_expert_parallel"),
            engine.get("enable_expert_parallel"),
            default=False,
        ),
    )


def _build_diffusion_config(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    engine: Mapping[str, Any],
    *,
    model: str | None,
    quantization_config: Any,
) -> DiffusionConfig | None:
    if topology.execution_type != StageExecutionType.DIFFUSION:
        return None

    diffusion_kwargs = {
        name: _copy_value(engine[name])
        for name in _DIFFUSION_STAGE_ENGINE_FIELDS
        if name in engine and engine[name] is not None
    }
    diffusion_kwargs["stage_id"] = topology.stage_id
    diffusion_kwargs["model_arch"] = _first_defined(
        diffusion_kwargs.get("model_arch"),
        topology.model_arch,
        pipeline.model_arch,
    )
    if "dtype" not in diffusion_kwargs and deploy.dtype is not None:
        diffusion_kwargs["dtype"] = _copy_value(deploy.dtype)
    if "trust_remote_code" not in diffusion_kwargs and deploy.trust_remote_code is not None:
        diffusion_kwargs["trust_remote_code"] = _copy_value(deploy.trust_remote_code)
    if "distributed_executor_backend" not in diffusion_kwargs and deploy.distributed_executor_backend is not None:
        diffusion_kwargs["distributed_executor_backend"] = _copy_value(deploy.distributed_executor_backend)
    if model is not None:
        diffusion_kwargs["model"] = model
    if quantization_config is not None:
        diffusion_kwargs["quantization_config"] = _copy_value(quantization_config)

    return DiffusionConfig(**{k: v for k, v in diffusion_kwargs.items() if v is not None})


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniConfig:
    """Top-level structured Omni config built once from registry inputs."""

    pipeline_config: PipelineConfig
    stage_configs: tuple[VllmOmniStageConfig, ...]
    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    @property
    def pipeline(self) -> PipelineConfig:
        """Compatibility alias for the RFC sketch."""
        return self.pipeline_config

    def stage_by_id(self, stage_id: int) -> VllmOmniStageConfig:
        for stage in self.stage_configs:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"no stage {stage_id}")

    @classmethod
    def from_registry(
        cls,
        model_type: str,
        deploy_config_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> VllmOmniConfig:
        """Create a structured config from a registered pipeline and deploy YAML."""
        if cli_overrides is None:
            cli_overrides = {}

        deploy_path = _resolve_deploy_path(model_type, deploy_config_path)
        loaded_deploy_config_path = str(deploy_path) if deploy_path.exists() else None
        if loaded_deploy_config_path is not None:
            deploy = load_deploy_config(deploy_path)
        else:
            deploy = DeployConfig()

        pipeline_key = deploy.pipeline or model_type
        if pipeline_key not in _PIPELINE_REGISTRY:
            raise KeyError(
                f"Pipeline {pipeline_key!r} not in registry "
                f"(resolved from {deploy_path.name!r}). Available: "
                f"{sorted(_PIPELINE_REGISTRY.keys())}"
            )
        pipeline = _PIPELINE_REGISTRY[pipeline_key]

        deploy_for_registry = copy.deepcopy(deploy)
        if cli_overrides.get("async_chunk") is not None:
            deploy_for_registry.async_chunk = bool(cli_overrides["async_chunk"])
        for name in _PIPELINE_DEPLOY_CLI_FIELDS:
            if cli_overrides.get(name) is not None:
                setattr(deploy_for_registry, name, _copy_value(cli_overrides[name]))

        deploy_for_registry = _apply_platform_overrides_to_deploy(deploy_for_registry)
        _validate_async_chunk_support(pipeline, deploy_for_registry)
        deploy_by_id = {stage.stage_id: stage for stage in deploy_for_registry.stages}
        model = cli_overrides.get("model")

        stage_configs = tuple(
            _build_stage_config(
                pipeline,
                deploy_for_registry,
                topology,
                deploy_by_id.get(topology.stage_id),
                _stage_engine_values(
                    deploy_by_id.get(topology.stage_id),
                    _stage_cli_overrides(topology.stage_id, cli_overrides),
                ),
                model=model,
            )
            for topology in pipeline.stages
        )

        worker_backend = (
            cli_overrides.get("distributed_executor_backend")
            or deploy_for_registry.distributed_executor_backend
            or "multi_process"
        )
        orchestrator_config = OrchestratorConfig(
            worker_backend=worker_backend,
            deploy_config_path=loaded_deploy_config_path,
        )
        return cls(
            pipeline_config=pipeline,
            stage_configs=stage_configs,
            orchestrator_config=orchestrator_config,
        )


__all__ = [
    "CacheConfig",
    "ConnectorConfig",
    "DiffusionConfig",
    "LoadConfig",
    "ModelConfig",
    "OrchestratorConfig",
    "ParallelConfig",
    "RuntimeConfig",
    "SchedulerConfig",
    "VllmOmniConfig",
    "VllmOmniStageConfig",
]
