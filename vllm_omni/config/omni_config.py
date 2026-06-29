# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured vLLM-Omni configuration classes.

This module is additive for Phase 2 of RFC #4021.  ``VllmOmniConfig.from_registry``
builds the structured view directly from the pipeline registry and deploy config
so parity can be proven before later PRs cut consumers over to these classes.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

from pydantic import ConfigDict, Field
from vllm.config.utils import config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    _STAGE_DEPLOY_FIELDS,
    PIPELINE_WIDE_ENGINE_FIELDS,
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    _apply_platform_overrides,
    _resolve_scheduler,
    _scheduler_path,
    _select_processor_funcs,
    build_stage_runtime_overrides,
    load_deploy_config,
)

_EXECUTION_TYPE_TO_STAGE_WORKER: dict[StageExecutionType, tuple[StageType, str | None]] = {
    StageExecutionType.LLM_AR: (StageType.LLM, "ar"),
    StageExecutionType.LLM_GENERATION: (StageType.LLM, "generation"),
    StageExecutionType.DIFFUSION: (StageType.DIFFUSION, None),
}

_PIPELINE_DEPLOY_CLI_FIELDS = PIPELINE_WIDE_ENGINE_FIELDS

_QuantizationConfigType: TypeAlias = QuantizationConfig | str | Mapping[str, Any] | None


class _QuantizationEngineOverrides(TypedDict, total=False):
    quantization_config: _QuantizationConfigType
    quantization: str


class _ModelEngineOverrides(TypedDict, total=False):
    active_stream_window: int
    enable_sleep_mode: bool
    subtalker_sampling_params: dict[str, Any]
    has_sampling_extra_args: bool
    custom_voice_dir: str
    task_type: str
    codec_frame_rate_hz: float
    enforce_eager: bool
    enable_flashinfer_autotune: bool
    compilation_config: dict[str, Any]
    enable_multithread_weight_load: bool
    num_weight_load_threads: int
    disable_autocast: bool


class _LoadEngineOverrides(TypedDict, total=False):
    load_format: str
    tokenizer_mode: str
    config_format: str
    skip_mm_profiling: bool


class _CacheEngineOverrides(TypedDict, total=False):
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    disable_hybrid_kv_cache_manager: bool
    mm_processor_cache_gb: float


class _SchedulerEngineOverrides(TypedDict, total=False):
    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    enable_chunked_prefill: bool
    async_scheduling: bool


class _RuntimeEngineOverrides(TypedDict, total=False):
    devices: str
    num_replicas: int
    env: dict[str, Any]
    num_gpus: int
    log_level: str
    log_stats: bool
    profiler_config: dict[str, Any]


class _ParallelConfigEngineOverrides(TypedDict, total=False):
    pipeline_parallel_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    sequence_parallel_size: int
    ulysses_degree: int
    ring_degree: int
    ulysses_mode: str
    cfg_parallel_size: int
    vae_patch_parallel_size: int
    use_hsdp: bool
    mask_sp_padding: bool
    hsdp_shard_size: int
    hsdp_replicate_size: int
    enable_expert_parallel: bool


class _ParallelEngineOverrides(_ParallelConfigEngineOverrides, total=False):
    parallel_config: _ParallelConfigEngineOverrides | Mapping[str, Any]


@dataclass(frozen=True)
class _StageEngineValues:
    """Typed projections of legacy flat per-stage ``yaml_engine_args``."""

    quantization: _QuantizationEngineOverrides
    model: _ModelEngineOverrides
    load: _LoadEngineOverrides
    cache: _CacheEngineOverrides
    scheduler: _SchedulerEngineOverrides
    runtime: _RuntimeEngineOverrides
    parallel: _ParallelEngineOverrides
    diffusion: _DiffusionEngineOverrides


@dataclass(frozen=True)
class _DiffusionEngineOverrides:
    """Validated diffusion projection of legacy flat per-stage engine args."""

    _values: dict[str, Any]

    @classmethod
    def from_engine(cls, engine: Mapping[str, Any]) -> _DiffusionEngineOverrides:
        return cls(_select_engine_overrides(engine, _DIFFUSION_STAGE_ENGINE_FIELDS))

    def to_kwargs(self) -> dict[str, Any]:
        return {name: _copy_value(value) for name, value in self._values.items()}


_IMMUTABLE_CONFIG_VALUE_TYPES = (str, int, float, bool, bytes, type(None))


def _can_share_config_value(value: Any) -> bool:
    if isinstance(value, _IMMUTABLE_CONFIG_VALUE_TYPES):
        return True
    if isinstance(value, tuple):
        return all(_can_share_config_value(item) for item in value)
    return False


def _copy_value(value: Any) -> Any:
    """Copy nested config values so the structured view owns its data."""
    if _can_share_config_value(value):
        return value
    return copy.deepcopy(value)


def _config_kwargs(overrides: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _copy_value(value) for name, value in overrides.items() if value is not None}


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return _copy_value(value)
    return None


def _validate_async_chunk_support(pipeline: PipelineConfig, deploy: DeployConfig) -> None:
    has_inter_stage_edges = any(stage.input_sources for stage in pipeline.stages)
    if (
        deploy.async_chunk
        and has_inter_stage_edges
        and not any(stage.async_chunk_process_next_stage_input_func for stage in pipeline.stages)
    ):
        raise ValueError(
            f"Pipeline {pipeline.model_type!r} has async_chunk=True in deploy but no stage "
            "declares a dedicated async-chunk next-stage processor "
            "(``async_chunk_process_next_stage_input_func``). "
            "Either set async_chunk=False or implement an async-chunk producer on the pipeline."
        )


def _resolve_execution_mode(execution_type: StageExecutionType) -> tuple[StageType, str | None]:
    try:
        return _EXECUTION_TYPE_TO_STAGE_WORKER[execution_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported stage execution type: {execution_type!r}") from exc


def _resolve_scheduler_path(execution_type: StageExecutionType, async_scheduling: bool = True) -> str | None:
    return _scheduler_path(_resolve_scheduler(execution_type, async_scheduling))


def _stage_cli_overrides(stage_id: int, cli_overrides: Mapping[str, Any]) -> dict[str, Any]:
    runtime_overrides = build_stage_runtime_overrides(stage_id, dict(cli_overrides))
    global_stage_fields = _global_stage_cli_fields()
    result: dict[str, Any] = {}
    for key, value in runtime_overrides.items():
        if key in global_stage_fields or f"stage_{stage_id}_{key}" in cli_overrides:
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


def _registered_pipeline_keys() -> list[str]:
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    return sorted(OMNI_PIPELINES)


def _resolve_registered_pipeline(model_type: str) -> PipelineConfig | None:
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    registered = OMNI_PIPELINES[model_type]
    return registered(None) if callable(registered) else registered


@config
class OmniStageModelConfig:
    """Per-stage model behavior."""

    active_stream_window: int = Field(default=0, ge=0)
    enable_sleep_mode: bool = False
    default_sampling_params: dict[str, Any] | None = None
    subtalker_sampling_params: dict[str, Any] | None = None
    has_sampling_extra_args: bool = False
    custom_voice_dir: str | None = None
    task_type: str | None = None
    codec_frame_rate_hz: float | None = None
    enforce_eager: bool = False
    enable_flashinfer_autotune: bool | None = None
    compilation_config: dict[str, Any] | None = None
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = Field(default=4, ge=1)
    disable_autocast: bool = False


@config
class OmniStageLoadConfig:
    """Per-stage loading behavior."""

    load_format: str = "auto"
    tokenizer_mode: str = "auto"
    config_format: str | None = None
    skip_mm_profiling: bool | None = None


@config
class OmniStageCacheConfig:
    """Per-stage engine cache and memory behavior.

    This is separate from ``_DiffusionConfigProjection.cache_config``, which configures
    vLLM-Omni diffusion-specific cache backends such as TeaCache and Cache-DiT.
    """

    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    enable_prefix_caching: bool = False
    disable_hybrid_kv_cache_manager: bool = False
    mm_processor_cache_gb: float | None = Field(default=None, ge=0.0)


@config
class OmniStageSchedulerConfig:
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
class OmniStageConnectorConfig:
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
class OmniStageRuntimeConfig:
    """Per-stage process placement and runtime behavior."""

    devices: str | None = None
    num_replicas: int = Field(default=1, ge=1)
    env: dict[str, Any] | None = None
    num_gpus: int = Field(default=1, ge=1)
    log_level: str = "info"
    log_stats: bool = False
    profiler_config: dict[str, Any] | None = None


@config
class OmniStageParallelConfig:
    """Common per-stage distributed parallelism behavior."""

    pipeline_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    enable_expert_parallel: bool = False
    world_size: int = Field(default=1, ge=1, init=False)

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * self.data_parallel_size * self.tensor_parallel_size


@config
class OmniStageDiffusionParallelConfig(OmniStageParallelConfig):
    """Diffusion-stage distributed parallelism behavior."""

    sequence_parallel_size: int = Field(default=1, ge=1, init=False)
    ulysses_degree: int = Field(default=1, ge=1)
    ring_degree: int = Field(default=1, ge=1)
    ulysses_mode: str = "strict"
    cfg_parallel_size: int = Field(default=1, ge=1, le=3)
    vae_patch_parallel_size: int = Field(default=1, ge=1)
    vae_parallel_mode: str = "tile"
    use_hsdp: bool = False
    mask_sp_padding: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = Field(default=1, ge=1)

    def __post_init__(self) -> None:
        self.sequence_parallel_size = self.ulysses_degree * self.ring_degree
        if self.ulysses_mode not in {"strict", "advanced_uaa"}:
            raise ValueError("ulysses_mode must be 'strict' or 'advanced_uaa'")
        if self.vae_parallel_mode not in {"tile", "spatial_shard_height", "spatial_shard_width"}:
            raise ValueError(
                "vae_parallel_mode must be one of {'tile', 'spatial_shard_height', 'spatial_shard_width'}, "
                f"but got {self.vae_parallel_mode!r}."
            )

        other_parallel_world_size = (
            self.pipeline_parallel_size
            * self.data_parallel_size
            * self.tensor_parallel_size
            * self.sequence_parallel_size
            * self.cfg_parallel_size
        )
        if self.use_hsdp:
            if self.tensor_parallel_size > 1 or self.data_parallel_size > 1:
                raise ValueError(
                    "HSDP (use_hsdp=True) cannot be used with TP or DP "
                    f"(tensor_parallel_size={self.tensor_parallel_size}, "
                    f"data_parallel_size={self.data_parallel_size})"
                )
            if self.hsdp_shard_size == -1:
                if other_parallel_world_size == 1:
                    raise ValueError("Cannot auto-calculate hsdp_shard_size when other parallelism is all 1")
                if other_parallel_world_size % self.hsdp_replicate_size != 0:
                    raise ValueError(
                        f"hsdp_replicate_size ({self.hsdp_replicate_size}) must evenly divide "
                        f"world_size ({other_parallel_world_size}) when hsdp_shard_size is -1"
                    )
                self.hsdp_shard_size = other_parallel_world_size // self.hsdp_replicate_size
                self.world_size = other_parallel_world_size
            else:
                if self.hsdp_shard_size <= 0:
                    raise ValueError("hsdp_shard_size must be > 0 when use_hsdp=True")
                hsdp_world_size = self.hsdp_replicate_size * self.hsdp_shard_size
                if other_parallel_world_size == 1:
                    self.world_size = hsdp_world_size
                else:
                    if hsdp_world_size != other_parallel_world_size:
                        raise ValueError(
                            f"HSDP dimensions ({self.hsdp_replicate_size} x {self.hsdp_shard_size} = "
                            f"{hsdp_world_size}) must equal world_size from other parallelism "
                            f"({other_parallel_world_size})"
                        )
                    self.world_size = other_parallel_world_size
        else:
            self.world_size = other_parallel_world_size


@config(config=ConfigDict(arbitrary_types_allowed=True))
class _DiffusionConfigProjection:
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
    quantization_config: _QuantizationConfigType = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> _DiffusionConfigProjection:
        from vllm_omni.diffusion.data import normalize_omni_diffusion_kwargs

        normalized_kwargs = normalize_omni_diffusion_kwargs(kwargs)
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in normalized_kwargs.items() if k in valid_fields})

    def __post_init__(self) -> None:
        # Keep diffusion imports lazy so importing vllm_omni.config does not
        # pull in the full diffusion stack unless a diffusion stage is built.
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
            # Import torch only when string dtype normalization is needed.
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


_DIFFUSION_CONFIG_FIELDS = frozenset(f.name for f in fields(_DiffusionConfigProjection))

# Current OmniDiffusionConfig still contains a flat mix of shared engine,
# runtime, parallel, and diffusion-specific knobs. Keep this classification
# explicit while Phase 2 is additive; later cutover PRs can move or remove
# fields without rediscovering the current boundary.
_DIFFUSION_SHARED_CONFIG_FIELDS = frozenset(
    {
        "stage_id",
        "model",
        "model_arch",
        "dtype",
        "trust_remote_code",
        "revision",
        "distributed_executor_backend",
        "dist_timeout",
        "model_config",
        "quantization_config",
    }
)
_DIFFUSION_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "host",
        "port",
        "nccl_port",
        "master_port",
        "worker_extension_cls",
        "enable_stage_verification",
        "prompt_file_path",
    }
)
_DIFFUSION_ONLY_CONFIG_FIELDS = (
    _DIFFUSION_CONFIG_FIELDS - _DIFFUSION_SHARED_CONFIG_FIELDS - _DIFFUSION_RUNTIME_CONFIG_FIELDS
)
_DIFFUSION_MOVED_SHARED_FIELDS = frozenset(
    {
        "parallel_config",
        "num_gpus",
        "log_level",
        "profiler_config",
        "omni_kv_config",
        "cfg_kv_collect_func",
        "max_num_seqs",
        "enable_sleep_mode",
        "enforce_eager",
        "enable_multithread_weight_load",
        "num_weight_load_threads",
        "disable_autocast",
    }
)


_STAGE_DEPLOY_ENGINE_FIELDS: tuple[str, ...] = tuple(_STAGE_DEPLOY_FIELDS)

_DIFFUSION_BACKCOMPAT_ENGINE_FIELDS = frozenset(
    {
        "diffusion_attention_backend",
        "kv_cache_dtype",
        "kv_cache_skip_layers",
        "kv_cache_skip_steps",
        "static_lora_scale",
    }
)
_DIFFUSION_STAGE_ENGINE_FIELDS = (_DIFFUSION_CONFIG_FIELDS | _DIFFUSION_BACKCOMPAT_ENGINE_FIELDS) - {
    "model",
    "stage_id",
}

_QUANTIZATION_ENGINE_FIELDS = frozenset(_QuantizationEngineOverrides.__annotations__)
_MODEL_ENGINE_FIELDS = frozenset(_ModelEngineOverrides.__annotations__)
_LOAD_ENGINE_FIELDS = frozenset(_LoadEngineOverrides.__annotations__)
_CACHE_ENGINE_FIELDS = frozenset(_CacheEngineOverrides.__annotations__)
_SCHEDULER_ENGINE_FIELDS = frozenset(_SchedulerEngineOverrides.__annotations__)
_RUNTIME_ENGINE_FIELDS = frozenset(_RuntimeEngineOverrides.__annotations__)
_PARALLEL_CONFIG_ENGINE_FIELDS = frozenset(_ParallelConfigEngineOverrides.__annotations__)
_PARALLEL_ENGINE_FIELDS = _PARALLEL_CONFIG_ENGINE_FIELDS | {"parallel_config"}


def _global_stage_cli_fields() -> frozenset[str]:
    # Lazy import avoids vllm_omni.config -> omni_config -> engine.arg_utils ->
    # vllm_omni.config during package-level config imports.
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    return (
        frozenset(f.name for f in fields(OmniEngineArgs))
        | frozenset(_STAGE_DEPLOY_ENGINE_FIELDS)
        | frozenset(_PIPELINE_DEPLOY_CLI_FIELDS)
    ) - {"model", "stage_id", "stage_configs_path", "async_chunk"}


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _select_engine_overrides(engine: Mapping[str, Any], keys: set[str] | frozenset[str]) -> dict[str, Any]:
    return {name: _copy_value(engine[name]) for name in keys if name in engine and engine[name] is not None}


def _stage_engine_overrides(stage_deploy: StageDeployConfig | None) -> dict[str, Any]:
    if stage_deploy is None:
        return {}

    overrides: dict[str, Any] = {}
    for name in _STAGE_DEPLOY_ENGINE_FIELDS:
        value = getattr(stage_deploy, name)
        if value is not None:
            overrides[name] = _copy_value(value)
    overrides.update(_copy_value(stage_deploy.engine_extras))
    return overrides


def _stage_engine_values(
    stage_deploy: StageDeployConfig | None,
    stage_cli_overrides: Mapping[str, Any] | None = None,
) -> _StageEngineValues:
    engine = _stage_engine_overrides(stage_deploy)
    if stage_cli_overrides:
        engine.update(_copy_value(stage_cli_overrides))
    return _StageEngineValues(
        quantization=cast(
            _QuantizationEngineOverrides,
            _select_engine_overrides(engine, _QUANTIZATION_ENGINE_FIELDS),
        ),
        model=cast(_ModelEngineOverrides, _select_engine_overrides(engine, _MODEL_ENGINE_FIELDS)),
        load=cast(_LoadEngineOverrides, _select_engine_overrides(engine, _LOAD_ENGINE_FIELDS)),
        cache=cast(_CacheEngineOverrides, _select_engine_overrides(engine, _CACHE_ENGINE_FIELDS)),
        scheduler=cast(
            _SchedulerEngineOverrides,
            _select_engine_overrides(engine, _SCHEDULER_ENGINE_FIELDS),
        ),
        runtime=cast(_RuntimeEngineOverrides, _select_engine_overrides(engine, _RUNTIME_ENGINE_FIELDS)),
        parallel=cast(_ParallelEngineOverrides, _select_engine_overrides(engine, _PARALLEL_ENGINE_FIELDS)),
        diffusion=_DiffusionEngineOverrides.from_engine(engine),
    )


def _stage_sampling_params(
    stage_deploy: StageDeployConfig | None,
    topology: StagePipelineConfig,
) -> dict[str, Any] | None:
    sampling: dict[str, Any] = {}
    if stage_deploy is not None and stage_deploy.default_sampling_params:
        sampling.update(_copy_value(stage_deploy.default_sampling_params))
    sampling.update(_copy_value(topology.sampling_constraints))
    return sampling or None


def _orchestrator_cli_overrides(cli_overrides: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for config_field in fields(VllmOmniOrchestratorConfig):
        name = config_field.name
        if name == "deploy_config_path":
            continue
        if cli_overrides.get(name) is not None:
            overrides[name] = _copy_value(cli_overrides[name])
    return overrides


@config
class VllmOmniOrchestratorConfig:
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
class BaseVllmOmniStageConfig:
    """Common structured config contract shared by all Omni stage realizations."""

    stage_pipeline_config: StagePipelineConfig
    model_config: OmniStageModelConfig = field(default_factory=OmniStageModelConfig)
    load_config: OmniStageLoadConfig = field(default_factory=OmniStageLoadConfig)
    cache_config: OmniStageCacheConfig = field(default_factory=OmniStageCacheConfig)
    scheduler_config: OmniStageSchedulerConfig = field(default_factory=OmniStageSchedulerConfig)
    connector_config: OmniStageConnectorConfig = field(default_factory=OmniStageConnectorConfig)
    runtime_config: OmniStageRuntimeConfig = field(default_factory=OmniStageRuntimeConfig)
    parallel_config: OmniStageParallelConfig = field(default_factory=OmniStageParallelConfig)
    quantization_config: _QuantizationConfigType = None

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


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniARStageConfig(BaseVllmOmniStageConfig):
    """Structured config for autoregressive LLM stages."""


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniGenerationStageConfig(BaseVllmOmniStageConfig):
    """Structured config for generation LLM stages."""


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniDiffusionStageConfig(BaseVllmOmniStageConfig):
    """Structured config for diffusion stages."""

    parallel_config: OmniStageDiffusionParallelConfig = field(default_factory=OmniStageDiffusionParallelConfig)
    diffusion_config: _DiffusionConfigProjection = field(default_factory=_DiffusionConfigProjection)


StageConfigType: TypeAlias = VllmOmniARStageConfig | VllmOmniGenerationStageConfig | VllmOmniDiffusionStageConfig


def _build_common_stage_config_kwargs(
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: _StageEngineValues,
    parallel_config_cls: type[OmniStageParallelConfig] = OmniStageParallelConfig,
) -> tuple[dict[str, Any], str | None, str | None]:
    input_proc, next_stage_proc = _select_processor_funcs(topology, bool(deploy.async_chunk))
    quantization_config = _build_quantization_config(deploy, engine.quantization)
    parallel_config = _build_parallel_config(deploy, engine.parallel, parallel_config_cls)

    return (
        {
            "stage_pipeline_config": topology,
            "model_config": _build_model_config(topology, stage_deploy, engine.model),
            "load_config": _build_load_config(engine.load),
            "cache_config": _build_cache_config(deploy, engine.cache),
            "scheduler_config": _build_scheduler_config(deploy, engine.scheduler),
            "connector_config": _build_connector_config(stage_deploy),
            "runtime_config": _build_runtime_config(stage_deploy, engine.runtime, parallel_config),
            "parallel_config": parallel_config,
            "quantization_config": _copy_value(quantization_config),
        },
        input_proc,
        next_stage_proc,
    )


def _with_resolved_processors(
    stage_config: StageConfigType,
    input_proc: str | None,
    next_stage_proc: str | None,
) -> StageConfigType:
    stage_config._resolved_custom_process_input_func = input_proc
    stage_config._resolved_custom_process_next_stage_input_func = next_stage_proc
    return stage_config


def _build_ar_stage_config(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: _StageEngineValues,
    *,
    model: str | None,
) -> VllmOmniARStageConfig:
    common_kwargs, input_proc, next_stage_proc = _build_common_stage_config_kwargs(
        deploy,
        topology,
        stage_deploy,
        engine,
    )
    return cast(
        VllmOmniARStageConfig,
        _with_resolved_processors(
            VllmOmniARStageConfig(**common_kwargs),
            input_proc,
            next_stage_proc,
        ),
    )


def _build_generation_stage_config(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: _StageEngineValues,
    *,
    model: str | None,
) -> VllmOmniGenerationStageConfig:
    common_kwargs, input_proc, next_stage_proc = _build_common_stage_config_kwargs(
        deploy,
        topology,
        stage_deploy,
        engine,
    )
    return cast(
        VllmOmniGenerationStageConfig,
        _with_resolved_processors(
            VllmOmniGenerationStageConfig(**common_kwargs),
            input_proc,
            next_stage_proc,
        ),
    )


def _build_diffusion_stage_config(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: _StageEngineValues,
    *,
    model: str | None,
) -> VllmOmniDiffusionStageConfig:
    common_kwargs, input_proc, next_stage_proc = _build_common_stage_config_kwargs(
        deploy,
        topology,
        stage_deploy,
        engine,
        OmniStageDiffusionParallelConfig,
    )
    common_kwargs["diffusion_config"] = _build_diffusion_config_projection(
        pipeline,
        deploy,
        topology,
        engine.diffusion,
        model=model,
        quantization_config=common_kwargs["quantization_config"],
    )
    return cast(
        VllmOmniDiffusionStageConfig,
        _with_resolved_processors(
            VllmOmniDiffusionStageConfig(**common_kwargs),
            input_proc,
            next_stage_proc,
        ),
    )


_STAGE_CONFIG_BUILDERS = {
    StageExecutionType.LLM_AR: _build_ar_stage_config,
    StageExecutionType.LLM_GENERATION: _build_generation_stage_config,
    StageExecutionType.DIFFUSION: _build_diffusion_stage_config,
}


def _build_stage_config(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: _StageEngineValues,
    *,
    model: str | None,
) -> StageConfigType:
    try:
        builder = _STAGE_CONFIG_BUILDERS[topology.execution_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported stage execution type: {topology.execution_type!r}") from exc
    return builder(
        pipeline,
        deploy,
        topology,
        stage_deploy,
        engine,
        model=model,
    )


def _build_quantization_config(
    deploy: DeployConfig,
    engine: _QuantizationEngineOverrides,
) -> _QuantizationConfigType:
    return _first_defined(
        engine.get("quantization_config"),
        engine.get("quantization"),
        deploy.quantization,
    )


def _build_model_config(
    topology: StagePipelineConfig,
    stage_deploy: StageDeployConfig | None,
    engine: _ModelEngineOverrides,
) -> OmniStageModelConfig:
    default_sampling_params = _stage_sampling_params(stage_deploy, topology)
    kwargs = _config_kwargs(engine)
    if "has_sampling_extra_args" not in kwargs:
        kwargs["has_sampling_extra_args"] = bool((default_sampling_params or {}).get("extra_args"))
    return OmniStageModelConfig(
        default_sampling_params=default_sampling_params,
        **kwargs,
    )


def _build_load_config(engine: _LoadEngineOverrides) -> OmniStageLoadConfig:
    return OmniStageLoadConfig(**_config_kwargs(engine))


def _build_cache_config(
    deploy: DeployConfig,
    engine: _CacheEngineOverrides,
) -> OmniStageCacheConfig:
    kwargs = _config_kwargs(engine)
    if "enable_prefix_caching" not in kwargs and deploy.enable_prefix_caching is not None:
        kwargs["enable_prefix_caching"] = _copy_value(deploy.enable_prefix_caching)
    return OmniStageCacheConfig(**kwargs)


def _build_scheduler_config(
    deploy: DeployConfig,
    engine: _SchedulerEngineOverrides,
) -> OmniStageSchedulerConfig:
    kwargs = _config_kwargs(engine)
    if "enable_chunked_prefill" not in kwargs and deploy.enable_chunked_prefill is not None:
        kwargs["enable_chunked_prefill"] = _copy_value(deploy.enable_chunked_prefill)
    return OmniStageSchedulerConfig(**kwargs)


def _build_connector_config(stage_deploy: StageDeployConfig | None) -> OmniStageConnectorConfig:
    output_connectors = stage_deploy.output_connectors if stage_deploy is not None else None
    input_connectors = stage_deploy.input_connectors if stage_deploy is not None else None
    return OmniStageConnectorConfig(
        output_connectors=_copy_value(output_connectors) if output_connectors else None,
        input_connectors=_copy_value(input_connectors) if input_connectors else None,
    )


def _build_runtime_config(
    stage_deploy: StageDeployConfig | None,
    engine: _RuntimeEngineOverrides,
    parallel_config: OmniStageParallelConfig,
) -> OmniStageRuntimeConfig:
    kwargs = _config_kwargs(engine)
    if "devices" not in kwargs and stage_deploy is not None and stage_deploy.devices is not None:
        kwargs["devices"] = _copy_value(stage_deploy.devices)
    if "num_replicas" not in kwargs and stage_deploy is not None:
        kwargs["num_replicas"] = stage_deploy.num_replicas
    if "env" not in kwargs and stage_deploy is not None and stage_deploy.env is not None:
        kwargs["env"] = _copy_value(stage_deploy.env)
    kwargs["num_gpus"] = parallel_config.world_size
    return OmniStageRuntimeConfig(**kwargs)


def _build_parallel_config(
    deploy: DeployConfig,
    engine: _ParallelEngineOverrides,
    config_cls: type[OmniStageParallelConfig] = OmniStageParallelConfig,
) -> OmniStageParallelConfig:
    parallel_config = _mapping_or_empty(engine.get("parallel_config"))
    config_fields = {
        config_field.name
        for config_field in fields(config_cls)
        if getattr(config_field.default, "init", None) is not False
    }
    kwargs = {
        name: _copy_value(value) for name in _PARALLEL_CONFIG_ENGINE_FIELDS if (value := engine.get(name)) is not None
    }
    kwargs = {name: value for name, value in kwargs.items() if name in config_fields}
    kwargs.update({name: value for name, value in _config_kwargs(parallel_config).items() if name in config_fields})
    if "pipeline_parallel_size" not in kwargs and deploy.pipeline_parallel_size is not None:
        kwargs["pipeline_parallel_size"] = _copy_value(deploy.pipeline_parallel_size)
    if "data_parallel_size" not in kwargs and deploy.data_parallel_size is not None:
        kwargs["data_parallel_size"] = _copy_value(deploy.data_parallel_size)
    return config_cls(**kwargs)


def _build_diffusion_config_projection(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    topology: StagePipelineConfig,
    engine: _DiffusionEngineOverrides,
    *,
    model: str | None,
    quantization_config: _QuantizationConfigType,
) -> _DiffusionConfigProjection:
    diffusion_kwargs = engine.to_kwargs()
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

    return _DiffusionConfigProjection.from_kwargs(**{k: v for k, v in diffusion_kwargs.items() if v is not None})


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniConfig:
    """Top-level structured Omni config built once from registry inputs."""

    pipeline_config: PipelineConfig
    stage_configs: tuple[StageConfigType, ...]
    orchestrator_config: VllmOmniOrchestratorConfig = field(default_factory=VllmOmniOrchestratorConfig)

    def stage_by_id(self, stage_id: int) -> StageConfigType:
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
        if pipeline_key not in _registered_pipeline_keys():
            raise KeyError(
                f"Pipeline {pipeline_key!r} not in registry "
                f"(resolved from {deploy_path.name!r}). Available: "
                f"{_registered_pipeline_keys()}"
            )
        pipeline = _resolve_registered_pipeline(pipeline_key)
        if pipeline is None:
            raise ValueError(
                f"Pipeline {pipeline_key!r} did not resolve to a concrete PipelineConfig without an HF config"
            )

        deploy_for_registry = copy.deepcopy(deploy)
        if cli_overrides.get("async_chunk") is not None:
            deploy_for_registry.async_chunk = bool(cli_overrides["async_chunk"])
        for name in _PIPELINE_DEPLOY_CLI_FIELDS:
            if cli_overrides.get(name) is not None:
                setattr(deploy_for_registry, name, _copy_value(cli_overrides[name]))

        deploy_for_registry = _apply_platform_overrides(deploy_for_registry)
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

        orchestrator_config = VllmOmniOrchestratorConfig(
            deploy_config_path=loaded_deploy_config_path,
            **_orchestrator_cli_overrides(cli_overrides),
        )
        return cls(
            pipeline_config=pipeline,
            stage_configs=stage_configs,
            orchestrator_config=orchestrator_config,
        )


__all__ = [
    "OmniStageCacheConfig",
    "OmniStageConnectorConfig",
    "BaseVllmOmniStageConfig",
    "OmniStageLoadConfig",
    "OmniStageModelConfig",
    "VllmOmniOrchestratorConfig",
    "OmniStageDiffusionParallelConfig",
    "OmniStageParallelConfig",
    "OmniStageRuntimeConfig",
    "OmniStageSchedulerConfig",
    "StageConfigType",
    "VllmOmniARStageConfig",
    "VllmOmniConfig",
    "VllmOmniDiffusionStageConfig",
    "VllmOmniGenerationStageConfig",
]
