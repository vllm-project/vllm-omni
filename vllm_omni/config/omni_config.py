# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured vLLM-Omni configuration classes.

This module is additive: current consumers still use ``StageConfig`` and
``merge_pipeline_deploy``.  ``VllmOmniConfig.from_registry`` builds the new
structured view from the existing merge path so Phase 2 can prove field parity
before later PRs cut consumers over to these classes.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import field, fields
from typing import Any

from pydantic import ConfigDict, Field
from vllm.config.utils import config

from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    merge_pipeline_deploy,
)


def _copy_value(value: Any) -> Any:
    """Copy nested config values so the structured view owns its data."""
    return copy.deepcopy(value)


def _get_with_default(values: dict[str, Any], name: str, default: Any) -> Any:
    value = values.get(name, default)
    return default if value is None else value


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
    profiler_config: dict[str, Any] | Any | None = None


@config
class ParallelConfig:
    """Per-stage distributed parallelism behavior."""

    pipeline_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    sequence_parallel_size: int | None = Field(default=None, ge=1)
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

    def __post_init__(self) -> None:
        if self.sequence_parallel_size is None:
            self.sequence_parallel_size = self.ulysses_degree * self.ring_degree
        if self.sequence_parallel_size != self.ulysses_degree * self.ring_degree:
            raise ValueError(
                f"sequence_parallel_size ({self.sequence_parallel_size}) must equal "
                f"ulysses_degree * ring_degree ({self.ulysses_degree * self.ring_degree})"
            )
        if self.ulysses_mode not in {"strict", "advanced_uaa"}:
            raise ValueError("ulysses_mode must be 'strict' or 'advanced_uaa'")

        base_world_size = (
            self.pipeline_parallel_size
            * self.data_parallel_size
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
        legacy = getattr(self, "_legacy_stage_config", None)
        if legacy is not None:
            return legacy.stage_type
        return (
            StageType.DIFFUSION
            if self.stage_pipeline_config.execution_type == StageExecutionType.DIFFUSION
            else StageType.LLM
        )

    @property
    def worker_type(self) -> str | None:
        legacy = getattr(self, "_legacy_stage_config", None)
        return legacy.worker_type if legacy is not None else None

    @property
    def scheduler_cls(self) -> str | None:
        legacy = getattr(self, "_legacy_stage_config", None)
        return legacy.scheduler_cls if legacy is not None else None

    @property
    def custom_process_input_func(self) -> str | None:
        legacy = getattr(self, "_legacy_stage_config", None)
        return (
            legacy.custom_process_input_func
            if legacy is not None
            else self.stage_pipeline_config.custom_process_input_func
        )

    @property
    def is_comprehension(self) -> bool:
        legacy = getattr(self, "_legacy_stage_config", None)
        return legacy.is_comprehension if legacy is not None else self.stage_pipeline_config.owns_tokenizer

    @property
    def engine_args(self) -> dict[str, Any]:
        legacy = getattr(self, "_legacy_stage_config", None)
        return _copy_value(legacy.yaml_engine_args) if legacy is not None else {}

    @property
    def runtime(self) -> dict[str, Any]:
        legacy = getattr(self, "_legacy_stage_config", None)
        return _copy_value(legacy.yaml_runtime) if legacy is not None else {}

    @property
    def extras(self) -> dict[str, Any]:
        legacy = getattr(self, "_legacy_stage_config", None)
        return _copy_value(legacy.yaml_extras) if legacy is not None else {}

    @property
    def runtime_overrides(self) -> dict[str, Any]:
        legacy = getattr(self, "_legacy_stage_config", None)
        return _copy_value(legacy.runtime_overrides) if legacy is not None else {}

    @classmethod
    def from_stage_config(
        cls,
        legacy: StageConfig,
        topology: StagePipelineConfig,
        *,
        model: str | None = None,
    ) -> VllmOmniStageConfig:
        """Project one legacy ``StageConfig`` into the structured shape."""
        engine_args = _copy_value(legacy.yaml_engine_args)
        runtime = _copy_value(legacy.yaml_runtime)
        extras = _copy_value(legacy.yaml_extras)

        quantization_config = engine_args.get("quantization_config", engine_args.get("quantization"))

        model_config = ModelConfig(
            enable_sleep_mode=_get_with_default(engine_args, "enable_sleep_mode", False),
            default_sampling_params=_copy_value(extras.get("default_sampling_params")),
            subtalker_sampling_params=_copy_value(engine_args.get("subtalker_sampling_params")),
            has_sampling_extra_args=_get_with_default(engine_args, "has_sampling_extra_args", False),
            task_type=engine_args.get("task_type"),
            codec_frame_rate_hz=engine_args.get("codec_frame_rate_hz"),
            enforce_eager=_get_with_default(engine_args, "enforce_eager", False),
            enable_flashinfer_autotune=engine_args.get("enable_flashinfer_autotune"),
            compilation_config=_copy_value(engine_args.get("compilation_config")),
            enable_multithread_weight_load=_get_with_default(
                engine_args,
                "enable_multithread_weight_load",
                True,
            ),
            num_weight_load_threads=_get_with_default(engine_args, "num_weight_load_threads", 4),
            disable_autocast=_get_with_default(engine_args, "disable_autocast", False),
        )
        load_config = LoadConfig(
            load_format=_get_with_default(engine_args, "load_format", "auto"),
            tokenizer_mode=_get_with_default(engine_args, "tokenizer_mode", "auto"),
            config_format=engine_args.get("config_format"),
            skip_mm_profiling=engine_args.get("skip_mm_profiling"),
        )
        cache_config = CacheConfig(
            gpu_memory_utilization=_get_with_default(engine_args, "gpu_memory_utilization", 0.90),
            enable_prefix_caching=_get_with_default(engine_args, "enable_prefix_caching", False),
            disable_hybrid_kv_cache_manager=_get_with_default(
                engine_args,
                "disable_hybrid_kv_cache_manager",
                False,
            ),
            mm_processor_cache_gb=engine_args.get("mm_processor_cache_gb"),
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=_get_with_default(engine_args, "max_num_seqs", 128),
            max_num_batched_tokens=engine_args.get("max_num_batched_tokens"),
            max_model_len=engine_args.get("max_model_len"),
            enable_chunked_prefill=_get_with_default(engine_args, "enable_chunked_prefill", False),
            async_scheduling=_get_with_default(engine_args, "async_scheduling", True),
        )
        connector_config = ConnectorConfig(
            output_connectors=_copy_value(extras.get("output_connectors")),
            input_connectors=_copy_value(extras.get("input_connectors")),
        )
        runtime_config = RuntimeConfig(
            devices=runtime.get("devices"),
            num_replicas=_get_with_default(runtime, "num_replicas", 1),
            env=_copy_value(runtime.get("env")),
            num_gpus=_get_with_default(engine_args, "num_gpus", 1),
            log_level=_get_with_default(engine_args, "log_level", "info"),
            log_stats=_get_with_default(engine_args, "log_stats", False),
            profiler_config=_copy_value(engine_args.get("profiler_config")),
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=_get_with_default(engine_args, "pipeline_parallel_size", 1),
            data_parallel_size=_get_with_default(engine_args, "data_parallel_size", 1),
            tensor_parallel_size=_get_with_default(engine_args, "tensor_parallel_size", 1),
            sequence_parallel_size=engine_args.get("sequence_parallel_size"),
            ulysses_degree=_get_with_default(engine_args, "ulysses_degree", 1),
            ring_degree=_get_with_default(engine_args, "ring_degree", 1),
            ulysses_mode=_get_with_default(engine_args, "ulysses_mode", "strict"),
            cfg_parallel_size=_get_with_default(engine_args, "cfg_parallel_size", 1),
            vae_patch_parallel_size=_get_with_default(engine_args, "vae_patch_parallel_size", 1),
            use_hsdp=_get_with_default(engine_args, "use_hsdp", False),
            hsdp_shard_size=_get_with_default(engine_args, "hsdp_shard_size", -1),
            hsdp_replicate_size=_get_with_default(engine_args, "hsdp_replicate_size", 1),
            enable_expert_parallel=_get_with_default(engine_args, "enable_expert_parallel", False),
        )

        diffusion_config = None
        if topology.execution_type == StageExecutionType.DIFFUSION:
            diffusion_kwargs = {
                name: _copy_value(engine_args[name])
                for name in _DIFFUSION_CONFIG_FIELDS
                if name in engine_args and engine_args[name] is not None
            }
            diffusion_kwargs["stage_id"] = legacy.stage_id
            if model is not None:
                diffusion_kwargs["model"] = model
            diffusion_kwargs.setdefault("model_arch", engine_args.get("model_arch"))
            diffusion_kwargs.setdefault("quantization_config", quantization_config)
            diffusion_config = DiffusionConfig(**diffusion_kwargs)

        stage = cls(
            stage_pipeline_config=topology,
            model_config=model_config,
            load_config=load_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            connector_config=connector_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
            diffusion_config=diffusion_config,
            quantization_config=_copy_value(quantization_config),
        )
        stage._legacy_stage_config = copy.deepcopy(legacy)
        return stage

    def to_legacy_stage_config(self) -> StageConfig:
        """Return a legacy ``StageConfig`` view with parity to the old merge path."""
        legacy = getattr(self, "_legacy_stage_config", None)
        if legacy is not None:
            return copy.deepcopy(legacy)
        return StageConfig(
            stage_id=self.stage_id,
            model_stage=self.model_stage,
            stage_type=self.stage_type,
            input_sources=self.input_sources,
            custom_process_input_func=self.custom_process_input_func,
            final_output=self.final_output,
            final_output_type=self.final_output_type,
            worker_type=self.worker_type,
            scheduler_cls=self.scheduler_cls,
            hf_config_name=self.hf_config_name,
            is_comprehension=self.is_comprehension,
            yaml_engine_args=_copy_value(self.engine_args),
            yaml_runtime=_copy_value(self.runtime),
            yaml_extras=_copy_value(self.extras),
            runtime_overrides=_copy_value(self.runtime_overrides),
        )


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
        pipeline: PipelineConfig,
        deploy: DeployConfig,
        cli_overrides: dict[str, Any] | None = None,
    ) -> VllmOmniConfig:
        """Create a structured config from pipeline topology and deploy config."""
        if cli_overrides is None:
            cli_overrides = {}

        deploy_for_merge = copy.deepcopy(deploy)
        if cli_overrides.get("async_chunk") is not None:
            deploy_for_merge.async_chunk = bool(cli_overrides["async_chunk"])

        legacy_stages = merge_pipeline_deploy(pipeline, deploy_for_merge, cli_overrides)
        topology_by_id = {stage.stage_id: stage for stage in pipeline.stages}
        model = cli_overrides.get("model")

        stage_configs = tuple(
            VllmOmniStageConfig.from_stage_config(
                legacy_stage,
                topology_by_id[legacy_stage.stage_id],
                model=model,
            )
            for legacy_stage in legacy_stages
        )

        worker_backend = (
            cli_overrides.get("distributed_executor_backend")
            or deploy_for_merge.distributed_executor_backend
            or "multi_process"
        )
        orchestrator_config = OrchestratorConfig(worker_backend=worker_backend)
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
