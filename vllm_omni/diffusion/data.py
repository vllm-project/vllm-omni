# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import diffusers
import torch
from PIL import Image
from pydantic import model_validator
from typing_extensions import Self
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.utils.network_utils import is_port_available
from vllm_omni.quantization import build_quant_config

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

# Import after TYPE_CHECKING to avoid circular imports at runtime
# The actual import is deferred to __post_init__ to avoid import order issues

logger = init_logger(__name__)


@config
@dataclass
class DiffusionParallelConfig:
    """Configuration for diffusion model distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel stages."""

    data_parallel_size: int = 1
    """Number of data parallel groups."""

    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""

    enable_expert_parallel: bool = False
    """Enable expert parallelism for MoE layers (TP is still used for non-MoE layers)."""

    sequence_parallel_size: int | None = None
    """Number of sequence parallel groups. sequence_parallel_size = ring_degree * ulysses_degree"""

    ulysses_degree: int = 1
    """Number of GPUs used for ulysses sequence parallelism."""

    ring_degree: int = 1
    """Number of GPUs used for ring sequence parallelism."""

    ulysses_mode: str = "strict"
    """Ulysses sequence-parallel mode.

    - "strict": Require divisibility constraints (fastest, default).
    - "advanced_uaa": Enable UAA ("Ulysses Anything Attention") to support
      uneven sequence lengths and non-divisible head counts.

    Note:
    - Ring attention does not support `attention_mask`, so models that rely on
      mask-based auto-padding are still incompatible with Ring.
    - When used in hybrid Ulysses+Ring, Ring requires consistent per-rank
      sequence shapes across the ring group.
    """

    cfg_parallel_size: int = 1
    """Number of Classifier Free Guidance (CFG) parallel groups."""

    vae_patch_parallel_size: int = 1
    """Number of ranks used for VAE patch/tile parallelism (decode/encode)."""

    use_hsdp: bool = False
    """Enable Hybrid Sharded Data Parallel (HSDP) for model weight sharding."""

    hsdp_shard_size: int = -1
    """Number of GPUs to shard weights across within each replica group. -1 means auto-calculate."""

    hsdp_replicate_size: int = 1
    """Number of replica groups for HSDP. Each replica holds a full sharded copy."""

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        """Validates the config relationships among the parallel strategies."""
        assert self.pipeline_parallel_size > 0, "Pipeline parallel size must be > 0"
        assert self.data_parallel_size > 0, "Data parallel size must be > 0"
        assert self.tensor_parallel_size > 0, "Tensor parallel size must be > 0"
        assert self.sequence_parallel_size > 0, "Sequence parallel size must be > 0"
        assert self.ulysses_degree > 0, "Ulysses degree must be > 0"
        assert self.ring_degree > 0, "Ring degree must be > 0"
        assert self.cfg_parallel_size > 0, "CFG parallel size must be > 0"
        assert self.cfg_parallel_size in [1, 2, 3], (
            f"CFG parallel size must be 1, 2, or 3, but got {self.cfg_parallel_size}"
        )
        assert self.vae_patch_parallel_size > 0, "VAE patch parallel size must be > 0"
        assert self.sequence_parallel_size == self.ulysses_degree * self.ring_degree, (
            "Sequence parallel size must be equal to the product of ulysses degree and ring degree,"
            f" but got {self.sequence_parallel_size} != {self.ulysses_degree} * {self.ring_degree}"
        )
        assert self.ulysses_mode in {"strict", "advanced_uaa"}, (
            f"ulysses_mode must be one of {{'strict','advanced_uaa'}}, but got {self.ulysses_mode!r}."
        )

        # Validate HSDP configuration
        if self.use_hsdp:
            assert self.hsdp_replicate_size > 0, "HSDP replicate size must be > 0"
            assert self.hsdp_shard_size > 0, "HSDP shard size must be > 0 (should be set in __post_init__)"
        return self

    def __post_init__(self) -> None:
        if self.sequence_parallel_size is None:
            self.sequence_parallel_size = self.ulysses_degree * self.ring_degree

        # Calculate world_size from other parallelism dimensions
        other_parallel_world_size = (
            self.pipeline_parallel_size
            * self.data_parallel_size
            * self.tensor_parallel_size
            * self.ulysses_degree
            * self.ring_degree
            * self.cfg_parallel_size
        )

        # Handle HSDP configuration
        # HSDP can work in two modes:
        # 1. Standalone: when other parallelism is all 1, HSDP determines world_size
        # 2. Combined: HSDP overlays on top of other parallelism
        if self.use_hsdp:
            if self.tensor_parallel_size > 1 or self.data_parallel_size > 1:
                raise ValueError(
                    "HSDP (use_hsdp=True) cannot be used with TP or DP "
                    f"(tensor_parallel_size={self.tensor_parallel_size}, "
                    f"data_parallel_size={self.data_parallel_size}). "
                    "Set tensor_parallel_size=1 and data_parallel_size=1 when using HSDP."
                )
            if self.hsdp_shard_size == -1:
                # Auto-calculate: use other_parallel_world_size as shard_size
                if self.hsdp_replicate_size <= 0:
                    raise ValueError("hsdp_replicate_size must be > 0")
                if other_parallel_world_size == 1:
                    raise ValueError(
                        "Cannot auto-calculate hsdp_shard_size when other parallelism is all 1. "
                        "Please specify hsdp_shard_size explicitly for standalone HSDP."
                    )
                if other_parallel_world_size % self.hsdp_replicate_size != 0:
                    raise ValueError(
                        f"Invalid HSDP configuration: replicate_size ({self.hsdp_replicate_size}) "
                        f"must evenly divide world_size ({other_parallel_world_size}) when shard_size is -1."
                    )
                self.hsdp_shard_size = other_parallel_world_size // self.hsdp_replicate_size
                self.world_size = other_parallel_world_size
            else:
                # Explicit shard_size: HSDP can work standalone or combined
                hsdp_world_size = self.hsdp_replicate_size * self.hsdp_shard_size
                if other_parallel_world_size == 1:
                    # Standalone HSDP: world_size is determined by HSDP
                    self.world_size = hsdp_world_size
                else:
                    # Combined: HSDP must match other parallelism world_size
                    if hsdp_world_size != other_parallel_world_size:
                        raise ValueError(
                            f"HSDP dimensions "
                            f"({self.hsdp_replicate_size} × {self.hsdp_shard_size} = {hsdp_world_size}) "
                            f"must equal world_size from other parallelism ({other_parallel_world_size})"
                        )
                    self.world_size = other_parallel_world_size
        else:
            self.world_size = other_parallel_world_size

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiffusionParallelConfig":
        """
        Create DiffusionParallelConfig from a dictionary.

        Args:
            data: Dictionary containing parallel configuration parameters

        Returns:
            DiffusionParallelConfig instance with parameters set from dict
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected parallel config dict, got {type(data)!r}")
        return cls(**data)


@dataclass
class TransformerConfig:
    """Container for raw transformer configuration dictionaries."""

    params: dict[str, Any] = field(default_factory=dict)
    quant_method: str | None = None
    quant_config: "QuantizationConfig | None" = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerConfig":
        if not isinstance(data, dict):
            raise TypeError(f"Expected transformer config dict, got {type(data)!r}")
        params = dict(data)  # copy to avoid mutating caller's dict

        quant_method: str | None = None
        quant_config: QuantizationConfig | None = None
        disk_qc = params.get("quantization_config")
        if isinstance(disk_qc, dict) and "quant_method" in disk_qc:
            quant_method = disk_qc["quant_method"]
            kwargs = {k: v for k, v in disk_qc.items() if k != "quant_method"}
            quant_config = build_quant_config(quant_method, **kwargs)

        return cls(params=params, quant_method=quant_method, quant_config=quant_config)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.params.get(key, default)

    def __getattr__(self, item: str) -> Any:
        params = object.__getattribute__(self, "params")
        try:
            return params[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


@dataclass
class DiffusionCacheConfig:
    """
    Configuration for cache adapters (TeaCache, cache-dit, etc.).

    This dataclass provides a unified interface for cache configuration parameters.
    It can be initialized from a dictionary and accessed via attributes.

    Common parameters:
        - TeaCache: rel_l1_thresh, coefficients (optional)
        - cache-dit: Fn_compute_blocks, Bn_compute_blocks, max_warmup_steps,
                    residual_diff_threshold, enable_taylorseer, taylorseer_order,
                    scm_steps_mask_policy, scm_steps_policy

    Example:
        >>> # From dict (user-facing API) - partial config uses defaults for missing keys
        >>> config = DiffusionCacheConfig.from_dict({"rel_l1_thresh": 0.3})
        >>> # Access via attribute
        >>> print(config.rel_l1_thresh)  # 0.3 (from dict)
        >>> print(config.Fn_compute_blocks)  # 8 (default)
        >>> # Empty dict uses all defaults
        >>> default_config = DiffusionCacheConfig.from_dict({})
        >>> print(default_config.rel_l1_thresh)  # 0.2 (default)
    """

    # TeaCache parameters [tea_cache only]
    # Default: 0.2 provides ~1.5x speedup with minimal quality loss (optimal balance)
    rel_l1_thresh: float = 0.2
    coefficients: list[float] | None = None  # Uses model-specific defaults if None

    # cache-dit parameters [cache-dit only]
    # Default: 1 forward compute block (optimized for single-transformer models)
    # Use 1 as default instead of cache-dit's 8, optimized for single-transformer models
    # This provides better performance while maintaining quality for most use cases
    Fn_compute_blocks: int = 1
    # Default: 0 backward compute blocks (no fusion by default)
    Bn_compute_blocks: int = 0
    # Default: 4 warmup steps (optimized for few-step distilled models like Z-Image with 8 steps)
    # Use 4 as default warmup steps instead of 8 in cache-dit, making DBCache work
    # for few-step distilled models (e.g., Z-Image with 8 steps)
    max_warmup_steps: int = 4
    # Default: -1 (unlimited cached steps) - DBCache disables caching when previous cached steps exceed this value
    # to prevent precision degradation. Set to -1 for unlimited caching (cache-dit default).
    max_cached_steps: int = -1
    # Default: 0.24 residual difference threshold (higher for more aggressive caching)
    # Use a relatively higher residual diff threshold (0.24) as default to allow more
    # aggressive caching. This is safe because we have max_continuous_cached_steps limit.
    # Without this limit, a lower threshold like 0.12 would be needed.
    residual_diff_threshold: float = 0.24
    # Default: Limit consecutive cached steps to 3 to prevent precision degradation
    # This allows us to use a higher residual_diff_threshold for more aggressive caching
    max_continuous_cached_steps: int = 3
    # Default: Disable TaylorSeer (not suitable for few-step distilled models)
    # TaylorSeer is not suitable for few-step distilled models, so we disable it by default.
    # References:
    # - From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers
    # - Forecast then Calibrate: Feature Caching as ODE for Efficient Diffusion Transformers
    enable_taylorseer: bool = False
    # Default: 1st order TaylorSeer polynomial
    taylorseer_order: int = 1
    # Default: None SCM mask policy (disabled by default)
    scm_steps_mask_policy: str | None = None
    # Default: "dynamic" steps policy for adaptive caching
    scm_steps_policy: str = "dynamic"
    # Used by cache-dit for scm mask generation. If this value changes during inference,
    # we will re-generate the scm mask and refresh the cache context.
    num_inference_steps: int | None = None
    # Force refresh the cache at a specific step index hint, useful for models like
    # GLM-Image (image preprocessing step in editing mode).
    force_refresh_step_hint: int | None = None
    # Policy for force refresh: "once" refreshes only at the hint step,
    # "repeat" refreshes every force_refresh_step_hint steps.
    force_refresh_step_policy: str = "once"

    # Additional parameters that may be passed but not explicitly defined
    _extra_params: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiffusionCacheConfig":
        """
        Create DiffusionCacheConfig from a dictionary.

        Args:
            data: Dictionary containing cache configuration parameters

        Returns:
            DiffusionCacheConfig instance with parameters set from dict
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected cache config dict, got {type(data)!r}")

        # Get all dataclass field names automatically
        field_names = {f.name for f in fields(cls)}

        # Extract parameters that match dataclass fields (excluding private fields)
        known_params = {k: v for k, v in data.items() if k in field_names and not k.startswith("_")}

        # Store extra parameters
        extra_params = {k: v for k, v in data.items() if k not in field_names}

        # Create instance with known params (missing ones will use defaults)
        # Then update _extra_params after creation since it's a private field
        instance = cls(**known_params, _extra_params=extra_params)
        return instance

    def __getattr__(self, item: str) -> Any:
        """
        Allow access to extra parameters via attribute access.

        This enables accessing parameters that weren't explicitly defined
        in the dataclass fields but were passed in the dict.
        """
        if item == "_extra_params" or item.startswith("_"):
            return object.__getattribute__(self, item)

        extra = object.__getattribute__(self, "_extra_params")
        if item in extra:
            return extra[item]

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


@dataclass
class OmniDiffusionConfig:
    # Model and path configuration (for convenience)
    stage_id: int = 0

    model: str | None = None

    model_class_name: str | None = None

    dtype: torch.dtype = torch.bfloat16

    model_config: dict[str, Any] = field(default_factory=dict)
    tf_model_config: TransformerConfig = field(default_factory=TransformerConfig)

    # Attention
    attention_backend: str | None = None

    # Running mode
    # mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    # workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy (legacy)
    cache_strategy: str = "none"
    parallel_config: DiffusionParallelConfig = field(default_factory=DiffusionParallelConfig)

    # Cache backend configuration (NEW)
    cache_backend: str = "none"  # "tea_cache", "deep_cache", etc.
    cache_config: DiffusionCacheConfig | dict[str, Any] = field(default_factory=dict)
    enable_cache_dit_summary: bool = False

    # Distributed executor backend
    distributed_executor_backend: str = "mp"
    nccl_port: int | None = None

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    num_gpus: int | None = None

    dist_timeout: int | None = None  # timeout for torch.distributed

    # pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

    # LoRA parameters
    lora_path: str | None = None
    lora_scale: float = 1.0
    max_cpu_loras: int | None = None

    output_type: str = "pil"

    # CPU offload parameters
    # When enabled, DiT and encoders swap GPU access (mutual exclusion):
    # - Text encoders run on GPU while DiT is on CPU
    # - DiT runs on GPU while encoders are on CPU
    enable_cpu_offload: bool = False
    # Layer-wise offloading (block-level offloading) parameters
    enable_layerwise_offload: bool = False

    pin_cpu_memory: bool = True  # Use pinned memory for faster transfers when offloading

    # VAE memory optimization parameters
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    # STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enforce_eager: bool = False

    # Parallel weight loading (for faster diffusion model startup)
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = 4

    # Enable sleep mode
    enable_sleep_mode: bool = False

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    # moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed inference
    # TODO: do not hard code
    master_port: int | None = None

    # Worker extension class for custom functionality
    worker_extension_cls: str | None = None

    # Custom pipeline arguments for custom pipelines
    custom_pipeline_args: dict[str, Any] | None = None

    # Diffusion model loading format
    # "default", "custom_pipeline", "dummy", "diffusers" (HF diffusers adapter)
    diffusion_load_format: str = "default"

    # Diffusers adapter kwargs
    # kwargs forwarded to DiffusionPipeline.from_pretrained()
    diffusers_load_kwargs: dict[str, Any] = field(default_factory=dict)
    # kwargs forwarded to pipeline.__call__()
    diffusers_call_kwargs: dict[str, Any] = field(default_factory=dict)
    # Actual diffusers pipeline object (to determine inputs of the dummy run)
    diffusers_pipeline_cls: type[diffusers.DiffusionPipeline] | None = None  # pyright: ignore[reportPrivateImportUsage]

    # http server endpoint config, would be ignored in local mode
    host: str | None = None
    port: int | None = None

    scheduler_port: int = 5555

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_file_path: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
        }
    )
    override_transformer_cls_name: str | None = None

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = None
    # Scheduler flow_shift for Wan2.2 (12.0 for 480p, 5.0 for 720p)
    flow_shift: float | None = None

    # Support multi-image inputs and expose any model-specific request limit
    # through a generic config field so serving code stays model-agnostic.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None

    log_level: str = "info"

    # Omni configuration (injected from stage config)
    omni_kv_config: dict[str, Any] = field(default_factory=dict)

    profiler_config: "ProfilerConfig | dict[str, Any] | None" = None

    # Model-specific function for collecting CFG KV caches (set at runtime)
    cfg_kv_collect_func: Any | None = None

    # Quantization: str method name, dict config, QuantizationConfig, or None.
    # str is resolved to {"method": <str>} internally.
    # Per-component: {"transformer": {"method": "fp8"}, "vae": None}
    quantization_config: str | QuantizationConfig | dict[str, Any] | None = None

    # Diffusion pipeline Profiling config
    enable_diffusion_pipeline_profiler: bool = False

    # Step mode settings
    step_execution: bool = False

    # sleep mode
    enable_sleep_mode: bool = False
    # Maximum number of sequences to generate in a batch
    max_num_seqs: int = 1

    @property
    def is_moe(self) -> bool:
        num_experts = self.tf_model_config.get("num_experts", None)
        if not isinstance(num_experts, (list, tuple, int)):
            return False
        if isinstance(num_experts, int):
            return num_experts > 0

        if isinstance(num_experts, (list, tuple)):
            return any(isinstance(n, int) and n > 0 for n in num_experts)

        return False

    def settle_port(self, port: int, port_inc: int = 42, max_attempts: int = 100) -> int:
        """
        Find an available port with retry logic.

        Args:
            port: Initial port to check
            port_inc: Port increment for each attempt
            max_attempts: Maximum number of attempts to find an available port

        Returns:
            An available port number

        Raises:
            RuntimeError: If no available port is found after max_attempts
        """
        attempts = 0
        original_port = port

        while attempts < max_attempts:
            if is_port_available(port):
                if attempts > 0:
                    logger.info(f"Port {original_port} was unavailable, using port {port} instead")
                return port

            attempts += 1
            if port < 60000:
                port += port_inc
            else:
                # Wrap around with randomization to avoid collision
                port = 5000 + random.randint(0, 1000)

        raise RuntimeError(
            f"Failed to find available port after {max_attempts} attempts (started from port {original_port})"
        )

    def __post_init__(self):
        # TODO: remove hard code
        initial_master_port = (self.master_port or 30005) + random.randint(0, 100)
        self.master_port = self.settle_port(initial_master_port, 37)

        if isinstance(self.profiler_config, dict):
            from vllm.config import ProfilerConfig

            self.profiler_config = ProfilerConfig(**self.profiler_config)

        # Convert parallel_config dict/DictConfig to DiffusionParallelConfig
        # Use Mapping to handle both plain dicts and OmegaConf DictConfig
        if isinstance(self.parallel_config, Mapping):
            self.parallel_config = DiffusionParallelConfig.from_dict(dict(self.parallel_config))
        elif not isinstance(self.parallel_config, DiffusionParallelConfig):
            self.parallel_config = DiffusionParallelConfig()

        if self.num_gpus is None:
            if self.parallel_config is not None:
                self.num_gpus = self.parallel_config.world_size
            else:
                self.num_gpus = 1

        if self.num_gpus < self.parallel_config.world_size:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) < parallel_config.world_size ({self.parallel_config.world_size})"
            )

        # Convert string dtype to torch.dtype if needed
        if isinstance(self.dtype, str):
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
            dtype_lower = self.dtype.lower()
            if dtype_lower in dtype_map:
                self.dtype = dtype_map[dtype_lower]
            else:
                logger.warning(f"Unknown dtype string '{self.dtype}', defaulting to bfloat16")
                self.dtype = torch.bfloat16

        # Convert cache_config dict to DiffusionCacheConfig if needed
        if isinstance(self.cache_config, dict):
            self.cache_config = DiffusionCacheConfig.from_dict(self.cache_config)
        elif not isinstance(self.cache_config, DiffusionCacheConfig):
            # If it's neither dict nor DiffusionCacheConfig, convert to empty config
            self.cache_config = DiffusionCacheConfig()

        # Auto-detect quantization from TransformerConfig if not explicitly set.
        # This covers the case where tf_model_config is passed at construction
        # time.  For late (post-construction) assignment, callers should use
        # set_tf_model_config() which propagates quant_config automatically.
        if self.quantization_config is None and self.tf_model_config.quant_config is not None:
            self.quantization_config = self.tf_model_config.quant_config
            logger.info(
                "Auto-detected quantization '%s' from model config",
                self.tf_model_config.quant_method,
            )

        # Resolve quantization_config: str/dict -> QuantizationConfig via build_quant_config.
        if self.quantization_config is not None:
            if isinstance(self.quantization_config, QuantizationConfig):
                pass  # Already built
            elif isinstance(self.quantization_config, str):
                self.quantization_config = build_quant_config(self.quantization_config)
            elif isinstance(self.quantization_config, Mapping):
                self.quantization_config = build_quant_config(dict(self.quantization_config))
            else:
                raise TypeError(
                    f"quantization_config must be str, dict, QuantizationConfig, or None, "
                    f"got {type(self.quantization_config)!r}"
                )

        if self.max_cpu_loras is None:
            self.max_cpu_loras = 1
        elif self.max_cpu_loras < 1:
            raise ValueError("max_cpu_loras must be >= 1 for diffusion LoRA")

        if self.diffusion_load_format != "diffusers" and (self.diffusers_load_kwargs or self.diffusers_call_kwargs):
            raise ValueError(
                "diffusers_load_kwargs and diffusers_call_kwargs are only "
                "valid together with diffusion_load_format=diffusers"
            )

    def set_tf_model_config(self, tf_config: "TransformerConfig") -> None:
        """Assign `tf_model_config` and propagate quantization if detected.

        In the normal startup flow `OmniDiffusionConfig` is created
        *before* the transformer `config.json` is loaded from disk, so
        `__post_init__` sees an empty `TransformerConfig`.  Callers
        that load the config later should use this method instead of bare
        assignment so that an embedded `quant_config` is propagated to
        `self.quantization_config` automatically.

        Args:
            tf_config: Transformer configuration, typically built via
                `TransformerConfig.from_dict`.
        """
        self.tf_model_config = tf_config
        if self.quantization_config is None and tf_config.quant_config is not None:
            self.quantization_config = tf_config.quant_config
            logger.info(
                "Auto-detected quantization '%s' from model config",
                tf_config.quant_method,
            )

    def update_multimodal_support(self) -> None:
        # Resolve serving-visible multimodal behavior from shared metadata
        # instead of importing concrete pipeline modules into the config layer.
        metadata = get_diffusion_model_metadata(self.model_class_name)
        self.supports_multimodal_inputs = metadata.supports_multimodal_inputs
        self.max_multimodal_image_inputs = metadata.max_multimodal_image_inputs

    def enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate config fields.

        Diffusers-style models expose ``model_index.json`` with ``_class_name``.
        Non-diffusers models (e.g. Bagel, NextStep) only have ``config.json``,
        so we fall back to reading that and mapping model_type manually.
        """
        from vllm.transformers_utils.config import get_hf_file_to_dict

        # Default model_class_name for diffusers adapter
        if self.model_class_name is None and self.diffusion_load_format == "diffusers":
            self.model_class_name = "DiffusersAdapterPipeline"

        try:
            config_dict = get_hf_file_to_dict("model_index.json", self.model)
            if config_dict is not None:
                if self.model_class_name is None:
                    self.model_class_name = config_dict.get("_class_name", None)
                self.update_multimodal_support()

                # Skip transformer config loading for diffusers adapter
                # (non-DiT models don't have a separate transformer folder/config)
                if self.diffusion_load_format == "diffusers":
                    self.tf_model_config = TransformerConfig()
                    try:
                        diffusers_pipeline_cls_name = config_dict["_class_name"]
                        self.diffusers_pipeline_cls = getattr(diffusers, diffusers_pipeline_cls_name)
                    except (KeyError, AttributeError) as exc:
                        logger.warning(
                            f"Could not find valid _class_name for diffusers pipeline in model_index.json: {exc}. "
                            "Without knowing the underlying diffusers pipeline class, the dummy run will input only "
                            "text prompt, which may cause errors for pipelines that require additional inputs."
                        )
                else:
                    tf_config_dict = get_hf_file_to_dict("transformer/config.json", self.model)
                    self.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            # Skip transformer config loading for diffusers adapter
            # (non-DiT models don't have a separate transformer folder/config)
            if self.diffusion_load_format == "diffusers":
                self.tf_model_config = TransformerConfig()
                logger.warning(
                    "Could not find valid model_index.json per diffusers format. "
                    "This model is suspectedly unsupported by the diffusers backend. "
                    "Also, without knowing the underlying diffusers pipeline class from model_index.json, "
                    "the dummy run will input only text prompt, which may cause errors for pipelines "
                    "that require additional inputs."
                )
            else:
                cfg = get_hf_file_to_dict("config.json", self.model)
                if cfg is None:
                    raise ValueError(f"Could not find config.json or model_index.json for model {self.model}")

                self.tf_model_config = TransformerConfig.from_dict(cfg)
                model_type = cfg.get("model_type")
                architectures = cfg.get("architectures") or []

                if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                    self.model_class_name = "BagelPipeline"
                    self.tf_model_config = TransformerConfig()
                    self.update_multimodal_support()
                elif model_type == "nextstep":
                    if self.model_class_name is None:
                        self.model_class_name = "NextStep11Pipeline"
                    self.tf_model_config = TransformerConfig()
                    self.update_multimodal_support()
                elif model_type == "s2v":
                    if self.model_class_name is None:
                        self.model_class_name = "WanS2VPipeline"
                    self.tf_model_config = TransformerConfig()
                    self.update_multimodal_support()
                elif architectures and len(architectures) == 1:
                    self.model_class_name = architectures[0]
                else:
                    raise

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OmniDiffusionConfig":
        # Backwards-compatibility: older callers may use a diffusion-specific
        # "static_lora_scale" kwarg. Normalize it to the canonical "lora_scale"
        # before constructing the dataclass to avoid TypeError on unknown fields.
        if "static_lora_scale" in kwargs:
            if "lora_scale" not in kwargs:
                kwargs["lora_scale"] = kwargs["static_lora_scale"]
            kwargs.pop("static_lora_scale", None)

        # Backwards-compatibility: map "quantization" to "quantization_config"
        # so callers using the old field name still work.
        if "quantization" in kwargs and kwargs.get("quantization_config", None) is None:
            kwargs["quantization_config"] = kwargs.pop("quantization")
        else:
            kwargs.pop("quantization", None)

        # Check environment variable as fallback for cache_backend
        # Support both old DIFFUSION_CACHE_ADAPTER and new DIFFUSION_CACHE_BACKEND for backwards compatibility
        if "cache_backend" not in kwargs:
            cache_backend = os.environ.get("DIFFUSION_CACHE_BACKEND") or os.environ.get("DIFFUSION_CACHE_ADAPTER")
            kwargs["cache_backend"] = cache_backend.lower() if cache_backend else "none"

        # Falsy-value check for not-None fields (convert potential None values in YAML config to empty containers)
        if "diffusers_load_kwargs" in kwargs and kwargs["diffusers_load_kwargs"] is None:
            kwargs["diffusers_load_kwargs"] = {}
        if "diffusers_call_kwargs" in kwargs and kwargs["diffusers_call_kwargs"] is None:
            kwargs["diffusers_call_kwargs"] = {}

        # Forward top-level parallel knobs (e.g. --tensor-parallel-size from CLI)
        # into parallel_config so the diffusion engine sees them.
        par = kwargs.get("parallel_config", {})
        if isinstance(par, Mapping):
            par = dict(par)
            if par.get("tensor_parallel_size") is None:
                par.pop("tensor_parallel_size", None)
            tensor_parallel_size = kwargs.get("tensor_parallel_size")
            if tensor_parallel_size is not None and "tensor_parallel_size" not in par:
                par["tensor_parallel_size"] = tensor_parallel_size
            kwargs["parallel_config"] = par

        # Filter kwargs to only include valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return cls(**filtered_kwargs)


@dataclass
class DiffusionOutput:
    """
    Final output (after pipeline completion)
    """

    # Fields may be replaced with SHM handle dicts by ipc.pack_diffusion_output_shm
    output: torch.Tensor | dict | None = None
    trajectory_timesteps: torch.Tensor | dict | None = None
    trajectory_latents: torch.Tensor | dict | None = None
    trajectory_log_probs: torch.Tensor | dict | None = None
    trajectory_decoded: list[Image.Image] | None = None
    error: str | None = None
    aborted: bool = False
    abort_message: str | None = None

    post_process_func: Callable[..., Any] | None = None

    # Extra custom output data (e.g. latent trajectories, prompt embeds)
    # passed through to OmniRequestOutput.custom_output
    custom_output: dict[str, Any] = field(default_factory=dict)

    # logged timings info, directly from Req.timings
    # timings: Optional["RequestTimings"] = None

    # logged duration of stages
    stage_durations: dict[str, float] = field(default_factory=dict)

    # memory usage info
    peak_memory_mb: float = 0.0


class DiffusionRequestAbortedError(RuntimeError):
    """Raised when a diffusion request ends via user-visible abort."""


class AttentionBackendEnum(enum.Enum):
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_THREE = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    VMOBA_ATTN = enum.auto()
    AITER = enum.auto()
    NO_ATTENTION = enum.auto()

    def __str__(self):
        return self.name.lower()


@dataclass
class OmniACK:
    """
    Handshake payload from Workers to Orchestrator.
    """

    task_id: str
    status: str
    stage_id: int | None = None
    rank: int | None = None
    freed_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Additional telemetry such as:
    - max_contiguous_block: for fragmentation analysis.
    - cuda_graph_recalled: boolean if graphs were successfully destroyed/rebuilt.
    - latency_ms: time taken for the D2H/H2D transfer.
    """
    error_msg: str | None = None


@dataclass
class OmniSleepTask:
    """Structured sleep instruction."""

    task_id: str
    level: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniWakeTask:
    """Structured wake-up instruction."""

    task_id: str
    tags: list[str] | None = None


# Special message broadcast via scheduler queues to signal worker shutdown.
SHUTDOWN_MESSAGE = {"type": "shutdown"}
