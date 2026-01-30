# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from pydantic import model_validator
from typing_extensions import Self
from vllm.config.utils import config
from vllm.logger import init_logger

from vllm_omni.diffusion.utils.network_utils import is_port_available

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

    sequence_parallel_size: int | None = None
    """Number of sequence parallel groups. sequence_parallel_size = ring_degree * ulysses_degree"""

    ulysses_degree: int = 1
    """Number of GPUs used for ulysses sequence parallelism."""

    ring_degree: int = 1
    """Number of GPUs used for ring sequence parallelism."""

    cfg_parallel_size: int = 1
    """Number of Classifier Free Guidance (CFG) parallel groups."""

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
        assert self.sequence_parallel_size == self.ulysses_degree * self.ring_degree, (
            "Sequence parallel size must be equal to the product of ulysses degree and ring degree,"
            f" but got {self.sequence_parallel_size} != {self.ulysses_degree} * {self.ring_degree}"
        )
        return self

    def __post_init__(self) -> None:
        if self.sequence_parallel_size is None:
            self.sequence_parallel_size = self.ulysses_degree * self.ring_degree
        self.world_size = (
            self.pipeline_parallel_size
            * self.data_parallel_size
            * self.tensor_parallel_size
            * self.ulysses_degree
            * self.ring_degree
            * self.cfg_parallel_size
        )

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerConfig":
        if not isinstance(data, dict):
            raise TypeError(f"Expected transformer config dict, got {type(data)!r}")
        return cls(params=dict(data))

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
    model: str | None = None

    model_class_name: str | None = None

    dtype: torch.dtype = torch.bfloat16

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

    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
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
    # Number of transformer blocks ready for computation to keep on GPU
    layerwise_num_gpu_layers: int = 1

    use_fsdp_inference: bool = False
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

    # support multi images input
    supports_multimodal_inputs: bool = False

    # Logging
    log_level: str = "info"

    # Omni configuration (injected from stage config)
    omni_kv_config: dict[str, Any] = field(default_factory=dict)

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

        # Convert parallel_config dict to DiffusionParallelConfig if needed
        # This must be done before accessing parallel_config.world_size
        if isinstance(self.parallel_config, dict):
            self.parallel_config = DiffusionParallelConfig.from_dict(self.parallel_config)
        elif not isinstance(self.parallel_config, DiffusionParallelConfig):
            # If it's neither dict nor DiffusionParallelConfig, use default config
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

        if self.max_cpu_loras is None:
            self.max_cpu_loras = 1
        elif self.max_cpu_loras < 1:
            raise ValueError("max_cpu_loras must be >= 1 for diffusion LoRA")

    def update_multimodal_support(self) -> None:
        self.supports_multimodal_inputs = self.model_class_name in {"QwenImageEditPlusPipeline"}

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OmniDiffusionConfig":
        # Backwards-compatibility: older callers may use a diffusion-specific
        # "static_lora_scale" kwarg. Normalize it to the canonical "lora_scale"
        # before constructing the dataclass to avoid TypeError on unknown fields.
        if "static_lora_scale" in kwargs:
            if "lora_scale" not in kwargs:
                kwargs["lora_scale"] = kwargs["static_lora_scale"]
            kwargs.pop("static_lora_scale", None)

        # Check environment variable as fallback for cache_backend
        # Support both old DIFFUSION_CACHE_ADAPTER and new DIFFUSION_CACHE_BACKEND for backwards compatibility
        if "cache_backend" not in kwargs:
            cache_backend = os.environ.get("DIFFUSION_CACHE_BACKEND") or os.environ.get("DIFFUSION_CACHE_ADAPTER")
            kwargs["cache_backend"] = cache_backend.lower() if cache_backend else "none"

        # Filter kwargs to only include valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return cls(**filtered_kwargs)


@dataclass
class DiffusionOutput:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | None = None
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_decoded: list[torch.Tensor] | None = None
    error: str | None = None

    post_process_func: Callable[..., Any] | None = None

    # logged timings info, directly from Req.timings
    # timings: Optional["RequestTimings"] = None


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


# Special message broadcast via scheduler queues to signal worker shutdown.
SHUTDOWN_MESSAGE = {"type": "shutdown"}
