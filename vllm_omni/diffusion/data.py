# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, Any

import diffusers
import torch
from PIL import Image
from pydantic import Field, model_validator
from typing_extensions import Self
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.utils.network_utils import is_port_available
from vllm_omni.errors import client_error_metadata
from vllm_omni.quantization import build_quant_config

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

# Import after TYPE_CHECKING to avoid circular imports at runtime
# The actual import is deferred to __post_init__ to avoid import order issues

logger = init_logger(__name__)


def normalize_omni_diffusion_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy diffusion kwargs before config construction."""
    normalized = dict(kwargs)

    # Backwards-compatibility: older callers may use a diffusion-specific
    # "static_lora_scale" kwarg. Normalize it to the canonical "lora_scale".
    if "static_lora_scale" in normalized:
        if "lora_scale" not in normalized:
            normalized["lora_scale"] = normalized["static_lora_scale"]
        normalized.pop("static_lora_scale", None)

    # Backwards-compatibility: map "quantization" to "quantization_config"
    # so callers using the old field name still work.
    if "quantization" in normalized and normalized.get("quantization_config", None) is None:
        normalized["quantization_config"] = normalized.pop("quantization")
    else:
        normalized.pop("quantization", None)

    # Renamed from kv_cache_* to avoid clashing with vLLM's --kv-cache-dtype.
    if normalized.get("diffusion_kv_cache_dtype") is None and "kv_cache_dtype" in normalized:
        normalized["diffusion_kv_cache_dtype"] = normalized.pop("kv_cache_dtype")
    else:
        normalized.pop("kv_cache_dtype", None)
    if normalized.get("diffusion_kv_cache_skip_steps") is None and "kv_cache_skip_steps" in normalized:
        normalized["diffusion_kv_cache_skip_steps"] = normalized.pop("kv_cache_skip_steps")
    else:
        normalized.pop("kv_cache_skip_steps", None)
    if normalized.get("diffusion_kv_cache_skip_layers") is None and "kv_cache_skip_layers" in normalized:
        normalized["diffusion_kv_cache_skip_layers"] = normalized.pop("kv_cache_skip_layers")
    else:
        normalized.pop("kv_cache_skip_layers", None)

    # Handle "diffusion_attention_backend" shorthand: merge into
    # diffusion_attention_config before field filtering.
    diffusion_attn_backend = normalized.pop("diffusion_attention_backend", None)
    if diffusion_attn_backend is not None:
        existing = normalized.get("diffusion_attention_config")
        normalized["diffusion_attention_config"] = parse_attention_config(
            existing,
            attention_backend=diffusion_attn_backend,
        )

    # Check environment variable as fallback for cache_backend.
    # Support both old DIFFUSION_CACHE_ADAPTER and new DIFFUSION_CACHE_BACKEND.
    if "cache_backend" not in normalized:
        cache_backend = os.environ.get("DIFFUSION_CACHE_BACKEND") or os.environ.get("DIFFUSION_CACHE_ADAPTER")
        normalized["cache_backend"] = cache_backend.lower() if cache_backend else "none"

    # Convert optional YAML null values to empty containers.
    for key in ("diffusers_load_kwargs", "diffusers_call_kwargs"):
        if key in normalized and normalized[key] is None:
            normalized[key] = {}

    return normalized


def parse_kv_cache_skip_selector(
    selector: str | list[int] | tuple[int, ...] | set[int] | None,
) -> set[int] | None:
    """Parse a non-negative index selector such as "0-9,20,25-30"."""
    if selector is None:
        return None
    if isinstance(selector, set):
        values = selector
    elif isinstance(selector, (list, tuple)):
        values = set(selector)
    elif isinstance(selector, str):
        text = selector.strip()
        if not text:
            return None
        values: set[int] = set()
        for chunk in text.split(","):
            token = chunk.strip()
            if not token:
                continue
            if "-" in token:
                start_str, end_str = token.split("-", 1)
                try:
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                except ValueError as exc:
                    raise ValueError(f"Invalid range token '{token}' in selector '{selector}'.") from exc
                if start < 0 or end < 0 or start > end:
                    raise ValueError(f"Invalid range token '{token}' in selector '{selector}'.")
                values.update(range(start, end + 1))
            else:
                try:
                    index = int(token)
                except ValueError as exc:
                    raise ValueError(f"Invalid index token '{token}' in selector '{selector}'.") from exc
                if index < 0:
                    raise ValueError(f"Negative index '{index}' is not allowed in selector '{selector}'.")
                values.add(index)
    else:
        raise TypeError(f"Unsupported selector type: {type(selector)!r}")

    for idx in values:
        if not isinstance(idx, int):
            raise TypeError(f"Selector index must be int, got {type(idx)!r}")
        if idx < 0:
            raise ValueError("Selector indices must be non-negative.")
    return values


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

    vae_parallel_mode: str = "tile"
    """VAE parallel decode strategy.

    - "tile": Patch/tile parallel decode (default). Each rank decodes a subset
      of spatial tiles and the results are stitched on rank 0.
    - "spatial_shard_height": Spatially-sharded decode that splits decoder
      feature maps along height and exchanges halo rows around spatial
      convolutions.
    - "spatial_shard_width": Same as "spatial_shard_height" but sharded along width.

    The "spatial_shard_*" modes are decode-only and currently require
    ``vae_patch_parallel_size`` to match the DiT group size; otherwise the VAE
    falls back to tile-parallel decode at runtime.
    """

    use_hsdp: bool = False
    """Enable Hybrid Sharded Data Parallel (HSDP) for model weight sharding."""

    mask_sp_padding: bool = False
    """If True, generate a boolean attention mask for zero-padded SP tokens
    when sequence length is not divisible by the SP world size. The mask
    routes attention through the varlen path (unpad→kernel→repad), which is
    correct but carries additional overhead. When False (default), padding
    tokens are left unmasked; since _shard_with_auto_pad always pads with
    zeros, their contribution to attention output is negligible."""

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
        assert self.vae_parallel_mode in {"tile", "spatial_shard_height", "spatial_shard_width"}, (
            "vae_parallel_mode must be one of {'tile', 'spatial_shard_height', 'spatial_shard_width'}, "
            f"but got {self.vae_parallel_mode!r}."
        )
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
        if isinstance(disk_qc, dict):
            raw_quant_method = disk_qc.get("quant_method", disk_qc.get("method"))
            quant_config = build_quant_config(disk_qc)
            if quant_config is not None:
                quant_method = raw_quant_method if raw_quant_method is not None else quant_config.get_name()

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
    Configuration for cache adapters (TeaCache, cache-dit, MagCache, etc.).

    This dataclass provides a unified interface for cache configuration parameters.
    It can be initialized from a dictionary and accessed via attributes.

    Common parameters:
        - TeaCache: rel_l1_thresh, coefficients (optional)
        - cache-dit: Fn_compute_blocks, Bn_compute_blocks, max_warmup_steps,
                    residual_diff_threshold, enable_taylorseer, taylorseer_order,
                    scm_steps_mask_policy, scm_steps_policy
        - MagCache: mag_threshold, mag_max_skip_steps, mag_retention_ratio,
                    mag_ratios, mag_calibrate
        - step_cache: step_cache_dit_enabled, velocity_sim_thresholds,
                          velocity_skip_countdowns, step_cache_dit_min_history

    Example:
        >>> # From dict (user-facing API) - partial config uses defaults for missing keys
        >>> config = DiffusionCacheConfig.from_dict({"rel_l1_thresh": 0.3})
        >>> # Access via attribute
        >>> print(config.rel_l1_thresh)  # 0.3 (from dict)
        >>> print(config.Fn_compute_blocks)  # 8 (default)
        >>> # Empty dict uses all defaults
        >>> default_config = DiffusionCacheConfig.from_dict({})
        >>> print(config.rel_l1_thresh)  # 0.2 (default)
    """

    # TeaCache parameters [tea_cache only]
    # Default: 0.2 provides ~1.5x speedup with minimal quality loss (optimal balance)
    rel_l1_thresh: float = 0.2
    coefficients: list[float] | None = None  # Uses model-specific defaults if None

    # MagCache parameters [mag_cache only]
    # Default: 0.24 threshold for accumulated magnitude error
    mag_threshold: float = 0.24
    # Default: 5 maximum consecutive skip steps (K)
    mag_max_skip_steps: int = 5
    # Default: 0.1 fraction of initial steps where skipping is disabled (stability)
    mag_retention_ratio: float = 0.1
    # Default: None magnitude ratios (model-specific, required for inference)
    mag_ratios: list[float] | None = None
    # Default: False calibration mode (computes mag_ratios on first run)
    mag_calibrate: bool = False

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

    # step_cache parameters [step_cache only] — DreamZero velocity schedule
    step_cache_dit_enabled: bool = True
    velocity_sim_thresholds: list[float] = field(default_factory=lambda: [0.95, 0.93])
    velocity_skip_countdowns: list[int] = field(default_factory=lambda: [4, 2])
    step_cache_dit_min_history: int = 2
    step_cache_dit_max_history: int = 2

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


def resolve_model_class_name(model: str | None, diffusion_load_format: str = "default") -> str | None:
    """Resolve the diffusion pipeline class name from the model config.

    Read-only counterpart of ``OmniDiffusionConfig.enrich_config``, safe to call
    client-side. Returns ``None`` if the pipeline can't be determined.
    """
    from vllm.transformers_utils.config import get_hf_file_to_dict

    if not model:
        return None

    is_lance_subfolder = os.path.basename(str(model).rstrip("/")) in {"Lance_3B", "Lance_3B_Video"}

    # Diffusers models: read _class_name from model_index.json.
    try:
        model_index = get_hf_file_to_dict("model_index.json", model)
    except Exception:
        model_index = None
    if model_index is not None:
        return model_index.get("_class_name")
    if diffusion_load_format == "diffusers":
        return "DiffusersAdapterPipeline"

    # Other models: map model_type / architecture from config.json.
    try:
        cfg = get_hf_file_to_dict("config.json", model) or {}
    except Exception:
        cfg = {}
    model_type = cfg.get("model_type")
    architectures = cfg.get("architectures") or []

    if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
        return "BagelPipeline"
    if (
        model_type == "lance"
        or "LancePipeline" in architectures
        or cfg.get("model_name") == "Lance"
        or is_lance_subfolder
    ):
        return "LancePipeline"
    if model_type == "neo_chat":
        return "SenseNovaU1Pipeline"
    if "BailingMM2NativeForConditionalGeneration" in architectures or model_type in (
        "bailingmm_moe_v2_lite",
        "ming_flash_omni",
        "ming_flash_omni_thinker",
    ):
        return "MingImagePipeline"
    if model_type == "nextstep":
        return "NextStep11Pipeline"
    if model_type == "s2v":
        return "WanS2VPipeline"
    if model_type == "vla":
        from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero

        return "DreamZeroPipeline" if _looks_like_dreamzero(model) else None
    if len(architectures) == 1:
        return architectures[0]
    return None


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
    diffusion_attention_config: "AttentionConfig" = field(default_factory=lambda: AttentionConfig())

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

    # Prompt-embedding cache. When enabled, ``DiffusionModelRunner`` wraps the
    # pipeline's ``encode_prompt`` so repeated calls with identical prompt
    # arguments (e.g. GRPO rollouts that sample the same prompt many times
    # with different seeds) reuse the text-encoder output instead of re-running
    # it. Safe against inputs that cannot be hashed (tensors, PIL images):
    # those calls transparently bypass the cache.
    enable_prompt_embed_cache: bool = False
    prompt_embed_cache_size: int = 32

    # Distributed executor backend
    distributed_executor_backend: str = "mp"
    nccl_port: int | None = None

    # Engine backend selection, resolved by ``DiffusionEngine.resolve_engine_class``
    # (mirrors ``DiffusionExecutor.get_class``). Config files use a string:
    # "default" -> DiffusionEngine, or an import path (set e.g. by a deploy
    # config). Programmatic callers may pass a DiffusionEngine subclass
    # directly — hence ``str | type`` (structured-config mirrors should expose
    # the string form only).
    engine_backend: str | type = "default"

    # Optional override for the diffusion model runner class (import path).
    # Precedence in the worker: this override > the runner declared by the
    # selected engine class (``default_diffusion_model_runner_cls``) > the
    # platform default. Never mutated by engines.
    diffusion_model_runner_cls: str | None = None

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

    # MoE kernel backend selection
    moe_backend: str = "auto"

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
    additional_config: dict[str, Any] = field(default_factory=dict)

    profiler_config: "ProfilerConfig | dict[str, Any] | None" = None

    # Model-specific function for collecting CFG KV caches (set at runtime)
    cfg_kv_collect_func: Any | None = None

    # Quantization: str method name, dict config, QuantizationConfig, or None.
    # str is resolved to {"method": <str>} internally.
    # Per-component: {"transformer": {"method": "fp8"}, "vae": None}
    quantization_config: str | QuantizationConfig | dict[str, Any] | None = None
    # Explicit runtime override for ModelOpt FP8 diffusion checkpoints. This
    # does not enable FP8 by itself; it only selects CUTLASS once the checkpoint
    # has already resolved to vLLM's ModelOpt FP8 linear method.
    force_cutlass_fp8: bool = False

    # Diffusion attention KV cache dtype (not vLLM's --kv-cache-dtype for AR models).
    # None = native dtype (no quantization).
    # "fp8" = dynamic FP8 (float8_e4m3fn) quantization per forward pass.
    # On Hopper+FA3: native FP8 attention (memory + compute savings).
    # On other backends: no benefit, backends skip quantization.
    diffusion_kv_cache_dtype: str | None = None
    # Optional skip selectors for KV-cache quantization. Format: "0-9,20,25-30".
    # Listed steps/layers skip quantization; others keep quantized execution.
    diffusion_kv_cache_skip_steps: str | None = None
    diffusion_kv_cache_skip_layers: str | None = None
    diffusion_kv_cache_skip_step_indices: set[int] | None = None
    diffusion_kv_cache_skip_layer_indices: set[int] | None = None

    # Diffusion pipeline Profiling config
    enable_diffusion_pipeline_profiler: bool = False

    # Step mode settings
    step_execution: bool = False

    # Streaming mode settings
    streaming_output: bool = False  # Start (video) generation with initial prompt, but streaming output in chunks

    # Maximum number of sequences to generate in a batch
    max_num_seqs: int = 1

    # Request-mode batch admission: wait briefly for compatible requests to
    # accumulate in the scheduler waiting queue before the first schedule() of
    # a wave.  Improves fused forward batch sizes under bursty HTTP ingress.
    # 0 disables admission (default; no added latency).
    request_batch_max_wait_ms: float = 0.0

    # Supplementary model specific parameters
    extras: dict[str, Any] = Field(default_factory=dict)

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

    def _resolve_master_port(self) -> int:
        """Resolve torch.distributed master port without unnecessary random jitter.

        Precedence:
        1. ``MASTER_PORT`` environment variable (set by orchestrators for multi-replica launch).
        2. Explicit ``master_port`` passed at construction time.
        3. An OS-assigned ephemeral port when neither is provided.
        """
        from vllm.utils.network_utils import get_open_port

        from vllm_omni.diffusion import envs

        env_port = envs.MASTER_PORT
        if env_port is not None:
            return self.settle_port(env_port, port_inc=37)
        if self.master_port is not None:
            return self.settle_port(self.master_port, port_inc=37)
        return self.settle_port(get_open_port(), port_inc=37)

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
        self.master_port = self._resolve_master_port()
        self.request_batch_max_wait_ms = float(self.request_batch_max_wait_ms or 0.0)
        if self.request_batch_max_wait_ms < 0:
            raise ValueError(f"request_batch_max_wait_ms must be non-negative, got {self.request_batch_max_wait_ms}.")

        if isinstance(self.profiler_config, dict):
            from vllm.config import ProfilerConfig

            self.profiler_config = ProfilerConfig(**self.profiler_config)

        if self.additional_config is None:
            self.additional_config = {}
        elif isinstance(self.additional_config, Mapping):
            self.additional_config = dict(self.additional_config)
        else:
            raise TypeError(f"additional_config must be a mapping or None, got {type(self.additional_config)!r}")

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
        # time. For late (post-construction) assignment, callers should use
        # set_tf_model_config() which propagates quant_config automatically.
        self._propagate_quantization_from_tf_config(self.tf_model_config)

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

        # Match vLLM's config flow: parse entrypoint shorthands before the
        # config object is built, and keep a single runtime truth source.
        self.diffusion_attention_config = build_attention_config(self.diffusion_attention_config)
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

    def _propagate_quantization_from_tf_config(self, tf_config: "TransformerConfig") -> None:
        if tf_config.quant_config is None:
            return

        is_checkpoint_fp8 = bool(getattr(tf_config.quant_config, "is_checkpoint_fp8_serialized", False))
        is_checkpoint_nvfp4 = bool(getattr(tf_config.quant_config, "is_checkpoint_nvfp4_serialized", False))
        should_use_checkpoint_config = (
            self.quantization_config is None
            or (is_checkpoint_fp8 and self._is_generic_fp8_quant_config(self.quantization_config))
            or (is_checkpoint_nvfp4 and self._is_generic_nvfp4_quant_config(self.quantization_config))
        )
        if should_use_checkpoint_config:
            self.quantization_config = tf_config.quant_config
            logger.info(
                "Auto-detected quantization '%s' from model config",
                tf_config.quant_method,
            )

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
        self._propagate_quantization_from_tf_config(tf_config)

    def update_multimodal_support(self) -> None:
        # Resolve serving-visible multimodal behavior from shared metadata
        # instead of importing concrete pipeline modules into the config layer.
        metadata = get_diffusion_model_metadata(self.model_class_name)
        self.supports_multimodal_inputs = metadata.supports_multimodal_inputs
        self.max_multimodal_image_inputs = metadata.max_multimodal_image_inputs

    @staticmethod
    def _looks_like_lance_subfolder(model: str | None) -> bool:
        """Return True when ``--model`` points at a Lance per-component subfolder.

        Lance's HF repo bundles ``Lance_3B/``, ``Lance_3B_Video/`` and
        ``Qwen2.5-VL-ViT/`` under a single top-level ``config.json``; users may
        reasonably hand the AR-style sub-checkpoint path directly.  The
        ``LancePipeline`` constructor knows to walk up to the repo root from
        either subfolder name.
        """
        if not model:
            return False
        base = os.path.basename(str(model).rstrip("/"))
        return base in {"Lance_3B", "Lance_3B_Video"}

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
                    self.set_tf_model_config(TransformerConfig())
                    try:
                        diffusers_pipeline_cls_name = config_dict["_class_name"]
                        self.diffusers_pipeline_cls = getattr(diffusers, diffusers_pipeline_cls_name)
                    except (KeyError, AttributeError) as exc:
                        logger.warning(
                            "Could not find valid _class_name for diffusers pipeline in model_index.json: %s. "
                            "Without the underlying pipeline class the dummy run may omit required inputs.",
                            exc,
                        )
                else:
                    tf_config_dict = get_hf_file_to_dict("transformer/config.json", self.model)
                    if tf_config_dict is None:
                        tf_config_dict = get_hf_file_to_dict("unet/config.json", self.model)
                    if tf_config_dict is not None:
                        self.set_tf_model_config(TransformerConfig.from_dict(tf_config_dict))
                    else:
                        self.set_tf_model_config(TransformerConfig())
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            # Skip transformer config loading for diffusers adapter
            # (non-DiT models don't have a separate transformer folder/config)
            if self.diffusion_load_format == "diffusers":
                self.set_tf_model_config(TransformerConfig())
                logger.warning(
                    "Could not find valid model_index.json per diffusers format. "
                    "This model is likely unsupported by the diffusers backend. "
                    "Also, without knowing the underlying diffusers pipeline class from model_index.json, "
                    "the dummy run will input only text prompt, which may cause errors for pipelines "
                    "that require additional inputs."
                )
            else:
                cfg = get_hf_file_to_dict("config.json", self.model)
                if cfg is None:
                    # Lance ships its top-level config.json one directory above
                    # the per-checkpoint subfolders (``Lance_3B/`` or
                    # ``Lance_3B_Video/``).  Try to recover that case before
                    # raising.
                    if self._looks_like_lance_subfolder(self.model):
                        self.model_class_name = "LancePipeline"
                        self.set_tf_model_config(TransformerConfig())
                        self.update_multimodal_support()
                        return
                    raise ValueError(f"Could not find config.json or model_index.json for model {self.model}")

                self.set_tf_model_config(TransformerConfig.from_dict(cfg))
                model_type = cfg.get("model_type")
                architectures = cfg.get("architectures") or []

                if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                    self.model_class_name = "BagelPipeline"
                    self.set_tf_model_config(TransformerConfig())
                    self.update_multimodal_support()
                elif (
                    model_type == "lance"
                    or "LancePipeline" in architectures
                    or cfg.get("model_name") == "Lance"
                    or self._looks_like_lance_subfolder(self.model)
                ):
                    # Lance ships a non-HF top-level config.json (model_name only)
                    # plus per-component subfolders; resolve to the Lance pipeline.
                    # Also accept --model pointing directly at the ``Lance_3B`` or
                    # ``Lance_3B_Video`` subfolder by walking up to the repo root.
                    self.model_class_name = "LancePipeline"
                    self.set_tf_model_config(TransformerConfig())
                    self.update_multimodal_support()
                elif model_type == "neo_chat":
                    self.model_class_name = "SenseNovaU1Pipeline"
                    self.tf_model_config = TransformerConfig()
                    self.update_multimodal_support()
                elif "BailingMM2NativeForConditionalGeneration" in architectures or model_type in (
                    "bailingmm_moe_v2_lite",
                    "ming_flash_omni",
                    "ming_flash_omni_thinker",
                ):
                    # Ming-flash-omni-2.0 — imagegen stage uses the custom
                    # ``MingImagePipeline`` (ZImage DiT + Qwen2 connector). See
                    # vllm_omni/diffusion/models/ming_flash_omni/pipeline_ming_imagegen.py.
                    self.model_class_name = "MingImagePipeline"
                    self.tf_model_config = TransformerConfig()
                    self.update_multimodal_support()
                elif model_type == "nextstep":
                    if self.model_class_name is None:
                        self.model_class_name = "NextStep11Pipeline"
                    self.set_tf_model_config(TransformerConfig())
                    self.update_multimodal_support()
                elif model_type == "s2v":
                    if self.model_class_name is None:
                        self.model_class_name = "WanS2VPipeline"
                    self.tf_model_config = TransformerConfig()
                    self.update_multimodal_support()
                elif model_type == "Gr00tN1d7" or "Gr00tN1d7" in architectures:
                    self.model_class_name = "Gr00tN1d7Pipeline"
                    self.set_tf_model_config(TransformerConfig())
                    self.update_multimodal_support()
                elif model_type == "vla":
                    from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero

                    if _looks_like_dreamzero(self.model):
                        self.model_class_name = "DreamZeroPipeline"
                        self.set_tf_model_config(TransformerConfig())
                        self.update_multimodal_support()
                    else:
                        raise
                elif architectures and len(architectures) == 1:
                    architecture = architectures[0]
                    from vllm_omni.diffusion.registry import DiffusionModelRegistry

                    if (
                        self.model_class_name is None
                        or DiffusionModelRegistry._try_load_model_cls(architecture) is not None
                    ):
                        self.model_class_name = architecture
                else:
                    raise

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OmniDiffusionConfig":
        kwargs = normalize_omni_diffusion_kwargs(kwargs)

        # Filter kwargs to only include valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        instance = cls(**filtered_kwargs)
        return instance


@dataclass
class DiffusionOutput:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | tuple[Any, ...] | dict[str, Any] | None = None

    # Legacy compatibility fields. New pipeline-specific payloads should be
    # carried by output["payload"] instead.
    trajectory_timesteps: torch.Tensor | dict[str, Any] | None = None
    trajectory_latents: torch.Tensor | dict[str, Any] | None = None
    trajectory_log_probs: torch.Tensor | dict[str, Any] | None = None
    trajectory_decoded: list[Image.Image] | None = None
    error: str | None = None
    error_status_code: int | None = None
    error_type: str | None = None
    aborted: bool = False
    abort_message: str | None = None

    post_process_func: Callable[..., Any] | None = None

    # logged timings info, directly from Req.timings
    # timings: Optional["RequestTimings"] = None

    # Streaming info (the defaults should make sense for non-streaming mode)
    finished: bool = True
    chunk_index: int = 0
    total_chunks: int = 1

    # logged duration of stages
    stage_durations: dict[str, float] = field(default_factory=dict)

    # memory usage info
    peak_memory_mb: float = 0.0

    # When True, move tensor fields to CPU at construction time. Useful when
    # the output is shipped across process boundaries (e.g. step-execution
    # mode) and the receiving side must not initialise a stray CUDA context.
    to_cpu: bool = False

    def __post_init__(self) -> None:
        if not self.to_cpu:
            return

        def _maybe_to_cpu(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: _maybe_to_cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_maybe_to_cpu(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_maybe_to_cpu(item) for item in value)
            return value

        self.output = _maybe_to_cpu(self.output)
        self.trajectory_timesteps = _maybe_to_cpu(self.trajectory_timesteps)
        self.trajectory_latents = _maybe_to_cpu(self.trajectory_latents)
        self.trajectory_log_probs = _maybe_to_cpu(self.trajectory_log_probs)

    @classmethod
    def from_exception(cls, exc: BaseException) -> "DiffusionOutput":
        status_code, error_type = client_error_metadata(exc)
        return cls(
            error=str(exc),
            error_status_code=status_code,
            error_type=error_type,
        )


class DiffusionRequestAbortedError(RuntimeError):
    """Raised when a diffusion request ends via user-visible abort."""


@dataclass
class AttentionSpec:
    """Specifies a backend and its backend-specific parameters for one attention role."""

    backend: str  # registry name, e.g. "FLASH_ATTN"
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific kwargs

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError(f"Expected str for AttentionSpec.backend, got {type(self.backend)!r}")

        if self.extra is None:
            self.extra = {}
        elif isinstance(self.extra, Mapping):
            self.extra = dict(self.extra)
        else:
            raise TypeError(f"Expected dict for AttentionSpec.extra, got {type(self.extra)!r}")


@dataclass
class AttentionConfig:
    """Per-role attention backend configuration.

    Lookup precedence for a given (role, role_category):
      1. per_role[role]         — exact match
      2. per_role[role_category] — category fallback (e.g. "ltx2.audio_to_video" → "cross")
      3. default                — global default
      4. platform default       — unchanged platform logic
    """

    default: AttentionSpec | None = None
    per_role: dict[str, AttentionSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default is not None:
            self.default = self._coerce_spec_or_none(self.default, "default")

        normalized_per_role: dict[str, AttentionSpec] = {}
        for role_key, spec_data in self._normalize_per_role_mapping(self.per_role).items():
            spec = self._coerce_spec_or_none(spec_data, f"per_role[{role_key!r}]")
            if spec is not None:
                normalized_per_role[role_key] = spec
        self.per_role = normalized_per_role

    @staticmethod
    def _coerce_spec(spec_data: Any, field_name: str) -> AttentionSpec:
        if isinstance(spec_data, AttentionSpec):
            return spec_data
        if isinstance(spec_data, str):
            return AttentionSpec(backend=spec_data)
        if isinstance(spec_data, Mapping):
            return AttentionSpec(**dict(spec_data))
        raise TypeError(f"Expected str, dict, or AttentionSpec for {field_name}, got {type(spec_data)!r}")

    @classmethod
    def _coerce_spec_or_none(cls, spec_data: Any, field_name: str) -> AttentionSpec | None:
        spec = cls._coerce_spec(spec_data, field_name)
        if spec.backend.lower() == "auto":
            return None
        return spec

    @classmethod
    def _normalize_per_role_mapping(cls, raw_per_role: Any) -> dict[str, Any]:
        if raw_per_role is None:
            return {}
        if not isinstance(raw_per_role, Mapping):
            raise TypeError(f"Expected dict for AttentionConfig.per_role, got {type(raw_per_role)!r}")

        normalized: dict[str, Any] = {}
        for role_key, spec_data in raw_per_role.items():
            cls._flatten_per_role_entry([role_key], spec_data, normalized)
        return normalized

    @classmethod
    def _flatten_per_role_entry(
        cls,
        path: list[str],
        node: Any,
        normalized: dict[str, Any],
    ) -> None:
        role = ".".join(path)
        if not isinstance(node, Mapping):
            normalized[role] = node
            return

        spec_keys = {"backend", "extra"}
        node_dict = dict(node)
        node_keys = set(node_dict)
        if node_keys & spec_keys:
            if not node_keys <= spec_keys:
                raise ValueError(
                    f"Invalid per_role entry for role {role!r}: cannot mix backend/extra with nested role keys."
                )
            normalized[role] = node_dict
            return

        if not node_dict:
            raise ValueError(f"Empty per_role entry for role {role!r}")

        for child_key, child_value in node_dict.items():
            cls._flatten_per_role_entry([*path, child_key], child_value, normalized)

    def resolve_with_source(
        self,
        role: str = "self",
        role_category: str | None = None,
    ) -> tuple[AttentionSpec | None, str | None]:
        """Resolve the AttentionSpec and report which config entry matched."""
        spec = self.per_role.get(role)
        if spec is not None:
            return spec, f"attention_config.per_role[{role!r}]"
        if role_category is not None:
            spec = self.per_role.get(role_category)
            if spec is not None:
                return spec, f"attention_config.per_role[{role_category!r}] (role_category fallback)"
        if self.default is not None:
            return self.default, "attention_config.default"
        return None, None


def parse_attention_config(
    attention_config: AttentionConfig | Mapping[str, Any] | None = None,
    *,
    attention_backend: str | None = None,
) -> AttentionConfig:
    """Pure type-conversion: coerce *attention_config* to an AttentionConfig.

    Optionally merges an ``attention_backend`` shorthand into the config's
    ``default`` field.  This does **not** read environment variables —
    use :func:`build_attention_config` for the full normalisation that
    should happen exactly once in ``OmniDiffusionConfig.__post_init__``.
    """
    if attention_config is None:
        normalized = AttentionConfig()
    elif isinstance(attention_config, AttentionConfig):
        normalized = copy.deepcopy(attention_config)
    elif isinstance(attention_config, Mapping):
        normalized = AttentionConfig(**dict(attention_config))
    else:
        raise TypeError(
            f"attention_config must be an AttentionConfig, mapping, or None; got {type(attention_config)!r}"
        )

    if attention_backend is not None:
        if normalized.default is not None:
            raise ValueError(
                "--diffusion-attention-backend is mutually exclusive with --diffusion-attention-config.default.backend."
            )
        if attention_backend.lower() != "auto":
            normalized.default = AttentionSpec(backend=attention_backend)

    return normalized


def build_attention_config(
    attention_config: AttentionConfig | Mapping[str, Any] | None = None,
) -> AttentionConfig:
    """Normalize diffusion attention config — the single authoritative entry point.

    Called exactly once in ``OmniDiffusionConfig.__post_init__``.
    Handles type-conversion **and** env-var fallback
    (``DIFFUSION_ATTENTION_BACKEND``).
    """
    normalized = parse_attention_config(attention_config)

    if normalized.default is not None:
        return normalized

    env_attention_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    if env_attention_backend is None:
        return normalized

    if env_attention_backend.lower() == "auto":
        return normalized

    normalized.default = AttentionSpec(backend=env_attention_backend)
    logger.info(
        "Parsed attention config from DIFFUSION_ATTENTION_BACKEND '%s': default=%s, per_role=%s",
        env_attention_backend,
        normalized.default,
        {k: v.backend for k, v in normalized.per_role.items()},
    )
    return normalized


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


class CuMemTag(str, Enum):
    """Tags representing specific CuMem allocations for sleep/wake state tracking."""

    WEIGHTS = "weights"
    KV_CACHE = "kv_cache"


# Special message broadcast via scheduler queues to signal worker shutdown.
SHUTDOWN_MESSAGE = {"type": "shutdown"}
