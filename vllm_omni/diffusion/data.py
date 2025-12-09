# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import torch
from pydantic import Field, model_validator
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
        if self.sequence_parallel_size is None:
            self.sequence_parallel_size = self.ulysses_degree * self.ring_degree
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
        self.world_size = (
            self.pipeline_parallel_size
            * self.data_parallel_size
            * self.tensor_parallel_size
            * self.ulysses_degree
            * self.ring_degree
            * self.cfg_parallel_size
        )


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
class OmniDiffusionConfig:
    # Model and path configuration (for convenience)
    model: str

    model_class_name: str | None = None

    dtype: torch.dtype = torch.bfloat16

    tf_model_config: TransformerConfig = field(default_factory=TransformerConfig)

    # Attention
    # attention_backend: str = None

    # Running mode
    # mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    # workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy
    cache_strategy: str = "none"
    parallel_config: DiffusionParallelConfig = Field(default_factory=DiffusionParallelConfig)

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
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = False
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # VAE memory optimization parameters
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    # STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False

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

    # Logging
    log_level: str = "info"

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
        if self.num_gpus is None:
            self.num_gpus = 1
        if self.num_gpus < self.parallel_config.world_size:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) < parallel_config.world_size ({self.parallel_config.world_size})"
            )

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OmniDiffusionConfig":
        return cls(**kwargs)


_current_omni_diffusion_config: OmniDiffusionConfig | None = None
_current_prefix: str | None = None


@contextmanager
def set_current_vllm_config(omni_diffusion_config: OmniDiffusionConfig, check_compile=False, prefix: str | None = None):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_omni_diffusion_config, _current_prefix
    old_omni_diffusion_config = _current_omni_diffusion_config
    old_prefix = _current_prefix
    # from vllm.compilation.counter import compilation_counter

    # num_models_seen = compilation_counter.num_models_seen
    try:
        _current_omni_diffusion_config = omni_diffusion_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            raise RuntimeError("Compilation is not yet supported for OmniDiffusion")
    finally:
        _current_omni_diffusion_config = old_omni_diffusion_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_omni_diffusion_config()"""
    return get_current_omni_diffusion_config().compilation_config


def get_current_omni_diffusion_config() -> OmniDiffusionConfig:
    if _current_omni_diffusion_config is None:
        logger.warning("Current OmniDiffusionConfig is not set.")
        return OmniDiffusionConfig()
    return _current_omni_diffusion_config


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
