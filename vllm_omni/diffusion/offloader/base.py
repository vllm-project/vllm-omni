# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig, validate_dlo_host_registration_options

from .chunked_transport import TransportBackendKind

logger = init_logger(__name__)


@runtime_checkable
class SupportsModelCpuOffload(Protocol):
    """Pipeline-owned lifecycle for model-level CPU offload.

    Pipelines with non-forward component entry points (for example VAE
    ``decode_latent`` methods) need to activate those stages explicitly, so
    generic forward-hook discovery cannot manage their full lifecycle.
    """

    def enable_omni_model_cpu_offload(
        self,
        *,
        device: torch.device,
        pin_memory: bool,
        use_hsdp: bool,
    ) -> None: ...

    def disable_omni_model_cpu_offload(self) -> None: ...


class OffloadStrategy(Enum):
    NONE = "none"
    MODEL_LEVEL = "model_level"  # Sequential offloading between DiT and encoders
    LAYER_WISE = "layer_wise"  # Block-level
    DISTRIBUTED_LAYER_WISE = "distributed_layer_wise"  # Block-level with DP sharding + H2D/AllGather overlap


@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    use_hsdp: bool = False
    dp_size: int = 1  # derived from parallel_config, not user-configurable
    # True: add DP sharding + AllGather. False: stream complete rank-local
    # blocks from the loader-selected host backing with H2D only.
    dlo_use_allgather: bool = True
    dlo_resident_layers: int = 0  # leading DiT layers kept on device
    # Stage-1 chunked H2D + FS AllGather overlap
    chunk_size_bytes: int = 64 * 1024 * 1024  # 64 MiB full-chunk target
    dlo_pin_budget_bytes: int | None = None  # None = unlimited
    dlo_pin_failure_policy: str = "fail"  # "fail" | "whole_block_fallback"
    dlo_transport_backend: str = "auto"
    # Resolved FS group (set by _init_weight_shard_group, not user-configurable)
    weight_shard_size: int = 1
    weight_shard_rank: int = 0
    weight_shard_group: Any | None = None
    weight_shard_cpu_group: Any | None = None
    # Optional per-worker ceiling for registering an HWR mmap. Zero means no
    # additional ceiling; pin_cpu_memory controls whether registration is tried.
    dlo_host_registration_limit_gib: float = 0.0

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> "OffloadConfig":
        """Extract and validate offload settings from OmniDiffusionConfig.

        Enforces mutual exclusion among the three offload strategies.
        Distributed layer-wise takes the highest priority, then layer-wise,
        then model-level.

        The ``dp_size`` is automatically derived from ``parallel_config`` —
        it is NOT a user-configurable parameter. The distributed layerwise
        offload works with whatever DP/SP parallelism is already set up.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
        enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
        enable_distributed_layerwise_offload = getattr(od_config, "enable_distributed_layerwise_offload", False)
        pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

        parallel_config = getattr(od_config, "parallel_config", None)
        use_hsdp = getattr(parallel_config, "use_hsdp", False) if parallel_config else False
        # Derive dp_size from parallel_config — not user-configurable.
        # The offload adapts to whatever DP/SP is already configured.
        dp_size = 1
        if parallel_config is not None:
            dp_size = getattr(parallel_config, "data_parallel_size", 1)
            # HSDP shard and replica sizes determine the effective group size.
            hsdp_shard_size = getattr(parallel_config, "hsdp_shard_size", -1) if use_hsdp else -1
            hsdp_replicate_size = getattr(parallel_config, "hsdp_replicate_size", 1) if use_hsdp else 1
            if use_hsdp and hsdp_shard_size > 0:
                dp_size = hsdp_shard_size * hsdp_replicate_size

            # When there is no DP but SP > 1, shard weights across SP ranks.
            # AllGather reconstructs full weights per layer; each rank then
            # computes on its SP portion of the sequence.  This gives N×
            # compute parallelism with 1/N H2D transfer, reusing the exact
            # same AllGather code path — only the process group changes.
            if dp_size <= 1:
                sp_size = getattr(parallel_config, "sequence_parallel_size", 1)
                if sp_size and sp_size > 1:
                    dp_size = sp_size

        # Determine strategy (mutual exclusion, distributed layer-wise takes priority)
        if enable_distributed_layerwise_offload:
            strategy = OffloadStrategy.DISTRIBUTED_LAYER_WISE
            if enable_layerwise_offload or enable_cpu_offload:
                logger.info("Distributed layer-wise offloading takes priority, disabling other offloading strategies.")
        elif enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
            if enable_cpu_offload:
                logger.info(
                    "Both model-level and layer-wise offloading enabled. "
                    "Layer-wise takes priority, disabling model-level offloading."
                )
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE

        # With dlo_use_allgather=False, do not add another DP shard. Each rank
        # streams the tensors produced by the standard loader, which may
        # already be TP-local shards. This avoids AllGather synchronization
        # requirements (concurrent requests, dummy run skip).
        dlo_use_allgather = getattr(od_config, "dlo_use_allgather", True)
        dlo_resident_layers = int(getattr(od_config, "dlo_resident_layers", 0))
        dlo_host_registration_limit_gib = validate_dlo_host_registration_options(
            limit_gib=getattr(od_config, "dlo_host_registration_limit_gib", 0.0),
            enable_dlo=enable_distributed_layerwise_offload,
            use_allgather=dlo_use_allgather,
            hwr_mode=getattr(od_config, "host_weight_runtime_mode", "disabled"),
        )
        if dlo_resident_layers < 0:
            raise ValueError(f"dlo_resident_layers must be >= 0, got {dlo_resident_layers}")
        if dlo_resident_layers and dlo_use_allgather:
            raise ValueError(
                "dlo_resident_layers currently requires --dlo-no-use-allgather so "
                "resident blocks use weights prepared by the standard TP-aware loader"
            )

        # If dlo_use_allgather=False, force dp_size=1 (each rank independent)
        if enable_distributed_layerwise_offload and not dlo_use_allgather:
            dp_size = 1
            logger.info(
                "Distributed layerwise offload: dlo_use_allgather=False, "
                "streaming complete rank-local blocks (no DLO shard or AllGather); "
                "the backend will select mmap or standard-loader host storage"
            )

        # HSDP already shards parameters into DTensors. Running distributed
        # layerwise offload with AllGather on top would shard each local tensor
        # again and produce incorrect reconstruction. Keep this combination
        # rejected; HSDP with rank-local DLO remains supported.
        if enable_distributed_layerwise_offload and use_hsdp and dlo_use_allgather:
            raise ValueError(
                "Distributed layerwise offload with AllGather is incompatible with "
                "HSDP: HSDP parameters are already sharded DTensors, and the offloader "
                "would double-shard them. Use --dlo-no-use-allgather (standard-loader "
                "rank-local weights) or disable HSDP."
            )

        weight_shard_size = dp_size

        chunk_size_mb = int(getattr(od_config, "dlo_chunk_size_mb", 64))
        if chunk_size_mb <= 0:
            raise ValueError(f"dlo_chunk_size_mb must be > 0, got {chunk_size_mb}")
        chunk_size_bytes = chunk_size_mb * 1024 * 1024

        pin_budget_gb = getattr(od_config, "dlo_pin_budget_gb", None)
        dlo_pin_budget_bytes = None if pin_budget_gb is None else int(float(pin_budget_gb) * 1024**3)

        dlo_pin_failure_policy = getattr(od_config, "dlo_pin_failure_policy", "fail")
        if dlo_pin_failure_policy not in ("fail", "whole_block_fallback"):
            raise ValueError(
                "dlo_pin_failure_policy must be one of 'fail' or 'whole_block_fallback', "
                f"got {dlo_pin_failure_policy!r}"
            )

        dlo_transport_backend = getattr(od_config, "dlo_transport_backend", TransportBackendKind.AUTO.value)
        try:
            TransportBackendKind(dlo_transport_backend)
        except ValueError as exc:
            choices = ", ".join(backend.value for backend in TransportBackendKind)
            raise ValueError(f"dlo_transport_backend must be one of {choices}, got {dlo_transport_backend!r}") from exc

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            use_hsdp=use_hsdp,
            dp_size=dp_size,
            dlo_use_allgather=dlo_use_allgather,
            dlo_resident_layers=dlo_resident_layers,
            chunk_size_bytes=chunk_size_bytes,
            dlo_pin_budget_bytes=dlo_pin_budget_bytes,
            dlo_pin_failure_policy=dlo_pin_failure_policy,
            dlo_transport_backend=dlo_transport_backend,
            weight_shard_size=weight_shard_size,
            dlo_host_registration_limit_gib=dlo_host_registration_limit_gib,
        )


class OffloadBackend(ABC):
    """Base class for CPU offload backends"""

    def __init__(self, config: OffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """Enable offloading on the pipeline.

        Discovers modules, moves them to appropriate devices, and
        registers forward hooks for swapping/prefetching.

        Args:
            pipeline: Diffusion pipeline model (e.g., Wan22Pipeline)
        """
        raise NotImplementedError

    @abstractmethod
    def disable(self) -> None:
        """Disable offloading and cleanup resources.

        Removes all registered hooks. Does NOT move modules back to
        original devices (caller responsible for that).
        """
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled
