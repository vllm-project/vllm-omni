# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

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

        # HSDP already shards parameters into DTensors.  Running distributed
        # layerwise offload on top would shard each to_local() again, producing
        # incorrect reconstruction after AllGather.  Reject this combination.
        if enable_distributed_layerwise_offload and use_hsdp and dlo_use_allgather:
            raise ValueError(
                "Distributed layerwise offload with AllGather is incompatible with "
                "HSDP: HSDP parameters are already sharded DTensors, and the offloader "
                "would double-shard them. Use --dlo-no-use-allgather (standard-loader "
                "rank-local weights) or disable HSDP."
            )

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            use_hsdp=use_hsdp,
            dp_size=dp_size,
            dlo_use_allgather=dlo_use_allgather,
            dlo_resident_layers=dlo_resident_layers,
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
