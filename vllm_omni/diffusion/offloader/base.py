# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig, validate_dlo_host_registration_options

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

        **New in RFC #7204:** accepts the canonical offload policy via
        ``diffusion_offload_shortcut`` or ``diffusion_offload_config`` fields
        in addition to the legacy boolean flags.  Canonical policy takes
        priority; legacy flags remain supported for compatibility.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        # --- Canonical policy (RFC #7204) — takes priority ---
        from vllm_omni.diffusion.offloader.config_parser import (
            OffloadMode,
            WeightTransfer,
            parse_offload_policy,
            parse_hwr_policy,
            validate_joint,
        )

        shortcut = getattr(od_config, "diffusion_offload_shortcut", None)
        config_dict = getattr(od_config, "diffusion_offload_config", None)

        if shortcut is not None or config_dict is not None:
            canonical = parse_offload_policy(
                diffusion_offload=shortcut,
                diffusion_offload_config=config_dict,
                pin_cpu_memory=getattr(od_config, "pin_cpu_memory", True),
            )

            # Also parse joint HWR policy if supplied via new config.
            hwr_dict = getattr(od_config, "host_weight_runtime_config", None)
            if hwr_dict is not None:
                hwr_policy = parse_hwr_policy(hwr_dict)
                validate_joint(canonical, hwr_policy)
                dlo_host_registration_limit_gib = hwr_policy.registration_limit_gib
            else:
                dlo_host_registration_limit_gib = validate_dlo_host_registration_options(
                    limit_gib=getattr(od_config, "dlo_host_registration_limit_gib", 0.0),
                    enable_dlo=canonical.mode is OffloadMode.LAYER
                    and canonical.has_dit
                    and canonical.dit_transfer() is WeightTransfer.RANK_LOCAL
                    and canonical.dit_resident_layers() == 0,
                    use_allgather=canonical.dit_transfer() is WeightTransfer.ALLGATHER,
                    hwr_mode=getattr(od_config, "host_weight_runtime_mode", "disabled"),
                )

            # Map canonical policy → legacy OffloadStrategy enum.
            if canonical.mode is OffloadMode.NONE or not canonical.is_active:
                strategy = OffloadStrategy.NONE
                dlo_use_allgather = True
                dlo_resident_layers = 0
            elif canonical.mode is OffloadMode.MODULE:
                strategy = OffloadStrategy.MODEL_LEVEL
                dlo_use_allgather = True
                dlo_resident_layers = 0
            else:  # OffloadMode.LAYER
                has_allgather = canonical.dit_transfer() is WeightTransfer.ALLGATHER
                has_resident = canonical.dit_resident_layers() > 0
                # DistributedLayerwiseOffloadBackend when AllGather or resident layers.
                if has_allgather or has_resident:
                    strategy = OffloadStrategy.DISTRIBUTED_LAYER_WISE
                else:
                    # Pure layer-wise rank-local without AllGather or resident
                    # → plain LayerWiseOffloadBackend (ordinary loader path).
                    strategy = OffloadStrategy.LAYER_WISE
                dlo_use_allgather = has_allgather
                dlo_resident_layers = canonical.dit_resident_layers()

            pin_cpu_memory = canonical.pin_memory
            hsdp_enable_distributed = (
                strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
                and not dlo_use_allgather
            )

        else:
            # --- Legacy boolean flags (pre-RFC #7204 path) ---
            enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
            enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
            enable_distributed_layerwise_offload = getattr(
                od_config, "enable_distributed_layerwise_offload", False
            )
            pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

            # Determine strategy (mutual exclusion, distributed layer-wise takes priority)
            if enable_distributed_layerwise_offload:
                strategy = OffloadStrategy.DISTRIBUTED_LAYER_WISE
                if enable_layerwise_offload or enable_cpu_offload:
                    logger.info(
                        "Distributed layer-wise offloading takes priority, disabling other offloading strategies."
                    )
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

            dlo_use_allgather = getattr(od_config, "dlo_use_allgather", True)
            dlo_resident_layers = int(getattr(od_config, "dlo_resident_layers", 0))
            dlo_host_registration_limit_gib = validate_dlo_host_registration_options(
                limit_gib=getattr(od_config, "dlo_host_registration_limit_gib", 0.0),
                enable_dlo=enable_distributed_layerwise_offload,
                use_allgather=dlo_use_allgather,
                hwr_mode=getattr(od_config, "host_weight_runtime_mode", "disabled"),
            )
            if dlo_resident_layers < 0:
                raise ValueError(
                    f"dlo_resident_layers must be >= 0, got {dlo_resident_layers}"
                )
            if dlo_resident_layers and dlo_use_allgather:
                raise ValueError(
                    "dlo_resident_layers currently requires --dlo-no-use-allgather so "
                    "resident blocks use weights prepared by the standard TP-aware loader"
                )

            hsdp_enable_distributed = enable_distributed_layerwise_offload

        # --- Common logic (derived fields / cross-checks) ---
        parallel_config = getattr(od_config, "parallel_config", None)
        use_hsdp = getattr(parallel_config, "use_hsdp", False) if parallel_config else False
        # Derive dp_size from parallel_config — not user-configurable.
        dp_size = 1
        if parallel_config is not None:
            dp_size = getattr(parallel_config, "data_parallel_size", 1)
            hsdp_shard_size = (
                getattr(parallel_config, "hsdp_shard_size", -1) if use_hsdp else -1
            )
            hsdp_replicate_size = (
                getattr(parallel_config, "hsdp_replicate_size", 1) if use_hsdp else 1
            )
            if use_hsdp and hsdp_shard_size > 0:
                dp_size = hsdp_shard_size * hsdp_replicate_size
            if dp_size <= 1:
                sp_size = getattr(parallel_config, "sequence_parallel_size", 1)
                if sp_size and sp_size > 1:
                    dp_size = sp_size

        # If dlo_use_allgather=False, force dp_size=1 (each rank independent)
        if (
            strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
            and not dlo_use_allgather
        ):
            dp_size = 1
            logger.info(
                "Distributed layerwise offload: dlo_use_allgather=False, "
                "streaming complete rank-local blocks (no DLO shard or AllGather); "
                "the backend will select mmap or standard-loader host storage"
            )

        # HSDP already shards parameters into DTensors.  Running distributed
        # layerwise offload on top would shard each to_local() again, producing
        # incorrect reconstruction after AllGather.  Reject this combination.
        if (
            hsdp_enable_distributed
            and use_hsdp
            and dlo_use_allgather
        ):
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
