# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, OffloadStrategy, SupportsModelCpuOffload
from .block_discovery import get_blocks_attr_names, get_blocks_from_dit, set_blocks_attr_names
from .distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
    apply_distributed_block_hook,
    remove_distributed_block_hook,
)
from .layerwise_backend import LayerWiseOffloadBackend
from .module_residency import BoundedAllocatorCache, PinnedModuleStager
from .offload_plan import OffloadPlan, get_offload_plan
from .sequential_backend import (
    ModelLevelOffloadBackend,
    apply_sequential_offload,
    remove_sequential_offload,
    sequential_offload_component,
)
from .startup import OffloadStartupState, take_offload_startup_state
from .tensor_utils import (
    dtype_size,
    is_dtensor,
    is_materialized_tensor,
    make_offload_placeholder,
    set_tensor_storage,
)

logger = init_logger(__name__)

__all__ = [
    "OffloadBackend",
    "OffloadConfig",
    "OffloadPlan",
    "OffloadStrategy",
    "SupportsModelCpuOffload",
    "LayerWiseOffloadBackend",
    "DistributedLayerwiseOffloadBackend",
    "DistributedLayerwiseOffloadHook",
    "ModelLevelOffloadBackend",
    "BoundedAllocatorCache",
    "PinnedModuleStager",
    "apply_sequential_offload",
    "remove_sequential_offload",
    "sequential_offload_component",
    "apply_distributed_block_hook",
    "remove_distributed_block_hook",
    "get_offload_backend",
    "enable_offload_backend",
    "OffloadStartupState",
    "take_offload_startup_state",
    "get_offload_plan",
    "get_blocks_attr_names",
    "get_blocks_from_dit",
    "set_blocks_attr_names",
    "dtype_size",
    "is_dtensor",
    "is_materialized_tensor",
    "make_offload_placeholder",
    "set_tensor_storage",
]


def get_offload_backend(
    od_config: OmniDiffusionConfig,
    device: torch.device | None = None,
    host_weight_plan: HostWeightPlan | None = None,
) -> OffloadBackend | None:
    """Create appropriate offload backend based on configuration.

    Args:
        od_config: OmniDiffusionConfig with offload settings
        device: Target device (auto-detected if None)
        host_weight_plan: Exact loader-produced backing plan, if ordinary
            weight materialization was skipped.

    Returns:
        OffloadBackend instance or None if offloading disabled

    Example:
        >>> backend = get_offload_backend(od_config, device=torch.device("cuda:0"))
        >>> if backend:
        ...     backend.enable(pipeline)
    """
    # Extract and validate configuration
    config = OffloadConfig.from_od_config(od_config)

    if host_weight_plan is not None and config.strategy != OffloadStrategy.DISTRIBUTED_LAYER_WISE:
        raise RuntimeError(
            "A loader-owned DLO host-weight plan was produced, but distributed layerwise offload is not selected"
        )

    # Return None if no offloading requested
    if config.strategy == OffloadStrategy.NONE:
        return None

    # Validate platform (CUDA required for now)
    if not current_omni_platform.supports_cpu_offload() or current_omni_platform.get_device_count() < 1:
        if host_weight_plan is not None:
            raise RuntimeError(
                "The loader skipped ordinary weight materialization for DLO, "
                "but this platform cannot create the required offload backend"
            )
        logger.warning(
            "Current device: %s does not support CPU offloading. Skipping offloading.",
            current_omni_platform.get_device_name(),
        )
        return None

    # Detect device if not provided
    if device is None:
        try:
            device = current_omni_platform.get_torch_device()
        except (NotImplementedError, AttributeError) as exc:
            if host_weight_plan is not None:
                raise RuntimeError(
                    "The loader skipped ordinary weight materialization for DLO, "
                    "but the target device could not be resolved"
                ) from exc
            logger.error("Failed to detect device: %s. Skipping offloading.", exc)
            return None

    # Create appropriate backend
    if config.strategy == OffloadStrategy.MODEL_LEVEL:
        return ModelLevelOffloadBackend(config, device)
    elif config.strategy == OffloadStrategy.LAYER_WISE:
        return LayerWiseOffloadBackend(config, device)
    elif config.strategy == OffloadStrategy.DISTRIBUTED_LAYER_WISE:
        return DistributedLayerwiseOffloadBackend(
            config,
            device,
            host_weight_plan=host_weight_plan,
        )
    else:
        logger.error("Unknown offload strategy: %s", config.strategy)
        return None


def enable_offload_backend(
    od_config: OmniDiffusionConfig,
    pipeline: torch.nn.Module,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, OffloadBackend | None]:
    """Create and enable the loader-selected backend transactionally.

    The model runner only calls this generic offloader boundary. Loader-owned
    host plans and optional fresh-model recovery callbacks stay inside the
    startup state consumed here.
    """

    def enable_once(model: torch.nn.Module, state: OffloadStartupState | None) -> OffloadBackend | None:
        backend: OffloadBackend | None = None
        try:
            backend = get_offload_backend(
                od_config,
                device=device,
                host_weight_plan=state.host_weight_plan if state is not None else None,
            )
            if backend is not None:
                logger.info("Enabling offloader backend: %s", backend.__class__.__name__)
                backend.enable(model)
            elif state is not None:
                state.close_loader_ownership()
            return backend
        except Exception:
            if backend is not None:
                try:
                    backend.disable()
                except Exception:
                    logger.exception("Failed to clean up the offload backend after startup failure")
            if state is not None:
                state.close_loader_ownership()
            raise

    startup_state = take_offload_startup_state(pipeline)
    try:
        return pipeline, enable_once(pipeline, startup_state)
    except Exception:
        if startup_state is None or not startup_state.allow_fresh_retry:
            raise
        assert startup_state.fresh_model_loader is not None
        logger.warning(
            "Loader-backed offload startup failed; retrying with a fresh canonical model",
            exc_info=True,
        )
        del pipeline
        pipeline = startup_state.fresh_model_loader()
        return pipeline, enable_once(pipeline, take_offload_startup_state(pipeline))
