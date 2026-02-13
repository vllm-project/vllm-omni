# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, OffloadStrategy
from .layerwise_backend import LayerWiseOffloadBackend
from .sequential_backend import (
    ModelLevelOffloadBackend,
    apply_sequential_offload,
    remove_sequential_offload,
)

logger = init_logger(__name__)

__all__ = [
    "OffloadBackend",
    "OffloadConfig",
    "OffloadStrategy",
    "LayerWiseOffloadBackend",
    "ModelLevelOffloadBackend",
    "apply_sequential_offload",
    "remove_sequential_offload",
    "get_offload_backend",
]


def get_offload_backend(
    od_config: OmniDiffusionConfig,
    device: torch.device | None = None,
) -> OffloadBackend | None:
    """Create appropriate offload backend based on configuration.

    Args:
        od_config: OmniDiffusionConfig with offload settings
        device: Target device (auto-detected if None)

    Returns:
        OffloadBackend instance or None if offloading disabled

    Example:
        >>> backend = get_offload_backend(od_config, device=torch.device("cuda:0"))
        >>> if backend:
        ...     backend.enable(pipeline)
    """
    # Extract and validate configuration
    config = OffloadConfig.from_od_config(od_config)

    # Return None if no offloading requested
    if config.strategy == OffloadStrategy.NONE:
        return None

    # Validate platform (CUDA required for now)
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 1:
        logger.warning("CPU offloading requires CUDA devices. Skipping offloading.")
        return None

    # Detect device if not provided
    if device is None:
        try:
            device = current_omni_platform.get_torch_device()
        except (NotImplementedError, AttributeError) as exc:
            logger.error("Failed to detect device: %s. Skipping offloading.", exc)
            return None

    # Create appropriate backend
    if config.strategy == OffloadStrategy.MODEL_LEVEL:
        return ModelLevelOffloadBackend(config, device)
    elif config.strategy == OffloadStrategy.LAYER_WISE:
        return LayerWiseOffloadBackend(config, device)
    else:
        logger.error("Unknown offload strategy: %s", config.strategy)
        return None
