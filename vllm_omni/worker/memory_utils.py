# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU memory utilities for vLLM Omni workers.

Includes a tolerant version of the upstream request_memory() that handles
multi-stage GPU sharing by capping the memory budget to available free
memory instead of raising ValueError.
"""

from __future__ import annotations

import math

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import MemorySnapshot, format_gib

logger = init_logger(__name__)


def request_memory_tolerant(
    init_snapshot: MemorySnapshot,
    cache_config: CacheConfig,
) -> int:
    """Calculate the amount of memory required for this stage.

    Like upstream ``request_memory()`` but tolerates multi-stage GPU sharing:
    if ``free_memory < requested_memory`` (because another stage on the same
    GPU has already consumed memory), caps the requested budget to the actual
    free memory instead of raising ``ValueError``.  The downstream
    ``OmniGPUWorkerBase.determine_available_memory()`` already does per-process
    NVML accounting and correctly computes the KV cache budget regardless.

    A firing cap is logged at ERROR level with both the requested and the
    granted budget: it is not a normal condition. It means the resolved
    per-stage ``gpu_memory_utilization`` values of the stages sharing this
    GPU do not fit (co-located stages must sum to at most 1.0 of the device),
    or an unaccounted consumer occupies the device. The stage still starts
    because the downstream ``OmniGPUWorkerBase.determine_available_memory()``
    accounting sizes the KV cache from what is actually free, but the
    operator should fix the deploy config rather than rely on the cap.
    """
    requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)

    if init_snapshot.free_memory < requested_memory:
        capped = init_snapshot.free_memory
        logger.error(
            "GPU memory budget capped on device %s: requested %s GiB "
            "(gpu_memory_utilization=%.2f of %s GiB) but only %s GiB is free at startup; "
            "granting %s GiB. The stages sharing this GPU over-commit it "
            "(their resolved gpu_memory_utilization must sum to <= 1.0) or another "
            "process holds memory here. Fix the deploy config; the KV cache is sized "
            "from actual free memory, so this stage will run with less than configured.",
            init_snapshot.device_,
            format_gib(requested_memory),
            cache_config.gpu_memory_utilization,
            format_gib(init_snapshot.total_memory),
            format_gib(init_snapshot.free_memory),
            format_gib(capped),
        )
        return capped

    return requested_memory
