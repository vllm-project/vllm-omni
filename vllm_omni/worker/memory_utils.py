# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU memory utilities for vLLM Omni workers.

Discrete GPUs tolerate multi-stage sharing by capping the memory budget to
available free memory. Integrated GPUs share memory with the host and retain
upstream request_memory() validation instead of assigning all free RAM.
"""

from __future__ import annotations

import math

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.worker.utils import request_memory

logger = init_logger(__name__)


def request_memory_tolerant(
    init_snapshot: MemorySnapshot,
    cache_config: CacheConfig,
) -> int:
    """Calculate the amount of memory required for this stage.

    On discrete GPUs, tolerates multi-stage sharing by capping the requested
    budget to free memory instead of raising ``ValueError``. The downstream
    memory profiler subtracts the model and runtime allocations from this
    capped budget when sizing the KV cache.

    Integrated GPUs use upstream validation: an oversized budget raises
    ``ValueError`` rather than consuming all available shared host memory.

    Logs a warning when the budget is capped so operators can detect
    under-provisioned GPU memory.
    """
    if current_platform.is_integrated_gpu(init_snapshot.device_.index):
        return request_memory(init_snapshot, cache_config)

    requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)

    if init_snapshot.free_memory < requested_memory:
        capped = init_snapshot.free_memory
        logger.warning(
            "Free memory on device %s (%s/%s GiB) on startup is less than "
            "desired GPU memory utilization (%.2f, %s GiB). "
            "Capping requested memory to available free memory (%s GiB). "
            "This is expected when multiple Omni stages share a GPU; "
            "determine_available_memory() will subtract the model and runtime "
            "allocations from this capped budget when sizing the KV cache.",
            init_snapshot.device_,
            format_gib(init_snapshot.free_memory),
            format_gib(init_snapshot.total_memory),
            cache_config.gpu_memory_utilization,
            format_gib(requested_memory),
            format_gib(capped),
        )
        return capped

    return requested_memory
