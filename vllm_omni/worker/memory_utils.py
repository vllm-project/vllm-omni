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

    Logs a warning when the budget is capped so operators can detect
    under-provisioned GPU memory.
    """
    requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)

    if init_snapshot.free_memory < requested_memory:
        capped = init_snapshot.free_memory
        logger.warning(
            "Free memory on device %s (%s/%s GiB) on startup is less than "
            "desired GPU memory utilization (%.2f, %s GiB). "
            "Capping requested memory to available free memory (%s GiB). "
            "This is expected when multiple Omni stages share a GPU; "
            "the per-process NVML accounting in determine_available_memory() "
            "will compute the correct KV cache budget.",
            init_snapshot.device_,
            format_gib(init_snapshot.free_memory),
            format_gib(init_snapshot.total_memory),
            cache_config.gpu_memory_utilization,
            format_gib(requested_memory),
            format_gib(capped),
        )
        return capped

    return requested_memory
