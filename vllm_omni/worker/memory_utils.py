# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU memory utilities for vLLM Omni workers.

Discrete GPUs tolerate multi-stage sharing by capping the memory budget to
available free memory. Integrated GPUs share memory with the host and retain
upstream request_memory() validation instead of assigning all free RAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.worker.utils import request_memory

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class GPUMemoryBudgetSnapshot:
    """Immutable startup accounting for one Omni worker rank.

    The profile fields describe the same components used by
    ``OmniGPUWorkerBase.determine_available_memory``. ``activation_peak_bytes``
    is the profiling-window ``torch_peak_increase`` used in the budget; the
    separate ``transient_peak_headroom_bytes`` field preserves the upstream
    warm-up recommendation value without making the accounting ambiguous.

    ``shortfall_bytes`` is positive when profiled usage exceeds the requested
    budget. In that case ``available_kv_cache_memory_bytes`` is zero and the
    shortfall keeps the snapshot's accounting explicit.
    """

    stage_id: int | None = None
    rank: int | None = None
    device: str | None = None
    total_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    uncapped_requested_memory_bytes: int | None = None
    requested_memory_bytes: int | None = None
    budget_was_capped: bool | None = None
    weights_memory_bytes: int | None = None
    preprofile_buffers_bytes: int | None = None
    activation_peak_bytes: int | None = None
    transient_peak_headroom_bytes: int | None = None
    non_torch_memory_bytes: int | None = None
    cuda_graph_memory_bytes: int | None = None
    auxiliary_graph_memory_bytes: int | None = None
    profiled_usage_bytes: int | None = None
    available_kv_cache_memory_bytes: int | None = None
    shortfall_bytes: int | None = None
    accounting_mode: str = "unavailable"
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a transport-safe mapping containing only scalar fields."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


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

    utilization = cache_config.resolved_gpu_memory_utilization
    requested_memory = math.ceil(init_snapshot.total_memory * utilization)

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
            utilization,
            format_gib(requested_memory),
            format_gib(capped),
        )
        return capped

    return requested_memory
