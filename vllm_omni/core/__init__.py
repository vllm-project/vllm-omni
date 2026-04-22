"""Core components for vllm-omni."""

from vllm_omni.core.gpu_memory_pool import (
    GPUMemoryPool,
    GPUMemoryPoolConfig,
    MemoryPoolMetrics,
)
from vllm_omni.core.request_deduplicator import (
    DeduplicationConfig,
    DeduplicationMetrics,
    RequestDeduplicator,
)

__all__ = [
    "GPUMemoryPool",
    "GPUMemoryPoolConfig",
    "MemoryPoolMetrics",
    "RequestDeduplicator",
    "DeduplicationConfig",
    "DeduplicationMetrics",
]
