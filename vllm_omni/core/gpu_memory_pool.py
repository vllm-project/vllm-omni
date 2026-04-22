"""GPU memory pool for efficient allocation and reuse."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class MemoryBlockStatus(Enum):
    """Memory block status."""

    FREE = "free"
    ALLOCATED = "allocated"


@dataclass
class MemoryBlock:
    """Memory block for GPU allocation."""

    block_id: str
    size: int
    offset: int
    status: MemoryBlockStatus = MemoryBlockStatus.FREE
    owner_request_id: str | None = None
    last_used: float = field(default_factory=time.time)


@dataclass
class GPUMemoryPoolConfig:
    """Configuration for GPU memory pool."""

    total_size: int = 8 * 1024 * 1024 * 1024  # 8GB default
    block_size_granularity: int = 1024 * 1024  # 1MB granularity
    enable_tracking: bool = True


@dataclass
class MemoryPoolMetrics:
    """Metrics for memory pool."""

    total_allocations: int = 0
    total_frees: int = 0
    current_allocated: int = 0


class GPUMemoryPool:
    """GPU memory pool for efficient block allocation."""

    def __init__(self, config: GPUMemoryPoolConfig):
        self._config = config
        self._lock = threading.RLock()
        self._blocks: dict[str, MemoryBlock] = {}
        self._free_blocks: dict[int, list[str]] = {}
        self._allocated: dict[str, str] = {}
        self._metrics = MemoryPoolMetrics()
        self._init_pool()

    @property
    def config(self) -> GPUMemoryPoolConfig:
        return self._config

    def _init_pool(self) -> None:
        """Initialize memory pool."""
        num_blocks = self._config.total_size // self._config.block_size_granularity

        for i in range(num_blocks):
            block_id = f"block_{i}"
            block = MemoryBlock(
                block_id=block_id,
                size=self._config.block_size_granularity,
                offset=i * self._config.block_size_granularity,
            )
            self._blocks[block_id] = block

            if self._config.block_size_granularity not in self._free_blocks:
                self._free_blocks[self._config.block_size_granularity] = []
            self._free_blocks[self._config.block_size_granularity].append(block_id)

    def allocate(self, size: int, request_id: str) -> tuple[str, int, int] | None:
        """Allocate a memory block.

        Args:
            size: Size in bytes
            request_id: Request identifier

        Returns:
            Tuple of (block_id, offset, size) or None if failed
        """
        with self._lock:
            aligned_size = self._align_size(size)

            if aligned_size not in self._free_blocks or not self._free_blocks[aligned_size]:
                logger.warning(f"No free blocks for size {aligned_size}")
                return None

            block_id = self._free_blocks[aligned_size].pop(0)
            block = self._blocks[block_id]

            block.status = MemoryBlockStatus.ALLOCATED
            block.owner_request_id = request_id
            block.last_used = time.time()

            self._allocated[request_id] = block_id
            self._metrics.total_allocations += 1
            self._metrics.current_allocated += aligned_size

            return block_id, block.offset, aligned_size

    def free(self, request_id: str) -> bool:
        """Free allocated block.

        Args:
            request_id: Request identifier

        Returns:
            True if freed successfully
        """
        with self._lock:
            block_id = self._allocated.pop(request_id, None)
            if not block_id:
                return False

            block = self._blocks[block_id]
            block.status = MemoryBlockStatus.FREE
            block.owner_request_id = None

            self._free_blocks[block.size].append(block_id)
            self._metrics.current_allocated -= block.size
            self._metrics.total_frees += 1

            return True

    def get_metrics(self) -> dict[str, Any]:
        """Get pool metrics."""
        with self._lock:
            return {
                "total_size": self._config.total_size,
                "current_allocated": self._metrics.current_allocated,
                "total_allocations": self._metrics.total_allocations,
                "total_frees": self._metrics.total_frees,
                "num_free_blocks": sum(len(blocks) for blocks in self._free_blocks.values()),
            }

    def _align_size(self, size: int) -> int:
        """Align size to block granularity."""
        if size % self._config.block_size_granularity == 0:
            return size
        return ((size // self._config.block_size_granularity) + 1) * self._config.block_size_granularity
