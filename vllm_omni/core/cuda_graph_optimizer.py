"""CUDA graph optimization for reduced kernel overhead."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class GraphCaptureMode(Enum):
    """CUDA graph capture modes."""

    EAGER = "eager"
    LAZY = "lazy"
    ADAPTIVE = "adaptive"


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA graph optimization."""

    enable_cuda_graph: bool = True
    capture_mode: GraphCaptureMode = GraphCaptureMode.ADAPTIVE
    max_graphs: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 32
    enable_mixed_batch: bool = True
    graph_timeout_ms: int = 1000


@dataclass
class CUDAGraphMetrics:
    """Metrics for CUDA graph."""

    graphs_captured: int = 0
    graphs_launched: int = 0
    capture_failures: int = 0
    total_capture_time_ms: float = 0.0


class CUDAGraphOptimizer:
    """Optimize inference with CUDA graphs."""

    def __init__(self, config: CUDAGraphConfig | None = None):
        self._config = config or CUDAGraphConfig()
        self._graphs: dict[int, Any] = {}
        self._metrics = CUDAGraphMetrics()
        self._enabled = config.enable_cuda_graph if config else True

    @property
    def config(self) -> CUDAGraphConfig:
        return self._config

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def should_use_graph(self, batch_size: int) -> bool:
        """Determine if CUDA graph should be used."""
        if not self._enabled:
            return False

        if batch_size < self._config.min_batch_size:
            return False

        if batch_size > self._config.max_batch_size:
            return False

        if batch_size in self._graphs:
            return True

        if len(self._graphs) < self._config.max_graphs:
            return True

        return False

    def get_or_create_graph(self, batch_size: int) -> Any | None:
        """Get existing graph or create new one."""
        if not self.should_use_graph(batch_size):
            return None

        if batch_size in self._graphs:
            self._metrics.graphs_launched += 1
            return self._graphs[batch_size]

        return None

    def register_graph(self, batch_size: int, graph: Any) -> None:
        """Register a captured CUDA graph."""
        if batch_size in self._graphs:
            logger.warning(f"Graph for batch size {batch_size} already exists, replacing")
            del self._graphs[batch_size]

        self._graphs[batch_size] = graph
        self._metrics.graphs_captured += 1
        logger.info(f"Registered CUDA graph for batch size {batch_size}")

    def clear_graphs(self) -> None:
        """Clear all cached graphs."""
        count = len(self._graphs)
        self._graphs.clear()
        logger.info(f"Cleared {count} CUDA graphs")

    def should_rebuild(self, batch_size: int, graph: Any) -> bool:
        """Determine if graph needs rebuilding."""
        if self._config.capture_mode == GraphCaptureMode.EAGER:
            return False
        if self._config.capture_mode == GraphCaptureMode.LAZY:
            return True
        return self._metrics.capture_failures > 3

    def get_metrics(self) -> dict[str, Any]:
        """Get CUDA graph metrics."""
        return {
            "graphs_captured": self._metrics.graphs_captured,
            "graphs_launched": self._metrics.graphs_launched,
            "capture_failures": self._metrics.capture_failures,
            "cached_graphs": len(self._graphs),
            "enabled": self._enabled,
        }

    def record_capture_failure(self) -> None:
        """Record a capture failure."""
        self._metrics.capture_failures += 1

    def record_capture_time(self, time_ms: float) -> None:
        """Record capture time."""
        self._metrics.total_capture_time_ms += time_ms
