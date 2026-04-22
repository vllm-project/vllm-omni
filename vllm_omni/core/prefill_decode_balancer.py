"""Intelligent prefill-decode balancing for optimal latency/throughput."""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class SchedulingStrategy(Enum):
    """Scheduling strategies."""

    MIN_LATENCY = "min_latency"
    MAX_THROUGHPUT = "max_throughput"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class PrefillDecodeConfig:
    """Configuration for prefill-decode balancer."""

    strategy: SchedulingStrategy = SchedulingStrategy.BALANCED
    max_prefill_batch_size: int = 8
    max_decode_batch_size: int = 32
    prefill_chunk_size: int = 512
    target_prefill_ratio: float = 0.3
    latency_sla_ms: float = 2000.0


@dataclass
class BalancerMetrics:
    """Metrics for balancer."""

    avg_prefill_latency: float = 0.0
    avg_decode_latency: float = 0.0
    total_prefill_requests: int = 0
    total_decode_requests: int = 0


class PrefillDecodeBalancer:
    """Balances prefill and decode workloads."""

    def __init__(self, config: PrefillDecodeConfig):
        self._config = config
        self._lock = threading.RLock()
        self._pending_prefill: list[str] = []
        self._pending_decode: list[str] = []
        self._metrics = BalancerMetrics()
        self._current_ratio = config.target_prefill_ratio

    @property
    def config(self) -> PrefillDecodeConfig:
        return self._config

    def add_request(self, request_id: str, is_prefill: bool = True) -> None:
        """Add request to appropriate queue."""
        with self._lock:
            if is_prefill:
                self._pending_prefill.append(request_id)
            else:
                self._pending_decode.append(request_id)

    def get_next_batch(self, prefill_ratio: float | None = None) -> tuple[list[str], bool]:
        """Get next batch applying strategy.

        Returns:
            Tuple of (batch_request_ids, is_prefill_batch)
        """
        with self._lock:
            target = prefill_ratio or self._current_ratio

            if self._config.strategy == SchedulingStrategy.MAX_THROUGHPUT:
                return self._get_max_throughput_batch()
            if self._config.strategy == SchedulingStrategy.MIN_LATENCY:
                return self._get_min_latency_batch()
            return self._get_balanced_batch(target)

    def _get_max_throughput_batch(self) -> tuple[list[str], bool]:
        """Prioritize decode for throughput."""
        if self._pending_decode:
            batch_size = min(self._config.max_decode_batch_size, len(self._pending_decode))
            batch = self._pending_decode[:batch_size]
            self._pending_decode = self._pending_decode[batch_size:]
            return batch, False
        if self._pending_prefill:
            batch_size = min(self._config.max_prefill_batch_size, len(self._pending_prefill))
            batch = self._pending_prefill[:batch_size]
            self._pending_prefill = self._pending_prefill[batch_size:]
            return batch, True
        return [], True

    def _get_min_latency_batch(self) -> tuple[list[str], bool]:
        """Prioritize prefill for latency."""
        if self._pending_prefill:
            batch = [self._pending_prefill.pop(0)]
            return batch, True
        if self._pending_decode:
            return [self._pending_decode.pop(0)], False
        return [], True

    def _get_balanced_batch(self, target_ratio: float) -> tuple[list[str], bool]:
        """Balanced prefill-decode scheduling."""
        total = len(self._pending_prefill) + len(self._pending_decode)
        if total == 0:
            return [], True

        current_ratio = len(self._pending_prefill) / total

        if current_ratio >= target_ratio and self._pending_decode:
            batch_size = min(self._config.max_decode_batch_size, len(self._pending_decode))
            batch = self._pending_decode[:batch_size]
            self._pending_decode = self._pending_decode[batch_size:]
            return batch, False

        if self._pending_prefill:
            batch_size = min(self._config.max_prefill_batch_size, len(self._pending_prefill))
            batch = self._pending_prefill[:batch_size]
            self._pending_prefill = self._pending_prefill[batch_size:]
            return batch, True

        if self._pending_decode:
            return [self._pending_decode.pop(0)], False
        return [], True

    def get_pending_counts(self) -> dict[str, int]:
        """Get pending request counts."""
        with self._lock:
            return {"prefill": len(self._pending_prefill), "decode": len(self._pending_decode)}

    def get_metrics(self) -> dict[str, Any]:
        """Get balancer metrics."""
        with self._lock:
            return {
                "pending_prefill": len(self._pending_prefill),
                "pending_decode": len(self._pending_decode),
                "current_ratio": self._current_ratio,
                "avg_prefill_latency": self._metrics.avg_prefill_latency,
                "avg_decode_latency": self._metrics.avg_decode_latency,
            }
