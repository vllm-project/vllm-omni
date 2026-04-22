"""Adaptive batch scheduler for dynamic request grouping."""

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class RequestPriority(IntEnum):
    """Request priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    REALTIME = 3


@dataclass
class BatchSchedulingConfig:
    """Configuration for batch scheduler."""

    max_batch_size: int = 32
    max_num_seqs: int = 256
    max_model_len: int = 8192
    batch_timeout_ms: int = 100
    enable_adaptive_batching: bool = True
    enable_request_priority: bool = True


@dataclass
class RequestGroup:
    """Group of compatible requests for batch processing."""

    request_ids: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    priority: RequestPriority = RequestPriority.NORMAL


@dataclass
class SchedulerMetrics:
    """Metrics for scheduler performance."""

    requests_pending: int = 0
    requests_batched: int = 0
    batches_created: int = 0
    scheduling_errors: int = 0


class AdaptiveBatchScheduler:
    """
    Adaptive batch scheduler that dynamically groups requests for
    optimal throughput while maintaining latency targets.
    """

    def __init__(self, config: BatchSchedulingConfig):
        self._config = config
        self._lock = threading.RLock()
        self._pending_requests: dict[str, RequestGroup] = {}
        self._active_batches: list[RequestGroup] = []
        self._metrics = SchedulerMetrics()
        self._running = False
        self._schedule_thread: threading.Thread | None = None

    @property
    def config(self) -> BatchSchedulingConfig:
        return self._config

    def start(self) -> None:
        """Start the batch scheduling loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._schedule_thread = threading.Thread(target=self._schedule_loop, daemon=True, name="batch-scheduler")
            self._schedule_thread.start()
            logger.info("Batch scheduler started")

    def stop(self) -> None:
        """Stop the batch scheduling loop."""
        with self._lock:
            self._running = False
        if self._schedule_thread:
            self._schedule_thread.join(timeout=2.0)
        logger.info("Batch scheduler stopped")

    def add_request(
        self,
        request_id: str,
        prompt_length: int = 0,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> bool:
        """
        Add a new request to the scheduler.

        Args:
            request_id: Unique identifier for the request
            prompt_length: Number of text tokens
            image_tokens: Number of image tokens (each image = 128 tokens)
            audio_tokens: Number of audio tokens
            priority: Request priority level

        Returns:
            True if request was added successfully
        """
        if not request_id:
            return False

        with self._lock:
            if request_id in self._pending_requests:
                return False

            estimated_tokens = self._estimate_tokens(prompt_length, image_tokens, audio_tokens)
            request_group = RequestGroup(request_ids=[request_id], estimated_tokens=estimated_tokens, priority=priority)
            self._pending_requests[request_id] = request_group
            self._metrics.requests_pending += 1
            return True

    def remove_request(self, request_id: str) -> bool:
        """Remove a request from the scheduler."""
        with self._lock:
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
                return True
            return False

    def get_next_batch(self) -> list[str] | None:
        """Get next batch of request IDs."""
        with self._lock:
            if not self._active_batches:
                return None
            batch = self._active_batches.pop(0)
            return batch.request_ids

    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        with self._lock:
            return len(self._pending_requests)

    def get_active_batch_count(self) -> int:
        """Get number of active batches."""
        with self._lock:
            return len(self._active_batches)

    def get_metrics(self) -> dict[str, Any]:
        """Get scheduler metrics."""
        with self._lock:
            return {
                "requests_pending": self._metrics.requests_pending,
                "requests_batched": self._metrics.requests_batched,
                "batches_created": self._metrics.batches_created,
                "scheduling_errors": self._metrics.scheduling_errors,
                "active_batches": len(self._active_batches),
            }

    def _estimate_tokens(self, prompt_length: int, image_tokens: int, audio_tokens: int) -> int:
        """Estimate total token count for request."""
        return prompt_length + (image_tokens * 128) + audio_tokens

    def _schedule_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            try:
                with self._lock:
                    self._process_pending_requests()
                time.sleep(self._config.batch_timeout_ms / 1000.0)
            except Exception as e:
                logger.error(f"Scheduling error: {e}")
                self._metrics.scheduling_errors += 1

    def _process_pending_requests(self) -> None:
        """Process and batch pending requests."""
        if not self._pending_requests:
            return

        groups = self._create_compatible_groups()
        for group in groups:
            if group.request_ids:
                self._active_batches.append(group)
                self._metrics.batches_created += 1

    def _create_compatible_groups(self) -> list[RequestGroup]:
        """Create compatible request groups."""
        groups = []
        sorted_requests = sorted(self._pending_requests.items(), key=lambda x: (x[1].priority.value, x[1].created_at))

        current_group = RequestGroup()
        for request_id, group in sorted_requests:
            if self._can_add_to_group(current_group, group):
                current_group.request_ids.append(request_id)
                current_group.estimated_tokens += group.estimated_tokens
            else:
                if current_group.request_ids:
                    groups.append(current_group)
                    self._metrics.requests_pending -= len(current_group.request_ids)
                    self._metrics.requests_batched += len(current_group.request_ids)
                current_group = RequestGroup(
                    request_ids=[request_id], estimated_tokens=group.estimated_tokens, priority=group.priority
                )

        if current_group.request_ids:
            groups.append(current_group)
            self._metrics.requests_pending -= len(current_group.request_ids)
            self._metrics.requests_batched += len(current_group.request_ids)

        for group in groups:
            self._pending_requests.pop(group.request_ids[0], None)

        return groups

    def _can_add_to_group(self, current_group: RequestGroup, new_group: RequestGroup) -> bool:
        """Check if request can be added to group."""
        if not current_group.request_ids:
            return True

        if len(current_group.request_ids) >= self._config.max_batch_size:
            return False

        total_tokens = current_group.estimated_tokens + new_group.estimated_tokens
        if total_tokens > self._config.max_model_len:
            return False

        return True
