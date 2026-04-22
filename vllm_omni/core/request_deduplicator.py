"""Request deduplication layer."""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for request deduplication."""

    enable_content_dedup: bool = True
    max_cache_size: int = 1024
    cache_ttl_seconds: float = 60.0
    similarity_threshold: float = 0.95


@dataclass
class DeduplicationMetrics:
    """Metrics for deduplication."""

    duplicates_found: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class RequestHasher:
    """Generate content-based hashes."""

    @staticmethod
    def hash_request(
        prompt: str, multimodal_data: dict[str, Any] | None = None, sampling_params: dict[str, Any] | None = None
    ) -> str:
        """Generate deterministic hash."""
        components = [
            prompt or "",
            str(sorted(multimodal_data.items())) if multimodal_data else "",
            str(sorted(sampling_params.items())) if sampling_params else "",
        ]
        content = "|".join(components).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def compute_similarity(hash1: str, hash2: str) -> float:
        """Compute hash similarity."""
        if len(hash1) != len(hash2):
            return 0.0
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)


class RequestDeduplicator:
    """Deduplicate identical requests within TTL window."""

    def __init__(self, config: DeduplicationConfig):
        self._config = config
        self._lock = threading.RLock()
        self._hash_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._request_tracking: dict[str, str] = {}
        self._metrics = DeduplicationMetrics()

    @property
    def config(self) -> DeduplicationConfig:
        return self._config

    def check_and_register(
        self,
        request_id: str,
        prompt: str,
        multimodal_data: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str | None:
        """Check for duplicate and register if unique.

        Args:
            request_id: Unique request identifier
            prompt: Text prompt
            multimodal_data: Multi-modal data
            sampling_params: Sampling parameters

        Returns:
            Duplicate request_id if duplicate, None otherwise
        """
        if not self._config.enable_content_dedup:
            return None

        with self._lock:
            request_hash = RequestHasher.hash_request(prompt, multimodal_data, sampling_params)

            existing_id = self._find_duplicate(request_hash)
            if existing_id:
                self._metrics.duplicates_found += 1
                logger.info(f"Duplicate request found: {existing_id}")
                return existing_id

            self._register_request(request_id, request_hash, prompt)
            return None

    def _find_duplicate(self, request_hash: str) -> str | None:
        """Find duplicate in cache."""
        current_time = time.time()

        for cached_hash, cached_info in self._hash_cache.items():
            age = current_time - cached_info["timestamp"]
            if age > self._config.cache_ttl_seconds:
                del self._hash_cache[cached_hash]
                continue

            similarity = RequestHasher.compute_similarity(cached_hash, request_hash)
            if similarity >= self._config.similarity_threshold:
                self._metrics.cache_hits += 1
                return cached_info["request_id"]

        self._metrics.cache_misses += 1
        return None

    def _register_request(self, request_id: str, request_hash: str, prompt: str) -> None:
        """Register new request in cache."""
        self._hash_cache[request_hash] = {"request_id": request_id, "timestamp": time.time(), "prompt": prompt}
        self._request_tracking[request_id] = request_hash

        while len(self._hash_cache) > self._config.max_cache_size:
            self._hash_cache.popitem(last=False)

    def unregister(self, request_id: str) -> None:
        """Unregister request."""
        with self._lock:
            request_hash = self._request_tracking.pop(request_id, None)
            if request_hash:
                self._hash_cache.pop(request_hash, None)

    def get_metrics(self) -> dict[str, Any]:
        """Get deduplication metrics."""
        with self._lock:
            return {
                "cache_size": len(self._hash_cache),
                "duplicates_found": self._metrics.duplicates_found,
                "cache_hits": self._metrics.cache_hits,
                "cache_misses": self._metrics.cache_misses,
            }
