# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process content-addressed embedding cache (RFC #3427, P0).

Design constraints:
- Per-encoder-worker singleton; no cross-process state.
- Thread-safe: encoder forward runs in the model's inference thread;
  the eviction background thread must not race on entry access.
- LRU + TTL eviction: an entry is eligible for eviction when its TTL
  expires OR when total memory exceeds max_bytes (LRU ordering).
- Stored tensors stay on the device they were produced on (GPU).
  Cache keys are content-addressed on CPU bytes (hasher.py).

Disabled by default unless VLLM_OMNI_EMBEDDING_CACHE=1 or
--embedding-cache-enabled is passed to the engine.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)

# Defaults; overridden by EmbeddingCacheConfig / env vars.
_DEFAULT_TTL_SECONDS = 30
_DEFAULT_MAX_BYTES = 4 * 1024**3  # 4 GB
_EVICTION_INTERVAL_SECONDS = 5


@dataclass
class _Entry:
    embedding: "torch.Tensor"
    size_bytes: int
    inserted_at: float = field(default_factory=time.monotonic)
    last_hit: float = field(default_factory=time.monotonic)
    hits: int = 0


class EmbeddingCache:
    """Thread-safe LRU+TTL in-process embedding cache.

    Usage::

        cache = EmbeddingCache(ttl_seconds=30, max_bytes=4 * 1024**3)
        key = hash_audio_features(input_features)
        hit = cache.get(key)
        if hit is None:
            result = encoder(input_features)
            cache.put(key, result, result.nbytes)
        else:
            result = hit
    """

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        # OrderedDict preserves insertion order; move_to_end() gives LRU.
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._total_bytes = 0
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "inserts": 0}

        self._stop = threading.Event()
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop,
            name="embedding-cache-evictor",
            daemon=True,
        )
        self._eviction_thread.start()
        logger.info(
            "EmbeddingCache started: ttl=%.0fs max_bytes=%d MB",
            ttl_seconds,
            max_bytes // 1024**2,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> "torch.Tensor | None":
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            now = time.monotonic()
            if now - entry.inserted_at > self._ttl:
                # TTL expired — treat as miss and evict immediately.
                self._evict_key(key)
                self._stats["misses"] += 1
                return None
            # Refresh LRU position.
            self._store.move_to_end(key)
            entry.last_hit = now
            entry.hits += 1
            self._stats["hits"] += 1
            return entry.embedding

    def put(self, key: str, embedding: "torch.Tensor", size_bytes: int) -> None:
        with self._lock:
            if key in self._store:
                return  # Another thread inserted while we were encoding.
            self._store[key] = _Entry(
                embedding=embedding,
                size_bytes=size_bytes,
            )
            self._total_bytes += size_bytes
            self._stats["inserts"] += 1
            # Evict oldest entries if over memory budget.
            while self._total_bytes > self._max_bytes and self._store:
                oldest_key, _ = next(iter(self._store.items()))
                self._evict_key(oldest_key)
                self._stats["evictions"] += 1

    def stats(self) -> dict:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total else 0.0
            return {
                **self._stats,
                "entries": len(self._store),
                "total_bytes": self._total_bytes,
                "hit_rate": round(hit_rate, 4),
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._total_bytes = 0

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _evict_key(self, key: str) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.size_bytes

    def _eviction_loop(self) -> None:
        while not self._stop.wait(timeout=_EVICTION_INTERVAL_SECONDS):
            now = time.monotonic()
            with self._lock:
                expired = [k for k, e in self._store.items() if now - e.inserted_at > self._ttl]
                for k in expired:
                    self._evict_key(k)
                    self._stats["evictions"] += 1
            if expired:
                logger.debug("EmbeddingCache: evicted %d expired entries", len(expired))


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------
# Lazily initialised on first use; config comes from env vars or engine args.
# Each encoder worker process gets its own instance (no shared state).

_singleton: EmbeddingCache | None = None
_singleton_lock = threading.Lock()


def _is_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_EMBEDDING_CACHE", "0") == "1"


def get_embedding_cache() -> EmbeddingCache | None:
    """Return the process-level singleton, or None if cache is disabled."""
    global _singleton
    if not _is_enabled():
        return None
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                ttl = float(os.environ.get("VLLM_OMNI_EMBEDDING_CACHE_TTL", str(_DEFAULT_TTL_SECONDS)))
                max_gb = float(os.environ.get("VLLM_OMNI_EMBEDDING_CACHE_MAX_GB", "4"))
                _singleton = EmbeddingCache(
                    ttl_seconds=ttl,
                    max_bytes=int(max_gb * 1024**3),
                )
    return _singleton
