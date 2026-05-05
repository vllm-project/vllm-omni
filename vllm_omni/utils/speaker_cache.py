"""Process-wide thread-safe LRU cache for speaker extraction artifacts.

Keyed by ``(model_type, speaker_name, created_at)`` so each upload generation
has its own slot. Access via :func:`get_speaker_cache`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_BYTES = 512 * 1024**2  # 512 MiB

_SINGLETON: SpeakerEmbeddingCache | None = None
_SINGLETON_LOCK = threading.Lock()


def _estimate_tensor_bytes(obj: object) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_estimate_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_bytes(item) for item in obj)
    return 0


class SpeakerEmbeddingCache:
    """Thread-safe in-memory LRU cache for speaker extraction artifacts."""

    def __init__(self, *, max_bytes: int = _MAX_BYTES):
        self._cache: OrderedDict[tuple[str, str, int], dict[str, Any]] = OrderedDict()
        self._sizes: dict[tuple[str, str, int], int] = {}
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._max_bytes = max_bytes
        logger.info("Speaker cache ready (max_bytes=%d)", self._max_bytes)

    @staticmethod
    def make_cache_key(speaker_name: str, model_type: str, created_at: int = 0) -> tuple[str, str, int]:
        """Build a cache key. ``created_at=0`` for built-in speakers (no upload).

        Names are normalized (stripped + lowercased) so delete/clear paths that
        normalize to lowercase match entries put with mixed-case names.
        """
        if not speaker_name or not speaker_name.strip():
            raise ValueError("speaker_name is required")
        if not model_type:
            raise ValueError("model_type is required")
        return (model_type, speaker_name.strip().lower(), int(created_at))

    def get(self, key: tuple[str, str, int]) -> dict[str, Any] | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: tuple[str, str, int], artifacts: dict[str, Any]) -> None:
        with self._lock:
            self._insert_locked(key, artifacts)

    def _insert_locked(self, key: tuple[str, str, int], artifacts: dict[str, Any]) -> None:
        size = _estimate_tensor_bytes(artifacts)
        if size > self._max_bytes:
            logger.warning("Speaker cache skip: entry %s size=%dB exceeds max_bytes=%dB", key, size, self._max_bytes)
            return
        if key in self._cache:
            self._total_bytes -= self._sizes.pop(key, 0)
            del self._cache[key]
        self._cache[key] = artifacts
        self._sizes[key] = size
        self._total_bytes += size
        self._cache.move_to_end(key)
        while self._cache and self._total_bytes > self._max_bytes:
            evict_key, _ = self._cache.popitem(last=False)
            self._total_bytes -= self._sizes.pop(evict_key, 0)
            logger.debug("Speaker cache EVICT: key=%s", evict_key)

    def clear(self, speaker_name: str | None = None) -> int:
        """Remove entries. With a name, drops matches across model types and generations."""
        with self._lock:
            if speaker_name is None:
                removed = len(self._cache)
                self._cache.clear()
                self._sizes.clear()
                self._total_bytes = 0
                self._hits = 0
                self._misses = 0
                return removed

            if not speaker_name or not speaker_name.strip():
                raise ValueError("speaker_name cannot be an empty string")
            normalized = speaker_name.strip().lower()
            removed = 0
            for k in list(self._cache.keys()):
                if isinstance(k, tuple) and len(k) >= 2 and k[1] == normalized:
                    self._total_bytes -= self._sizes.pop(k, 0)
                    del self._cache[k]
                    removed += 1
            return removed

    def memory_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "memory_bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
                "memory_mb": round(self._total_bytes / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
            }


def get_speaker_cache() -> SpeakerEmbeddingCache:
    """Return the process-wide speaker cache singleton."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = SpeakerEmbeddingCache()
    return _SINGLETON
