"""In-memory LRU cache for voice extraction artifacts.

Keyed by voice name + extraction mode (e.g. ``"alice:icl"``).
Only named voices are cached; inline ``ref_audio`` without a voice
name is not cached.

Usage::

    key = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False)
    cached = cache.get(key)
    if cached is None:
        # ... extract ...
        cache.put(key, {"artifact": result})
"""

import os
import threading
from collections import OrderedDict
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_MAX_ENTRIES = 128


class VoiceEmbeddingCache:
    """LRU cache for voice extraction outputs.

    Each entry stores a ``dict[str, Any]`` whose contents are model-specific.
    Thread-safe via a lightweight ``threading.Lock``.
    """

    def __init__(self, max_entries: int | None = None):
        if max_entries is None:
            max_entries = int(os.environ.get("VOICE_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info("Voice embedding cache initialized (max_entries=%d)", max_entries)

    @staticmethod
    def make_cache_key(voice_name: str, xvec_only: bool, created_at: float = 0.0) -> str:
        """Build a cache key from a voice name, upload timestamp, and extraction mode.

        Args:
            voice_name: The speaker/voice name (case-insensitive, lowered
                by the caller).
            xvec_only: True for speaker-embedding-only mode, False for
                ICL mode (speaker embedding + ref_code).
            created_at: Upload timestamp from metadata. Prevents stale cache
                hits after a voice is deleted and re-uploaded with the same
                name but different audio.
        """
        mode = "xvec" if xvec_only else "icl"
        return f"{voice_name}:{created_at:.6f}:{mode}"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached artifacts or ``None`` on miss.  Promotes to MRU on hit."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug("Voice cache HIT (key=%s, hits=%d)", key, self._hits)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, artifacts: dict[str, Any]) -> None:
        """Store *artifacts* under *key*, evicting the LRU entry if full."""
        with self._lock:
            self._cache[key] = artifacts
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Voice cache EVICT (key=%s)", evicted_key)

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }
