"""In-memory LRU cache for voice extraction artifacts.

Model-agnostic: stores ``dict[str, Any]`` keyed by a cache key that
combines a voice identifier with the extraction mode.

For uploaded voices the identifier is the voice name (cheap, no hashing).
For inline ref_audio the identifier is a content hash (SHA-256 prefix).

Usage::

    key = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False)
    cached = cache.get(key)
    if cached is None:
        # ... extract ...
        cache.put(key, {"artifact": result})
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


class VoiceEmbeddingCache:
    """LRU cache for voice extraction outputs.

    Each entry stores a ``dict[str, Any]`` whose contents are model-specific.
    Thread-safe via a lightweight ``threading.Lock``.
    """

    def __init__(self, max_entries: int = 128):
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def compute_audio_hash(wav: np.ndarray, sr: int) -> str:
        """Compute a 16-char hex hash from normalised audio + sample rate.

        Only needed for inline ref_audio (no voice name).
        """
        h = hashlib.sha256(wav.astype(np.float32).tobytes())
        h.update(str(sr).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def make_cache_key(identifier: str, xvec_only: bool) -> str:
        """Build a cache key from a voice identifier and extraction mode.

        Args:
            identifier: Voice name (for uploaded voices) or audio content
                hash (for inline ref_audio).
            xvec_only: True for speaker-embedding-only mode, False for
                ICL mode (speaker embedding + ref_code).
        """
        mode = "xvec" if xvec_only else "icl"
        return f"{identifier}:{mode}"

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
