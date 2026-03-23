"""In-memory LRU cache for voice extraction artifacts.

Model-agnostic: stores ``dict[str, Any]`` keyed by audio content hash.
Any TTS model can use this with the 3-line pattern::

    audio_hash = VoiceEmbeddingCache.compute_audio_hash(wav, sr)
    cached = cache.get(audio_hash)
    if cached is None:
        # ... extract ...
        cache.put(audio_hash, {"artifact": result})
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


class VoiceEmbeddingCache:
    """LRU cache for voice extraction outputs, keyed by audio content hash.

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
        """Compute a 16-char hex hash from normalised audio + sample rate."""
        h = hashlib.sha256(wav.astype(np.float32).tobytes())
        h.update(str(sr).encode())
        return h.hexdigest()[:16]

    def get(self, audio_hash: str) -> dict[str, Any] | None:
        """Return cached artifacts or ``None`` on miss.  Promotes to MRU on hit."""
        with self._lock:
            if audio_hash in self._cache:
                self._cache.move_to_end(audio_hash)
                self._hits += 1
                logger.debug(
                    "Voice cache HIT (hash=%s, hits=%d)",
                    audio_hash[:8],
                    self._hits,
                )
                return self._cache[audio_hash]
            self._misses += 1
            return None

    def put(self, audio_hash: str, artifacts: dict[str, Any]) -> None:
        """Store *artifacts* under *audio_hash*, evicting the LRU entry if full."""
        with self._lock:
            self._cache[audio_hash] = artifacts
            self._cache.move_to_end(audio_hash)
            while len(self._cache) > self._max_entries:
                evicted_hash, _ = self._cache.popitem(last=False)
                logger.debug("Voice cache EVICT (hash=%s)", evicted_hash[:8])

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }
