"""KV cache compression for memory optimization."""

import zlib
from dataclasses import dataclass
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)


class CompressionLevel(Enum):
    """Compression levels."""

    NONE = 0
    FAST = 1
    BALANCED = 6
    BEST = 9


@dataclass
class KVCacheCompressionConfig:
    """Configuration for KV cache compression."""

    compression_level: CompressionLevel = CompressionLevel.BALANCED
    enable_adaptive: bool = True
    min_cache_size: int = 1024
    similarity_threshold: float = 0.9


class KVCacheCompressor:
    """Compress KV cache entries for memory optimization."""

    def __init__(self, config: KVCacheCompressionConfig | None = None):
        self._config = config or KVCacheCompressionConfig()

    @property
    def config(self) -> KVCacheCompressionConfig:
        return self._config

    def compress(self, data: bytes) -> bytes:
        """Compress KV cache data."""
        if self._config.compression_level == CompressionLevel.NONE:
            return data

        level = self._config.compression_level.value
        return zlib.compress(data, level=level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress KV cache data."""
        if self._config.compression_level == CompressionLevel.NONE:
            return data

        return zlib.decompress(data)

    def should_compress(self, cache_size: int, similarity: float) -> bool:
        """Determine if compression is beneficial."""
        if not self._config.enable_adaptive:
            return True

        if cache_size < self._config.min_cache_size:
            return False

        if similarity >= self._config.similarity_threshold:
            return True

        return cache_size > self._config.min_cache_size * 2

    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio."""
        if not original:
            return 1.0
        return len(compressed) / len(original)
