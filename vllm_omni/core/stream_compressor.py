"""Streaming response compression for bandwidth optimization."""

import base64
import json
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)


class CompressionType(Enum):
    """Available compression types."""

    NONE = "none"
    GZIP = "gzip"


@dataclass
class StreamCompressorConfig:
    """Configuration for stream compressor."""

    compression_type: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    min_size_for_compression: int = 256


class StreamCompressor:
    """Compress streaming responses for bandwidth optimization."""

    def __init__(self, config: StreamCompressorConfig | None = None):
        self._config = config or StreamCompressorConfig()

    @property
    def config(self) -> StreamCompressorConfig:
        return self._config

    def compress(self, data: bytes) -> bytes:
        """Compress data using configured compression type."""
        if self._config.compression_type == CompressionType.NONE:
            return data

        if self._config.compression_type == CompressionType.GZIP:
            return zlib.compress(data, level=self._config.compression_level)

        return data

    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if self._config.compression_type == CompressionType.NONE:
            return data

        return zlib.decompress(data)

    async def compress_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Compress an async string stream.

        Args:
            stream: Async iterator of string chunks

        Yields:
            Compressed chunks as JSON strings
        """
        buffer = []
        chunk_index = 0

        async for chunk in stream:
            buffer.append(chunk)

            if len("".join(buffer)) >= self._config.min_size_for_compression:
                combined = "".join(buffer).encode()
                compressed = self.compress(combined)
                yield json.dumps(
                    {"data": base64.b64encode(compressed).decode(), "index": chunk_index, "size": len(combined)}
                )
                buffer = []
                chunk_index += 1

        if buffer:
            combined = "".join(buffer).encode()
            if len(combined) >= 64:
                compressed = self.compress(combined)
                yield json.dumps(
                    {
                        "data": base64.b64encode(compressed).decode(),
                        "index": chunk_index,
                        "is_final": True,
                        "size": len(combined),
                    }
                )
            else:
                yield json.dumps(
                    {
                        "data": base64.b64encode(combined).decode(),
                        "index": chunk_index,
                        "is_final": True,
                        "size": len(combined),
                    }
                )

    @staticmethod
    def decompress_chunk(chunk_json: str) -> str:
        """Decompress a single chunk JSON string."""
        try:
            data = json.loads(chunk_json)
            compressed = base64.b64decode(data["data"])
            decompressed = zlib.decompress(compressed)
            return decompressed.decode("utf-8")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to decompress chunk: {e}")
            return chunk_json
