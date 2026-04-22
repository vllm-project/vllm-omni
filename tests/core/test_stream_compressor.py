"""Unit tests for stream compressor."""

import pytest

from vllm_omni.core.stream_compressor import (
    CompressionType,
    StreamCompressor,
    StreamCompressorConfig,
)


class TestStreamCompressor:
    """Tests for StreamCompressor."""

    def test_compress_decompress(self):
        """Test compression and decompression."""
        config = StreamCompressorConfig(compression_type=CompressionType.GZIP, compression_level=6)
        compressor = StreamCompressor(config)

        data = b"Hello, World!" * 10
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_compress_none(self):
        """Test no compression."""
        config = StreamCompressorConfig(compression_type=CompressionType.NONE)
        compressor = StreamCompressor(config)

        data = b"Test data"
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert compressed == data
        assert decompressed == data

    @pytest.mark.asyncio
    async def test_compress_stream(self):
        """Test streaming compression."""
        config = StreamCompressorConfig(min_size_for_compression=32)
        compressor = StreamCompressor(config)

        async def generate():
            for i in range(5):
                yield f"chunk{i}_"

        results = []
        async for compressed in compressor.compress_stream(generate()):
            results.append(compressed)

        assert len(results) > 0

    def test_decompress_chunk(self):
        """Test chunk decompression."""
        import base64
        import json

        original = "Test message"
        chunk_json = json.dumps({"data": base64.b64encode(original.encode()).decode(), "index": 0, "is_final": True})

        decompressed = StreamCompressor.decompress_chunk(chunk_json)
        assert decompressed == original

    def test_decompress_invalid_chunk(self):
        """Test handling invalid chunk."""
        result = StreamCompressor.decompress_chunk("invalid json")
        assert result == "invalid json"


class TestStreamCompressorConfig:
    """Tests for StreamCompressorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StreamCompressorConfig()

        assert config.compression_type == CompressionType.GZIP
        assert config.compression_level == 6
        assert config.min_size_for_compression == 256

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamCompressorConfig(compression_type=CompressionType.NONE, compression_level=9)

        assert config.compression_type == CompressionType.NONE
        assert config.compression_level == 9
