"""Unit tests for KV cache compression."""

from vllm_omni.core.kv_cache_compression import (
    CompressionLevel,
    KVCacheCompressionConfig,
    KVCacheCompressor,
)


class TestKVCacheCompressor:
    """Tests for KVCacheCompressor."""

    def test_compress_decompress(self):
        """Test compression and decompression."""
        config = KVCacheCompressionConfig(compression_level=CompressionLevel.BALANCED)
        compressor = KVCacheCompressor(config)

        data = b"x" * 1000
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_compress_none(self):
        """Test no compression."""
        config = KVCacheCompressionConfig(compression_level=CompressionLevel.NONE)
        compressor = KVCacheCompressor(config)

        data = b"test data"
        compressed = compressor.compress(data)

        assert compressed == data

    def test_should_compress_disabled(self):
        """Test should_compress when disabled."""
        config = KVCacheCompressionConfig(enable_adaptive=False)
        compressor = KVCacheCompressor(config)

        assert compressor.should_compress(100, 0.5) is True

    def test_should_compress_small_cache(self):
        """Test should_compress for small cache."""
        config = KVCacheCompressionConfig(min_cache_size=1024, enable_adaptive=True)
        compressor = KVCacheCompressor(config)

        assert compressor.should_compress(512, 0.9) is False

    def test_should_compress_high_similarity(self):
        """Test should_compress for high similarity."""
        config = KVCacheCompressionConfig(similarity_threshold=0.9, enable_adaptive=True)
        compressor = KVCacheCompressor(config)

        assert compressor.should_compress(2000, 0.95) is True

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        config = KVCacheCompressionConfig()
        compressor = KVCacheCompressor(config)

        original = b"x" * 1000
        compressed = compressor.compress(original)
        ratio = compressor.get_compression_ratio(original, compressed)

        assert 0 < ratio < 1.0

    def test_compression_levels(self):
        """Test different compression levels."""
        for level in CompressionLevel:
            config = KVCacheCompressionConfig(compression_level=level)
            compressor = KVCacheCompressor(config)

            data = b"y" * 500
            compressed = compressor.compress(data)
            decompressed = compressor.decompress(compressed)

            assert decompressed == data


class TestKVCacheCompressionConfig:
    """Tests for KVCacheCompressionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = KVCacheCompressionConfig()

        assert config.compression_level == CompressionLevel.BALANCED
        assert config.enable_adaptive is True
        assert config.min_cache_size == 1024

    def test_custom_config(self):
        """Test custom configuration."""
        config = KVCacheCompressionConfig(compression_level=CompressionLevel.FAST, similarity_threshold=0.8)

        assert config.compression_level == CompressionLevel.FAST
        assert config.similarity_threshold == 0.8
