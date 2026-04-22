"""Unit tests for GPU memory pool."""

from vllm_omni.core.gpu_memory_pool import (
    GPUMemoryPool,
    GPUMemoryPoolConfig,
)


class TestGPUMemoryPool:
    """Tests for GPUMemoryPool."""

    def test_allocate(self):
        """Test memory allocation."""
        config = GPUMemoryPoolConfig(total_size=1024 * 1024, block_size_granularity=512 * 1024)
        pool = GPUMemoryPool(config)

        result = pool.allocate(size=256 * 1024, request_id="req1")

        assert result is not None
        block_id, offset, size = result
        assert size == 512 * 1024  # Aligned

    def test_free(self):
        """Test memory free."""
        config = GPUMemoryPoolConfig(total_size=1024 * 1024, block_size_granularity=512 * 1024)
        pool = GPUMemoryPool(config)

        pool.allocate(size=256 * 1024, request_id="req1")
        assert pool.free("req1")

    def test_allocate_no_space(self):
        """Test allocation with no space."""
        config = GPUMemoryPoolConfig(total_size=512 * 1024, block_size_granularity=512 * 1024)
        pool = GPUMemoryPool(config)

        result1 = pool.allocate(size=256 * 1024, request_id="req1")
        result2 = pool.allocate(size=256 * 1024, request_id="req2")

        assert result1 is not None
        assert result2 is not None

        result3 = pool.allocate(size=256 * 1024, request_id="req3")
        assert result3 is None

    def test_metrics(self):
        """Test metrics tracking."""
        config = GPUMemoryPoolConfig(total_size=1024 * 1024, block_size_granularity=512 * 1024)
        pool = GPUMemoryPool(config)

        pool.allocate(size=256 * 1024, request_id="req1")
        pool.free("req1")

        metrics = pool.get_metrics()
        assert metrics["total_allocations"] == 1
        assert metrics["total_frees"] == 1


class TestGPUMemoryPoolConfig:
    """Tests for GPUMemoryPoolConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GPUMemoryPoolConfig()

        assert config.total_size == 8 * 1024 * 1024 * 1024
        assert config.block_size_granularity == 1024 * 1024

    def test_custom_config(self):
        """Test custom configuration."""
        config = GPUMemoryPoolConfig(total_size=4 * 1024 * 1024 * 1024, block_size_granularity=2 * 1024 * 1024)

        assert config.total_size == 4 * 1024 * 1024 * 1024
        assert config.block_size_granularity == 2 * 1024 * 1024
