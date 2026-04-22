"""Unit tests for CUDA graph optimizer."""

from vllm_omni.core.cuda_graph_optimizer import (
    CUDAGraphConfig,
    CUDAGraphOptimizer,
    GraphCaptureMode,
)


class TestCUDAGraphOptimizer:
    """Tests for CUDAGraphOptimizer."""

    def test_enabled_by_default(self):
        """Test optimizer is enabled by default."""
        config = CUDAGraphConfig(enable_cuda_graph=True)
        optimizer = CUDAGraphOptimizer(config)

        assert optimizer.is_enabled is True

    def test_disabled(self):
        """Test optimizer can be disabled."""
        config = CUDAGraphConfig(enable_cuda_graph=False)
        optimizer = CUDAGraphOptimizer(config)

        assert optimizer.is_enabled is False

    def test_should_use_graph_in_range(self):
        """Test graph usage within batch size range."""
        config = CUDAGraphConfig(enable_cuda_graph=True, min_batch_size=1, max_batch_size=32)
        optimizer = CUDAGraphOptimizer(config)

        assert optimizer.should_use_graph(8) is True
        assert optimizer.should_use_graph(16) is True

    def test_should_not_use_graph_outside_range(self):
        """Test graph not used outside batch size range."""
        config = CUDAGraphConfig(enable_cuda_graph=True, min_batch_size=4, max_batch_size=16)
        optimizer = CUDAGraphOptimizer(config)

        assert optimizer.should_use_graph(2) is False
        assert optimizer.should_use_graph(32) is False

    def test_should_use_graph_disabled(self):
        """Test graph not used when disabled."""
        config = CUDAGraphConfig(enable_cuda_graph=False)
        optimizer = CUDAGraphOptimizer(config)

        assert optimizer.should_use_graph(8) is False

    def test_get_existing_graph(self):
        """Test getting existing graph."""
        config = CUDAGraphConfig(max_graphs=32)
        optimizer = CUDAGraphOptimizer(config)

        optimizer._graphs[8] = "mock_graph"
        graph = optimizer.get_or_create_graph(8)

        assert graph == "mock_graph"

    def test_register_graph(self):
        """Test graph registration."""
        config = CUDAGraphConfig()
        optimizer = CUDAGraphOptimizer(config)

        optimizer.register_graph(8, "mock_graph")

        assert 8 in optimizer._graphs
        assert optimizer._metrics.graphs_captured == 1

    def test_clear_graphs(self):
        """Test clearing graphs."""
        config = CUDAGraphConfig()
        optimizer = CUDAGraphOptimizer(config)

        optimizer._graphs[8] = "graph1"
        optimizer._graphs[16] = "graph2"
        optimizer.clear_graphs()

        assert len(optimizer._graphs) == 0

    def test_metrics(self):
        """Test metrics tracking."""
        config = CUDAGraphConfig()
        optimizer = CUDAGraphOptimizer(config)

        optimizer._metrics.graphs_captured = 5
        optimizer._metrics.graphs_launched = 10

        metrics = optimizer.get_metrics()

        assert metrics["graphs_captured"] == 5
        assert metrics["graphs_launched"] == 10
        assert metrics["cached_graphs"] == 0


class TestCUDAGraphConfig:
    """Tests for CUDAGraphConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CUDAGraphConfig()

        assert config.enable_cuda_graph is True
        assert config.capture_mode == GraphCaptureMode.ADAPTIVE
        assert config.max_graphs == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = CUDAGraphConfig(capture_mode=GraphCaptureMode.EAGER, max_graphs=16)

        assert config.capture_mode == GraphCaptureMode.EAGER
        assert config.max_graphs == 16
