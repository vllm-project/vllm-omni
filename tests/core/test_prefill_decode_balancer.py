"""Unit tests for prefill-decode balancer."""

from vllm_omni.core.prefill_decode_balancer import (
    PrefillDecodeBalancer,
    PrefillDecodeConfig,
    SchedulingStrategy,
)


class TestPrefillDecodeBalancer:
    """Tests for PrefillDecodeBalancer."""

    def test_add_prefill_request(self):
        """Test adding prefill request."""
        config = PrefillDecodeConfig()
        balancer = PrefillDecodeBalancer(config)

        balancer.add_request("req1", is_prefill=True)
        counts = balancer.get_pending_counts()

        assert counts["prefill"] == 1
        assert counts["decode"] == 0

    def test_add_decode_request(self):
        """Test adding decode request."""
        config = PrefillDecodeConfig()
        balancer = PrefillDecodeBalancer(config)

        balancer.add_request("req1", is_prefill=False)
        counts = balancer.get_pending_counts()

        assert counts["prefill"] == 0
        assert counts["decode"] == 1

    def test_max_throughput_strategy(self):
        """Test max throughput strategy prioritizes decode."""
        config = PrefillDecodeConfig(strategy=SchedulingStrategy.MAX_THROUGHPUT)
        balancer = PrefillDecodeBalancer(config)

        balancer.add_request("pref1", is_prefill=True)
        balancer.add_request("dec1", is_prefill=False)

        batch, is_prefill = balancer.get_next_batch()

        assert is_prefill is False
        assert "dec1" in batch

    def test_min_latency_strategy(self):
        """Test min latency strategy prioritizes prefill."""
        config = PrefillDecodeConfig(strategy=SchedulingStrategy.MIN_LATENCY)
        balancer = PrefillDecodeBalancer(config)

        balancer.add_request("pref1", is_prefill=True)
        balancer.add_request("dec1", is_prefill=False)

        batch, is_prefill = balancer.get_next_batch()

        assert is_prefill is True
        assert "pref1" in batch

    def test_balanced_strategy(self):
        """Test balanced strategy."""
        config = PrefillDecodeConfig(strategy=SchedulingStrategy.BALANCED, target_prefill_ratio=0.5)
        balancer = PrefillDecodeBalancer(config)

        for i in range(4):
            balancer.add_request(f"pref{i}", is_prefill=True)
        for i in range(4):
            balancer.add_request(f"dec{i}", is_prefill=False)

        counts = balancer.get_pending_counts()
        assert counts["prefill"] == 4
        assert counts["decode"] == 4

    def test_metrics(self):
        """Test metrics tracking."""
        config = PrefillDecodeConfig()
        balancer = PrefillDecodeBalancer(config)

        balancer.add_request("req1", is_prefill=True)
        metrics = balancer.get_metrics()

        assert "pending_prefill" in metrics
        assert "pending_decode" in metrics
        assert "current_ratio" in metrics


class TestPrefillDecodeConfig:
    """Tests for PrefillDecodeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PrefillDecodeConfig()

        assert config.strategy == SchedulingStrategy.BALANCED
        assert config.max_prefill_batch_size == 8
        assert config.max_decode_batch_size == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = PrefillDecodeConfig(strategy=SchedulingStrategy.MAX_THROUGHPUT, target_prefill_ratio=0.5)

        assert config.strategy == SchedulingStrategy.MAX_THROUGHPUT
        assert config.target_prefill_ratio == 0.5
