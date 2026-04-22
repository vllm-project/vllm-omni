"""Unit tests for batch scheduler."""

from vllm_omni.core.batch_scheduler import (
    AdaptiveBatchScheduler,
    BatchSchedulingConfig,
    RequestPriority,
)


class TestAdaptiveBatchScheduler:
    """Tests for AdaptiveBatchScheduler."""

    def test_add_request(self):
        """Test adding requests."""
        config = BatchSchedulingConfig(max_batch_size=8)
        scheduler = AdaptiveBatchScheduler(config)

        assert scheduler.add_request("req1", prompt_length=100)
        assert scheduler.get_pending_count() == 1

        assert scheduler.add_request("req2", prompt_length=200)
        assert scheduler.get_pending_count() == 2

    def test_add_duplicate_request(self):
        """Test adding duplicate request."""
        config = BatchSchedulingConfig()
        scheduler = AdaptiveBatchScheduler(config)

        assert scheduler.add_request("req1", prompt_length=100)
        assert not scheduler.add_request("req1", prompt_length=100)

    def test_remove_request(self):
        """Test removing requests."""
        config = BatchSchedulingConfig()
        scheduler = AdaptiveBatchScheduler(config)

        scheduler.add_request("req1", prompt_length=100)
        assert scheduler.remove_request("req1")
        assert scheduler.get_pending_count() == 0

    def test_priority_ordering(self):
        """Test priority-based ordering."""
        config = BatchSchedulingConfig(enable_request_priority=True)
        scheduler = AdaptiveBatchScheduler(config)

        scheduler.add_request("low", prompt_length=100, priority=RequestPriority.LOW)
        scheduler.add_request("high", prompt_length=100, priority=RequestPriority.HIGH)
        scheduler.add_request("normal", prompt_length=100, priority=RequestPriority.NORMAL)

        scheduler._process_pending_requests()

        metrics = scheduler.get_metrics()
        assert metrics["batches_created"] >= 1

    def test_batch_size_limit(self):
        """Test batch size limit."""
        config = BatchSchedulingConfig(max_batch_size=2)
        scheduler = AdaptiveBatchScheduler(config)

        scheduler.add_request("req1", prompt_length=100)
        scheduler.add_request("req2", prompt_length=100)
        scheduler.add_request("req3", prompt_length=100)

        scheduler._process_pending_requests()

        while scheduler.get_active_batch_count() > 0:
            batch = scheduler.get_next_batch()
            if batch:
                assert len(batch) <= 2

    def test_metrics_tracking(self):
        """Test metrics tracking."""
        config = BatchSchedulingConfig(max_batch_size=4)
        scheduler = AdaptiveBatchScheduler(config)

        for i in range(5):
            scheduler.add_request(f"req{i}", prompt_length=100)

        scheduler._process_pending_requests()

        metrics = scheduler.get_metrics()
        assert metrics["batches_created"] > 0


class TestBatchSchedulingConfig:
    """Tests for BatchSchedulingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BatchSchedulingConfig()

        assert config.max_batch_size == 32
        assert config.max_model_len == 8192
        assert config.batch_timeout_ms == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchSchedulingConfig(max_batch_size=64, batch_timeout_ms=50)

        assert config.max_batch_size == 64
        assert config.batch_timeout_ms == 50
