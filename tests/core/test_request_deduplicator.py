"""Unit tests for request deduplicator."""

import time

from vllm_omni.core.request_deduplicator import (
    DeduplicationConfig,
    RequestDeduplicator,
)


class TestRequestDeduplicator:
    """Tests for RequestDeduplicator."""

    def test_new_request(self):
        """Test new request registration."""
        config = DeduplicationConfig()
        dedup = RequestDeduplicator(config)

        result = dedup.check_and_register(request_id="req1", prompt="What is AI?")

        assert result is None
        metrics = dedup.get_metrics()
        assert metrics["cache_size"] == 1

    def test_duplicate_detected(self):
        """Test duplicate detection."""
        config = DeduplicationConfig(cache_ttl_seconds=60.0)
        dedup = RequestDeduplicator(config)

        dedup.check_and_register(request_id="req1", prompt="What is AI?")
        result = dedup.check_and_register(request_id="req2", prompt="What is AI?")

        assert result == "req1"
        metrics = dedup.get_metrics()
        assert metrics["duplicates_found"] == 1

    def test_no_duplicate_different_prompt(self):
        """Test no duplicate for different prompt."""
        config = DeduplicationConfig()
        dedup = RequestDeduplicator(config)

        dedup.check_and_register(request_id="req1", prompt="What is AI?")
        result = dedup.check_and_register(request_id="req2", prompt="What is ML?")

        assert result is None

    def test_ttl_expiry(self):
        """Test TTL expiry."""
        config = DeduplicationConfig(cache_ttl_seconds=0.1)
        dedup = RequestDeduplicator(config)

        dedup.check_and_register(request_id="req1", prompt="Test")
        time.sleep(0.2)

        result = dedup.check_and_register(request_id="req2", prompt="Test")
        assert result is None

    def test_unregister(self):
        """Test request unregistration."""
        config = DeduplicationConfig()
        dedup = RequestDeduplicator(config)

        dedup.check_and_register(request_id="req1", prompt="Test")
        dedup.unregister("req1")

        metrics = dedup.get_metrics()
        assert metrics["cache_size"] == 0


class TestDeduplicationConfig:
    """Tests for DeduplicationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeduplicationConfig()

        assert config.enable_content_dedup is True
        assert config.max_cache_size == 1024
        assert config.cache_ttl_seconds == 60.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeduplicationConfig(max_cache_size=512, cache_ttl_seconds=30.0)

        assert config.max_cache_size == 512
        assert config.cache_ttl_seconds == 30.0
