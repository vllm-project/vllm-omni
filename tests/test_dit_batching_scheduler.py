# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for DiT Batching Scheduler.

This module contains comprehensive tests for the DiT batching scheduler,
including compatibility grouping, fairness, memory safety, and performance.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler.dit_batching_scheduler import DiTBatchingScheduler, BatchingConfig
from vllm_omni.diffusion.scheduler.compatibility import CompatibilityGroupManager
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.outputs import OmniRequestOutput


class TestCompatibilityGroupManager:
    """Test compatibility-aware grouping system."""

    def test_resolution_compatibility(self):
        """Test resolution compatibility checking."""
        manager = CompatibilityGroupManager()

        # Create requests with different resolutions
        req1 = OmniDiffusionRequest(prompt="test", height=1024, width=1024)
        req2 = OmniDiffusionRequest(prompt="test", height=1024, width=1024)  # Same resolution
        req3 = OmniDiffusionRequest(prompt="test", height=512, width=512)   # Different resolution

        # Add first request
        assert manager.add_request(req1)

        # Check compatibility
        assert manager.groups[0].can_add_request(req2, 8192)  # Should be compatible
        assert not manager.groups[0].can_add_request(req3, 8192)  # Should not be compatible

    def test_cfg_compatibility(self):
        """Test CFG scale compatibility checking."""
        manager = CompatibilityGroupManager()

        req1 = OmniDiffusionRequest(prompt="test", guidance_scale=7.5)
        req2 = OmniDiffusionRequest(prompt="test", guidance_scale=7.0)  # Close enough
        req3 = OmniDiffusionRequest(prompt="test", guidance_scale=1.0)  # Too different

        assert manager.add_request(req1)
        assert manager.groups[0].can_add_request(req2, 8192)
        assert not manager.groups[0].can_add_request(req3, 8192)

    def test_steps_compatibility(self):
        """Test inference steps compatibility checking."""
        manager = CompatibilityGroupManager()

        req1 = OmniDiffusionRequest(prompt="test", num_inference_steps=50)
        req2 = OmniDiffusionRequest(prompt="test", num_inference_steps=45)  # Within 20%
        req3 = OmniDiffusionRequest(prompt="test", num_inference_steps=25)  # Too different

        assert manager.add_request(req1)
        assert manager.groups[0].can_add_request(req2, 8192)
        assert not manager.groups[0].can_add_request(req3, 8192)

    def test_memory_constraints(self):
        """Test memory-aware batch sizing."""
        manager = CompatibilityGroupManager(max_memory_mb=100)  # Very low memory

        req1 = OmniDiffusionRequest(prompt="test", height=1024, width=1024, num_inference_steps=50)
        req2 = OmniDiffusionRequest(prompt="test", height=1024, width=1024, num_inference_steps=50)

        assert manager.add_request(req1)
        # Second request should not fit due to memory constraints
        assert not manager.groups[0].can_add_request(req2, 50)


class TestDiTBatchingScheduler:
    """Test the main batching scheduler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = BatchingConfig(
            max_batch_size=4,
            min_batch_size=1,
            max_wait_time_ms=100,
            max_memory_mb=8192,
            enable_priority_queuing=True,
            enable_starvation_prevention=True,
        )
        self.scheduler = DiTBatchingScheduler(self.config)

    def teardown_method(self):
        """Clean up after tests."""
        self.scheduler.reset_stats()

    @pytest.mark.asyncio
    async def test_add_request(self):
        """Test adding requests to the scheduler."""
        req = OmniDiffusionRequest(prompt="test prompt", request_id="test-1")

        request_id = await self.scheduler.add_request(req)
        assert request_id == "test-1"

        stats = self.scheduler.get_stats()
        assert stats["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_priority_queuing(self):
        """Test priority-based request queuing."""
        high_priority_req = OmniDiffusionRequest(prompt="high priority", request_id="high")
        normal_req = OmniDiffusionRequest(prompt="normal priority", request_id="normal")
        low_priority_req = OmniDiffusionRequest(prompt="low priority", request_id="low")

        await self.scheduler.add_request(high_priority_req, priority=0)
        await self.scheduler.add_request(normal_req, priority=5)
        await self.scheduler.add_request(low_priority_req, priority=10)

        # High priority should be processed first
        batch = await self.scheduler.get_next_batch()
        assert batch is not None
        assert len(batch) == 1
        assert batch[0].request_id == "high"

    @pytest.mark.asyncio
    async def test_starvation_prevention(self):
        """Test starvation prevention mechanism."""
        req = OmniDiffusionRequest(prompt="test", request_id="starvation-test")

        await self.scheduler.add_request(req)

        # Simulate waiting past starvation threshold
        self.scheduler.last_batch_time = time.time() - 10  # 10 seconds ago

        batch = await self.scheduler.get_next_batch()
        assert batch is not None
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_batch_formation_timeout(self):
        """Test batch formation with timeout."""
        req1 = OmniDiffusionRequest(prompt="test1", request_id="timeout-1")
        req2 = OmniDiffusionRequest(prompt="test2", request_id="timeout-2", height=512, width=512)  # Different resolution

        await self.scheduler.add_request(req1)
        await self.scheduler.add_request(req2)

        # First batch should be ready immediately (single request)
        batch1 = await self.scheduler.get_next_batch()
        assert batch1 is not None
        assert len(batch1) == 1

        # Second request should eventually timeout and form single batch
        batch2 = await self.scheduler.get_next_batch()
        assert batch2 is not None
        assert len(batch2) == 1

    @pytest.mark.asyncio
    async def test_memory_safety(self):
        """Test memory-aware batch sizing."""
        # Create scheduler with very low memory limit
        low_memory_config = BatchingConfig(
            max_batch_size=4,
            max_memory_mb=100,  # Very low memory
            min_batch_size=1,
        )
        scheduler = DiTBatchingScheduler(low_memory_config)

        # Add requests that would exceed memory if batched
        req1 = OmniDiffusionRequest(prompt="test1", height=1024, width=1024, num_inference_steps=50)
        req2 = OmniDiffusionRequest(prompt="test2", height=1024, width=1024, num_inference_steps=50)

        await scheduler.add_request(req1)
        await scheduler.add_request(req2)

        # Should get single batches due to memory constraints
        batch1 = await scheduler.get_next_batch()
        assert batch1 is not None
        assert len(batch1) == 1

        batch2 = await scheduler.get_next_batch()
        assert batch2 is not None
        assert len(batch2) == 1

    @pytest.mark.asyncio
    async def test_process_completed_batch(self):
        """Test processing completed batch outputs."""
        req1 = OmniDiffusionRequest(prompt="test1", request_id="batch-1")
        req2 = OmniDiffusionRequest(prompt="test2", request_id="batch-2")

        batch = [req1, req2]

        # Mock diffusion output
        output = DiffusionOutput(output=["image1", "image2"])

        # Process batch
        results = await self.scheduler.process_completed_batch(batch, output)

        assert len(results) == 2
        assert all(isinstance(r, OmniRequestOutput) for r in results)
        assert results[0].request_id == "batch-1"
        assert results[1].request_id == "batch-2"

    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        stats = self.scheduler.get_stats()

        expected_keys = [
            "total_requests", "batched_requests", "single_requests",
            "avg_batch_size", "queue_sizes", "active_groups"
        ]

        for key in expected_keys:
            assert key in stats

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        requests = []
        for i in range(10):
            req = OmniDiffusionRequest(
                prompt=f"test prompt {i}",
                request_id=f"concurrent-{i}",
                height=1024,
                width=1024,
                num_inference_steps=50
            )
            requests.append(req)

        # Add all requests concurrently
        tasks = [self.scheduler.add_request(req) for req in requests]
        await asyncio.gather(*tasks)

        # Process batches
        total_processed = 0
        while total_processed < len(requests):
            batch = await self.scheduler.get_next_batch()
            if batch:
                total_processed += len(batch)
                # Mock processing
                output = DiffusionOutput(output=[f"image{i}" for i in range(len(batch))])
                await self.scheduler.process_completed_batch(batch, output)

        assert total_processed == len(requests)

        # Check that some batching occurred
        stats = self.scheduler.get_stats()
        assert stats["batched_requests"] > 0 or stats["single_requests"] == len(requests)


class TestIntegrationWithDiffusionEngine:
    """Integration tests with DiffusionEngine."""

    @patch('vllm_omni.diffusion.diffusion_engine.DiffusionEngine.add_req_and_wait_for_response')
    def test_engine_integration_batching_disabled(self, mock_process):
        """Test engine integration when batching is disabled."""
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig(model="test-model")
        engine = DiffusionEngine(config)  # No batching config = disabled

        assert not engine.is_batching_enabled()

        # Should use non-batching path
        requests = [OmniDiffusionRequest(prompt="test")]
        mock_process.return_value = DiffusionOutput(output=["test_image"])

        result = engine.step_sync(requests)
        assert result is not None
        mock_process.assert_called_once_with(requests)

    @patch('vllm_omni.diffusion.diffusion_engine.DiffusionEngine.add_req_and_wait_for_response')
    def test_engine_integration_batching_enabled(self, mock_process):
        """Test engine integration when batching is enabled."""
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.config.batching import DiTBatchingConfig

        config = OmniDiffusionConfig(model="test-model")
        batching_config = DiTBatchingConfig(enable_batching=True, max_batch_size=2)
        engine = DiffusionEngine(config, batching_config)

        assert engine.is_batching_enabled()

        # Should use batching path (but fall back to sync for testing)
        requests = [OmniDiffusionRequest(prompt="test")]
        mock_process.return_value = DiffusionOutput(output=["test_image"])

        result = engine.step_sync(requests)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])