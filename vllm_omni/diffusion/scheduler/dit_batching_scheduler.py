# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
DiT Batching Scheduler for vLLM-Omni.

This module implements dynamic request-level batching for diffusion workloads,
inspired by vLLM's LLMEngine + Scheduler architecture for autoregressive models.
"""

import asyncio
import time
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Deque
import heapq

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler.compatibility import CompatibilityGroupManager, CompatibilityGroup
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.outputs import OmniRequestOutput

logger = logging.getLogger(__name__)


@dataclass
class BatchingConfig:
    """Configuration for DiT batching behavior."""
    max_batch_size: int = 8
    min_batch_size: int = 1
    max_wait_time_ms: int = 50  # Maximum time to wait for batch formation
    max_memory_mb: float = 8192  # Maximum memory per batch
    enable_priority_queuing: bool = True
    enable_starvation_prevention: bool = True
    batch_timeout_strategy: str = "adaptive"  # "adaptive", "fixed", "aggressive"


@dataclass
class PendingRequest:
    """Wrapper for a pending diffusion request with metadata."""
    request: OmniDiffusionRequest
    arrival_time: float
    priority: int = 0  # Lower numbers = higher priority
    timeout: Optional[float] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.arrival_time < other.arrival_time


class DiTBatchingScheduler:
    """
    Dynamic batching scheduler for DiT diffusion workloads.
    
    Inspired by vLLM's LLMEngine + Scheduler, this scheduler enables:
    - Dynamic request-level batching
    - Compatibility-aware grouping
    - Fair scheduling with priority queuing
    - Memory-aware batch sizing
    - Low-latency optimization
    """
    
    def __init__(self, config: BatchingConfig):
        self.config = config
        self.group_manager = CompatibilityGroupManager(
            max_batch_size=config.max_batch_size,
            max_memory_mb=config.max_memory_mb
        )
        
        # Queues for different priority levels
        self.high_priority_queue: Deque[PendingRequest] = deque()
        self.normal_priority_queue: Deque[PendingRequest] = deque()
        self.low_priority_queue: Deque[PendingRequest] = deque()
        
        # Statistics and monitoring
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "single_requests": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
            "memory_efficiency": 0.0
        }
        
        # Starvation prevention
        self.last_batch_time = time.time()
        self.starvation_threshold = 5.0  # seconds
        
        # Adaptive timeout tracking
        self.adaptive_timeout = config.max_wait_time_ms / 1000.0
        self.performance_history = deque(maxlen=100)
    
    async def add_request(self, request: OmniDiffusionRequest, 
                         priority: int = 0) -> str:
        """Add a request to the scheduler."""
        self.stats["total_requests"] += 1
        
        pending_req = PendingRequest(
            request=request,
            arrival_time=time.time(),
            priority=priority
        )
        
        # Add to appropriate queue based on priority
        if priority <= 0:
            self.high_priority_queue.append(pending_req)
        elif priority <= 5:
            self.normal_priority_queue.append(pending_req)
        else:
            self.low_priority_queue.append(pending_req)
        
        # Add to compatibility groups
        self.group_manager.add_request(request)
        
        logger.debug(f"Added request {request.request_id} with priority {priority}")
        return request.request_id
    
    async def get_next_batch(self) -> Optional[List[OmniDiffusionRequest]]:
        """
        Get the next batch of requests ready for processing.
        
        Returns:
            List of requests to batch together, or None if no batch ready.
        """
        current_time = time.time()
        
        # Check for starvation and adjust strategy
        if self._should_prevent_starvation(current_time):
            batch = await self._get_starvation_batch()
            if batch:
                return batch
        
        # Get ready groups from compatibility manager
        ready_groups = self.group_manager.get_ready_groups(self.config.min_batch_size)
        
        if ready_groups:
            # Select the best group based on size and efficiency
            best_group = self._select_best_group(ready_groups, current_time)
            if best_group:
                self.group_manager.remove_group(best_group)
                self.stats["batched_requests"] += len(best_group.requests)
                self.last_batch_time = current_time
                
                logger.info(f"Selected batch of {len(best_group.requests)} requests")
                return best_group.requests
        
        # Check if we should form a smaller batch due to time constraints
        if self._should_form_small_batch(current_time):
            batch = await self._form_small_batch()
            if batch:
                self.stats["single_requests"] += len(batch)
                self.last_batch_time = current_time
                return batch
        
        return None
    
    async def process_completed_batch(self, batch: List[OmniDiffusionRequest], 
                                     output: DiffusionOutput) -> List[OmniRequestOutput]:
        """
        Process the output of a completed batch.
        
        Args:
            batch: The batch that was processed
            output: The diffusion output for the batch
            
        Returns:
            List of individual request outputs
        """
        # Update performance metrics
        batch_size = len(batch)
        self._update_performance_metrics(batch_size)
        
        # Convert batch output to individual outputs
        if batch_size == 1:
            # Single request - convert directly
            request = batch[0]
            return [self._convert_to_output(request, output)]
        else:
            # Multiple requests - need to split output
            return self._split_batch_output(batch, output)
    
    def _select_best_group(self, ready_groups: List[CompatibilityGroup], 
                          current_time: float) -> Optional[CompatibilityGroup]:
        """Select the best group from ready groups based on various criteria."""
        if not ready_groups:
            return None
        
        # Score groups based on multiple factors
        best_group = None
        best_score = float('-inf')
        
        for group in ready_groups:
            score = self._calculate_group_score(group, current_time)
            if score > best_score:
                best_score = score
                best_group = group
        
        return best_group
    
    def _calculate_group_score(self, group: CompatibilityGroup, 
                              current_time: float) -> float:
        """Calculate a score for a group to determine processing priority."""
        # Base score: larger batches are preferred (higher throughput)
        base_score = len(group.requests) * 10
        
        # Age penalty: older groups get higher priority
        oldest_request = min(group.requests, key=lambda r: getattr(r, 'arrival_time', current_time))
        age = current_time - getattr(oldest_request, 'arrival_time', current_time)
        age_bonus = min(age * 5, 50)  # Cap age bonus
        
        # Memory efficiency bonus
        memory_efficiency = min(group.estimated_memory_mb / self.config.max_memory_mb, 1.0)
        efficiency_bonus = (1.0 - memory_efficiency) * 20
        
        # Compatibility strength bonus
        compatibility_bonus = len(group.requests) * 2
        
        return base_score + age_bonus + efficiency_bonus + compatibility_bonus
    
    def _should_prevent_starvation(self, current_time: float) -> bool:
        """Check if starvation prevention should be triggered."""
        if not self.config.enable_starvation_prevention:
            return False
        
        time_since_last_batch = current_time - self.last_batch_time
        return time_since_last_batch > self.starvation_threshold
    
    async def _get_starvation_batch(self) -> Optional[List[OmniDiffusionRequest]]:
        """Get a batch to prevent starvation of waiting requests."""
        # Get the oldest request from any queue
        all_queues = [self.high_priority_queue, self.normal_priority_queue, self.low_priority_queue]
        
        oldest_request = None
        for queue in all_queues:
            if queue:
                req = queue[0]  # Oldest in FIFO
                if oldest_request is None or req.arrival_time < oldest_request.arrival_time:
                    oldest_request = req
        
        if oldest_request:
            # Remove from queue and return as single request batch
            for queue in all_queues:
                if queue and queue[0] == oldest_request:
                    queue.popleft()
                    break
            
            logger.info("Starvation prevention: processing single request")
            return [oldest_request.request]
        
        return None
    
    def _should_form_small_batch(self, current_time: float) -> bool:
        """Determine if we should form a smaller batch due to time constraints."""
        if not self.high_priority_queue and not self.normal_priority_queue:
            return False
        
        # Check the oldest request's wait time
        oldest_req = None
        for queue in [self.high_priority_queue, self.normal_priority_queue]:
            if queue:
                req = queue[0]
                if oldest_req is None or req.arrival_time < oldest_req.arrival_time:
                    oldest_req = req
        
        if not oldest_req:
            return False
        
        wait_time = (current_req.arrival_time - oldest_req.arrival_time) * 1000  # ms
        
        # Use adaptive timeout strategy
        if self.config.batch_timeout_strategy == "adaptive":
            target_timeout = self.adaptive_timeout * 1000
        elif self.config.batch_timeout_strategy == "aggressive":
            target_timeout = self.config.max_wait_time_ms * 0.5
        else:  # fixed
            target_timeout = self.config.max_wait_time_ms
        
        return wait_time > target_timeout
    
    async def _form_small_batch(self) -> Optional[List[OmniDiffusionRequest]]:
        """Form a smaller batch when time constraints require it."""
        # Try to get at least one request from high priority queue
        if self.high_priority_queue:
            req = self.high_priority_queue.popleft()
            return [req.request]
        
        # Fallback to normal priority
        if self.normal_priority_queue:
            req = self.normal_priority_queue.popleft()
            return [req.request]
        
        return None
    
    def _update_performance_metrics(self, batch_size: int):
        """Update performance tracking metrics."""
        # Update average batch size
        total_processed = self.stats["batched_requests"] + self.stats["single_requests"]
        if total_processed > 0:
            current_avg = self.stats["avg_batch_size"]
            self.stats["avg_batch_size"] = (current_avg * (total_processed - 1) + batch_size) / total_processed
    
    def _convert_to_output(self, request: OmniDiffusionRequest, 
                          output: DiffusionOutput) -> OmniRequestOutput:
        """Convert diffusion output for a single request."""
        # This would implement the conversion logic similar to DiffusionEngine.step()
        # For now, return a placeholder
        return OmniRequestOutput.from_diffusion(
            request_id=request.request_id or "",
            images=[],  # Would extract from output
            prompt=request.prompt,
            metrics={},
            latents=None
        )
    
    def _split_batch_output(self, batch: List[OmniDiffusionRequest], 
                           output: DiffusionOutput) -> List[OmniRequestOutput]:
        """Split batch output into individual request outputs."""
        # This would implement the logic from DiffusionEngine.step() for multiple requests
        results = []
        # Implementation would split output.images based on num_outputs_per_prompt for each request
        for request in batch:
            results.append(self._convert_to_output(request, output))
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current scheduler statistics."""
        stats = self.stats.copy()
        stats["queue_sizes"] = {
            "high_priority": len(self.high_priority_queue),
            "normal_priority": len(self.normal_priority_queue),
            "low_priority": len(self.low_priority_queue)
        }
        stats["active_groups"] = len(self.group_manager.groups)
        stats["adaptive_timeout_ms"] = self.adaptive_timeout * 1000
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "single_requests": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
            "memory_efficiency": 0.0
        }
        self.performance_history.clear()