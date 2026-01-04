# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Dynamic Batch Handler for GPU Workers.

This module handles dynamic batching of diffusion requests at the worker level,
enabling efficient GPU utilization while maintaining individual request semantics.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.data import DiffusionOutput

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for a processed batch."""
    batch_size: int
    processing_time: float
    memory_used_mb: float
    throughput_improvement: float = 0.0


class DynamicBatchHandler:
    """
    Handles dynamic batching of diffusion requests at the worker level.
    
    This class manages the actual execution of batched requests while
    preserving individual request semantics and results.
    """
    
    def __init__(self, device: torch.device, max_memory_gb: float = 8.0):
        self.device = device
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_memory_usage = 0
        self.batch_history = []
        
    def execute_batch(self, requests: List[OmniDiffusionRequest], 
                     pipeline) -> DiffusionOutput:
        """
        Execute a batch of requests using the provided pipeline.
        
        Args:
            requests: List of requests to batch
            pipeline: The diffusion pipeline to use
            
        Returns:
            DiffusionOutput containing batch results
        """
        if not requests:
            raise ValueError("Cannot execute empty batch")
        
        if len(requests) == 1:
            return self._execute_single_request(requests[0], pipeline)
        
        # For multiple requests, we need to handle batching carefully
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated(self.device)
        
        try:
            # Check if pipeline supports native batching
            if hasattr(pipeline, 'forward_batch') and self._can_batch_requests(requests):
                output = self._execute_native_batch(requests, pipeline)
            else:
                # Fall back to sequential execution with result aggregation
                output = self._execute_sequential_batch(requests, pipeline)
            
            processing_time = time.time() - start_time
            memory_after = torch.cuda.memory_allocated(self.device)
            memory_used = memory_after - memory_before
            
            # Record metrics
            metrics = BatchMetrics(
                batch_size=len(requests),
                processing_time=processing_time,
                memory_used_mb=memory_used / (1024 * 1024)
            )
            self._update_metrics(metrics)
            
            return output
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            # Fallback to individual execution
            return self._execute_fallback_batch(requests, pipeline)
    
    def _execute_single_request(self, request: OmniDiffusionRequest, 
                               pipeline) -> DiffusionOutput:
        """Execute a single request."""
        return pipeline.forward(request)
    
    def _can_batch_requests(self, requests: List[OmniDiffusionRequest]) -> bool:
        """Check if requests can be batched together."""
        if len(requests) <= 1:
            return True
        
        # Check compatibility for batching
        ref_req = requests[0]
        
        # Resolution compatibility
        ref_resolution = self._get_resolution(ref_req)
        for req in requests[1:]:
            if abs(self._get_resolution(req) - ref_resolution) / ref_resolution > 0.1:
                return False
        
        # Steps compatibility
        ref_steps = ref_req.num_inference_steps or 50
        for req in requests[1:]:
            steps = req.num_inference_steps or 50
            if abs(steps - ref_steps) / ref_steps > 0.2:
                return False
        
        # CFG compatibility
        ref_cfg = ref_req.guidance_scale or 1.0
        for req in requests[1:]:
            cfg = req.guidance_scale or 1.0
            if cfg != ref_cfg:
                return False
        
        return True
    
    def _get_resolution(self, request: OmniDiffusionRequest) -> int:
        """Get effective resolution for a request."""
        if request.height and request.width:
            return request.height * request.width
        elif request.resolution:
            return request.resolution * request.resolution
        else:
            return 1024 * 1024  # Default
    
    def _execute_native_batch(self, requests: List[OmniDiffusionRequest], 
                             pipeline) -> DiffusionOutput:
        """Execute requests using native pipeline batching."""
        # Prepare batch inputs
        batch_prompts = []
        batch_configs = []
        
        for req in requests:
            if isinstance(req.prompt, list):
                batch_prompts.extend(req.prompt)
            else:
                batch_prompts.append(req.prompt)
            
            batch_configs.append({
                'height': req.height,
                'width': req.width,
                'num_inference_steps': req.num_inference_steps,
                'guidance_scale': req.guidance_scale,
                'num_outputs_per_prompt': req.num_outputs_per_prompt,
                'seed': req.seed
            })
        
        # Execute batch
        if hasattr(pipeline, 'forward_batch'):
            return pipeline.forward_batch(batch_prompts, batch_configs)
        else:
            raise NotImplementedError("Pipeline does not support native batching")
    
    def _execute_sequential_batch(self, requests: List[OmniDiffusionRequest], 
                                 pipeline) -> DiffusionOutput:
        """Execute requests sequentially and aggregate results."""
        results = []
        all_outputs = []
        
        for i, request in enumerate(requests):
            try:
                output = pipeline.forward(request)
                all_outputs.append(output)
                
                # Extract images from output
                if output.output is not None:
                    if isinstance(output.output, list):
                        results.extend(output.output)
                    else:
                        results.append(output.output)
                
            except Exception as e:
                logger.error(f"Request {i} failed: {e}")
                # Add placeholder for failed request
                results.append(None)
        
        # Combine outputs
        return DiffusionOutput(
            output=results,
            trajectory_latents=all_outputs[0].trajectory_latents if all_outputs else None,
            trajectory_timesteps=all_outputs[0].trajectory_timesteps if all_outputs else None
        )
    
    def _execute_fallback_batch(self, requests: List[OmniDiffusionRequest], 
                               pipeline) -> DiffusionOutput:
        """Fallback execution when batch processing fails."""
        logger.warning("Falling back to individual request execution")
        
        results = []
        for request in requests:
            try:
                output = pipeline.forward(request)
                if output.output is not None:
                    if isinstance(output.output, list):
                        results.extend(output.output)
                    else:
                        results.append(output.output)
            except Exception as e:
                logger.error(f"Fallback execution failed for request: {e}")
                results.append(None)
        
        return DiffusionOutput(output=results)
    
    def _update_metrics(self, metrics: BatchMetrics):
        """Update performance metrics."""
        self.batch_history.append(metrics)
        
        # Keep only recent history
        if len(self.batch_history) > 1000:
            self.batch_history = self.batch_history[-500:]
        
        # Calculate throughput improvement
        if len(self.batch_history) >= 2:
            recent_batches = self.batch_history[-10:]
            single_request_times = [m.processing_time for m in recent_batches if m.batch_size == 1]
            batch_times = [m.processing_time / m.batch_size for m in recent_batches if m.batch_size > 1]
            
            if single_request_times and batch_times:
                avg_single = sum(single_request_times) / len(single_request_times)
                avg_batch_per_req = sum(batch_times) / len(batch_times)
                metrics.throughput_improvement = (avg_single - avg_batch_per_req) / avg_single * 100
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this handler."""
        if not self.batch_history:
            return {"status": "no_data"}
        
        batch_sizes = [m.batch_size for m in self.batch_history]
        processing_times = [m.processing_time for m in self.batch_history]
        memory_usage = [m.memory_used_mb for m in self.batch_history]
        throughput_improvements = [m.throughput_improvement for m in self.batch_history if m.throughput_improvement > 0]
        
        stats = {
            "total_batches": len(self.batch_history),
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
            "max_batch_size": max(batch_sizes),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "max_processing_time": max(processing_times),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "max_memory_mb": max(memory_usage),
            "batching_efficiency": len([s for s in batch_sizes if s > 1]) / len(batch_sizes) * 100
        }
        
        if throughput_improvements:
            stats["avg_throughput_improvement"] = sum(throughput_improvements) / len(throughput_improvements)
            stats["max_throughput_improvement"] = max(throughput_improvements)
        
        return stats
    
    def estimate_memory_requirements(self, requests: List[OmniDiffusionRequest]) -> float:
        """Estimate memory requirements for a batch of requests."""
        if not requests:
            return 0.0
        
        # Base memory per request
        base_memory_mb = 50.0  # Conservative estimate
        
        # Resolution scaling
        ref_resolution = self._get_resolution(requests[0])
        resolution_factor = ref_resolution / (1024 * 1024)  # Normalize to 1MP
        
        # Steps scaling
        steps = requests[0].num_inference_steps or 50
        steps_factor = steps / 50.0  # Normalize to 50 steps
        
        # Batch size factor
        batch_factor = len(requests)
        
        # Total estimate
        estimated_mb = base_memory_mb * resolution_factor * steps_factor * batch_factor
        
        return max(estimated_mb, base_memory_mb * batch_factor)  # Minimum per-request memory
    
    def can_fit_in_memory(self, requests: List[OmniDiffusionRequest]) -> bool:
        """Check if a batch can fit in available memory."""
        estimated_mb = self.estimate_memory_requirements(requests)
        estimated_bytes = estimated_mb * 1024 * 1024
        
        # Add safety margin
        safety_margin = 0.8  # Use only 80% of available memory
        available_bytes = self.max_memory_bytes * safety_margin
        
        return estimated_bytes <= (available_bytes - self.current_memory_usage)
    
    def optimize_batch_size(self, requests: List[OmniDiffusionRequest]) -> List[OmniDiffusionRequest]:
        """Optimize batch size based on memory constraints."""
        if len(requests) <= 1:
            return requests
        
        # Try to find the largest subset that fits in memory
        best_batch = requests[:1]
        
        for i in range(2, len(requests) + 1):
            candidate_batch = requests[:i]
            if self.can_fit_in_memory(candidate_batch):
                best_batch = candidate_batch
            else:
                break
        
        if len(best_batch) < len(requests):
            logger.info(f"Optimized batch size from {len(requests)} to {len(best_batch)} due to memory constraints")
        
        return best_batch