# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compatibility-aware grouping system for DiT batching.

This module provides the logic to determine which diffusion requests can be
batched together based on critical compatibility parameters.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityGroup:
    """Group of requests that can be batched together."""
    requests: List[OmniDiffusionRequest]
    compatibility_key: str
    max_batch_size: int
    estimated_memory_mb: float
    
    def can_add_request(self, request: OmniDiffusionRequest, 
                       max_memory_mb: float) -> bool:
        """Check if a request can be added to this group."""
        if len(self.requests) >= self.max_batch_size:
            return False
            
        # Check memory constraints
        new_memory = self._estimate_request_memory(request)
        if (self.estimated_memory_mb + new_memory) > max_memory_mb:
            return False
            
        # Check compatibility
        return self._are_compatible(self.requests[0], request)
    
    def _are_compatible(self, ref_req: OmniDiffusionRequest, 
                       new_req: OmniDiffusionRequest) -> bool:
        """Check if two requests are compatible for batching."""
        
        # Resolution compatibility (critical for memory and compute)
        if not self._resolution_compatible(ref_req, new_req):
            return False
            
        # CFG scale compatibility (affects denoising behavior)
        if not self._cfg_compatible(ref_req, new_req):
            return False
            
        # Number of inference steps (affects iteration count)
        if not self._steps_compatible(ref_req, new_req):
            return False
            
        # Cache backend compatibility
        if not self._cache_compatible(ref_req, new_req):
            return False
            
        # Model/pipeline compatibility
        if not self._model_compatible(ref_req, new_req):
            return False
            
        return True
    
    def _resolution_compatible(self, req1: OmniDiffusionRequest, 
                              req2: OmniDiffusionRequest) -> bool:
        """Check if resolutions are compatible for batching."""
        # Allow batching if resolutions are within 10% tolerance
        def get_resolution(req):
            if req.height and req.width:
                return req.height * req.width
            elif req.resolution:
                # Assume square resolution
                return req.resolution * req.resolution
            else:
                return 1024 * 1024  # Default
        
        res1 = get_resolution(req1)
        res2 = get_resolution(req2)
        
        if res1 == 0 or res2 == 0:
            return True  # Allow if one is default
            
        max_res = max(res1, res2)
        min_res = min(res1, res2)
        tolerance = 0.1  # 10% tolerance
        
        return (max_res - min_res) / max_res <= tolerance
    
    def _cfg_compatible(self, req1: OmniDiffusionRequest, 
                       req2: OmniDiffusionRequest) -> bool:
        """Check if CFG scales are compatible."""
        cfg1 = req1.guidance_scale or 1.0
        cfg2 = req2.guidance_scale or 1.0
        
        # Allow batching if CFG scales are within reasonable range
        max_cfg = max(cfg1, cfg2)
        min_cfg = min(cfg1, cfg2)
        tolerance = 0.2  # 20% tolerance
        
        if max_cfg <= 1.0:  # Both without CFG
            return True
            
        return (max_cfg - min_cfg) / max_cfg <= tolerance
    
    def _steps_compatible(self, req1: OmniDiffusionRequest, 
                         req2: OmniDiffusionRequest) -> bool:
        """Check if inference steps are compatible."""
        steps1 = req1.num_inference_steps or 50
        steps2 = req2.num_inference_steps or 50
        
        # Allow batching if step counts are within 20% of each other
        max_steps = max(steps1, steps2)
        min_steps = min(steps1, steps2)
        tolerance = 0.2
        
        return (max_steps - min_steps) / max_steps <= tolerance
    
    def _cache_compatible(self, req1: OmniDiffusionRequest, 
                         req2: OmniDiffusionRequest) -> bool:
        """Check if cache configurations are compatible."""
        # For now, assume cache compatibility is handled at config level
        return True
    
    def _model_compatible(self, req1: OmniDiffusionRequest, 
                         req2: OmniDiffusionRequest) -> bool:
        """Check if model/pipeline configurations are compatible."""
        # This would check if both requests use the same model/pipeline
        # For now, assume all models are compatible within the same engine
        return True
    
    def _estimate_request_memory(self, request: OmniDiffusionRequest) -> float:
        """Estimate memory usage for a single request."""
        # Basic memory estimation based on resolution and steps
        height = request.height or 1024
        width = request.width or 1024
        steps = request.num_inference_steps or 50
        
        # Rough estimation: resolution * steps * factor
        resolution_factor = (height * width) / (1024 * 1024)  # Normalize to 1MP
        memory_mb = resolution_factor * steps * 0.5  # Conservative estimate
        
        return max(memory_mb, 10.0)  # Minimum 10MB per request


class CompatibilityGroupManager:
    """Manages compatibility groups for efficient batching."""
    
    def __init__(self, max_batch_size: int = 8, max_memory_mb: float = 8192):
        self.max_batch_size = max_batch_size
        self.max_memory_mb = max_memory_mb
        self.groups: List[CompatibilityGroup] = []
    
    def add_request(self, request: OmniDiffusionRequest) -> bool:
        """Add a request to an appropriate compatibility group."""
        # Try to find an existing compatible group
        for group in self.groups:
            if group.can_add_request(request, self.max_memory_mb):
                group.requests.append(request)
                group.estimated_memory_mb += self._estimate_request_memory(request)
                return True
        
        # Create new group if no compatible group found
        new_group = CompatibilityGroup(
            requests=[request],
            compatibility_key=self._generate_compatibility_key(request),
            max_batch_size=self.max_batch_size,
            estimated_memory_mb=self._estimate_request_memory(request)
        )
        self.groups.append(new_group)
        return True
    
    def get_ready_groups(self, min_batch_size: int = 1) -> List[CompatibilityGroup]:
        """Get groups that are ready for processing."""
        ready_groups = []
        for group in self.groups:
            if len(group.requests) >= min_batch_size:
                ready_groups.append(group)
        return ready_groups
    
    def remove_group(self, group: CompatibilityGroup) -> None:
        """Remove a processed group."""
        if group in self.groups:
            self.groups.remove(group)
    
    def _generate_compatibility_key(self, request: OmniDiffusionRequest) -> str:
        """Generate a unique key for compatibility grouping."""
        height = request.height or 1024
        width = request.width or 1024
        steps = request.num_inference_steps or 50
        cfg = request.guidance_scale or 1.0
        
        # Round values to reduce granularity
        height_bucket = (height // 128) * 128
        width_bucket = (width // 128) * 128
        steps_bucket = (steps // 5) * 5
        cfg_bucket = round(cfg, 1)
        
        return f"{height_bucket}x{width_bucket}_steps{steps_bucket}_cfg{cfg_bucket}"
    
    def _estimate_request_memory(self, request: OmniDiffusionRequest) -> float:
        """Estimate memory usage for a single request."""
        height = request.height or 1024
        width = request.width or 1024
        steps = request.num_inference_steps or 50
        
        resolution_factor = (height * width) / (1024 * 1024)
        memory_mb = resolution_factor * steps * 0.5
        
        return max(memory_mb, 10.0)