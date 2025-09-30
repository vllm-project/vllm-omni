"""
DiT Cache Manager for vLLM-omni.

This module provides caching functionality for DiT (Diffusion Transformer) models
to optimize inference performance and memory usage.
"""

import time
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..config import DiTCacheConfig, DiTCacheTensor


class DiTCacheManager:
    """Manages DiT-specific caching for optimized inference."""
    
    def __init__(self, config: DiTCacheConfig):
        self.config = config
        self.cache_tensors: Dict[str, DiTCacheTensor] = {}
        self.cache_groups: List[DiTCacheTensor] = config.cache_tensors
        self.max_cache_size = config.max_cache_size
        self.current_cache_size = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_timeout = 3600  # 1 hour default timeout
    
    def allocate_cache(self, request_id: str, size: int) -> torch.Tensor:
        """Allocate cache for a specific request."""
        # Check if we have enough space
        if self.current_cache_size + size > self.max_cache_size:
            self._evict_cache()
        
        # Create tensor
        tensor = torch.zeros(size, dtype=torch.float32)
        
        # Store in cache
        cache_tensor = DiTCacheTensor(
            name=request_id,
            tensor=tensor,
            timestamp=time.time(),
            access_count=0,
            size_bytes=size * 4  # Assuming float32
        )
        
        self.cache_tensors[request_id] = cache_tensor
        self.current_cache_size += cache_tensor.size_bytes
        
        return tensor
    
    def get_cache(self, request_id: str) -> Optional[torch.Tensor]:
        """Get cached tensor for a request."""
        if request_id in self.cache_tensors:
            cache_tensor = self.cache_tensors[request_id]
            cache_tensor.access_count += 1
            self.cache_hits += 1
            return cache_tensor.tensor
        else:
            self.cache_misses += 1
            return None
    
    def store_cache(self, request_id: str, tensor: torch.Tensor) -> None:
        """Store a tensor in the cache."""
        if request_id in self.cache_tensors:
            # Update existing cache
            old_tensor = self.cache_tensors[request_id]
            self.current_cache_size -= old_tensor.size_bytes
            
            new_size_bytes = tensor.numel() * 4  # Assuming float32
            self.current_cache_size += new_size_bytes
            
            self.cache_tensors[request_id] = DiTCacheTensor(
                name=request_id,
                tensor=tensor,
                timestamp=time.time(),
                access_count=old_tensor.access_count,
                size_bytes=new_size_bytes
            )
        else:
            # Create new cache entry
            new_size_bytes = tensor.numel() * 4
            if self.current_cache_size + new_size_bytes > self.max_cache_size:
                self._evict_cache()
            
            self.cache_tensors[request_id] = DiTCacheTensor(
                name=request_id,
                tensor=tensor,
                timestamp=time.time(),
                access_count=0,
                size_bytes=new_size_bytes
            )
            self.current_cache_size += new_size_bytes
    
    def release_cache(self, request_id: str) -> None:
        """Release cache for a request."""
        if request_id in self.cache_tensors:
            cache_tensor = self.cache_tensors[request_id]
            self.current_cache_size -= cache_tensor.size_bytes
            del self.cache_tensors[request_id]
    
    def clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for request_id, cache_tensor in self.cache_tensors.items():
            if current_time - cache_tensor.timestamp > self.cache_timeout:
                expired_keys.append(request_id)
        
        for request_id in expired_keys:
            self.release_cache(request_id)
    
    def _evict_cache(self) -> None:
        """Evict cache entries based on strategy."""
        if self.config.cache_strategy == "fifo":
            self._evict_fifo()
        elif self.config.cache_strategy == "lru":
            self._evict_lru()
        elif self.config.cache_strategy == "lfu":
            self._evict_lfu()
        else:
            self._evict_fifo()  # Default to FIFO
    
    def _evict_fifo(self) -> None:
        """Evict cache entries using FIFO strategy."""
        if not self.cache_tensors:
            return
        
        # Find oldest entry
        oldest_key = min(self.cache_tensors.keys(), 
                        key=lambda k: self.cache_tensors[k].timestamp)
        self.release_cache(oldest_key)
    
    def _evict_lru(self) -> None:
        """Evict cache entries using LRU strategy."""
        if not self.cache_tensors:
            return
        
        # Find least recently used entry
        lru_key = min(self.cache_tensors.keys(),
                     key=lambda k: self.cache_tensors[k].access_count)
        self.release_cache(lru_key)
    
    def _evict_lfu(self) -> None:
        """Evict cache entries using LFU strategy."""
        if not self.cache_tensors:
            return
        
        # Find least frequently used entry
        lfu_key = min(self.cache_tensors.keys(),
                     key=lambda k: self.cache_tensors[k].access_count)
        self.release_cache(lfu_key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": self.current_cache_size,
            "max_cache_size": self.max_cache_size,
            "cache_utilization": self.current_cache_size / self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "num_cached_tensors": len(self.cache_tensors)
        }
    
    def clear_all_cache(self) -> None:
        """Clear all cache entries."""
        self.cache_tensors.clear()
        self.current_cache_size = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def set_cache_timeout(self, timeout: float) -> None:
        """Set cache timeout in seconds."""
        self.cache_timeout = timeout