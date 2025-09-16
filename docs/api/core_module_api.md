# Core Module API Design

## 1. Module Overview

**Purpose**: The core module provides fundamental scheduling, caching, and resource management functionality for vLLM-omni.

**Responsibilities**:
- Request scheduling and prioritization across different model types
- DiT cache management for diffusion models
- Resource allocation and coordination between workers
- Inter-module communication and coordination
- Queue management and load balancing

**Dependencies**:
- `vllm_omni.request` - Request handling and types
- `vllm_omni.config` - Configuration management
- `vllm_omni.utils` - Utility functions and helpers
- `vllm` - vLLM core components for integration

**Integration Points**:
- Receives requests from entrypoints
- Coordinates with engine modules for model execution
- Manages worker allocation and task distribution
- Provides status and metrics to monitoring systems
- Handles cache coordination between different model types

## 2. Core Classes/Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from vllm_omni.request import OmniRequest, RequestType, ProcessingStage

class SchedulerType(Enum):
    """Types of schedulers available."""
    FIFO = "fifo"
    PRIORITY = "priority"
    MULTIMODAL = "multimodal"
    DIFFUSION = "diffusion"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Cache strategies for different model types."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class SchedulerConfig:
    """Configuration for scheduler."""
    scheduler_type: SchedulerType = SchedulerType.FIFO
    max_queue_size: int = 1000
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "high": 2.0, "normal": 1.0, "low": 0.5
    })
    timeout_seconds: int = 300
    batch_size: int = 32
    enable_preemption: bool = True

@dataclass
class CacheConfig:
    """Configuration for cache management."""
    max_memory_gb: float = 8.0
    strategy: CacheStrategy = CacheStrategy.LRU
    ttl_seconds: int = 3600
    enable_compression: bool = True
    compression_ratio: float = 0.7

class BaseScheduler(ABC):
    """Abstract base class for all schedulers."""
    
    @abstractmethod
    async def schedule_request(self, request: OmniRequest) -> bool:
        """Schedule a request for processing."""
        pass
    
    @abstractmethod
    async def get_next_request(self) -> Optional[OmniRequest]:
        """Get the next request to process."""
        pass
    
    @abstractmethod
    async def remove_request(self, request_id: str) -> bool:
        """Remove a request from the queue."""
        pass
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size."""
        pass
    
    @abstractmethod
    def get_processed_count(self) -> int:
        """Get number of processed requests."""
        pass

class DiTCacheManager:
    """Manages DiT cache for diffusion models."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.memory_usage = 0.0
    
    async def get_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data."""
        pass
    
    async def set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache."""
        pass
    
    async def invalidate_cache(self, cache_key: str) -> None:
        """Invalidate cached data."""
        pass
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        pass
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        pass

class ResourceManager:
    """Manages system resources and allocation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allocated_resources = {}
        self.available_resources = {}
    
    async def allocate_resources(self, request: OmniRequest) -> Dict[str, Any]:
        """Allocate resources for a request."""
        pass
    
    async def deallocate_resources(self, request_id: str) -> None:
        """Deallocate resources for a request."""
        pass
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status."""
        pass
```

## 3. Public API Methods

### CoreScheduler Class

```python
class OmniScheduler(SchedulerInterface):
    """Main scheduler for vLLM-omni core module."""
    
    def __init__(self, config: SchedulerConfig, cache_config: CacheConfig):
        """
        Initialize the core scheduler.
        
        Args:
            config: Scheduler configuration
            cache_config: Cache configuration
        """
        self.config = config
        self.cache_manager = DiTCacheManager(cache_config)
        self.resource_manager = ResourceManager({})
        self.scheduler = self._create_scheduler()
        self._running = False
        self._stats = {
            "total_requests": 0,
            "processed_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0
        }
    
    def _create_scheduler(self) -> BaseScheduler:
        """Factory method to create appropriate scheduler."""
        if self.config.scheduler_type == SchedulerType.FIFO:
            return FIFOScheduler(self.config)
        elif self.config.scheduler_type == SchedulerType.PRIORITY:
            return PriorityScheduler(self.config)
        elif self.config.scheduler_type == SchedulerType.MULTIMODAL:
            return MultimodalScheduler(self.config)
        elif self.config.scheduler_type == SchedulerType.DIFFUSION:
            return DiffusionScheduler(self.config)
        elif self.config.scheduler_type == SchedulerType.HYBRID:
            return HybridScheduler(self.config)
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
```

#### Core Operations

```python
    async def schedule_request(self, request: OmniRequest) -> bool:
        """
        Schedule a request for processing.
        
        Args:
            request: The request to schedule
            
        Returns:
            bool: True if successfully scheduled, False otherwise
            
        Raises:
            SchedulerError: If scheduling fails
            QueueFullError: If queue is at capacity
            ResourceError: If insufficient resources
        """
        try:
            # Validate request
            if not self._validate_request(request):
                return False
            
            # Check resource availability
            if not await self.resource_manager.can_allocate(request):
                raise ResourceError("Insufficient resources")
            
            # Schedule the request
            success = await self.scheduler.schedule_request(request)
            if success:
                self._stats["total_requests"] += 1
                await self.resource_manager.allocate_resources(request)
            
            return success
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            raise SchedulerError(f"Failed to schedule request: {e}")
    
    async def get_next_request(self) -> Optional[OmniRequest]:
        """
        Get the next request to process.
        
        Returns:
            OmniRequest or None: Next request to process
            
        Raises:
            SchedulerError: If retrieval fails
        """
        try:
            request = await self.scheduler.get_next_request()
            if request:
                request.update_processing_stage(ProcessingStage.PREPROCESSING)
            return request
        except Exception as e:
            raise SchedulerError(f"Failed to get next request: {e}")
    
    async def process_request(self, request: OmniRequest) -> Any:
        """
        Process a request through the appropriate pipeline.
        
        Args:
            request: The request to process
            
        Returns:
            Any: Processing result
            
        Raises:
            ProcessingError: If processing fails
        """
        start_time = time.time()
        try:
            # Update processing stage
            request.update_processing_stage(ProcessingStage.AR_GENERATION)
            
            # Check cache first
            cache_key = request.generate_cache_key()
            cached_result = await self.cache_manager.get_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Process based on request type
            if request.request_type == RequestType.TEXT:
                result = await self._process_text_request(request)
            elif request.request_type == RequestType.IMAGE:
                result = await self._process_image_request(request)
            elif request.request_type == RequestType.MULTIMODAL:
                result = await self._process_multimodal_request(request)
            elif request.request_type == RequestType.DIFFUSION:
                result = await self._process_diffusion_request(request)
            else:
                raise ProcessingError(f"Unknown request type: {request.request_type}")
            
            # Cache the result
            await self.cache_manager.set_cache(cache_key, result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            request.update_processing_stage(ProcessingStage.COMPLETED)
            return result
            
        except Exception as e:
            request.add_error(str(e))
            self._stats["failed_requests"] += 1
            raise ProcessingError(f"Failed to process request: {e}")
        finally:
            # Cleanup resources
            await self.resource_manager.deallocate_resources(request.request_id)
```

#### Configuration Methods

```python
    def update_config(self, new_config: SchedulerConfig) -> None:
        """
        Update scheduler configuration.
        
        Args:
            new_config: New configuration to apply
            
        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            new_config.validate()
            self.config = new_config
            # Recreate scheduler with new config
            self.scheduler = self._create_scheduler()
        except Exception as e:
            raise ConfigError(f"Failed to update config: {e}")
    
    def get_config(self) -> SchedulerConfig:
        """Get current configuration."""
        return self.config
    
    def update_cache_config(self, new_config: CacheConfig) -> None:
        """Update cache configuration."""
        self.cache_manager.config = new_config
```

#### Lifecycle Management

```python
    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            raise SchedulerError("Scheduler is already running")
        
        self._running = True
        # Start background tasks
        asyncio.create_task(self._background_cleanup())
        asyncio.create_task(self._monitor_resources())
    
    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._running:
            return
        
        self._running = False
        # Wait for current requests to complete
        await self._wait_for_completion()
        # Cleanup resources
        await self._cleanup_resources()
    
    async def shutdown(self) -> None:
        """Force shutdown the scheduler."""
        self._running = False
        # Immediate cleanup
        await self._force_cleanup()
```

#### Monitoring Methods

```python
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "running": self._running,
            "queue_size": self.scheduler.get_queue_size(),
            "processed_requests": self.scheduler.get_processed_count(),
            "cache_hit_rate": self.cache_manager.get_hit_rate(),
            "memory_usage": self.cache_manager.get_memory_usage(),
            "resource_status": self.resource_manager.get_resource_status(),
            "stats": self._stats.copy()
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return {
            "throughput": self._calculate_throughput(),
            "average_latency": self._stats["average_processing_time"],
            "cache_hit_rate": self.cache_manager.get_hit_rate(),
            "memory_utilization": self.cache_manager.get_memory_usage() / self.cache_manager.config.max_memory_gb,
            "queue_utilization": self.scheduler.get_queue_size() / self.config.max_queue_size
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the scheduler."""
        return {
            "healthy": self._is_healthy(),
            "issues": self._get_health_issues(),
            "recommendations": self._get_recommendations()
        }
```

## 4. Configuration

```python
@dataclass
class CoreModuleConfig:
    """Complete configuration for the core module."""
    
    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Resource limits
    max_memory_gb: float = 16.0
    max_gpu_utilization: float = 0.8
    max_cpu_utilization: float = 0.9
    
    # Timeouts
    request_timeout: int = 300
    worker_timeout: int = 60
    cleanup_interval: int = 30
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: int = 10
    enable_health_checks: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 0 < self.max_gpu_utilization <= 1:
            raise ValueError("max_gpu_utilization must be between 0 and 1")
        if not 0 < self.max_cpu_utilization <= 1:
            raise ValueError("max_cpu_utilization must be between 0 and 1")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
```

## 5. Error Handling

```python
class CoreModuleError(Exception):
    """Base exception for core module errors."""
    pass

class SchedulerError(CoreModuleError):
    """Scheduler-related errors."""
    pass

class QueueFullError(SchedulerError):
    """Queue is at capacity."""
    pass

class CacheError(CoreModuleError):
    """Cache-related errors."""
    pass

class ResourceError(CoreModuleError):
    """Resource allocation errors."""
    pass

class ProcessingError(CoreModuleError):
    """Request processing errors."""
    pass

class ConfigError(CoreModuleError):
    """Configuration errors."""
    pass
```

## 6. Examples

### Basic Usage

```python
from vllm_omni.core import CoreScheduler, CoreModuleConfig
from vllm_omni.request import create_text_request

# Create configuration
config = CoreModuleConfig(
    scheduler=SchedulerConfig(scheduler_type=SchedulerType.FIFO),
    cache=CacheConfig(max_memory_gb=8.0)
)

# Initialize scheduler
scheduler = CoreScheduler(config.scheduler, config.cache)
await scheduler.start()

# Create and schedule a request
request = create_text_request(
    request_id="req_001",
    prompt="Hello, world!",
    sampling_params=sampling_params
)

success = await scheduler.schedule_request(request)
if success:
    result = await scheduler.process_request(request)
    print(f"Result: {result}")

# Get status
status = scheduler.get_status()
print(f"Queue size: {status['queue_size']}")
print(f"Cache hit rate: {status['cache_hit_rate']:.2%}")

await scheduler.stop()
```

### Advanced Usage

```python
# Custom scheduler with priority handling and monitoring
config = CoreModuleConfig(
    scheduler=SchedulerConfig(
        scheduler_type=SchedulerType.PRIORITY,
        priority_weights={"high": 3.0, "normal": 1.0, "low": 0.3},
        enable_preemption=True
    ),
    cache=CacheConfig(
        strategy=CacheStrategy.ADAPTIVE,
        max_memory_gb=16.0,
        enable_compression=True
    ),
    enable_metrics=True,
    enable_health_checks=True
)

scheduler = CoreScheduler(config.scheduler, config.cache)
await scheduler.start()

# Monitor scheduler
async def monitor_scheduler():
    while True:
        status = scheduler.get_status()
        metrics = scheduler.get_metrics()
        health = scheduler.get_health_status()
        
        print(f"Status: {status}")
        print(f"Metrics: {metrics}")
        print(f"Health: {health}")
        
        await asyncio.sleep(10)

# Start monitoring
asyncio.create_task(monitor_scheduler())
```

### Integration Example

```python
# Integration with other modules
from vllm_omni.engine import EngineManager
from vllm_omni.executor import ExecutorManager

class OmniSystem:
    def __init__(self, config: CoreModuleConfig):
        self.scheduler = CoreScheduler(config.scheduler, config.cache)
        self.engine_manager = EngineManager(config.engine)
        self.executor_manager = ExecutorManager(config.executor)
    
    async def start(self):
        await self.scheduler.start()
        await self.engine_manager.start()
        await self.executor_manager.start()
    
    async def process_request(self, request: OmniRequest):
        # Schedule request
        await self.scheduler.schedule_request(request)
        
        # Get next request
        next_request = await self.scheduler.get_next_request()
        if next_request:
            # Route to appropriate executor
            executor = self.executor_manager.get_executor(next_request)
            result = await executor.execute(next_request)
            return result
```
