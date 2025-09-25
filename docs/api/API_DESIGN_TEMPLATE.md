# API Design Template for vLLM-omni Modules

This template provides a standardized structure for designing APIs for core, engine, executor, and worker modules in vLLM-omni. Use this template to ensure consistency and completeness across all modules.

## Template Structure

### 1. Module Overview
- **Purpose**: What this module does
- **Responsibilities**: Key responsibilities of the module
- **Dependencies**: What other modules this depends on
- **Integration Points**: How it integrates with other modules

### 2. Core Classes/Interfaces
- **Base Classes**: Abstract base classes and interfaces
- **Implementation Classes**: Concrete implementations
- **Data Structures**: Key data structures and models

### 3. Public API Methods
- **Initialization**: Constructor and setup methods
- **Core Operations**: Main functionality methods
- **Configuration**: Configuration and parameter methods
- **Lifecycle Management**: Start, stop, cleanup methods
- **Monitoring**: Status, metrics, and debugging methods

### 4. Configuration
- **Configuration Classes**: Dataclasses or config objects
- **Required Parameters**: Must-have configuration
- **Optional Parameters**: Optional configuration with defaults
- **Validation**: Parameter validation rules

### 5. Error Handling
- **Custom Exceptions**: Module-specific exceptions
- **Error Codes**: Standardized error codes
- **Recovery Strategies**: How to handle and recover from errors

### 6. Examples
- **Basic Usage**: Simple usage examples
- **Advanced Usage**: Complex scenarios
- **Integration Examples**: How to use with other modules

---

## Example: Core Module Template

### 1. Module Overview

**Purpose**: The core module provides fundamental scheduling and caching functionality for vLLM-omni.

**Responsibilities**:
- Request scheduling and prioritization
- DiT cache management
- Resource allocation and coordination
- Inter-module communication

**Dependencies**:
- `vllm_omni.request` - Request handling
- `vllm_omni.config` - Configuration management
- `vllm_omni.utils` - Utility functions

**Integration Points**:
- Receives requests from entrypoints
- Coordinates with engine modules
- Manages worker allocation
- Provides status to monitoring systems

### 2. Core Classes/Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class DiTCacheManager:
    """Manages DiT cache for diffusion models."""
    
    def __init__(self, config: DiTCacheConfig):
        self.config = config
        self.cache = {}
        self.cache_stats = {}
    
    async def get_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data."""
        pass
    
    async def set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache."""
        pass
    
    async def invalidate_cache(self, cache_key: str) -> None:
        """Invalidate cached data."""
        pass
```

### 3. Public API Methods

#### Initialization
```python
class OmniScheduler():
    def __init__(self, config: SchedulerConfig):
        """Initialize the core scheduler."""
        self.config = config
        self.scheduler = self._create_scheduler()
        self.cache_manager = DiTCacheManager(config.dit_cache_config)
        self._running = False
    
    def _create_scheduler(self) -> BaseScheduler:
        """Factory method to create appropriate scheduler."""
        if self.config.scheduler_type == SchedulerType.FIFO:
            return FIFOScheduler(self.config)
        elif self.config.scheduler_type == SchedulerType.PRIORITY:
            return PriorityScheduler(self.config)
        # ... other scheduler types
```

#### Core Operations
```python
    async def schedule(self, request: OmniRequest) -> bool:
        """
        Schedule a request for processing.
        
        Args:
            request: The request to schedule
            
        Returns:
            bool: True if successfully scheduled, False otherwise
            
        Raises:
            SchedulerError: If scheduling fails
            QueueFullError: If queue is at capacity
        """
        pass
```

#### Configuration
```python
    def update_config(self, new_config: SchedulerConfig) -> None:
        """Update scheduler configuration."""
        pass
    
    def get_config(self) -> SchedulerConfig:
        """Get current configuration."""
        pass
```

#### Lifecycle Management
```python
    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        # Start background tasks
    
    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False
        # Cleanup resources
    
    async def shutdown(self) -> None:
        """Force shutdown the scheduler."""
        # Immediate cleanup
```

#### Monitoring
```python
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "running": self._running,
            "queue_size": self.scheduler.queue_size(),
            "processed_requests": self.scheduler.processed_count(),
            "cache_hit_rate": self.cache_manager.get_hit_rate()
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        pass
```

### 4. Configuration

```python
@dataclass
class OmniConfig:
    """Configuration for the core module."""
    
    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Cache configuration
    dit_cache: DiTCacheConfig = field(default_factory=DiTCacheConfig)
    
    # Resource limits
    max_memory_gb: float = 16.0
    max_gpu_utilization: float = 0.8
    
    # Timeouts
    request_timeout: int = 300
    worker_timeout: int = 60
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 0 < self.max_gpu_utilization <= 1:
            raise ValueError("max_gpu_utilization must be between 0 and 1")
```

### 5. Error Handling

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
```

### 6. Examples

#### Basic Usage
```python
from vllm_omni.core import CoreScheduler, CoreModuleConfig
from vllm_omni.request import create_text_request

# Create configuration
config = CoreModuleConfig(
    scheduler=SchedulerConfig(scheduler_type=SchedulerType.FIFO),
    max_memory_gb=8.0
)

# Initialize scheduler
scheduler = CoreScheduler(config)
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
```

#### Advanced Usage
```python
# Custom scheduler with priority handling
config = CoreModuleConfig(
    scheduler=SchedulerConfig(
        scheduler_type=SchedulerType.PRIORITY,
        priority_weights={"high": 2.0, "normal": 1.0, "low": 0.5}
    )
)

scheduler = CoreScheduler(config)
await scheduler.start()

# Monitor scheduler status
status = scheduler.get_status()
print(f"Queue size: {status['queue_size']}")
print(f"Cache hit rate: {status['cache_hit_rate']:.2%}")
```

---

## Module-Specific Guidelines

### Core Module
- Focus on scheduling, caching, and resource management
- Provide clear interfaces for other modules
- Handle concurrency and thread safety
- Implement comprehensive monitoring

### Engine Module
- Handle model loading and inference
- Support both AR and diffusion models
- Provide unified interface for different model types
- Implement efficient memory management

### Executor Module
- Coordinate between different engines
- Handle request routing and load balancing
- Manage execution pipelines
- Provide error recovery mechanisms

### Worker Module
- Handle actual model execution
- Manage GPU resources
- Implement batching strategies
- Provide performance optimization

---

## Checklist for API Design

- [ ] **Clear Purpose**: Module purpose is well-defined
- [ ] **Complete Interface**: All public methods are documented
- [ ] **Error Handling**: Comprehensive error handling strategy
- [ ] **Configuration**: Flexible configuration system
- [ ] **Examples**: Basic and advanced usage examples
- [ ] **Type Hints**: All methods have proper type hints
- [ ] **Documentation**: Comprehensive docstrings
- [ ] **Testing**: Unit tests for all public methods
- [ ] **Integration**: Clear integration points with other modules
- [ ] **Performance**: Performance considerations documented

---

## Submission Guidelines

1. Create a new file: `docs/api/[module_name]_api.md`
2. Follow the template structure exactly
3. Include all required sections
4. Provide working code examples
5. Ensure all methods have proper docstrings
6. Include error handling strategies
7. Submit for review before implementation

This template ensures consistency and completeness across all vLLM-omni modules.
