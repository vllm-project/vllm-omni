# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Configuration for DiT batching in vLLM-Omni.

This module provides configuration classes and utilities for managing
batching behavior in diffusion workloads.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DiTBatchingConfig:
    """
    Configuration for DiT diffusion request batching.
    
    This configuration controls how diffusion requests are grouped and
    processed together for improved throughput and GPU utilization.
    """
    
    # Core batching parameters
    enable_batching: bool = True
    max_batch_size: int = 8
    min_batch_size: int = 1
    max_wait_time_ms: int = 50
    
    # Memory management
    max_memory_mb: float = 8192.0
    memory_safety_margin: float = 0.2  # 20% safety margin
    
    # Scheduling behavior
    enable_priority_queuing: bool = True
    enable_starvation_prevention: bool = True
    starvation_threshold_seconds: float = 5.0
    batch_timeout_strategy: str = "adaptive"  # "adaptive", "fixed", "aggressive"
    
    # Compatibility grouping
    resolution_tolerance: float = 0.1  # 10% tolerance for resolution differences
    cfg_tolerance: float = 0.2  # 20% tolerance for CFG scale differences
    steps_tolerance: float = 0.2  # 20% tolerance for inference steps differences
    
    # Performance tuning
    adaptive_timeout_enabled: bool = True
    performance_history_size: int = 100
    throughput_improvement_threshold: float = 0.1  # 10% minimum improvement
    
    # Monitoring and logging
    enable_detailed_metrics: bool = True
    log_level: str = "INFO"
    metrics_export_interval_seconds: int = 30
    
    # Compatibility with existing features
    cache_backend_compatible: bool = True
    parallelism_compatible: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DiTBatchingConfig":
        """Create config from dictionary."""
        # Filter out None values and unknown fields
        filtered_config = {}
        for key, value in config_dict.items():
            if hasattr(cls, key) and value is not None:
                filtered_config[key] = value
        
        return cls(**filtered_config)
    
    @classmethod
    def from_env(cls) -> "DiTBatchingConfig":
        """Create config from environment variables."""
        config_dict = {}
        
        # Mapping of env vars to config fields
        env_mappings = {
            "VLLM_OMNI_DIT_BATCHING_ENABLED": ("enable_batching", lambda x: x.lower() == "true"),
            "VLLM_OMNI_DIT_MAX_BATCH_SIZE": ("max_batch_size", int),
            "VLLM_OMNI_DIT_MIN_BATCH_SIZE": ("min_batch_size", int),
            "VLLM_OMNI_DIT_MAX_WAIT_TIME_MS": ("max_wait_time_ms", int),
            "VLLM_OMNI_DIT_MAX_MEMORY_MB": ("max_memory_mb", float),
            "VLLM_OMNI_DIT_MEMORY_SAFETY_MARGIN": ("memory_safety_margin", float),
            "VLLM_OMNI_DIT_ENABLE_PRIORITY": ("enable_priority_queuing", lambda x: x.lower() == "true"),
            "VLLM_OMNI_DIT_ENABLE_STARVATION": ("enable_starvation_prevention", lambda x: x.lower() == "true"),
            "VLLM_OMNI_DIT_STARVATION_THRESHOLD": ("starvation_threshold_seconds", float),
            "VLLM_OMNI_DIT_TIMEOUT_STRATEGY": ("batch_timeout_strategy", str),
            "VLLM_OMNI_DIT_RESOLUTION_TOLERANCE": ("resolution_tolerance", float),
            "VLLM_OMNI_DIT_CFG_TOLERANCE": ("cfg_tolerance", float),
            "VLLM_OMNI_DIT_STEPS_TOLERANCE": ("steps_tolerance", float),
            "VLLM_OMNI_DIT_ADAPTIVE_TIMEOUT": ("adaptive_timeout_enabled", lambda x: x.lower() == "true"),
            "VLLM_OMNI_DIT_PERF_HISTORY_SIZE": ("performance_history_size", int),
            "VLLM_OMNI_DIT_METRICS_ENABLED": ("enable_detailed_metrics", lambda x: x.lower() == "true"),
            "VLLM_OMNI_DIT_LOG_LEVEL": ("log_level", str),
        }
        
        for env_var, (config_field, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if callable(converter):
                        config_dict[config_field] = converter(value)
                    else:
                        config_dict[config_field] = converter(value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value}. Using default.")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_batching": self.enable_batching,
            "max_batch_size": self.max_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_wait_time_ms": self.max_wait_time_ms,
            "max_memory_mb": self.max_memory_mb,
            "memory_safety_margin": self.memory_safety_margin,
            "enable_priority_queuing": self.enable_priority_queuing,
            "enable_starvation_prevention": self.enable_starvation_prevention,
            "starvation_threshold_seconds": self.starvation_threshold_seconds,
            "batch_timeout_strategy": self.batch_timeout_strategy,
            "resolution_tolerance": self.resolution_tolerance,
            "cfg_tolerance": self.cfg_tolerance,
            "steps_tolerance": self.steps_tolerance,
            "adaptive_timeout_enabled": self.adaptive_timeout_enabled,
            "performance_history_size": self.performance_history_size,
            "throughput_improvement_threshold": self.throughput_improvement_threshold,
            "enable_detailed_metrics": self.enable_detailed_metrics,
            "log_level": self.log_level,
            "metrics_export_interval_seconds": self.metrics_export_interval_seconds,
            "cache_backend_compatible": self.cache_backend_compatible,
            "parallelism_compatible": self.parallelism_compatible,
        }
    
    def validate(self) -> bool:
        """Validate configuration values."""
        if self.max_batch_size < self.min_batch_size:
            raise ValueError(f"max_batch_size ({self.max_batch_size}) must be >= min_batch_size ({self.min_batch_size})")
        
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
        
        if self.min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be positive, got {self.min_batch_size}")
        
        if self.max_wait_time_ms <= 0:
            raise ValueError(f"max_wait_time_ms must be positive, got {self.max_wait_time_ms}")
        
        if self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        
        if not 0.0 <= self.memory_safety_margin <= 1.0:
            raise ValueError(f"memory_safety_margin must be between 0.0 and 1.0, got {self.memory_safety_margin}")
        
        if self.batch_timeout_strategy not in ["adaptive", "fixed", "aggressive"]:
            raise ValueError(f"batch_timeout_strategy must be one of ['adaptive', 'fixed', 'aggressive'], got {self.batch_timeout_strategy}")
        
        if self.starvation_threshold_seconds <= 0:
            raise ValueError(f"starvation_threshold_seconds must be positive, got {self.starvation_threshold_seconds}")
        
        return True
    
    def get_optimized_defaults(self) -> "DiTBatchingConfig":
        """Get optimized default configuration based on hardware."""
        import torch
        
        config = self
        
        # Detect GPU memory and adjust batch size
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Scale max batch size based on GPU memory
            if gpu_memory_gb >= 24:  # High-end GPUs
                config.max_batch_size = min(config.max_batch_size, 12)
                config.max_memory_mb = min(config.max_memory_mb, 16384)  # 16GB
            elif gpu_memory_gb >= 12:  # Mid-range GPUs
                config.max_batch_size = min(config.max_batch_size, 8)
                config.max_memory_mb = min(config.max_memory_mb, 8192)  # 8GB
            else:  # Lower-end GPUs
                config.max_batch_size = min(config.max_batch_size, 4)
                config.max_memory_mb = min(config.max_memory_mb, 4096)  # 4GB
        
        return config


@dataclass
class BatchingMetrics:
    """Metrics for monitoring batching performance."""
    
    # Request statistics
    total_requests: int = 0
    batched_requests: int = 0
    single_requests: int = 0
    
    # Batch statistics
    total_batches: int = 0
    avg_batch_size: float = 0.0
    max_batch_size: int = 0
    batching_efficiency: float = 0.0  # Percentage of requests that were batched
    
    # Performance metrics
    avg_wait_time_ms: float = 0.0
    avg_processing_time: float = 0.0
    throughput_improvement: float = 0.0  # Percentage improvement over single requests
    
    # Memory metrics
    avg_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    memory_efficiency: float = 0.0
    
    # Compatibility metrics
    compatibility_groups_created: int = 0
    avg_group_size: float = 0.0
    compatibility_rate: float = 0.0  # Percentage of requests that found compatible groups
    
    def update(self, other: "BatchingMetrics"):
        """Update metrics with another metrics object."""
        # Simple averaging for demonstration - in practice, you'd want weighted averages
        for field_name in self.__dataclass_fields__:
            if hasattr(other, field_name):
                self_value = getattr(self, field_name)
                other_value = getattr(other, field_name)
                
                if isinstance(self_value, (int, float)) and isinstance(other_value, (int, float)):
                    # Average numeric fields
                    setattr(self, field_name, (self_value + other_value) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }