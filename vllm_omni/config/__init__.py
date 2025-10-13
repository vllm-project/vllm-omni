"""
Configuration module for vLLM-omni.
"""

from vllm.config import ModelConfig, VllmConfig
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from vllm.config.utils import config
import vllm_omni.model_executor.models as me_models

from .stage_config import (
    OmniStageConfig,
    DiTConfig,
    DiTCacheConfig,
    DiTCacheTensor,
    create_ar_stage_config,
    create_dit_stage_config,
)


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
    """Configuration for Omni models, extending the base ModelConfig."""
    
    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: Optional[str] = None
    
    @property
    def registry(self):
        return me_models.OmniModelRegistry
    
    @property
    def architectures(self) -> list[str]:
        return [self.model_arch]


@dataclass
class OmniConfig:
    """Configuration for vLLM-omni multi-stage processing."""
    
    vllm_config: Optional[VllmConfig] = None
    """Base vLLM configuration."""
    
    stage_configs: list[OmniStageConfig] = None
    """List of stage configurations."""
    
    dit_cache_config: Optional[DiTCacheConfig] = None
    """DiT cache configuration."""
    
    log_stats: bool = False
    """Whether to log statistics."""


__all__ = [
    "OmniModelConfig",
    "OmniConfig",
    "OmniStageConfig",
    "DiTConfig",
    "DiTCacheConfig",
    "DiTCacheTensor",
    "create_ar_stage_config",
    "create_dit_stage_config",
]