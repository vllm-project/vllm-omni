"""
Configuration module for vLLM-omni.
"""
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

from vllm.config import ModelConfig
from vllm.config import config

import vllm_omni.model_executor.models as me_models

from .stage_config import (
    DiTCacheConfig,
    DiTCacheTensor,
    DiTConfig,
    OmniStageConfig,
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


__all__ = [
    "OmniModelConfig",
    "OmniStageConfig",
    "DiTConfig",
    "DiTCacheConfig",
    "DiTCacheTensor",
    "create_ar_stage_config",
    "create_dit_stage_config",
]
