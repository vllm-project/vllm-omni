from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from vllm.config import ModelConfig, config

import vllm_omni.model_executor.models as me_models


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
    """Configuration for Omni models, extending the base ModelConfig."""

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    ode_solver_class: Optional[str] = None
    engine_output_type: Optional[str] = None

    @property
    def registry(self):
        return me_models.OmniModelRegistry

    @property
    def architectures(self) -> list[str]:
        return [self.model_arch]
