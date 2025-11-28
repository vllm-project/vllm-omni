from dataclasses import dataclass
from typing import Optional

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.async_llm import AsyncEngineArgs

from vllm_omni.config import OmniModelConfig


@dataclass
class OmniEngineArgs(EngineArgs):
    """Engine arguments for omni models, extending base EngineArgs.

    Adds omni-specific configuration fields for multi-stage pipeline
    processing and output type specification.

    Args:
        stage_id: Identifier for the stage in a multi-stage pipeline (default: 0)
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: Optional[str] = None

    def create_model_config(self) -> OmniModelConfig:
        """Create an OmniModelConfig from these engine arguments.

        Returns:
            OmniModelConfig instance with all configuration fields set
        """
        # First, get the base ModelConfig from the parent class
        base_config = super().create_model_config()

        # Create OmniModelConfig by copying all base config attributes
        # and adding the new omni-specific fields
        config_dict = base_config.__dict__.copy()

        # Add the new omni-specific fields
        config_dict["stage_id"] = self.stage_id
        config_dict["model_stage"] = self.model_stage
        config_dict["model_arch"] = self.model_arch
        config_dict["engine_output_type"] = self.engine_output_type

        # Create and return the OmniModelConfig instance
        omni_config = OmniModelConfig(**config_dict)
        omni_config.hf_config.architectures = omni_config.architectures

        return omni_config


@dataclass
class AsyncOmniEngineArgs(AsyncEngineArgs):
    """Async engine arguments for omni models, extending base AsyncEngineArgs.

    Adds omni-specific configuration fields for multi-stage pipeline
    processing and output type specification in async contexts.

    Args:
        stage_id: Identifier for the stage in a multi-stage pipeline (default: 0)
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: Optional[str] = None

    def create_model_config(self) -> OmniModelConfig:
        """Create an OmniModelConfig from these engine arguments.

        Returns:
            OmniModelConfig instance with all configuration fields set
        """
        # First, get the base ModelConfig from the parent class
        base_config = super().create_model_config()

        # Create OmniModelConfig by copying all base config attributes
        # and adding the new omni-specific fields
        config_dict = base_config.__dict__.copy()

        # Add the new omni-specific fields
        config_dict["stage_id"] = self.stage_id
        config_dict["model_stage"] = self.model_stage
        config_dict["model_arch"] = self.model_arch
        config_dict["engine_output_type"] = self.engine_output_type

        # Create and return the OmniModelConfig instance
        omni_config = OmniModelConfig(**config_dict)
        omni_config.hf_config.architectures = omni_config.architectures

        return omni_config
