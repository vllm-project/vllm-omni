from dataclasses import dataclass

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncEngineArgs

from vllm_omni.config import OmniModelConfig

logger = init_logger(__name__)


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
    engine_output_type: str | None = None
    hf_config_name: str | None = None

    def draw_hf_text_config(self, config_dict: dict) -> Qwen3OmniMoeTextConfig:
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        return getattr(config_dict["hf_config"], config_dict["hf_config_name"]).get_text_config()

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
        # FIXME(Isotr0py): This is a temporary workaround for multimodal_config
        config_dict = {
            **(getattr(mm := config_dict.pop("multimodal_config", None), "__dict__", mm or {})),
            **config_dict,
        }

        # Add the new omni-specific fields
        config_dict["stage_id"] = self.stage_id
        config_dict["model_stage"] = self.model_stage
        config_dict["model_arch"] = self.model_arch
        config_dict["engine_output_type"] = self.engine_output_type
        config_dict["hf_config_name"] = self.hf_config_name
        if self.hf_config_name is not None:
            config_dict["hf_text_config"] = self.draw_hf_text_config(config_dict)
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
    engine_output_type: str | None = None
    hf_config_name: str | None = None

    def draw_hf_text_config(self, config_dict: dict) -> Qwen3OmniMoeTextConfig:
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        return getattr(config_dict["hf_config"], config_dict["hf_config_name"]).get_text_config()

    def create_model_config(self) -> OmniModelConfig:
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

        config_dict["hf_config_name"] = self.hf_config_name
        if self.hf_config_name is not None:
            config_dict["hf_text_config"] = self.draw_hf_text_config(config_dict)
        # Create and return the OmniModelConfig instance
        omni_config = OmniModelConfig(**config_dict)
        omni_config.hf_config.architectures = omni_config.architectures

        return omni_config
