from dataclasses import dataclass, field

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.v1.engine.async_llm import AsyncEngineArgs

from vllm_omni.config import OmniModelConfig

logger = init_logger(__name__)


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    except ValueError:
        # Already registered elsewhere; ignore.
        return


def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    _register_omni_hf_configs()

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}")


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
        custom_process_next_stage_input_func: Optional path to a custom function for processing
            inputs from previous stages
            If None, default processing is used.
        stage_connector_spec: Extra configuration for stage connector
        async_chunk: If set to True, perform async chunk
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, any] = field(default_factory=dict)
    async_chunk: bool = False
    omni_kv_config: dict | None = None

    def draw_hf_text_config(self, config_dict: dict) -> Qwen3OmniMoeTextConfig:
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        hf_config = config_dict["hf_config"]
        hf_config_name = config_dict["hf_config_name"]
        try:
            # Try to get the stage-specific config (e.g., thinker_config, talker_config)
            stage_config = getattr(hf_config, hf_config_name)
            return stage_config.get_text_config()
        except AttributeError:
            # Fallback: if the attribute doesn't exist, use the default get_hf_text_config
            logger.warning(
                f"Config attribute '{hf_config_name}' not found in hf_config, "
                "falling back to default get_hf_text_config"
            )
            return get_hf_text_config(hf_config)

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def create_model_config(self) -> OmniModelConfig:
        """Create an OmniModelConfig from these engine arguments.
        Returns:
            OmniModelConfig instance with all configuration fields set
        """
        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()

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
        config_dict["async_chunk"] = self.async_chunk
        config_dict["model_stage"] = self.model_stage
        config_dict["model_arch"] = self.model_arch
        config_dict["engine_output_type"] = self.engine_output_type
        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id
        config_dict["stage_connector_config"] = stage_connector_config

        config_dict["hf_config_name"] = self.hf_config_name
        config_dict["custom_process_next_stage_input_func"] = self.custom_process_next_stage_input_func
        config_dict["omni_kv_config"] = self.omni_kv_config
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
        stage_connector_spec: Extra configuration for stage connector
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, any] = field(default_factory=dict)
    async_chunk: bool = False
    omni_kv_config: dict | None = None

    def draw_hf_text_config(self, config_dict: dict) -> Qwen3OmniMoeTextConfig:
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        hf_config = config_dict["hf_config"]
        hf_config_name = config_dict["hf_config_name"]
        try:
            # Try to get the stage-specific config (e.g., thinker_config, talker_config)
            stage_config = getattr(hf_config, hf_config_name)
            return stage_config.get_text_config()
        except AttributeError:
            # Fallback: if the attribute doesn't exist, use the default get_hf_text_config
            logger.warning(
                f"Config attribute '{hf_config_name}' not found in hf_config, "
                "falling back to default get_hf_text_config"
            )
            return get_hf_text_config(hf_config)

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def create_model_config(self) -> OmniModelConfig:
        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()
        # First, get the base ModelConfig from the parent class
        base_config = super().create_model_config()

        # Create OmniModelConfig by copying all base config attributes
        # and adding the new omni-specific fields
        config_dict = base_config.__dict__.copy()

        # Add the new omni-specific fields
        config_dict["stage_id"] = self.stage_id
        config_dict["async_chunk"] = self.async_chunk
        config_dict["model_stage"] = self.model_stage
        config_dict["model_arch"] = self.model_arch
        config_dict["engine_output_type"] = self.engine_output_type
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id
        config_dict["stage_connector_config"] = stage_connector_config

        config_dict["hf_config_name"] = self.hf_config_name
        config_dict["custom_process_next_stage_input_func"] = self.custom_process_next_stage_input_func
        config_dict["omni_kv_config"] = self.omni_kv_config
        if self.hf_config_name is not None:
            config_dict["hf_text_config"] = self.draw_hf_text_config(config_dict)
        # Create and return the OmniModelConfig instance
        omni_config = OmniModelConfig(**config_dict)
        omni_config.hf_config.architectures = omni_config.architectures

        return omni_config
