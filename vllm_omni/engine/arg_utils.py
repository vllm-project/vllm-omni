from dataclasses import dataclass, field
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.v1.engine.async_llm import AsyncEngineArgs

from vllm_omni.config import OmniModelConfig
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)

# Maps model architecture names to their HuggingFace model_type values.
# Used when auto-injecting hf_overrides for models with missing config.json.
_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "CosyVoice3Model": "cosyvoice3",
    "OmniVoiceModel": "omnivoice",
    "VoxCPM2TalkerForConditionalGeneration": "voxcpm2",
    "VoxCPMForConditionalGeneration": "voxcpm",
}

# Maps model architecture names to tokenizer subfolder paths within HF repos.
_TOKENIZER_SUBFOLDER_MAP: dict[str, str] = {
    "CosyVoice3Model": "CosyVoice-BlankEN",
}


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
        from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.voxtral_tts.configuration_voxtral_tts import (
            VoxtralTTSConfig,
        )
        from vllm_omni.transformers_utils.configs.voxcpm import VoxCPMConfig
        from vllm_omni.transformers_utils.configs.voxcpm2 import VoxCPM2Config
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    # Register with both transformers AutoConfig and vLLM's config registry
    # so models with empty/missing config.json (e.g. CosyVoice3) can be
    # resolved when model_type is injected via hf_overrides.
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except ImportError:
        _CONFIG_REGISTRY = None

    for model_type, config_cls in [
        ("qwen3_tts", Qwen3TTSConfig),
        ("cosyvoice3", CosyVoice3Config),
        ("omnivoice", OmniVoiceConfig),
        ("voxtral_tts", VoxtralTTSConfig),
        ("voxcpm", VoxCPMConfig),
        ("voxcpm2", VoxCPM2Config),
    ]:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            # Already registered elsewhere; ignore.
            pass
        if _CONFIG_REGISTRY is not None and model_type not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY[model_type] = config_cls

def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

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
        stage_id: Identifier for the stage in a multi-stage pipeline.
            Defaults to 0 for per-stage engine construction. The CLI-level
            single-stage selector remains optional on the parsed argparse
            namespace and should not be forwarded as a nullable per-stage
            engine argument.
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
        hf_config_name: Optional key for HF config subkey to be extracted
            for this stage, e.g., talker_config; If None, the default
            HF config will be used.
        custom_process_next_stage_input_func: Optional path to a custom function for processing
            inputs from previous stages
            If None, default processing is used.
        stage_connector_spec: Extra configuration for stage connector
        async_chunk: If set to True, perform async chunk
        worker_type: Model Type, e.g., "ar" or "generation"
        task_type: Default task type for TTS models (CustomVoice, VoiceDesign, or Base).
            If not specified, will be inferred from model path.
        omni_master_address: TCP address that the OmniMasterServer (running
            inside AsyncOmniEngine) listens on for engine core registrations.
            Required when single-stage mode is active.
        omni_master_port: TCP port for the OmniMasterServer registration
            socket.  Required when single-stage mode is active.
        stage_configs_path: Optional path to a JSON/YAML file containing
            stage configurations for the multi-stage pipeline. If None,
            stage configs are resolved from the model's default configuration.
        output_modalities: Optional list of output modality names to enable
            (e.g. ["text", "audio"]). If None, all modalities supported by
            the model are used.
        log_stats: Whether to log engine statistics. Defaults to False.
        custom_pipeline_args: Dictionary of arguments for custom pipeline
            initialization (e.g., ``{"pipeline_class": "my.Module"}``).
            Passed through to the diffusion stage engine.
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    async_chunk: bool = False
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    worker_type: str | None = None
    task_type: str | None = None
    omni_master_address: str | None = None
    omni_master_port: int | None = None
    stage_configs_path: str | None = None
    output_modalities: list[str] | None = None
    log_stats: bool = False
    custom_pipeline_args: dict[str, Any] | None = None

    def draw_hf_text_config(self, config_dict: dict) -> Any:
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

    @property
    def output_modality(self) -> OutputModality:
        """Parse engine_output_type into a type-safe OutputModality flag."""
        return OutputModality.from_string(self.engine_output_type)


@dataclass
class AsyncOmniEngineArgs(AsyncEngineArgs):
    """Async engine arguments for omni LLM stages.

    Extends AsyncEngineArgs with omni-specific multi-stage pipeline fields.
    Used when launching LLM stages (stage_type=llm) within an async context.
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    async_chunk: bool = False
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    worker_type: str | None = None

    def __post_init__(self) -> None:
        load_omni_general_plugins()
        super().__post_init__()

    def create_engine_config(self, usage_context=None, **kwargs):
        """Create engine config, injecting model_arch into hf_overrides if set."""
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])
        return super().create_engine_config(usage_context=usage_context, **kwargs)
