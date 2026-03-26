import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger

from vllm_omni.config import OmniModelConfig
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.voxtral_tts.configuration_voxtral_tts import (
            VoxtralTTSConfig,
        )
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    for model_type, config_cls in [
        ("qwen3_tts", Qwen3TTSConfig),
        ("cosyvoice3", CosyVoice3Config),
        ("mistral", VoxtralTTSConfig),
    ]:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            # Already registered elsewhere; ignore.
            pass


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
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    async_chunk: bool = False
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    worker_type: str | None = None
    task_type: str | None = None

    def __post_init__(self) -> None:
        load_omni_general_plugins()
        super().__post_init__()

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "OmniEngineArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
        return engine_args

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

        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        # If model_arch is specified, inject it into hf_overrides so vLLM can
        # resolve the architecture even when config.json lacks 'architectures'.
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])

        # Build the vLLM config first, then use it to create the Omni config.
        model_config = super().create_model_config()

        omni_config = OmniModelConfig.from_vllm_model_config(
            model_config=model_config,
            # All kwargs below are Omni specific
            stage_id=self.stage_id,
            async_chunk=self.async_chunk,
            model_stage=self.model_stage,
            model_arch=self.model_arch,
            worker_type=self.worker_type,
            engine_output_type=self.engine_output_type,
            hf_config_name=self.hf_config_name,
            custom_process_next_stage_input_func=self.custom_process_next_stage_input_func,
            stage_connector_config=stage_connector_config,
            omni_kv_config=self.omni_kv_config,
            task_type=self.task_type,
        )
        return omni_config

    @property
    def output_modality(self) -> OutputModality:
        """Parse engine_output_type into a type-safe OutputModality flag."""
        return OutputModality.from_string(self.engine_output_type)
