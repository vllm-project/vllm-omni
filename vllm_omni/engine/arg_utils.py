import argparse
import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger

from vllm_omni.config import OmniModelConfig
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)

# Maps model architecture names to their HuggingFace model_type values.
# Used when auto-injecting hf_overrides for models with missing config.json.
_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "CosyVoice3Model": "cosyvoice3",
    "OmniVoiceModel": "omnivoice",
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

    def _patch_empty_hf_config(self, model_type: str) -> None:
        """For models with empty config.json (e.g. CosyVoice3), create a
        patched config in a temp directory with model_type set so that
        transformers AutoConfig.from_pretrained can resolve the config class.
        Sets self.hf_config_path to point to the patched directory."""
        try:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(self.model)
            if config_dict.get("model_type"):
                return  # config.json already has model_type, no patching needed
        except Exception:
            return  # can't load config, let vLLM handle the error

        # Create a temp dir with a patched config.json
        temp_dir = tempfile.mkdtemp(prefix="omni_hf_config_")
        config_dict["model_type"] = model_type
        config_dict.setdefault("architectures", [self.model_arch])
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        self.hf_config_path = temp_dir
        self._temp_config_dir = temp_dir
        logger.info("Patched empty HF config with model_type=%s at %s", model_type, temp_dir)

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
        # Also inject model_type so AutoConfig can resolve the correct config
        # class for models with empty or missing config.json (e.g. CosyVoice3).
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])
                if "model_type" not in self.hf_overrides:
                    model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                    if model_type is not None:
                        self.hf_overrides.setdefault("model_type", model_type)

            # For models whose HF config.json is empty or lacks model_type
            # (e.g. CosyVoice3), AutoConfig.from_pretrained fails because it
            # cannot determine which config class to use from the empty dict.
            # hf_overrides alone is not enough since transformers reads
            # model_type from config_dict before applying overrides.
            # Workaround: create a patched config.json in a temp directory
            # and point hf_config_path to it so vLLM reads model_type from it.
            if not self.hf_config_path:
                model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                if model_type is not None:
                    self._patch_empty_hf_config(model_type)

        # Auto-detect tokenizer for models that store it in a subdirectory
        # rather than the root (e.g. CosyVoice3 uses CosyVoice-BlankEN/).
        if not self.tokenizer and self.model:
            model_path = self.model
            if os.path.isdir(model_path) and not os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
                for subfolder in sorted(os.listdir(model_path)):
                    candidate = os.path.join(model_path, subfolder)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                        self.tokenizer = candidate
                        logger.info("Auto-detected tokenizer at %s", candidate)
                        break
            elif not os.path.isdir(model_path):
                subfolder = _TOKENIZER_SUBFOLDER_MAP.get(self.model_arch)
                if subfolder:
                    # Download just the tokenizer files from the subfolder
                    try:
                        from huggingface_hub import snapshot_download

                        local_dir = snapshot_download(
                            model_path,
                            allow_patterns=[
                                f"{subfolder}/tokenizer*",
                                f"{subfolder}/special_tokens*",
                                f"{subfolder}/vocab*",
                                f"{subfolder}/merges*",
                                f"{subfolder}/added_tokens*",
                            ],
                        )
                        candidate = os.path.join(local_dir, subfolder)
                        if os.path.isdir(candidate):
                            self.tokenizer = candidate
                            logger.info("Downloaded tokenizer from %s/%s", model_path, subfolder)
                    except Exception as e:
                        logger.warning("Failed to download tokenizer subfolder: %s", e)

        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
        finally:
            # Clean up temp config dir if we created one
            if hasattr(self, "_temp_config_dir"):
                import shutil

                shutil.rmtree(self._temp_config_dir, ignore_errors=True)
                del self._temp_config_dir

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
