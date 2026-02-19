from dataclasses import dataclass, field
from typing import Any

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import is_gguf

from vllm_omni.config import OmniModelConfig
from vllm_omni.plugins import load_omni_general_plugins

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
        worker_type: Model Type, e.g., "ar" or "generation"
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
    worker_type: str | None = None

    def __post_init__(self) -> None:
        load_omni_general_plugins()
        super().__post_init__()

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
        # GGUF files need a specific model loader path in vLLM.
        if is_gguf(self.model):
            self.quantization = self.load_format = "gguf"

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()

        # Keep compatibility when async args are constructed from partial payloads.
        limit_mm_per_prompt = getattr(self, "limit_mm_per_prompt", {})
        enable_mm_embeds = getattr(self, "enable_mm_embeds", False)
        interleave_mm_strings = getattr(self, "interleave_mm_strings", False)
        media_io_kwargs = getattr(self, "media_io_kwargs", {})
        skip_mm_profiling = getattr(self, "skip_mm_profiling", False)
        mm_processor_kwargs = getattr(self, "mm_processor_kwargs", None)
        mm_processor_cache_gb = getattr(self, "mm_processor_cache_gb", 4)
        mm_processor_cache_type = getattr(self, "mm_processor_cache_type", None)
        mm_shm_cache_max_object_size_mb = getattr(self, "mm_shm_cache_max_object_size_mb", 128)
        mm_encoder_only = getattr(self, "mm_encoder_only", False)
        mm_encoder_tp_mode = getattr(self, "mm_encoder_tp_mode", "weights")
        mm_encoder_attn_backend = getattr(self, "mm_encoder_attn_backend", None)
        video_pruning_rate = getattr(self, "video_pruning_rate", 0.0)

        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        # Create OmniModelConfig directly from engine args
        # Note: We pass the actual init parameters matching vLLM's EngineArgs.create_model_config()
        omni_config = OmniModelConfig(
            # Base ModelConfig fields (matching vLLM's EngineArgs.create_model_config)
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            allowed_media_domains=self.allowed_media_domains,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            enable_return_routed_experts=self.enable_return_routed_experts,
            max_logprobs=self.max_logprobs,
            logprobs_mode=self.logprobs_mode,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_mm_embeds=enable_mm_embeds,
            interleave_mm_strings=interleave_mm_strings,
            media_io_kwargs=media_io_kwargs,
            skip_mm_profiling=skip_mm_profiling,
            config_format=self.config_format,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=mm_processor_cache_gb,
            mm_processor_cache_type=mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
            mm_encoder_only=mm_encoder_only,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            pooler_config=self.pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
            override_attention_dtype=self.override_attention_dtype,
            logits_processors=self.logits_processors,
            video_pruning_rate=video_pruning_rate,
            io_processor_plugin=self.io_processor_plugin,
            # Omni-specific fields
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
        )
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
        worker_type: Model Type, e.g., "ar" or "generation"
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
    worker_type: str | None = None

    def __post_init__(self) -> None:
        load_omni_general_plugins()
        super().__post_init__()

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
        # GGUF files need a specific model loader path in vLLM.
        if is_gguf(self.model):
            self.quantization = self.load_format = "gguf"

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()

        # Keep compatibility when async args are constructed from partial payloads.
        limit_mm_per_prompt = getattr(self, "limit_mm_per_prompt", {})
        enable_mm_embeds = getattr(self, "enable_mm_embeds", False)
        interleave_mm_strings = getattr(self, "interleave_mm_strings", False)
        media_io_kwargs = getattr(self, "media_io_kwargs", {})
        skip_mm_profiling = getattr(self, "skip_mm_profiling", False)
        mm_processor_kwargs = getattr(self, "mm_processor_kwargs", None)
        mm_processor_cache_gb = getattr(self, "mm_processor_cache_gb", 4)
        mm_processor_cache_type = getattr(self, "mm_processor_cache_type", None)
        mm_shm_cache_max_object_size_mb = getattr(self, "mm_shm_cache_max_object_size_mb", 128)
        mm_encoder_only = getattr(self, "mm_encoder_only", False)
        mm_encoder_tp_mode = getattr(self, "mm_encoder_tp_mode", "weights")
        mm_encoder_attn_backend = getattr(self, "mm_encoder_attn_backend", None)
        video_pruning_rate = getattr(self, "video_pruning_rate", 0.0)

        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        # Create OmniModelConfig directly from engine args
        # Note: We pass the actual init parameters matching vLLM's EngineArgs.create_model_config()
        omni_config = OmniModelConfig(
            # Base ModelConfig fields (matching vLLM's EngineArgs.create_model_config)
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            allowed_media_domains=self.allowed_media_domains,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            enable_return_routed_experts=self.enable_return_routed_experts,
            max_logprobs=self.max_logprobs,
            logprobs_mode=self.logprobs_mode,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_mm_embeds=enable_mm_embeds,
            interleave_mm_strings=interleave_mm_strings,
            media_io_kwargs=media_io_kwargs,
            skip_mm_profiling=skip_mm_profiling,
            config_format=self.config_format,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=mm_processor_cache_gb,
            mm_processor_cache_type=mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
            mm_encoder_only=mm_encoder_only,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            pooler_config=self.pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
            override_attention_dtype=self.override_attention_dtype,
            logits_processors=self.logits_processors,
            video_pruning_rate=video_pruning_rate,
            io_processor_plugin=self.io_processor_plugin,
            # Omni-specific fields
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
        )
        omni_config.hf_config.architectures = omni_config.architectures

        return omni_config
