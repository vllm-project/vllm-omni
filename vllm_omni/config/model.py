from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from vllm.config import ModelConfig
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_omni.model_executor.models as me_models

logger = init_logger(__name__)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
    """Configuration for Omni models, extending the base ModelConfig.

    This configuration class extends the base vLLM ModelConfig with
    omni-specific fields for multi-stage pipeline processing.

    Attributes:
        stage_id: Identifier for the stage in a multi-stage pipeline (default: 0)
        async_chunk: If set to True, perform async chunk
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        worker_type: Model Type, e.g., "ar" or "generation"
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
        stage_connector_config: Stage connector configuration dictionary.
            Contains "name" (connector name), "extra" (extra connector config).

    Example:
        >>> config = OmniModelConfig(
        ...     stage_id=0,
        ...     model_stage="thinker",
        ...     model_arch="Qwen2_5OmniForConditionalGeneration"
        ... )
    """

    stage_id: int = 0
    async_chunk: bool = False
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    worker_type: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "SharedMemoryConnector",
            "extra": {},
        }
    )
    omni_kv_config: dict | None = None

    @property
    def registry(self):
        return me_models.OmniModelRegistry

    @property
    def architectures(self) -> list[str]:
        return [self.model_arch]

    @property
    def embedding_size(self):
        if self.hf_config_name is not None:
            stage_config = getattr(self.hf_config, self.hf_config_name, None)
            override = getattr(stage_config, "embedding_size", None)
            if override is not None:
                return override
        return super().embedding_size

    def draw_hf_text_config(self):
        # transformers' get_text_config method is used to get the text config from thinker_config.
        # to handle the case that each model stage has their own text config,
        # we need to draw the text config from the corresponding model stage.
        if self.hf_config_name is None:
            return get_hf_text_config(self.hf_config)
        try:
            # Try to get the stage-specific config (e.g., thinker_config, talker_config)
            stage_config = getattr(self.hf_config, self.hf_config_name)
            return stage_config.get_text_config()
        except AttributeError:
            # Fallback: if the attribute doesn't exist, use the default get_hf_text_config
            logger.warning(
                f"Config attribute '{self.hf_config_name}' not found in hf_config, "
                "falling back to default get_hf_text_config"
            )
            return get_hf_text_config(self.hf_config)

    def __post_init__(
        self,
        # Multimodal config init vars
        limit_mm_per_prompt: dict[str, int | dict[str, int]] | None,
        enable_mm_embeds: bool | None,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
        mm_processor_kwargs: dict[str, Any] | None,
        mm_processor_cache_gb: float | None,
        mm_processor_cache_type: MMCacheType | None,
        mm_shm_cache_max_object_size_mb: int | None,
        mm_encoder_only: bool | None,
        mm_encoder_tp_mode: MMEncoderTPMode | None,
        mm_encoder_attn_backend: AttentionBackendEnum | str | None,
        interleave_mm_strings: bool | None,
        skip_mm_profiling: bool | None,
        video_pruning_rate: float | None,
    ) -> None:
        # Call parent's __post_init__ to handle all standard ModelConfig initialization
        super().__post_init__(
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_mm_embeds=enable_mm_embeds,
            media_io_kwargs=media_io_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_cache_gb=mm_processor_cache_gb,
            mm_processor_cache_type=mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
            mm_encoder_only=mm_encoder_only,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            interleave_mm_strings=interleave_mm_strings,
            skip_mm_profiling=skip_mm_profiling,
            video_pruning_rate=video_pruning_rate,
        )

        # Override hf_text_config with omni-specific logic for multi-stage models
        # (e.g., thinker_config, talker_config)
        new_hf_text_config = self.draw_hf_text_config()
        if new_hf_text_config is not self.hf_text_config:
            self.hf_text_config = new_hf_text_config
            # Recalculate dependent attributes
            self.attention_chunk_size = getattr(self.hf_text_config, "attention_chunk_size", None)
            # Recalculate max_model_len since it depends on hf_text_config
            self.max_model_len = self.get_and_verify_max_len(self.original_max_model_len)
            # Reset sliding_window if needed
            if self.disable_sliding_window:
                self.hf_text_config.sliding_window = None
