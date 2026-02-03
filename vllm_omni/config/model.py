import warnings
from dataclasses import field
from typing import Any

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from vllm.config import ModelConfig, config
from vllm.config.model import (
    _RUNNER_CONVERTS,
    _get_and_verify_dtype,
    get_served_model_name,
)
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode, MultiModalConfig
from vllm.config.pooler import PoolerConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    get_config,
    get_hf_image_processor_config,
    get_hf_text_config,
    get_pooling_config,
)
from vllm.transformers_utils.gguf_utils import is_gguf, maybe_patch_hf_config_from_gguf
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_omni.model_executor.models as me_models

logger = init_logger(__name__)


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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
        # Keep set served_model_name before maybe_model_redirect(self.model)
        self.served_model_name = get_served_model_name(self.model, self.served_model_name)
        self.model = maybe_model_redirect(self.model)
        # The tokenizer is consistent with the model by default.
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision
        self.tokenizer = maybe_model_redirect(self.tokenizer)

        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)

        if callable(self.hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = self.hf_overrides
            dict_overrides: dict[str, Any] = {}
        else:
            # Separate dict overrides from flat ones
            # We'll determine how to apply dict overrides after loading the config
            hf_overrides_kw = {}
            dict_overrides = {}
            for key, value in self.hf_overrides.items():
                if isinstance(value, dict):
                    dict_overrides[key] = value
                else:
                    hf_overrides_kw[key] = value
            hf_overrides_fn = None

        self.maybe_pull_model_tokenizer_for_runai(self.model, self.tokenizer)

        if self.override_attention_dtype is not None and not current_platform.is_rocm():
            warnings.warn(
                "override-attention-dtype is set but not using ROCm platform",
                stacklevel=2,
            )

        if self.enable_sleep_mode and not current_platform.is_sleep_mode_available():
            raise ValueError("Sleep mode is not supported on current platform.")

        hf_config = get_config(
            self.hf_config_path or self.model,
            self.trust_remote_code,
            self.revision,
            self.code_revision,
            self.config_format,
            hf_overrides_kw=hf_overrides_kw,
            hf_overrides_fn=hf_overrides_fn,
        )
        hf_config = maybe_patch_hf_config_from_gguf(
            self.model,
            hf_config,
        )

        self.hf_config = hf_config
        if dict_overrides:
            self._apply_dict_overrides(hf_config, dict_overrides)
        self.hf_text_config = self.draw_hf_text_config()
        self.attention_chunk_size = getattr(self.hf_text_config, "attention_chunk_size", None)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, hf_token=self.hf_token, revision=self.revision
        )
        self.model_arch_config = self.get_model_arch_config()

        if self.convert == "mm_encoder_only":
            logger.warning_once(
                "`--convert mm_encoder_only` is deprecated and "
                "will be removed in v0.15. "
                "Please use --mm-encoder-only` instead."
            )
            mm_encoder_only = True
            self.convert = "none"

        architectures = self.architectures
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(architectures, self)
        is_pooling_model = registry.is_pooling_model(architectures, self)

        self.runner_type = self._get_runner_type(architectures, self.runner)
        self.convert_type = self._get_convert_type(architectures, self.runner_type, self.convert)

        if self.runner_type == "generate" and not is_generative_model:
            generate_converts = _RUNNER_CONVERTS["generate"]
            if self.convert_type not in generate_converts:
                # Currently we don't have any converters for generative models
                raise ValueError("This model does not support `--runner generate`.")
        if self.runner_type == "pooling" and not is_pooling_model:
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model."
                )

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        model_info, arch = registry.inspect_model_cls(architectures, self)
        self._model_info = model_info
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        # Init pooler config if needed
        if self.runner_type == "pooling":
            if self.pooler_config is None:
                self.pooler_config = PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(self.pooler_config, k) is None:
                        setattr(self.pooler_config, k, v)

            default_seq_pooling_type = self._model_info.default_seq_pooling_type
            if self.pooler_config.seq_pooling_type is None:
                self.pooler_config.seq_pooling_type = default_seq_pooling_type
            default_tok_pooling_type = self._model_info.default_tok_pooling_type
            if self.pooler_config.tok_pooling_type is None:
                self.pooler_config.tok_pooling_type = default_tok_pooling_type

        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
        )

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)

        if self.is_encoder_decoder:
            self.mm_processor_cache_gb = 0
            logger.info("Encoder-decoder model detected, disabling mm processor cache.")

        # Init multimodal config if needed
        if self._model_info.supports_multimodal:
            if mm_encoder_tp_mode == "data" and not self._model_info.supports_multimodal_encoder_tp_data:
                logger.warning_once(
                    "This model does not support `--mm-encoder-tp-mode data`. "
                    "Falling back to `--mm-encoder-tp-mode weights`."
                )
                mm_encoder_tp_mode = "weights"

            mm_config_kwargs = dict(
                limit_per_prompt=limit_mm_per_prompt,
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

            mm_config_kwargs = {k: v for k, v in mm_config_kwargs.items() if v is not None}

            self.multimodal_config = MultiModalConfig(**mm_config_kwargs)

        # Multimodal GGUF models must use original repo for mm processing
        if is_gguf(self.tokenizer) and self.is_multimodal_model:
            raise ValueError(
                "Loading a multimodal GGUF model needs to use original "
                "tokenizer. Please specify the unquantized hf model's "
                "repo name or path using the --tokenizer argument."
            )

        if self.disable_sliding_window:
            # Set after get_and_verify_max_len to ensure that max_model_len
            # can be correctly capped to sliding window size
            self.hf_text_config.sliding_window = None

        # Avoid running try_verify_and_update_config multiple times
        self.config_updated = False
        self._try_verify_and_update_model_config()
        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()
