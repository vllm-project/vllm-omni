import warnings
from importlib.util import find_spec
from typing import Any

import torch
import vllm.envs as envs
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import ModelConfig, config
from vllm.config.model import (
    _RUNNER_CONVERTS,
    _RUNNER_TASKS,
    ConvertOption,
    ConvertType,
    RunnerOption,
    TaskOption,
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
from vllm.transformers_utils.gguf_utils import (
    maybe_patch_hf_config_from_gguf,
)
from vllm.transformers_utils.utils import (
    is_gguf,
    maybe_model_redirect,
)

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
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.

    Example:
        >>> config = OmniModelConfig(
        ...     stage_id=0,
        ...     model_stage="thinker",
        ...     model_arch="Qwen2_5OmniForConditionalGeneration"
        ... )
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: str | None = None
    hf_config_name: str | None = None

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
        return getattr(self.hf_config, self.hf_config_name).get_text_config()

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

        if (backend := envs.VLLM_ATTENTION_BACKEND) and backend == "FLASHINFER" and find_spec("flashinfer") is None:
            raise ValueError(
                "VLLM_ATTENTION_BACKEND is set to FLASHINFER, but flashinfer "
                "module was not found. See "
                "https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile "  # noqa: E501
                "for instructions on how to install it."
            )

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

        architectures = self.architectures
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(architectures, self)
        is_pooling_model = registry.is_pooling_model(architectures, self)

        def _task_to_convert(task: TaskOption) -> ConvertType:
            if task == "embedding" or task == "embed":
                return "embed"
            if task == "classify":
                return "classify"
            if task == "reward":
                return "reward"
            if task == "score":
                new_task = self._get_default_pooling_task(architectures)
                return "classify" if new_task == "classify" else "embed"

            return "none"

        if self.task is not None:
            runner: RunnerOption = "auto"
            convert: ConvertOption = "auto"
            msg_prefix = (
                "The 'task' option has been deprecated and will be removed in v0.13.0 or v1.0, whichever comes first."
            )
            msg_hint = "Please remove this option."

            is_generative_task = self.task in _RUNNER_TASKS["generate"]
            is_pooling_task = self.task in _RUNNER_TASKS["pooling"]

            if is_generative_model and is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = (
                        "Please replace this option with `--runner "
                        "generate` to continue using this model "
                        "as a generative model."
                    )
                elif is_pooling_task:
                    runner = "pooling"
                    convert = "auto"
                    msg_hint = (
                        "Please replace this option with `--runner "
                        "pooling` to continue using this model "
                        "as a pooling model."
                    )
                else:  # task == "auto"
                    pass
            elif is_generative_model or is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = "Please remove this option"
                elif is_pooling_task:
                    runner = "pooling"
                    convert = _task_to_convert(self.task)
                    msg_hint = (
                        "Please replace this option with `--convert "
                        f"{convert}` to continue using this model "
                        "as a pooling model."
                    )
                else:  # task == "auto"
                    pass
            else:
                # Neither generative nor pooling model - try to convert if possible
                if is_pooling_task:
                    runner = "pooling"
                    convert = _task_to_convert(self.task)
                    msg_hint = (
                        "Please replace this option with `--runner pooling "
                        f"--convert {convert}` to continue using this model "
                        "as a pooling model."
                    )
                else:
                    debug_info = {
                        "architectures": architectures,
                        "is_generative_model": is_generative_model,
                        "is_pooling_model": is_pooling_model,
                    }
                    raise AssertionError(
                        "The model should be a generative or "
                        "pooling model when task is set to "
                        f"{self.task!r}. Found: {debug_info}"
                    )

            self.runner = runner
            self.convert = convert

            msg = f"{msg_prefix} {msg_hint}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

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

            default_pooling_type = self._model_info.default_pooling_type
            if self.pooler_config.pooling_type is None:
                self.pooler_config.pooling_type = default_pooling_type

        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
        )

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
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

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()
