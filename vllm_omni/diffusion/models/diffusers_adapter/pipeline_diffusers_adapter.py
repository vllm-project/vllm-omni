# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusers backend adapter for vLLM-Omni.

Provides a black-box wrapper around any 🤗 Diffusers pipeline, enabling
vLLM-Omni to directly serve Diffusers models with near-zero per-model code.

The adapter delegates full pipeline execution to diffusers' ``__call__()``.
It does NOT support:
- CFG parallel (diffusers handles CFG via guidance_scale internally)
- Sequence parallel (requires model-specific attention surgery)
- TeaCache / Cache-DiT (requires hooking into transformer blocks)
- Step-wise execution (continuous batching)
"""

import inspect
import logging
import os
import re
from typing import Any, cast

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import BasePipelineUtils, get_pipeline_utils
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniPromptType, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


class DiffusersAdapterPipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """Black-box adapter that delegates full pipeline execution to a diffusers pipeline.

    Usage::

        adapter = DiffusersAdapterPipeline(od_config=od_config)
        adapter.load_weights()  # calls DiffusionPipeline.from_pretrained()
        output = adapter.forward(req)

    Step-wise execution is explicitly rejected — diffusers encapsulates the
    full denoising loop internally. Use native pipelines for continuous
    batching mode.
    """

    supports_step_execution: bool = False

    def __init__(self, *, od_config: OmniDiffusionConfig, device: torch.device | None = None):
        super().__init__()
        self._pipeline: DiffusionPipeline
        self._accept_call_kwargs: set[str] | None = None  # None to accept all kwargs
        self.od_config = od_config
        self.device = device
        self._capabilities: dict[str, Any] = {}
        self._pipeline_utils: BasePipelineUtils = BasePipelineUtils()
        self._raise_unsupported_features()

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
            profiler_targets=["forward"],
        )
        if od_config.enable_diffusion_pipeline_profiler:
            logger.info("Profiling enabled for DiffusersAdapterPipeline. Only 'forward' is supported.")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self) -> None:
        """Load the diffusers pipeline via ``DiffusionPipeline.from_pretrained()``."""

        model_id = self.od_config.model
        dtype = self.od_config.dtype

        load_kwargs = {
            "torch_dtype": dtype,
            **self.od_config.diffusers_load_kwargs,
        }
        logger.debug(f"Loading diffusers pipeline with kwargs: {load_kwargs}")

        pipeline_class = self.od_config.diffusers_pipeline_cls
        pipeline_class_name = pipeline_class.__name__ if pipeline_class is not None else None
        self._pipeline_utils = get_pipeline_utils(pipeline_class_name)
        self._pipeline_utils.update_load_kwargs(self.od_config, load_kwargs)

        self._pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        self._pipeline_utils.apply_post_load_updates(self._pipeline, self.od_config)

        self._pipeline.to(self.device)

        # Cache __call__kwargs signature introspection for later input validation
        self._accept_call_kwargs = set(inspect.signature(self._pipeline.__call__).parameters.keys())

        # CPU offloading
        if self.od_config.enable_layerwise_offload:
            self._pipeline.enable_sequential_cpu_offload()
        elif self.od_config.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

        # VAE slicing and tiling: try-catch because not all models have VAE
        if self.od_config.vae_use_slicing:
            try:
                self._pipeline.enable_vae_slicing()
            except Exception as e:
                logger.warning(
                    f"Failed to enable VAE slicing for diffusers pipeline {self._pipeline.__class__.__name__}: {e}"
                )
        if self.od_config.vae_use_tiling:
            try:
                self._pipeline.enable_vae_tiling()
            except Exception as e:
                logger.warning(
                    f"Failed to enable VAE tiling for diffusers pipeline {self._pipeline.__class__.__name__}: {e}"
                )

        # Attention backend
        self._set_attention_backend()

    # ------------------------------------------------------------------
    # Step-wise execution — explicitly rejected
    # ------------------------------------------------------------------

    def prepare_encode(self, **_: Any) -> Any:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    def denoise_step(self, **_: Any) -> torch.Tensor | None:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    def step_scheduler(self, **_: Any) -> None:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    def post_decode(self, **_: Any) -> Any:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Full delegation to diffusers ``pipeline.__call__()``."""

        kwargs = self._build_call_kwargs(req)
        logger.debug(f"Calling diffusers pipeline with kwargs: {kwargs}")

        with torch.inference_mode():
            output = self._pipeline(**kwargs)  # pyright: ignore[reportCallIssue]

        return self._wrap_output(output)

    # ------------------------------------------------------------------
    # Validation guards
    # ------------------------------------------------------------------

    def _raise_unsupported_features(self) -> None:
        """Raise an error for incompatible feature switches."""
        pc = self.od_config.parallel_config
        if pc.cfg_parallel_size > 1:
            raise NotImplementedError(
                "CFG parallel is not supported with the diffusers backend. "
                "Diffusers handles CFG internally via guidance_scale."
            )
        if pc.sequence_parallel_size is not None and pc.sequence_parallel_size > 1:
            raise NotImplementedError(
                "Sequence parallel is not supported with the diffusers backend. "
                "It requires model-specific attention surgery."
            )
        if self.od_config.cache_backend not in ("none", None):
            raise NotImplementedError(
                f"Cache backend '{self.od_config.cache_backend}' is not supported "
                "with the diffusers backend. TeaCache/Cache-DiT require hooking "
                "into individual transformer blocks."
            )
        if self.od_config.enforce_eager:
            raise NotImplementedError(
                "Eager execution is not supported with the diffusers backend. "
                "Use a native pipeline for continuous batching mode."
            )
        if self.od_config.quantization_config is not None:
            raise NotImplementedError(
                "Quantization is not supported with the diffusers backend. Use a native pipeline for quantization."
            )

    # ------------------------------------------------------------------
    # Wrap settings, inputs, and outputs
    # ------------------------------------------------------------------

    def _set_attention_backend(self) -> None:
        """Set the attention backend.

        Roughly follow the logic in vllm_omni/diffusion/attention/backends/utils/fa.py,
        But also consider the available attention backends in diffusers.
        (See: https://huggingface.co/docs/diffusers/optimization/attention_backends)
        """
        if not hasattr(self._pipeline, "transformer"):
            logging.info("No transformer found in diffusers pipeline. Skipping attention backend setting.")
            return

        attention_backend_config = self.od_config.attention_backend or os.environ.get("DIFFUSION_ATTENTION_BACKEND")
        attention_backend_attempts: list[str] = []
        match attention_backend_config:
            case "FLASH_ATTN" | None:
                if current_omni_platform.is_rocm():
                    attention_backend_attempts.append("aiter")
                elif current_omni_platform.is_xpu():
                    attention_backend_attempts.append("_native_xla")
                elif current_omni_platform.is_musa():
                    logger.warning(
                        "Unknown diffusers attention backend option for MUSA platform. Falling back to SDPA."
                    )
                    attention_backend_attempts.append("native")
                else:
                    attention_backend_attempts.extend(
                        [
                            "_flash_3_hub",
                            "_flash_3_varlen_hub",
                            "_flash_3",
                            "_flash_varlen_3",
                            "flash_hub",
                            "flash_varlen_hub",
                            "flash",
                            "flash_varlen",
                            "_native_flash",
                        ]
                    )
            case "SAGE_ATTN":
                attention_backend_attempts.extend(["sage_hub", "sage", "sage", "sage_varlen"])
            case "ASCEND":
                attention_backend_attempts.append("_native_npu")
            case "TORCH_SDPA":
                attention_backend_attempts.append("native")
            case _:
                logger.warning(f"Invalid attention backend: {attention_backend_config}. Falling back to SDPA.")
                attention_backend_attempts.append("native")

        attempt_errors: list[str] = []
        set_backend: str | None = None
        for backend in attention_backend_attempts:
            try:
                self._pipeline.transformer.set_attention_backend(backend)
                set_backend = backend
                break
            except Exception as e:
                attempt_errors.append(str(e))

        # If all attempts fail, fallback to SDPA and warn the user about the failures
        if len(attempt_errors) == len(attention_backend_attempts):
            self._pipeline.transformer.set_attention_backend("native")
            logger.warning(
                f"Failed to set attention backend '{attention_backend_config}' for "
                f"diffusers pipeline {self._pipeline.__class__.__name__}. "
                "Falling back to SDPA. "
                f"The following attempts were made: {dict(zip(attention_backend_attempts, attempt_errors))}"
            )
            return

        # If some attempts fail, only warn the user about the failures
        logger.info(
            f"Set diffusers attention backend to '{set_backend}', adapted from "
            f"user config value '{attention_backend_config}'."
        )
        if len(attempt_errors) > 0:
            logger.warning(
                f"The following failed attempts were made before choosing this diffusers backend: "
                f"{dict(zip(attention_backend_attempts, attempt_errors))}"
            )

    def _build_call_kwargs(self, req: OmniDiffusionRequest) -> dict[str, Any]:
        """Translate ``OmniDiffusionRequest`` into diffusers ``__call__`` kwargs."""
        sampling = req.sampling_params
        input_kwargs = self._extract_input(req.prompts)

        self._pipeline_utils.validate_runtime_sampling_params(sampling)

        # Merge user-provided call kwargs from stage/CLI defaults.
        # Load time defaults -> input kwargs (prompts, neg prompts, images...) -> request-time sampling params
        kwargs: dict[str, Any] = {}

        # Load time defaults
        for key, value in self.od_config.diffusers_call_kwargs.items():
            if self._accept_call_kwargs is None or key in self._accept_call_kwargs:
                kwargs[key] = value
            else:
                logger.warning(
                    f"Skipping unsupported diffusers pipeline __call__ argument `{key}` from "
                    f"diffusers_call_kwargs. Check out the documentation of {self._pipeline.__class__.__name__}."
                )

        # Input kwargs
        for key, value in input_kwargs.items():
            if self._accept_call_kwargs is None or key in self._accept_call_kwargs:
                kwargs[key] = value
            else:
                logger.warning(
                    f"Skipping unsupported diffusers pipeline __call__ argument `{key}` from prompt input."
                    f"Check out the documentation of {self._pipeline.__class__.__name__}."
                )

        # Request-time sampling params
        for key, value in sampling.__dict__.items():
            if value is None:
                continue
            if self._accept_call_kwargs is None or key in self._accept_call_kwargs:
                kwargs[key] = value

        # Special format fields in sampling params
        if output_type := sampling.output_type or self.od_config.output_type:
            kwargs["output_type"] = output_type

        if (num_outputs_per_prompt := sampling.num_outputs_per_prompt) > 0:
            # In diffusers, they are num_images_per_prompt, num_videos_per_prompt, etc.
            for key in self._accept_call_kwargs or ():
                if re.match(r"num_[a-z]+_per_prompt", key):
                    kwargs[key] = num_outputs_per_prompt

        if sampling.generator is not None:
            kwargs["generator"] = sampling.generator
        elif sampling.seed is not None:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device).manual_seed(sampling.seed)
        else:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device)

        logger.info(
            "Calling diffusers pipeline with kwargs: %s", DiffusersAdapterPipeline._summarize_call_kwargs_value(kwargs)
        )

        return kwargs

    def _extract_input(self, prompt_obj: list[OmniPromptType]) -> dict[str, Any]:
        """Extract the text prompts and negative prompts from a list of prompt objects."""
        if len(prompt_obj) == 1:
            if isinstance(prompt_obj[0], str):
                return {"prompt": prompt_obj[0]}
            else:
                obj = cast(OmniTextPrompt, prompt_obj[0])
                negative_prompt = obj.get("negative_prompt")
                multi_modal_data = obj.get("multi_modal_data") or {}
                kwargs = {
                    "prompt": obj.get("prompt", ""),
                    **multi_modal_data,
                }
                if negative_prompt is not None:
                    kwargs["negative_prompt"] = negative_prompt
                return kwargs

        # Check the first element for the presence of multimodal data.
        # The following elements should have the same multimodal data fields, or none has multimodal data.
        multi_modal_data_fields: list[str] = []
        if isinstance(prompt_obj[0], dict) and (multi_modal_data := prompt_obj[0].get("multi_modal_data")):
            multi_modal_data_fields = list(multi_modal_data.keys())
        if multi_modal_data_fields:
            for i, prompt in enumerate(prompt_obj):
                assert isinstance(prompt, dict) and (multi_modal_data := prompt.get("multi_modal_data")) is not None, (
                    "When there are multiple prompts and the first prompt has multimodal data, "
                    f"each prompt should also contain the same multimodal data fields, but prompt {i} does not."
                )
                for key in multi_modal_data_fields:
                    assert key in multi_modal_data, (
                        "When there are multiple prompts and the first prompt has multimodal data, each prompt should "
                        f"also contain the same multimodal data fields, but prompt {i} does not contain {key}."
                    )
                    assert not isinstance(multi_modal_data.get(key), list), (
                        f"When there are multiple prompts and each prompt has multiple {key} data, this input pattern "
                        "is ambiguous as diffusers accepts flattened lists of text prompts and multimodal data. "
                        f"To use multiple {key} data, please use one single prompt instead."
                    )

        input_kwargs = {"prompt": [], **{key: [] for key in multi_modal_data_fields}}

        # Negative prompt rule:
        # - If any OmniTextPrompt has a negative prompt, or diffusers_call_kwargs has `negative_prompt`,
        #     enforce a negative prompt input (list[str]) -> `kwargs_should_contain_negative_prompt=true`
        #     (Because the negative prompt must be str, list[str], or None. It cannot be list[str|None])
        # -   Further in this case, try:
        # -   1. negative prompt in this OmniTextPrompt (typed dict)
        # -   2. fallback negative prompt from diffusers_call_kwargs (single str or the i-th item in list[str])
        # -   3. empty string ""
        # - Otherwise, `kwargs_should_contain_negative_prompt=False`. Do not add "negative_prompt" key to input_kwargs.
        has_negative_prompt = any(isinstance(p, dict) and p.get("negative_prompt") is not None for p in prompt_obj)
        fallback_negative_prompt = self.od_config.diffusers_call_kwargs.get("negative_prompt")
        kwargs_should_contain_negative_prompt = has_negative_prompt or fallback_negative_prompt is not None
        if kwargs_should_contain_negative_prompt:
            input_kwargs["negative_prompt"] = []

        for i, prompt in enumerate(prompt_obj):
            this_fallback_negative_prompt = ""
            if isinstance(fallback_negative_prompt, str):
                this_fallback_negative_prompt = fallback_negative_prompt
            elif isinstance(fallback_negative_prompt, list):
                try:
                    this_fallback_negative_prompt = fallback_negative_prompt[i]
                except IndexError:
                    raise ValueError(
                        "The fallback negative_prompt in diffusers_call_kwargs is a list, but its length "
                        f"({len(fallback_negative_prompt)}) is less than the number of prompts ({len(prompt_obj)}). "
                        "Please provide a list with the same length as the number of prompts."
                    )

            if isinstance(prompt, str):
                input_kwargs["prompt"].append(prompt)
                if kwargs_should_contain_negative_prompt:
                    input_kwargs["negative_prompt"].append(this_fallback_negative_prompt)
                for key in multi_modal_data_fields:
                    input_kwargs[key].append(None)
            else:
                obj = cast(OmniTextPrompt, prompt)
                input_kwargs["prompt"].append(obj.get("prompt", ""))

                if kwargs_should_contain_negative_prompt:
                    negative_prompt: str = obj.get("negative_prompt", this_fallback_negative_prompt)
                    input_kwargs["negative_prompt"].append(negative_prompt)

                multi_modal_data = obj.get("multi_modal_data") or {}
                for key in multi_modal_data_fields:
                    input_kwargs[key].append(multi_modal_data.get(key))
        return input_kwargs

    def _wrap_output(self, output: Any) -> DiffusionOutput:
        """Convert diffusers pipeline output to ``DiffusionOutput``.

        Diffusers output types:
        - ``ImagePipelineOutput(images=...)`` — text2img, img2img
        - ``VideoPipelineOutput(frames=...)`` — text2vid, img2vid
        """
        from vllm_omni.diffusion.data import DiffusionOutput

        if hasattr(output, "images"):
            # Preserve diffusers image format (`output_type`)
            return DiffusionOutput(output=output.images)

        if hasattr(output, "frames"):
            # Preserve diffusers video format (`output_type`)
            return DiffusionOutput(output=output.frames)

        if hasattr(output, "audios"):
            return DiffusionOutput(output=output.audios)

        return DiffusionOutput(output=output)

    @staticmethod
    def _summarize_call_kwargs_value(value: Any) -> Any:
        """Return a sanitized summary of diffusers call kwargs for logging."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) < 20:
                return value
            return f"{value[:10]}...{value[-10:]}"
        if isinstance(value, torch.Tensor):
            return f"Tensor with shape {tuple(value.shape)}, dtype {value.dtype}, device {value.device}"
        if isinstance(value, torch.Generator):
            return f"Generator with seed {value.initial_seed()} on device {value.device}"
        if isinstance(value, dict):
            return {k: DiffusersAdapterPipeline._summarize_call_kwargs_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            if len(value) > 10:
                return f"{value.__class__.__name__} of length {len(value)}"
            return value.__class__([DiffusersAdapterPipeline._summarize_call_kwargs_value(v) for v in value])
        shape = getattr(value, "shape", None)
        size = getattr(value, "size", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": tuple(shape)}
        if size is not None and not callable(size):
            return {"type": type(value).__name__, "size": size}
        return f"<{type(value).__name__}>"
