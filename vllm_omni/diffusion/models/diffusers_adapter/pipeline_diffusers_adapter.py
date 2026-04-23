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

import logging
import os
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniPromptType
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
        self.od_config = od_config
        self.device = device
        self._capabilities: dict[str, Any] = {}
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

        self._pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            **load_kwargs,
        ).to(self.device)

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
        prompt, neg_prompt = self._extract_prompt(req.prompts)

        # Merge user-provided call kwargs from stage/CLI defaults.
        # Request-time parameters take precedence over stage-config defaults
        call_kwargs = self.od_config.diffusers_call_kwargs
        kwargs: dict[str, Any] = {
            **call_kwargs,
            "prompt": prompt,
            "num_inference_steps": sampling.num_inference_steps,
            "guidance_scale": sampling.guidance_scale,
            "output_type": sampling.output_type or self.od_config.output_type,
        }

        if sampling.height is not None:
            kwargs["height"] = sampling.height
        if sampling.width is not None:
            kwargs["width"] = sampling.width
        if sampling.num_frames is not None and sampling.num_frames > 1:
            kwargs["num_frames"] = sampling.num_frames
        if sampling.num_outputs_per_prompt is not None and sampling.num_outputs_per_prompt > 1:
            kwargs["num_images_per_prompt"] = sampling.num_outputs_per_prompt

        if neg_prompt is not None:
            kwargs["negative_prompt"] = neg_prompt

        if sampling.generator is not None:
            kwargs["generator"] = sampling.generator
        elif sampling.seed is not None:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device).manual_seed(sampling.seed)
        else:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device)

        if sampling.latents is not None:
            kwargs["latents"] = sampling.latents

        return kwargs

    @staticmethod
    def _extract_prompt(prompt_obj: list[OmniPromptType]) -> tuple[str | list[str], str | list[str] | None]:
        """Extract the text prompts and negative prompts from a list of prompt objects."""
        if len(prompt_obj) == 1:
            if isinstance(prompt_obj[0], str):
                return prompt_obj[0], None
            else:
                return prompt_obj[0].get("prompt", ""), prompt_obj[0].get("negative_prompt", None)

        prompts = []
        negative_prompts: list[str] | None = []
        for prompt in prompt_obj:
            if isinstance(prompt, str):
                prompts.append(prompt)
            else:
                prompts.append(prompt.get("prompt", ""))
                negative_prompts.append(prompt.get("negative_prompt", ""))
        if all(not np for np in negative_prompts):
            negative_prompts = None
        return prompts, negative_prompts

    @staticmethod
    def _extract_negative_prompt(prompt_obj: Any) -> str | None:
        """Extract the negative prompt from a prompt object, if present."""
        if isinstance(prompt_obj, dict):
            return prompt_obj.get("negative_prompt")
        return getattr(prompt_obj, "negative_prompt", None)

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
