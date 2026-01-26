# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Async entrypoint for vLLM-Omni diffusion model inference.

Provides an asynchronous interface for running diffusion models,
enabling concurrent request handling and streaming generation.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Any

from PIL import Image
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class AsyncOmniDiffusion:
    """Async entry point for vLLM-Omni diffusion model inference.

    This class provides an asynchronous interface for running diffusion models,
    enabling concurrent request handling. It wraps the DiffusionEngine and
    provides async methods for image generation.

    Args:
        model: Model name or path to load
        od_config: Optional OmniDiffusionConfig. If not provided, it will be
            created from kwargs
        **kwargs: Additional keyword arguments passed to OmniDiffusionConfig

    Example:
        >>> async_diffusion = AsyncOmniDiffusion(model="Qwen/Qwen-Image")
        >>> result = await async_diffusion.generate(
        ...     prompt="A beautiful sunset over the ocean",
        ...     request_id="req-1",
        ... )
        >>> print(result.images)
    """

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig | None = None,
        **kwargs: Any,
    ):
        self.model = model

        # Capture stage info from kwargs before they might be filtered out
        stage_id = kwargs.get("stage_id")
        engine_input_source = kwargs.get("engine_input_source")

        # Build config
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(model=model, **kwargs)
        elif isinstance(od_config, dict):
            # If config is dict, check it too (priority to kwargs if both exist)
            if stage_id is None:
                stage_id = od_config.get("stage_id")
            if engine_input_source is None:
                engine_input_source = od_config.get("engine_input_source")
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Inject stage info into omni_kv_config if present
        if stage_id is not None:
            self.od_config.omni_kv_config.setdefault("stage_id", stage_id)
        if engine_input_source is not None:
            self.od_config.omni_kv_config.setdefault("engine_input_source", engine_input_source)

        try:
            config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
            od_config.model_class_name = config_dict.get("_class_name", None)
            od_config.update_multimodal_support()

            tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
            od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
        except (AttributeError, OSError, ValueError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()

        # Initialize engine
        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

        # Thread pool for running sync engine in async context
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

        logger.info("AsyncOmniDiffusion initialized with model: %s", model)

    def _prepare_request(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> OmniDiffusionRequest:
        """Prepare a diffusion request from prompt and parameters.

        Args:
            prompt: Text prompt for image generation
            request_id: Optional unique identifier for the request
            **kwargs: Additional generation parameters

        Returns:
            OmniDiffusionRequest ready for processing
        """
        if request_id is None:
            request_id = f"diff-{uuid.uuid4().hex[:16]}"

        field_names = {f.name for f in fields(OmniDiffusionRequest)}

        init_kwargs = {
            "prompt": prompt,
            "request_id": request_id,
        }

        for key, value in kwargs.items():
            if key in field_names:
                init_kwargs[key] = value

        if "guidance_scale" in kwargs:
            init_kwargs["guidance_scale_provided"] = True

        return OmniDiffusionRequest(**init_kwargs)

    async def generate(
        self,
        prompt: str,
        request_id: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str | None = None,
        num_outputs_per_prompt: int = 1,
        seed: int | None = None,
        **kwargs: Any,
    ) -> OmniRequestOutput:
        """Generate images asynchronously from a text prompt.

        Args:
            prompt: Text prompt describing the desired image
            request_id: Optional unique identifier for tracking the request
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Classifier-free guidance scale (optional, uses model defaults if omitted)
            height: Optional image height in pixels
            width: Optional image width in pixels
            negative_prompt: Optional negative prompt for guidance
            num_outputs_per_prompt: Number of images to generate (default: 1)
            seed: Optional random seed for reproducibility
            **kwargs: Additional generation parameters

        Returns:
            OmniRequestOutput containing generated images

        Raises:
            RuntimeError: If generation fails
        """
        if request_id is None:
            request_id = f"diff-{uuid.uuid4().hex[:16]}"

        # Prepare request
        request_kwargs = {
            "prompt": prompt,
            "request_id": request_id,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "negative_prompt": negative_prompt,
            "num_outputs_per_prompt": num_outputs_per_prompt,
            "seed": seed,
            **kwargs,
        }
        if guidance_scale is not None:
            request_kwargs["guidance_scale"] = guidance_scale

        request = self._prepare_request(**request_kwargs)

        logger.debug("Starting generation for request %s", request_id)

        # Run engine in thread pool
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self.engine.step,
                [request],
            )
        except Exception as e:
            logger.error("Generation failed for request %s: %s", request_id, e)
            raise RuntimeError(f"Diffusion generation failed: {e}") from e

        # Check if result is already OmniRequestOutput
        if isinstance(result, OmniRequestOutput):
            # Update request_id if needed
            if not result.request_id:
                result.request_id = request_id
            return result

        # Process results if not OmniRequestOutput
        images: list[Image.Image] = []
        if result is not None:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, Image.Image):
                        images.append(item)
            elif isinstance(result, Image.Image):
                images.append(result)

        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=images,
            prompt=prompt,
            metrics={
                "num_inference_steps": num_inference_steps,
                "guidance_scale": request.guidance_scale,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate images with streaming progress updates.

        Currently, diffusion models don't support true streaming, so this
        yields a single result after generation completes. Future implementations
        may support step-by-step progress updates.

        Args:
            prompt: Text prompt describing the desired image
            request_id: Optional unique identifier for tracking the request
            **kwargs: Additional generation parameters

        Yields:
            OmniRequestOutput with generation progress/results
        """
        result = await self.generate(prompt=prompt, request_id=request_id, **kwargs)
        yield result

    def close(self) -> None:
        """Close the engine and release resources.

        Should be called when done using the AsyncOmniDiffusion instance.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self.engine.close()
        except Exception as e:
            logger.warning("Error closing diffusion engine: %s", e)

        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            logger.warning("Error shutting down executor: %s", e)

        logger.info("AsyncOmniDiffusion closed")

    def shutdown(self) -> None:
        """Alias for close() method."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request."""
        self.engine.abort(request_id)

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return not self._closed

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self._closed
