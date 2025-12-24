# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Async entrypoint for vLLM-Omni diffusion model inference.

Provides an asynchronous interface for running diffusion models,
enabling concurrent request handling and streaming generation.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Any

import numpy as np
import torch
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
    provides async methods for image/video generation.

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

        # Build config
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(model=model, **kwargs)
        elif isinstance(od_config, dict):
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Load model class name and transformer config
        config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
        od_config.model_class_name = config_dict.get("_class_name", None)
        od_config.update_multimodal_support()

        tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
        od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)

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

        return OmniDiffusionRequest(**init_kwargs)

    async def generate(
        self,
        prompt: str,
        request_id: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str | None = None,
        num_outputs_per_prompt: int = 1,
        seed: int | None = None,
        **kwargs: Any,
    ) -> OmniRequestOutput:
        """Generate images or videos asynchronously from a text prompt.

        Args:
            prompt: Text prompt describing the desired output
            request_id: Optional unique identifier for tracking the request
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Classifier-free guidance scale (default: 7.5)
            height: Optional image height in pixels
            width: Optional image width in pixels
            negative_prompt: Optional negative prompt for guidance
            num_outputs_per_prompt: Number of images to generate (default: 1)
            seed: Optional random seed for reproducibility
            **kwargs: Additional generation parameters

        Returns:
            OmniRequestOutput containing generated images or videos

        Raises:
            RuntimeError: If generation fails
        """
        if request_id is None:
            request_id = f"diff-{uuid.uuid4().hex[:16]}"

        # Prepare request
        request = self._prepare_request(
            prompt=prompt,
            request_id=request_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_outputs_per_prompt=num_outputs_per_prompt,
            seed=seed,
            **kwargs,
        )

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

        # Process results
        images, videos = self._extract_diffusion_outputs(result)

        logger.debug(
            "Generation completed for request %s, produced %d images and %d videos",
            request_id,
            len(images),
            len(videos),
        )

        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=images,
            videos=videos,
            prompt=prompt,
            metrics={
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate diffusion outputs with streaming progress updates.

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

    def _extract_diffusion_outputs(self, result: Any) -> tuple[list[Image.Image], list[Any]]:
        """Split diffusion outputs into image or video outputs."""
        images: list[Image.Image] = []
        videos: list[Any] = []

        if result is None:
            return images, videos

        if isinstance(result, Image.Image):
            return [result], videos

        if isinstance(result, list):
            if result and all(isinstance(item, Image.Image) for item in result):
                return list(result), videos
            for item in result:
                if isinstance(item, Image.Image):
                    images.append(item)
                else:
                    videos.extend(self._normalize_video_items(item))
            return images, videos

        if isinstance(result, (np.ndarray, torch.Tensor)):
            videos.extend(self._normalize_video_items(result))

        return images, videos

    def _normalize_video_items(self, item: Any) -> list[Any]:
        """Normalize possible video outputs into a list of video arrays/tensors."""
        if item is None:
            return []

        if isinstance(item, list):
            videos: list[Any] = []
            for sub_item in item:
                videos.extend(self._normalize_video_items(sub_item))
            return videos

        if isinstance(item, torch.Tensor):
            item = item.detach().cpu()

        if isinstance(item, np.ndarray):
            if item.ndim == 5:
                return [item[i] for i in range(item.shape[0])]
            if item.ndim in (4, 3):
                return [item]
            return [item]

        if isinstance(item, torch.Tensor):
            if item.ndim == 5:
                return [item[i] for i in range(item.shape[0])]
            if item.ndim in (4, 3):
                return [item]
            return [item]

        return []

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

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return not self._closed

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self._closed
