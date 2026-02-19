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
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.lora.request import LoRARequest
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
            elif architectures and len(architectures) == 1:
                od_config.model_class_name = architectures[0]
            else:
                raise

        # Initialize engine
        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

        # Thread pool for running sync engine in async context
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

        logger.info("AsyncOmniDiffusion initialized with model: %s", model)

    async def generate(
        self,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str | None = None,
        lora_request: LoRARequest | None = None,
    ) -> OmniRequestOutput:
        """Generate images asynchronously from a text prompt.

        Args:
            prompt: Text prompt describing the desired image
            sampling_params: Sampling parameters
            request_id: Optional unique identifier for tracking the request

        Returns:
            OmniRequestOutput containing generated images

        Raises:
            RuntimeError: If generation fails
        """
        if request_id is None:
            request_id = f"diff-{uuid.uuid4().hex[:16]}"

        if sampling_params.guidance_scale:
            sampling_params.guidance_scale_provided = True

        if lora_request is not None:
            sampling_params.lora_request = lora_request

        request = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[request_id],
        )

        logger.debug("Starting generation for request %s", request_id)

        # Run engine in thread pool
        loop = asyncio.get_event_loop()
        try:
            # In async mode, only a single request is submitted at a time
            result = await loop.run_in_executor(
                self._executor,
                self.engine.step,
                request,
            )
            result = result[0]
        except Exception as e:
            logger.error("Generation failed for request %s: %s", request_id, e)
            raise RuntimeError(f"Diffusion generation failed: {e}") from e

        # Update request_id if needed
        if not result.request_id:
            result.request_id = request_id
        return result

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

    async def remove_lora(self, adapter_id: int) -> bool:
        """Remove a LoRA"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "remove_lora",
            None,
            (adapter_id,),
            {},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def add_lora(self, lora_request: LoRARequest, lora_scale: float = 1.0) -> bool:
        """Add a LoRA adapter"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "add_lora",
            None,
            (),
            {"lora_request": lora_request, "lora_scale": lora_scale},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def list_loras(self) -> list[int]:
        """List all registered LoRA adapter IDs."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "list_loras",
            None,
            (),
            {},
            None,
        )
        # collective_rpc returns list from workers; flatten unique ids
        if not isinstance(results, list):
            return results or []
        merged: set[int] = set()
        for part in results:
            merged.update(part or [])
        return sorted(merged)

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "pin_lora",
            None,
            (),
            {"adapter_id": lora_id},
            None,
        )
        return all(results) if isinstance(results, list) else results
