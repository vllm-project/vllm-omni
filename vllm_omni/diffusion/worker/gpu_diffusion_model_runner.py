# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offload import apply_offload_hooks
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class GPUDiffusionModelRunner:
    """
    Model runner that handles model loading and execution for diffusion models.

    This class follows the AR pattern where the Runner handles all model-related
    operations including loading, compilation, offloading, caching, and execution.
    The Worker only handles infrastructure (device, distributed env).
    """

    def __init__(
        self,
        vllm_config,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ):
        """
        Initialize the diffusion model runner.

        Args:
            vllm_config: vLLM configuration.
            od_config: OmniDiffusion configuration.
            device: The device to run on.
        """
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None
        self.cache_backend = None

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
    ) -> None:
        """
        Load the diffusion model, apply compilation and offloading.

        Args:
            memory_pool_context_fn: Optional function that returns a context manager
                for memory pool allocation (used for sleep mode).
        """
        load_device = "cpu" if self.od_config.enable_cpu_offload else str(self.device)

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        # Load model within forward context
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            load_config = LoadConfig()
            model_loader = DiffusersPipelineLoader(load_config)
            time_before_load = time.perf_counter()

            with get_memory_context():
                with DeviceMemoryProfiler() as m:
                    self.pipeline = model_loader.load_model(
                        od_config=self.od_config,
                        load_device=load_device,
                    )
            time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info("Model runner: Model loaded successfully.")

        # Apply CPU offloading (DiT <-> encoders mutual exclusion)
        if self.od_config.enable_cpu_offload:
            for name in ["vae"]:
                module = getattr(self.pipeline, name, None)
                if module is None:
                    continue
                try:
                    module.to(self.device, non_blocking=True)
                except Exception as exc:
                    logger.debug("Failed to move %s to GPU: %s", name, exc)

            apply_offload_hooks(self.pipeline, self.od_config, device=self.device)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            try:
                self.pipeline.transformer = regionally_compile(
                    self.pipeline.transformer,
                    dynamic=True,
                )
                logger.info("Model runner: Model compiled with torch.compile.")
            except Exception as e:
                logger.warning(f"Model runner: torch.compile failed with error: {e}. Using eager mode.")

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)

        logger.info("Model runner: Initialization complete.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    @torch.inference_mode()
    def execute_model(self, reqs: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            reqs: List of diffusion requests to process.

        Returns:
            DiffusionOutput with generated results.
        """
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not reqs or len(reqs) == 0:
            raise ValueError("Cannot execute model with empty request list")

        # TODO: dealing with first req for now
        req = reqs[0]

        if req.generator is None and req.seed is not None:
            req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # Refresh cache context if needed
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.num_inference_steps)

        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            with record_function("pipeline_forward"):
                output = self.pipeline.forward(req)

        return output
