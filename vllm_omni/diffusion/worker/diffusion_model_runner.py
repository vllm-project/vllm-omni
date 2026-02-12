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

from vllm_omni.diffusion.cache.cache_dit_backend import cache_summary
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class DiffusionModelRunner:
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
        self.offload_backend = None

        # Initialize KV cache manager for connector management
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

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
        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        # Load model within forward context
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

        # Apply CPU offloading
        self.offload_backend = get_offload_backend(self.od_config, device=self.device)
        if self.offload_backend is not None:
            logger.info(f" Enabling offloader backend: {self.offload_backend.__class__.__name__}")
            self.offload_backend.enable(self.pipeline)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                try:
                    self.pipeline.transformer = regionally_compile(
                        self.pipeline.transformer,
                        dynamic=True,
                    )
                    logger.info("Model runner: Model compiled with torch.compile.")
                except Exception as e:
                    logger.warning(f"Model runner: torch.compile failed with error: {e}. Using eager mode.")
            else:
                logger.warning(
                    "Model runner: Platform %s does not support torch inductor, skipping torch.compile.",
                    current_omni_platform.get_torch_device(),
                )

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)

        logger.info("Model runner: Initialization complete.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    @torch.inference_mode()
    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            req: A diffusion request containing a list of prompts to process.

        Returns:
            DiffusionOutput with generated results.
        """
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if len(req.prompts) == 0:
            raise ValueError("Cannot execute model with empty request list")

        # The manager handles the check for need_recv_cache internally
        self.kv_transfer_manager.receive_kv_cache(req, target_device=getattr(self.pipeline, "device", None))

        if req.sampling_params.generator is None and req.sampling_params.seed is not None:
            if req.sampling_params.generator_device is not None:
                gen_device = req.sampling_params.generator_device
            elif self.device.type == "cpu":
                gen_device = "cpu"
            else:
                gen_device = self.device
            req.sampling_params.generator = torch.Generator(device=gen_device).manual_seed(req.sampling_params.seed)

        # Refresh cache context if needed
        if (
            not getattr(req, "skip_cache_refresh", False)
            and self.cache_backend is not None
            and self.cache_backend.is_enabled()
        ):
            self.cache_backend.refresh(self.pipeline, req.sampling_params.num_inference_steps)

        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            with record_function("pipeline_forward"):
                output = self.pipeline.forward(req)

            # NOTE:
            if self.od_config.cache_backend == "cache_dit" and self.od_config.enable_cache_dit_summary:
                cache_summary(self.pipeline, details=True)

        return output
