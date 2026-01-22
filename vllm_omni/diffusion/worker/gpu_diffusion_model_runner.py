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
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

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
        self.connector = None

        # Initialize OmniConnector after vllm_config is available (via init_device_and_model)
        self._init_omni_connector()

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

    def _init_omni_connector(self) -> None:
        # TODO(wzliu)! get real connector from yaml file instead of hardcode
        """Initialize OmniConnector for KV cache transfer."""
        try:
            connector_config = None

            # 1. Try to get from omni_kv_config (injected from YAML)
            # Use self.od_config because self.vllm_config is a dummy VllmConfig without model_config
            if self.od_config.omni_kv_config:
                connector_config = self.od_config.omni_kv_config.get("connector_config")

            if not connector_config:
                logger.warning("No OmniConnector config found, skipping initialization")
                return

            logger.info(f"Initializing OmniConnector with config: {connector_config}")

            c_type = connector_config.get("type")
            if not c_type:
                logger.error("Connector config missing 'type'")
                return

            c_extra = {k: v for k, v in connector_config.items() if k != "type"}
            connector_spec = ConnectorSpec(name=c_type, extra=c_extra)

            self.connector = OmniConnectorFactory.create_connector(connector_spec)

        except Exception as e:
            logger.error(f"Failed to initialize OmniConnector: {e}")
            import traceback

            traceback.print_exc()

    def _receive_kv_cache_for_request(self, req: OmniDiffusionRequest) -> None:
        """Receive KV cache for a request via OmniConnector."""
        # TODO(wzliu)! must get control info from stage queue instead of hardcode
        if not req.request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return

        try:
            logger.info(f"Attempting to receive KV cache for request {req.request_id}")

            # TODO: Key used for transfer (must match sender side)
            # key = f"kv_cache_{req.request_id}"

            # Get data from connector
            # Determine from_stage and to_stage dynamically
            omni_kv_config = self.od_config.omni_kv_config
            stage_id = omni_kv_config.get("stage_id")
            engine_input_source = omni_kv_config.get("engine_input_source", [])

            to_stage = stage_id
            # Default to stage_id - 1 if input source is not explicit
            if engine_input_source:
                from_stage = engine_input_source[0]
            elif isinstance(stage_id, int):
                from_stage = stage_id - 1
            else:
                raise ValueError("Invalid stage id")
            logger.info(f"Wait for KV cache for request {req.request_id} from stage {from_stage} to {to_stage}...")

            # Check if we should receive KV cache based on config
            need_recv_cache = omni_kv_config.get("need_recv_cache", False)
            if need_recv_cache:
                # Default timeout 30 seconds to prevent infinite hanging
                timeout = omni_kv_config.get("recv_timeout", 30.0)
                start_time = time.time()

                while True:
                    result = self.connector.get(
                        from_stage=from_stage,
                        to_stage=to_stage,
                        request_id=f"kv_cache_{req.request_id}",
                    )
                    if result:
                        break

                    if time.time() - start_time > timeout:
                        logger.error(f"Timeout waiting for KV cache for request {req.request_id} after {timeout}s")
                        result = None
                        break

                    time.sleep(0.5)
            else:
                logger.info(f"Skip receiving KV cache for {req.request_id} (need_recv_cache=False)")
                result = None

            if result:
                data, size = result
                logger.info(f"Successfully received KV cache for {req.request_id}")

                # Assume data structure matches KVCacheTransferData.to_dict()
                if isinstance(data, dict) and "layer_blocks" in data:
                    # Get layer blocks and ensure they are on the correct device
                    layer_blocks = data["layer_blocks"]

                    # Move tensors to GPU if needed (OmniSerializer should handle tensor reconstruction)
                    for cache_list in [layer_blocks["key_cache"], layer_blocks["value_cache"]]:
                        for i, tensor in enumerate(cache_list):
                            if isinstance(tensor, torch.Tensor) and tensor.device != self.pipeline.device:
                                cache_list[i] = tensor.to(self.pipeline.device).contiguous()
                    from types import SimpleNamespace

                    req.past_key_values = SimpleNamespace(**layer_blocks)

                if "metadata" in data:
                    req.kv_metadata = data["metadata"]

            else:
                logger.warning(f"No KV cache received for {req.request_id} (timeout or empty)")

        except Exception as e:
            logger.error(f"Error receiving KV cache for {req.request_id}: {e}")
            import traceback

            traceback.print_exc()

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

        # [Omni] KV Cache Receiving Logic
        if getattr(req, "need_kv_receive", False) and self.connector is not None:
            self._receive_kv_cache_for_request(req)

        if req.generator is None and req.seed is not None:
            req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # Refresh cache context if needed
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.num_inference_steps)

        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            with record_function("pipeline_forward"):
                output = self.pipeline.forward(req)

        return output
