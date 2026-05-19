# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

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
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, DiffusionRequestState, RunnerOutput
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


class DiffusionModelRunner(OmniConnectorModelRunnerMixin):
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

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, DiffusionRequestState] = {}

        # Initialize KV cache manager for connector management
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

    def _compile_transformer(self, attr_name: str) -> None:
        """Compile a transformer attribute on the pipeline with torch.compile."""
        model = getattr(self.pipeline, attr_name, None)
        if model is None:
            return
        try:
            setattr(self.pipeline, attr_name, regionally_compile(model, dynamic=True))
            logger.info("Model runner: %s compiled with torch.compile.", attr_name)
        except Exception as e:
            logger.warning(
                "Model runner: torch.compile for %s failed: %s. Using eager mode.",
                attr_name,
                e,
            )

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
        load_format: str | None = None,
        custom_pipeline_name: str | None = None,
    ) -> None:
        """
        Load the diffusion model, apply compilation and offloading.

        Args:
            memory_pool_context_fn: Optional function that returns a context manager
                for memory pool allocation (used for sleep mode).
            load_format: Format for loading model weights. Supported formats:
                - "default" (default): Automatically detect and use the default format based on configuration
                - "custom_pipeline": Init model from a custom pipeline class specified by `custom_pipeline_name`
                - "dummy": Skip actual weight loading, useful for testing and custom pipelines that
                    don't require default weights.
            custom_pipeline_name: Optional custom pipeline class name to use.
        """

        if load_format == "dummy":
            return

        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        # Load model within forward context
        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config, od_config=self.od_config)
        time_before_load = time.perf_counter()

        with get_memory_context():
            with DeviceMemoryProfiler() as m:
                self.pipeline = model_loader.load_model(
                    od_config=self.od_config,
                    load_device=load_device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                    device=self.device,
                )
        time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info("Model runner: Model loaded successfully.")

        if getattr(self.od_config, "step_execution", False) and not self.supports_step_mode():
            raise ValueError(
                "step_execution=True requires a pipeline implementing "
                "prepare_encode(), denoise_step(), step_scheduler(), and post_decode(); "
                f"{self.od_config.model_class_name} does not support that contract."
            )

        # Apply CPU offloading
        self.offload_backend = get_offload_backend(self.od_config, device=self.device)
        if self.offload_backend is not None:
            logger.info(f" Enabling offloader backend: {self.offload_backend.__class__.__name__}")
            self.offload_backend.enable(self.pipeline)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                self._compile_transformer("transformer")
                self._compile_transformer("transformer_2")
            else:
                logger.warning(
                    "Model runner: Platform %s does not support torch inductor, skipping torch.compile.",
                    current_omni_platform.get_torch_device(),
                )

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            if self.od_config.model_class_name in _NO_CACHE_ACCELERATION:
                logger.warning(
                    "Cache backend '%s' is not supported for %s; disabling cache acceleration.",
                    self.od_config.cache_backend,
                    self.od_config.model_class_name,
                )
                self.cache_backend = None
                self.od_config.cache_backend = None
            else:
                self.cache_backend.enable(self.pipeline)

        logger.info("Model runner: Initialization complete.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    def _record_peak_memory(self, output: DiffusionOutput) -> None:
        """Record peak GPU memory for the current forward pass into output.

        Must be called immediately after pipeline.forward(), with
        reset_peak_memory_stats() called just before it, so the measurement
        reflects this request only and not the global historical maximum.

        Uses max_memory_reserved (CUDA memory pool high-water mark) rather than
        max_memory_allocated so that allocator fragmentation is also visible.
        See: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
        """
        peak_reserved_bytes = current_omni_platform.max_memory_reserved()
        peak_allocated_bytes = current_omni_platform.max_memory_allocated()

        output.peak_memory_mb = peak_reserved_bytes / (1024**2)
        peak_reserved_gb = peak_reserved_bytes / (1024**3)
        peak_allocated_gb = peak_allocated_bytes / (1024**3)
        pool_overhead_gb = peak_reserved_gb - peak_allocated_gb

        logger.debug(
            "Peak GPU memory (this request): %.2f GB reserved, %.2f GB allocated, %.2f GB pool overhead (%.1f%%)",
            peak_reserved_gb,
            peak_allocated_gb,
            pool_overhead_gb,
            pool_overhead_gb / peak_reserved_gb * 100 if peak_reserved_gb > 0 else 0.0,
        )

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            req: A diffusion request containing a list of prompts to process.

        Returns:
            DiffusionOutput with generated results.

        Note:
            We use torch.no_grad() for HSDP because HSDP2's fully_shard requires access
            to tensor version counters in pre_forward hooks, which inference tensors do
            not track. For non-HSDP inference, we use torch.inference_mode() for better
            performance.
        """
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if len(req.prompts) == 0:
            raise ValueError("Cannot execute model with empty request list")

        # Use no_grad() for HSDP compatibility, inference_mode() otherwise for better perf
        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            # The manager handles the check for need_recv_cache internally
            self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                req,
                cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                target_device=getattr(self.pipeline, "device", None),
            )

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
                # FIXME (Alex): When num_inference_steps is None, we defer to
                # pipelines for default, but don't refresh the cache; the right
                # way to do this is to merge the sampling params first.
                #
                # For now, if num_inference_steps is not set, we pass 0 to allow
                # TeaCache to refresh to align with the param signature. This is
                # okay to force refresh TeaCache because the refresh does not use
                # num_inference_steps at all (i.e., just resets state and clears
                # stale residuals).
                num_inference_steps = req.sampling_params.num_inference_steps
                if self.od_config.cache_backend == "tea_cache" and num_inference_steps is None:
                    num_inference_steps = 0

                if num_inference_steps is not None:
                    self.cache_backend.refresh(self.pipeline, num_inference_steps)
                else:
                    logger.warning(
                        "Failed to refresh the diffusion transformer cache; backend %s "
                        "currently requires num_inference_steps to be passed explicitly",
                        self.od_config.cache_backend,
                    )

            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_primary:
                current_omni_platform.reset_peak_memory_stats()

            with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
                with record_function("pipeline_forward"):
                    output = self.pipeline.forward(req)

            if is_primary:
                self._record_peak_memory(output)

            # NOTE:
            if (
                self.cache_backend is not None
                and self.cache_backend.is_enabled()
                and self.od_config.cache_backend == "cache_dit"
                and self.od_config.enable_cache_dit_summary
            ):
                cache_summary(self.pipeline, details=True)

            return output

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def supports_step_mode(self) -> bool:
        """Return whether current pipeline supports step execution."""
        return self.pipeline is not None and supports_step_execution(self.pipeline)

    def _update_states(
        self, scheduler_output: DiffusionSchedulerOutput
    ) -> tuple[list[DiffusionRequestState], list[str]]:
        """Step-before update: cleanup finished requests and get/create one running state."""
        for req_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(req_id, None)

        resolved: list[DiffusionRequestState] = []
        new_req_id: list[str] = []
        try:
            # process new requests
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                # new_req_data = scheduler_output.scheduled_new_reqs[0]
                req_id = sched_new_req.sched_req_id
                req = sched_new_req.req
                new_req_id.append(req_id)
                if req_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {req_id}.")
                request_ids = req.request_ids or [req_id]
                if len(request_ids) != len(req.prompts):
                    raise ValueError(
                        f"request_ids length ({len(request_ids)}) does not match prompts length ({len(req.prompts)})"
                    )
                new_state = DiffusionRequestState(
                    req_id=req_id,
                    sampling=copy.deepcopy(req.sampling_params),
                    prompts=req.prompts,
                )
                self.state_cache[req_id] = new_state
                resolved.append(new_state)

            # process cached requests
            for req_id in scheduler_output.scheduled_cached_reqs.sched_req_ids:
                state = self.state_cache.get(req_id)
                if state is None:
                    raise ValueError(f"Missing cached state for request {req_id}.")
                resolved.append(state)
        except Exception:
            for req_id in new_req_id:
                self.state_cache.pop(req_id, None)
            raise

        return resolved, new_req_id

    def _prepare_batch_inputs(self, states: list[DiffusionRequestState], new_request_ids: list[str]) -> InputBatch:
        # process new reqs
        for state in states:
            if state.req_id in new_request_ids:
                # set generator
                if state.sampling.generator is None and state.sampling.seed is not None:
                    if state.sampling.generator_device is not None:
                        gen_device = state.sampling.generator_device
                    elif self.device.type == "cpu":
                        gen_device = "cpu"
                    else:
                        gen_device = self.device
                    state.sampling.generator = torch.Generator(device=gen_device).manual_seed(state.sampling.seed)
                # encode
                self.pipeline.prepare_encode(state)

        input_batch = InputBatch.make_batch(
            states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self.input_batch = input_batch
        return input_batch

    def _update_states_after(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
        interrupted: bool = False,
    ):
        """Step-after update: clear cached state for completed request."""
        gathered_latents = torch.cat([state.latents for state in states], dim=0)
        if (
            input_batch.latents.size() == gathered_latents.size()
            and input_batch.latents.dtype == gathered_latents.dtype
            and input_batch.latents.device == gathered_latents.device
        ):
            input_batch.latents.copy_(gathered_latents)
        else:
            input_batch.latents = gathered_latents.clone()

        self.input_batch = input_batch
        scatter_latents(states, input_batch)

        for state in states:
            if interrupted or state.denoise_completed:
                self.state_cache.pop(state.req_id, None)

    def _prepare_attn_metadata(self, input_batch: InputBatch) -> Any:
        model_state = getattr(self, "model_state", None)
        if model_state is None:
            return {}
        prepare_attn = getattr(model_state, "prepare_attn", None)
        if not callable(prepare_attn):
            return {}
        return prepare_attn(input_batch)

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one step for one scheduled request and return runner output."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        # Stepwise mode only supports the basic state-driven denoise path for now.
        # Request-mode extras such as cache backends, KV transfer, editing inputs,
        # and similar features are not supported here yet.
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            states, new_request_ids = self._update_states(scheduler_output)
            input_batch = self._prepare_batch_inputs(states, new_request_ids)
            attn_metadata = self._prepare_attn_metadata(input_batch)

            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
            ):
                noise_pred = self.pipeline.denoise_step(input_batch)

                runner_output_list = []
                pipeline_interrupted = getattr(self.pipeline, "interrupt", False)
                if noise_pred is None and pipeline_interrupted:
                    for state in states:
                        runner_output_list.append(
                            RunnerOutput(
                                req_id=state.req_id,
                                step_index=state.step_index,
                                finished=True,
                                result=DiffusionOutput(error="stepwise denoise interrupted"),
                            )
                        )

                else:
                    offset = 0
                    for req in states:
                        row_num = req.latents.shape[0]
                        self.pipeline.step_scheduler(
                            req, noise_pred[offset : offset + row_num] if noise_pred is not None else None
                        )
                        offset = offset + row_num
                        if req.denoise_completed:
                            result = self.pipeline.post_decode(req)
                        else:
                            result = None
                        runner_output_list.append(
                            RunnerOutput(
                                req_id=req.req_id,
                                step_index=req.step_index,
                                finished=req.denoise_completed,
                                result=result,
                            )
                        )

                    if noise_pred is not None and offset != noise_pred.shape[0]:
                        raise ValueError(
                            f"Stepwise noise_pred consumed {offset} rows, "
                            f"but batched noise_pred has {noise_pred.shape[0]} rows."
                        )

                self._update_states_after(states, input_batch, pipeline_interrupted)

                return BatchRunnerOutput.from_list(runner_output_list)
