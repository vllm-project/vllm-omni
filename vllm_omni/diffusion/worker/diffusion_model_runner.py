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
from vllm_omni.diffusion.cache.dit_cache_manager import DiTCacheManager
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
)
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
from vllm_omni.diffusion.worker.input_batch import InputBatch
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
        self.dit_cache_manager: DiTCacheManager | None = None
        self.offload_backend = None
        self.prompt_embed_cache = None

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, DiffusionRequestState] = {}

        # Initialize KV cache manager for connector management
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

    @staticmethod
    def _prompt_preview_for_log(prompts: list[Any] | None, max_length: int = 120) -> str:
        if not prompts:
            return "<none>"

        first_prompt = prompts[0]
        if isinstance(first_prompt, str):
            prompt_text = first_prompt
        elif isinstance(first_prompt, dict):
            prompt_text = first_prompt.get("prompt") or str(first_prompt)
        else:
            prompt_text = str(first_prompt)

        prompt_text = " ".join(prompt_text.split())
        if len(prompt_text) > max_length:
            prompt_text = f"{prompt_text[: max_length - 3]}..."
        if len(prompts) > 1:
            prompt_text = f"{prompt_text} (+{len(prompts) - 1} more)"
        return prompt_text

    @staticmethod
    def _sampling_seed_for_log(sampling: Any) -> str:
        seed = getattr(sampling, "seed", None)
        if seed is not None:
            return str(seed)
        if getattr(sampling, "generator", None) is not None:
            return "generator"
        return "auto"

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

    def _log_cache_dit_request_stats(self, req: OmniDiffusionRequest) -> None:
        if (
            self.pipeline is None
            or self.cache_backend is None
            or not self.cache_backend.is_enabled()
            or self.od_config.cache_backend != "cache_dit"
        ):
            return

        request_ids = getattr(req, "request_ids", None) or []
        request_id = request_ids[0] if request_ids else "unknown"
        if request_id == "dummy_req_id":
            return

        total_steps = int(getattr(req.sampling_params, "num_inference_steps", 0) or 0)
        prompt_preview = self._prompt_preview_for_log(req.prompts)
        seed_value = self._sampling_seed_for_log(req.sampling_params)
        seen_context_keys: set[tuple[int, str]] = set()
        found_stats = False

        candidate_modules = [
            self.pipeline,
            getattr(self.pipeline, "transformer", None),
            getattr(self.pipeline, "transformer_2", None),
            getattr(self.pipeline, "bagel", None),
        ]
        language_model = getattr(self.pipeline, "language_model", None)
        candidate_modules.extend([language_model, getattr(language_model, "model", None)])

        for module in candidate_modules:
            if module is None:
                continue
            context_manager = getattr(module, "_context_manager", None)
            context_names = tuple(getattr(module, "_context_names", ()) or ())
            if context_manager is None or not context_names:
                continue

            for context_name in context_names:
                context_key = (id(context_manager), context_name)
                if context_key in seen_context_keys:
                    continue
                seen_context_keys.add(context_key)
                try:
                    context = context_manager.get_context(context_name)
                except Exception:
                    continue
                if context is None:
                    continue
                found_stats = True

                context_total_steps = total_steps or (int(context.get_current_step()) + 1)
                cached_steps = list(context.get_cached_steps() or [])
                cfg_cached_steps = list(context.get_cfg_cached_steps() or [])
                skip_count = len(cached_steps)
                cfg_skip_count = len(cfg_cached_steps)
                skip_ratio = 100.0 * skip_count / context_total_steps if context_total_steps > 0 else 0.0
                cfg_skip_ratio = 100.0 * cfg_skip_count / context_total_steps if context_total_steps > 0 else 0.0

                logger.info(
                    "[Cache-DiT] Request %s seed=%s prompt=%s for %s: skipped %d / %d steps (%.2f%%).",
                    request_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    skip_count,
                    context_total_steps,
                    skip_ratio,
                )
                logger.info(
                    "[Cache-DiT] Request %s seed=%s prompt=%s for %s: skipped_step_ids=%s",
                    request_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    cached_steps,
                )
                if cfg_cached_steps:
                    logger.info(
                        "[Cache-DiT] Request %s seed=%s prompt=%s for %s: "
                        "cfg_skipped %d / %d steps (%.2f%%), cfg_skipped_step_ids=%s",
                        request_id,
                        seed_value,
                        prompt_preview,
                        context_name,
                        cfg_skip_count,
                        context_total_steps,
                        cfg_skip_ratio,
                        cfg_cached_steps,
                    )

        if not found_stats:
            logger.info("[Cache-DiT] Request %s: no live cache contexts found.", request_id)

    def _log_cache_dit_stepwise_request_stats(self, request_state: DiffusionRequestState) -> None:
        if (
            request_state.cache_slot is None
            or request_state.request_id == "dummy_req_id"
            or self.od_config.cache_backend != "cache_dit"
        ):
            return

        payload = request_state.cache_slot.payload
        if not isinstance(payload, tuple):
            return

        total_steps = request_state.total_steps or int(getattr(request_state.sampling, "num_inference_steps", 0) or 0)
        prompt_preview = self._prompt_preview_for_log(request_state.prompts)
        seed_value = self._sampling_seed_for_log(request_state.sampling)
        seen_context_ids: set[int] = set()
        found_stats = False

        for contexts in payload:
            if not isinstance(contexts, dict):
                continue
            for context_name, context in contexts.items():
                if context is None or id(context) in seen_context_ids:
                    continue
                seen_context_ids.add(id(context))
                found_stats = True

                cached_steps = list(context.get_cached_steps() or [])
                cfg_cached_steps = list(context.get_cfg_cached_steps() or [])
                skip_count = len(cached_steps)
                cfg_skip_count = len(cfg_cached_steps)
                skip_ratio = 100.0 * skip_count / total_steps if total_steps > 0 else 0.0
                cfg_skip_ratio = 100.0 * cfg_skip_count / total_steps if total_steps > 0 else 0.0

                logger.info(
                    "[Cache-DiT][stepwise] Request %s seed=%s prompt=%s for %s: skipped %d / %d steps (%.2f%%).",
                    request_state.request_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    skip_count,
                    total_steps,
                    skip_ratio,
                )
                logger.info(
                    "[Cache-DiT][stepwise] Request %s seed=%s prompt=%s for %s: skipped_step_ids=%s",
                    request_state.request_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    cached_steps,
                )
                if cfg_cached_steps:
                    logger.info(
                        "[Cache-DiT][stepwise] Request %s seed=%s prompt=%s for %s: "
                        "cfg_skipped %d / %d steps (%.2f%%), cfg_skipped_step_ids=%s",
                        request_state.request_id,
                        seed_value,
                        prompt_preview,
                        context_name,
                        cfg_skip_count,
                        total_steps,
                        cfg_skip_ratio,
                        cfg_cached_steps,
                    )

        if not found_stats:
            logger.info(
                "[Cache-DiT][stepwise] Request %s: no slot cache contexts found.",
                request_state.request_id,
            )

    def _should_log_on_this_rank(self) -> bool:
        try:
            return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        except Exception:
            return True

    def _log_stepwise_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        scheduled_states: list[DiffusionRequestState],
        new_req_ids: Iterable[str],
    ) -> None:
        if not scheduled_states or not self._should_log_on_this_rank():
            return

        new_req_id_set = set(new_req_ids)
        request_ids = [state.request_id for state in scheduled_states if state.request_id != "dummy_req_id"]
        if not request_ids:
            return

        new_batch_req_ids = [request_id for request_id in request_ids if request_id in new_req_id_set]
        cached_batch_req_ids = [request_id for request_id in request_ids if request_id not in new_req_id_set]
        per_req_progress = [
            f"{state.request_id}:{int(state.step_index) + 1}/{max(int(state.total_steps), 1)}"
            for state in scheduled_states
            if state.request_id != "dummy_req_id"
        ]
        request_meta = [
            f"{state.request_id}(seed={self._sampling_seed_for_log(state.sampling)}, "
            f"prompt={self._prompt_preview_for_log(state.prompts, max_length=64)})"
            for state in scheduled_states
            if state.request_id != "dummy_req_id"
        ]

        first_sampling = scheduled_states[0].sampling
        logger.info(
            "[StepBatch] scheduler_step=%d batch_size=%d req_ids=%s new_req_ids=%s cached_req_ids=%s "
            "progress=%s shape=%sx%s num_inference_steps=%s cache_backend=%s",
            scheduler_output.step_id,
            len(request_ids),
            request_ids,
            new_batch_req_ids,
            cached_batch_req_ids,
            per_req_progress,
            getattr(first_sampling, "width", None),
            getattr(first_sampling, "height", None),
            getattr(first_sampling, "num_inference_steps", None),
            self.od_config.cache_backend,
        )
        logger.info("[StepBatch] request_meta=%s", request_meta)

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
        load_format: str = "default",
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
        self.dit_cache_manager = None

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
                cache_pool_driver = self.cache_backend.create_state_driver(self.pipeline)
                if cache_pool_driver is not None:
                    self.dit_cache_manager = DiTCacheManager(cache_pool_driver)

        # Install prompt-embedding cache (transparent wrapper around
        # ``pipeline.encode_prompt``). Enabled via config or env var; a no-op
        # when the pipeline does not expose ``encode_prompt``.
        enable_pec, pec_size = resolve_prompt_embed_cache_config(
            enable=getattr(self.od_config, "enable_prompt_embed_cache", False),
            max_size=getattr(self.od_config, "prompt_embed_cache_size", 32),
        )
        if enable_pec:
            self.prompt_embed_cache = install_prompt_embed_cache(
                self.pipeline,
                max_size=pec_size,
                enabled=True,
                model_tag=self.od_config.model_class_name,
            )

        logger.info("Model runner: Initialization complete.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    def clear_prompt_embed_cache(self) -> None:
        """Evict all cached text-encoder outputs (e.g. between training epochs)."""
        if self.prompt_embed_cache is not None:
            self.prompt_embed_cache.clear()

    def get_prompt_embed_cache_stats(self) -> dict | None:
        """Return hit/miss statistics for the prompt-embedding cache, if enabled."""
        if self.prompt_embed_cache is None:
            return None
        return self.prompt_embed_cache.stats()

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

            # Log prompt-embed cache activity (hits/misses accumulate across requests).
            if is_primary and self.prompt_embed_cache is not None:
                logger.debug("prompt-embed cache: %s", self.prompt_embed_cache.stats())

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
        dit_cache_manager = getattr(self, "dit_cache_manager", None)
        for request_id in scheduler_output.finished_req_ids:
            state = self.state_cache.pop(request_id, None)
            if state is not None and dit_cache_manager is not None:
                dit_cache_manager.free(state)

        resolved: list[DiffusionRequestState] = []
        new_request_ids: list[str] = []
        try:
            # process new requests
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                req = sched_new_req.req
                new_request_ids.append(request_id)
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
                new_state = DiffusionRequestState(
                    request_id=request_id,
                    sampling=copy.deepcopy(req.sampling_params),
                    prompts=req.prompts,
                )
                self.state_cache[request_id] = new_state
                resolved.append(new_state)

            # process cached requests
            for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
                state = self.state_cache.get(request_id)
                if state is None:
                    raise ValueError(f"Missing cached state for request {request_id}.")
                resolved.append(state)
        except Exception:
            for request_id in new_request_ids:
                self.state_cache.pop(request_id, None)
            raise

        return resolved, new_request_ids

    def _prepare_batch_inputs(self, states: list[DiffusionRequestState], new_request_ids: list[str]) -> InputBatch:
        # process new reqs
        for state in states:
            if state.request_id in new_request_ids:
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
        dit_cache_manager = getattr(self, "dit_cache_manager", None)
        self.input_batch = input_batch

        for state in states:
            if interrupted or state.denoise_completed:
                removed = self.state_cache.pop(state.request_id, None)
                if removed is not None and dit_cache_manager is not None:
                    dit_cache_manager.free(state)

        if not self.state_cache:
            self.input_batch = None

    def _prepare_attn_metadata(self, input_batch: InputBatch) -> Any:
        model_state = getattr(self, "model_state", None)
        if model_state is None:
            return {}
        prepare_attn = getattr(model_state, "prepare_attn", None)
        if not callable(prepare_attn):
            return {}
        return prepare_attn(input_batch)

    @staticmethod
    def _build_stepwise_output(
        state: DiffusionRequestState,
        *,
        finished: bool,
        result: DiffusionOutput | None,
    ) -> RunnerOutput:
        return RunnerOutput(
            request_id=state.request_id,
            step_index=state.step_index,
            finished=finished,
            result=result,
        )

    def _build_stepwise_outputs(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
        noise_pred: torch.Tensor | None,
        pipeline_interrupted: bool,
    ) -> list[RunnerOutput]:
        runner_outputs: list[RunnerOutput] = []
        if noise_pred is None:
            error = "stepwise denoise interrupted" if pipeline_interrupted else "stepwise denoise returned None"
            for state in states:
                runner_outputs.append(
                    self._build_stepwise_output(
                        state,
                        finished=True,
                        result=DiffusionOutput(error=error),
                    )
                )
        else:
            offset = 0
            for state in states:
                next_offset = offset + state.latents.shape[0]
                self.pipeline.step_scheduler(state, noise_pred[offset:next_offset])
                offset = next_offset
                result = self.pipeline.post_decode(state) if state.denoise_completed else None
                runner_outputs.append(
                    self._build_stepwise_output(
                        state,
                        finished=state.denoise_completed,
                        result=result,
                    )
                )

            if offset != noise_pred.shape[0]:
                raise ValueError(
                    f"Stepwise noise_pred consumed {offset} rows, "
                    f"but batched noise_pred has {noise_pred.shape[0]} rows."
                )

        self._update_states_after(
            states,
            input_batch,
            interrupted=pipeline_interrupted or noise_pred is None,
        )
        return runner_outputs

    def _denoise_step_with_cache(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
        dit_cache_manager: DiTCacheManager | None,
    ) -> torch.Tensor | None:
        attn_metadata = self._prepare_attn_metadata(input_batch)
        with set_forward_context(
            vllm_config=self.vllm_config,
            omni_diffusion_config=self.od_config,
            attn_metadata=attn_metadata,
        ):
            try:
                if dit_cache_manager is not None:
                    dit_cache_manager.activate(states)
                return self.pipeline.denoise_step(input_batch)
            finally:
                if dit_cache_manager is not None:
                    dit_cache_manager.deactivate(states)

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one step for one scheduled request and return runner output."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        dit_cache_manager = getattr(self, "dit_cache_manager", None)
        if self.od_config.cache_backend not in (None, "none"):
            if dit_cache_manager is None:
                raise ValueError(
                    f"Step mode cache backend '{self.od_config.cache_backend}' has no resident-state driver."
                )

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        states: list[DiffusionRequestState] = []
        input_batch: InputBatch | None = None
        with grad_context:
            try:
                states, new_request_ids = self._update_states(scheduler_output)
                self._log_stepwise_batch(scheduler_output, states, new_request_ids)

                if (
                    dit_cache_manager is not None
                    and len(states) > 1
                    and not dit_cache_manager.supports_batch_activation
                ):
                    raise ValueError(
                        f"Cache backend '{dit_cache_manager.driver.backend_name}' "
                        "does not support batched slot activation."
                    )

                input_batch = self._prepare_batch_inputs(states, new_request_ids)
                noise_pred = self._denoise_step_with_cache(
                    states,
                    input_batch,
                    dit_cache_manager,
                )
                pipeline_interrupted = getattr(self.pipeline, "interrupt", False)
                runner_output_list = self._build_stepwise_outputs(
                    states,
                    input_batch,
                    noise_pred,
                    pipeline_interrupted,
                )

                return BatchRunnerOutput.from_list(runner_output_list)
            except Exception:
                if dit_cache_manager is not None:
                    dit_cache_manager.deactivate()
                for state in states:
                    self.state_cache.pop(state.request_id, None)
                    if dit_cache_manager is not None:
                        dit_cache_manager.free(state)
                self.input_batch = None
                raise
