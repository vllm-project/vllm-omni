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
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.cache_dit_backend import cache_summary
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    StageBoundary,
    StagePayload,
    supports_disaggregated_execution,
    supports_step_execution,
)
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput, KVPrefetchJob
from vllm_omni.diffusion.stage_payload import (
    STAGE_PAYLOAD_OUTPUT_KEY,
    STAGE_PAYLOAD_PROMPT_KEY,
    StagePayloadError,
)
from vllm_omni.diffusion.stage_roles import (
    EXECUTION_PATH_DECODE,
    EXECUTION_PATH_DENOISE,
    EXECUTION_PATH_ENCODE,
    EXECUTION_PATH_MODEL_DEFINED,
    EXECUTION_PATH_MONOLITHIC,
    is_disaggregated_role,
    normalize_stage_role,
    resolve_execution_path,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    RunnerOutput,
    DiffusionRequestState,
    attach_stage_durations,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


def _normalize_pipeline_outputs(
    outputs: object,
    *,
    expected_count: int,
    allow_single_output: bool,
    pipeline_name: str,
) -> list[DiffusionOutput]:
    if isinstance(outputs, DiffusionOutput):
        if allow_single_output and expected_count == 1:
            return [outputs]
        raise RuntimeError(
            f"{pipeline_name}.forward returned a single DiffusionOutput; "
            "request-batch forward must return list[DiffusionOutput]."
        )

    if not isinstance(outputs, list):
        raise RuntimeError(
            f"{pipeline_name}.forward returned {type(outputs).__name__}; "
            "expected DiffusionOutput or list[DiffusionOutput]."
        )

    if len(outputs) != expected_count:
        raise RuntimeError(
            f"{pipeline_name}.forward returned {len(outputs)} outputs for {expected_count} requests; "
            "expected exactly one DiffusionOutput per request."
        )

    bad_index = next((idx for idx, output in enumerate(outputs) if not isinstance(output, DiffusionOutput)), None)
    if bad_index is not None:
        raise RuntimeError(
            f"{pipeline_name}.forward returned list item {bad_index} with type "
            f"{type(outputs[bad_index]).__name__}; expected DiffusionOutput."
        )

    return outputs


class DiffusionModelRunner(OmniConnectorModelRunnerMixin):
    """
    Model runner that handles model loading and execution for diffusion models.

    This class follows the AR pattern where the Runner handles all model-related
    operations including loading, compilation, offloading, caching, and execution.
    The Worker only handles infrastructure (device, distributed env).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
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
        self.pipeline: Any | None = None
        self.cache_backend: Any | None = None
        self.offload_backend: Any | None = None
        self.prompt_embed_cache: Any | None = None
        self.input_batch: InputBatch | None = None

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, DiffusionRequestState] = {}

        # Initialize KV cache manager for connector management.
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

        # Prefetch covers TP / SP / CFG-Parallel / HSDP.  Disabled when a CFG
        # companion KV collector is set (that KV is not backgrounded).
        has_cfg_companion_kv = getattr(od_config, "cfg_kv_collect_func", None) is not None

        self._kv_prefetch_enabled = (
            bool(self.kv_transfer_manager.config.enable_kv_async_prefetch)
            and not has_cfg_companion_kv
            and self.kv_transfer_manager.config.need_recv_cache
        )

    @property
    def _target_device(self) -> torch.device | None:
        return getattr(self.pipeline, "device", None)

    # ------------------------------------------------------------------
    # Disaggregated-diffusion stage role (RFC #4590)
    # ------------------------------------------------------------------

    @property
    def model_stage(self) -> str:
        """Canonical stage role for this runner (``diffusion`` when unset).

        ``diffusion`` (or ``None``/empty) means the monolithic single-worker
        fallback: this runner owns the whole ``forward()`` and its execution
        path is unchanged from before RFC #4590.
        """
        return normalize_stage_role(getattr(self.od_config, "model_stage", None))

    @property
    def is_disaggregated_stage(self) -> bool:
        """True when this runner is an encode/denoise/decode stage."""
        return is_disaggregated_role(getattr(self.od_config, "model_stage", None))

    def supports_disaggregated_mode(self) -> bool:
        """Return whether the loaded pipeline can run as a disaggregated stage."""
        return self.pipeline is not None and supports_disaggregated_execution(self.pipeline)

    def _require_disaggregated_pipeline(self) -> None:
        """Raise a clear startup error if a disaggregated role lacks capability."""
        if not self.supports_disaggregated_mode():
            raise ValueError(
                "Stage requested disaggregated role "
                f"{self.model_stage!r} but pipeline "
                f"{type(self.pipeline).__name__ if self.pipeline else self.od_config.model_class_name} "
                "does not implement the DiffusionV2Atoms disaggregation contract "
                "(supports_disaggregated_execution flag + pack_stage_state / "
                "unpack_stage_state / required_components_for_stage)."
            )

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
        memory_pool_context_fn: Callable[[str], AbstractContextManager[Any]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        """
        Launch the diffusion pipeline, applying compilation, offloading, and caching.

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

        def get_memory_context() -> AbstractContextManager[Any]:
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn("weights")
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

        if self.od_config.streaming_output and not getattr(self.od_config, "step_execution", False):
            logger.warning("streaming_output=True requires step_execution=True; enabling step execution.")
            self.od_config.step_execution = True

        if getattr(self.od_config, "step_execution", False) and not self._supports_step_mode():
            raise ValueError(
                "step_execution=True requires a pipeline implementing the DiffusionV2Atoms "
                "step atoms or the legacy SupportsStepExecution contract; "
                f"{self.od_config.model_class_name} does not support that contract."
            )
        if self.od_config.streaming_output and not self._supports_step_mode():
            raise ValueError(
                "streaming_output=True requires step execution support; "
                f"{self.od_config.model_class_name} does not support that contract."
            )

        # Disaggregated-diffusion capability check (RFC #4590): a stage that
        # declares an encode/denoise/decode role requires a pipeline that
        # implements the disaggregated protocol. Fail fast at startup rather
        # than mid-forward.
        if self.is_disaggregated_stage:
            self._require_disaggregated_pipeline()
            # Stage-aware continuous batching is not implemented: a disaggregated
            # stage runs one request at a time through execute_model. Enforce
            # max_num_seqs==1 at startup so a misconfigured deploy fails fast here
            # rather than mid-serving when the scheduler tries to batch (the batch
            # path is also rejected at runtime in execute_model_batch).
            max_num_seqs = int(getattr(self.od_config, "max_num_seqs", 1) or 1)
            if max_num_seqs != 1:
                raise ValueError(
                    f"Disaggregated stage {self.model_stage!r} requires max_num_seqs==1 "
                    f"(stage-aware continuous batching is not implemented); got "
                    f"max_num_seqs={max_num_seqs}. Set max_num_seqs: 1 for this stage."
                )
        self._log_stage_startup()

        # Apply CPU offloading
        self.offload_backend = get_offload_backend(self.od_config, device=self.device)
        if self.offload_backend is not None:
            logger.info(f" Enabling offloader backend: {self.offload_backend.__class__.__name__}")
            self.offload_backend.enable(self.pipeline)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                if hasattr(self.pipeline, "setup_compile"):
                    try:
                        self.pipeline.setup_compile()
                    except Exception as exc:
                        logger.warning(
                            "Model runner: setup_compile() failed (%s); running without compile.",
                            exc,
                        )
                else:
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

    def clear_prompt_embed_cache(self) -> None:
        """Evict all cached text-encoder outputs (e.g. between training epochs).

        Kept primarily for extension purposes.
        """
        if self.prompt_embed_cache is not None:
            self.prompt_embed_cache.clear()

    def get_prompt_embed_cache_stats(self) -> dict | None:
        """Return hit/miss statistics for the prompt-embedding cache, if enabled.

        Kept primarily for extension purposes.
        """
        if self.prompt_embed_cache is None:
            return None
        return self.prompt_embed_cache.stats()

    def _sample_peak_memory_mb(self) -> float:
        """Return peak GPU memory for the current forward pass in MB.

        Must be called immediately after the measured forward/step work, with
        reset_peak_memory_stats() called just before it, so the measurement
        reflects the current execution slice and not the global historical
        maximum.

        Uses max_memory_reserved (CUDA memory pool high-water mark) rather than
        max_memory_allocated so that allocator fragmentation is also visible.
        See: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
        """
        peak_reserved_bytes = current_omni_platform.max_memory_reserved()
        peak_allocated_bytes = current_omni_platform.max_memory_allocated()

        peak_memory_mb = peak_reserved_bytes / (1024**2)
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
        return peak_memory_mb

    def _prepare_request_for_forward(
        self,
        req: OmniDiffusionRequest,
        *,
        od_config: OmniDiffusionConfig,
        kv_prefetch_job: KVPrefetchJob | None = None,
        use_prefetch: bool = False,
    ) -> None:
        # Receive AR KV. Single-request execution can use the prefetch path:
        # consume prior-forward payload, sync-fallback on miss; request-batch
        # execution keeps the synchronous per-request receive path.
        kv_recv_t0 = time.perf_counter()
        if use_prefetch and self._kv_prefetch_enabled:
            self.kv_transfer_manager.consume_and_distribute_kv_cache(
                req,
                target_device=self._target_device,
            )
        else:
            self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                req,
                cfg_kv_collect_func=getattr(od_config, "cfg_kv_collect_func", None),
                target_device=self._target_device if use_prefetch else getattr(self.pipeline, "device", None),
            )
        kv_recv_ms = (time.perf_counter() - kv_recv_t0) * 1000
        logger.debug("KV recv for %s %.1fms", req.request_id, kv_recv_ms)

        # Kick off the next request's prefetch (+ H2D) to overlap this forward.
        if use_prefetch and self._kv_prefetch_enabled and kv_prefetch_job is not None:
            self.kv_transfer_manager.start_prefetch(kv_prefetch_job, self._target_device)

        self._initialize_generator(req.sampling_params)

    def _initialize_generator(self, sampling_params: OmniDiffusionSamplingParams) -> None:
        if sampling_params.generator is None and sampling_params.seed is not None:
            if sampling_params.generator_device is not None:
                gen_device = sampling_params.generator_device
            elif self.device.type == "cpu":
                gen_device = "cpu"
            else:
                gen_device = self.device
            sampling_params.generator = torch.Generator(device=gen_device).manual_seed(sampling_params.seed)

    def _refresh_cache_for_requests(
        self,
        reqs: list[OmniDiffusionRequest],
        *,
        od_config: OmniDiffusionConfig,
    ) -> None:
        first_req = reqs[0]
        if self.cache_backend is None or not self.cache_backend.is_enabled():
            return

        # Refresh cache context if needed. Batch admission groups requests by
        # RequestBatchSamplingParamsKey, so the first request's num_inference_steps applies
        # to the whole runner batch.
        num_inference_steps = first_req.sampling_params.num_inference_steps
        if num_inference_steps is None and od_config.cache_backend in (
            "tea_cache",
            "step_cache",
        ):
            # When num_inference_steps is None, some pipelines defer to their
            # own defaults. TeaCache refresh ignores this value; step_cache
            # refresh is a no-op because per-chunk state resets in the denoise
            # loop. Use the pipeline default when available to keep refresh
            # behavior aligned with single-request execution.
            num_inference_steps = getattr(self.pipeline, "num_inference_steps", 0) or 0

        if num_inference_steps is not None:
            self.cache_backend.refresh(self.pipeline, num_inference_steps)
        else:
            logger.warning(
                "Failed to refresh the diffusion transformer cache; backend %s "
                "currently requires num_inference_steps to be passed explicitly",
                od_config.cache_backend,
            )

    def _runner_output_from_outputs(
        self,
        reqs: list[OmniDiffusionRequest],
        outputs: list[DiffusionOutput],
    ) -> BatchRunnerOutput:
        return BatchRunnerOutput.from_list(
            [
                RunnerOutput(
                    request_id=reqs[i].request_id,
                    step_index=None,
                    finished=True,
                    result=outputs[i],
                )
                for i in range(len(reqs))
            ]
        )

    def _execute_request_list(
        self,
        reqs: list[OmniDiffusionRequest],
        *,
        od_config: OmniDiffusionConfig,
        allow_single_output: bool,
        require_request_batch_support: bool,
        kv_prefetch_job: KVPrefetchJob | None = None,
        record_name: str,
    ) -> BatchRunnerOutput:
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not reqs:
            return BatchRunnerOutput.from_list([])
        for req in reqs:
            if req.prompt is None:
                raise ValueError("Cannot execute model with empty prompt")
        if require_request_batch_support and not getattr(self.pipeline, "supports_request_batch", False):
            raise RuntimeError(f"{type(self.pipeline).__name__} does not support request-batch forward.")

        # Use no_grad() for HSDP compatibility, inference_mode() otherwise for
        # better perf. HSDP2's fully_shard pre-forward hooks need tensor version
        # counters, which inference tensors do not track.
        use_hsdp = od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            for req in reqs:
                self._prepare_request_for_forward(
                    req,
                    od_config=od_config,
                    kv_prefetch_job=kv_prefetch_job,
                    use_prefetch=allow_single_output,
                )

            self._refresh_cache_for_requests(reqs, od_config=od_config)

            batch = DiffusionRequestBatch(requests=reqs)
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_primary:
                current_omni_platform.reset_peak_memory_stats()

            with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=od_config):
                with record_function(record_name):
                    raw_outputs = self.pipeline.forward(batch)
                    outputs = _normalize_pipeline_outputs(
                        raw_outputs,
                        expected_count=len(reqs),
                        allow_single_output=allow_single_output,
                        pipeline_name=type(self.pipeline).__name__,
                    )

            if is_primary and outputs:
                batch_peak_memory_mb = self._sample_peak_memory_mb()
                for output in outputs:
                    output.peak_memory_mb = max(output.peak_memory_mb, batch_peak_memory_mb)

            # Log prompt-embed cache activity; hits/misses accumulate across requests.
            prompt_embed_cache = getattr(self, "prompt_embed_cache", None)
            if is_primary and prompt_embed_cache is not None:
                logger.debug("prompt-embed cache: %s", prompt_embed_cache.stats())

            if (
                self.cache_backend is not None
                and self.cache_backend.is_enabled()
                and od_config.cache_backend == "cache_dit"
                and od_config.enable_cache_dit_summary
            ):
                cache_summary(self.pipeline, details=True)

        return self._runner_output_from_outputs(reqs, outputs)

    def _attach_stepwise_metrics(
        self,
        state: DiffusionRequestState,
        output: DiffusionOutput,
    ) -> None:
        merge_stage_durations(
            state,
            consume_pipeline_stage_durations(self.pipeline),
        )
        attach_stage_durations(state, output)

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
    ) -> DiffusionOutput:
        """
        Execute a forward pass for the given request.

        Dispatches on the stage role (RFC #4590):

        * ``diffusion`` / unset  -> the existing monolithic single-request path
          (unchanged behavior).
        * ``encode``             -> :meth:`execute_encode_stage`.
        * ``denoise``            -> :meth:`execute_denoise_stage`.
        * ``decode``             -> :meth:`execute_decode_stage`.
        * any other declared role -> :meth:`execute_model_defined_stage`.

        Args:
            req: A diffusion request containing a prompt to process.

        Returns:
            DiffusionOutput with generated results (encode/denoise stages return
            an intermediate output carrying a StagePayload; decode and
            monolithic stages return the user-visible result).

        Note:
            We use torch.no_grad() for HSDP because HSDP2's fully_shard requires access
            to tensor version counters in pre_forward hooks, which inference tensors do
            not track. For non-HSDP inference, we use torch.inference_mode() for better
            performance.
        """
        role = self.model_stage
        path = resolve_execution_path(role)
        if path == EXECUTION_PATH_ENCODE:
            return self.execute_encode_stage(req)
        if path == EXECUTION_PATH_DENOISE:
            return self.execute_denoise_stage(req)
        if path == EXECUTION_PATH_DECODE:
            return self.execute_decode_stage(req)
        if path == EXECUTION_PATH_MODEL_DEFINED:
            # A model-declared custom role (not one of the built-in three).
            return self.execute_model_defined_stage(role, req)

        assert path == EXECUTION_PATH_MONOLITHIC
        return self._execute_monolithic(req, kv_prefetch_job=kv_prefetch_job)

    def _execute_monolithic(
        self, req: OmniDiffusionRequest, kv_prefetch_job: KVPrefetchJob | None = None
    ) -> DiffusionOutput:
        """The original monolithic single-request path (unchanged)."""
        runner_output = self._execute_request_list(
            [req],
            od_config=self.od_config,
            allow_single_output=True,
            require_request_batch_support=False,
            kv_prefetch_job=kv_prefetch_job,
            record_name="pipeline_forward",
        )
        output = runner_output.runner_outputs[0].result
        assert output is not None
        return output

    def execute_model_defined_stage(self, role: str, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Hook for model-declared custom stage roles.

        The base runner has no built-in path for roles outside
        encode/denoise/decode. A pipeline that declares support for a custom
        role should subclass the runner (or override this) to handle it. Raising
        here keeps unknown roles from silently taking the monolithic path.
        """
        raise ValueError(
            f"Stage role {role!r} has no built-in execution path and pipeline "
            f"{type(self.pipeline).__name__ if self.pipeline else self.od_config.model_class_name} "
            "does not override execute_model_defined_stage()."
        )

    # ------------------------------------------------------------------
    # Disaggregated-stage execution (RFC #4590)
    # ------------------------------------------------------------------

    def _log_stage_startup(self) -> None:
        """Log stage role, capability, and (when known) loaded components."""
        components = "n/a"
        if self.supports_disaggregated_mode():
            try:
                spec = type(self.pipeline).required_components_for_stage(self.model_stage)
                components = spec.describe()
            except Exception:  # pragma: no cover - logging must not fail startup
                components = "unknown"
        logger.info(
            "Diffusion stage startup: pipeline=%s stage_id=%s model_stage=%s disaggregated_capable=%s components=%s",
            type(self.pipeline).__name__ if self.pipeline else self.od_config.model_class_name,
            getattr(self.od_config, "stage_id", 0),
            self.model_stage,
            self.supports_disaggregated_mode(),
            components,
        )

    def _grad_context(self):
        """Return the grad context used by all forward paths (HSDP-aware)."""
        use_hsdp = self.od_config.parallel_config.use_hsdp
        return torch.no_grad() if use_hsdp else torch.inference_mode()

    def _create_state_from_request(self, req: OmniDiffusionRequest) -> DiffusionRequestState:
        """Build a fresh runner-local state from a raw request (encode stage)."""
        state = DiffusionRequestState(
            request_id=req.request_id,
            sampling=copy.deepcopy(req.sampling_params),
            prompt=req.prompt,
            kv_sender_info=req.kv_sender_info,
        )
        # Seed the per-request RNG generator via the shared upstream helper
        # (_initialize_generator takes the sampling params directly).
        self._initialize_generator(state.sampling)
        return state

    def _extract_incoming_payload(self, req: OmniDiffusionRequest) -> StagePayload:
        """Pull the upstream StagePayload out of a request prompt.

        The generic transition processor places it in the prompt's ``extra``
        sub-dict under :data:`STAGE_PAYLOAD_PROMPT_KEY`. Raise a clear,
        request-scoped error if it is missing or malformed.

        The request crosses the out-of-process (multiproc) stage boundary via
        msgpack, which does not preserve dataclass identity: the payload
        arrives here as a plain ``dict`` rather than a ``StagePayload``
        instance. Rehydrate it before validating.
        """
        prompt = req.prompt
        extra = None
        if isinstance(prompt, dict):
            extra = prompt.get("extra")
        elif hasattr(prompt, "extra"):
            extra = getattr(prompt, "extra")
        payload = extra.get(STAGE_PAYLOAD_PROMPT_KEY) if isinstance(extra, dict) else None
        if payload is None:
            raise StagePayloadError(
                f"Request {req.request_id!r} reached stage {self.model_stage!r} without a "
                f"StagePayload in prompt['extra'][{STAGE_PAYLOAD_PROMPT_KEY!r}]; "
                f"pipeline={type(self.pipeline).__name__}."
            )
        if isinstance(payload, dict) and not isinstance(payload, StagePayload):
            payload = StagePayload.from_dict(payload)
        if not isinstance(payload, StagePayload):
            raise StagePayloadError(
                f"Request {req.request_id!r} stage {self.model_stage!r}: payload is "
                f"{type(payload).__name__}, expected StagePayload."
            )
        payload.validate()
        # Request-identity cross-check: the payload the transition processor
        # routed into this request must belong to this request. A mismatch means
        # request A's stage output was delivered to request B (a routing/queue
        # bug) — fail loud before restoring the wrong request's state or KV,
        # rather than silently denoising A's conditions under B's session.
        if payload.request_id != req.request_id:
            raise StagePayloadError(
                f"Stage {self.model_stage!r} received a StagePayload for request "
                f"{payload.request_id!r} on request {req.request_id!r} "
                f"(boundary={payload.boundary.value}); cross-request payload routing is a bug."
            )
        return payload

    def _intermediate_output(
        self,
        state: DiffusionRequestState,
        payload: StagePayload,
    ) -> DiffusionOutput:
        """Wrap an exported payload as an intermediate (non-final) stage output.

        The payload rides in ``custom_output`` under
        :data:`STAGE_PAYLOAD_OUTPUT_KEY`; the generic transition processor reads
        it there. ``to_cpu=True`` guarantees the receiving process never touches
        a live device tensor (payload tensors are already host-resident, but
        stage_durations and any stray tensors are normalized here too).
        """
        output = DiffusionOutput(
            output=None,
            custom_output={STAGE_PAYLOAD_OUTPUT_KEY: payload},
            finished=True,
            to_cpu=True,
        )
        attach_stage_durations(state, output)
        output.peak_memory_mb = max(output.peak_memory_mb, state.peak_memory_mb)
        return output

    def _reset_peak_memory(self) -> bool:
        """Reset peak-memory stats on the primary rank; return is_primary."""
        is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_primary and current_omni_platform.is_available():
            current_omni_platform.reset_peak_memory_stats()
        return is_primary

    def execute_encode_stage(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Encode stage: validate + encode conditions + prepare latents/timesteps.

        Produces an intermediate output carrying an encode->denoise payload. Does
        NOT run the DiT, advance the scheduler, or decode. Runner-local state is
        released once the payload is handed off (the payload is self-contained).
        """
        self._require_disaggregated_pipeline()
        with self._grad_context():
            is_primary = self._reset_peak_memory()
            clear_pipeline_stage_durations(self.pipeline)
            state = self._create_state_from_request(req)
            try:
                # Run under the diffusion forward context for parity with the
                # monolithic/denoise/decode paths (encoders may consult it).
                # Drive the DiffusionV2Atoms encode chain explicitly:
                # init_state -> check_inputs -> encode -> prepare.
                with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
                    with record_function("pipeline_encode_stage"):
                        state = self.pipeline.init_state(state)
                        state = self.pipeline.check_inputs(state)
                        state = self.pipeline.encode(state)
                        state = self.pipeline.prepare(state)
                merge_stage_durations(state, consume_pipeline_stage_durations(self.pipeline))
                payload = self.pipeline.pack_stage_state(state, StageBoundary.ENCODE_TO_DIT)
            finally:
                # Encode owns no persistent state; drop any cache entry eagerly.
                self.state_cache.pop(req.request_id, None)
            if is_primary:
                state.peak_memory_mb = max(state.peak_memory_mb, self._sample_peak_memory_mb())
            logger.debug("encode stage: request %s payload exported (%s)", req.request_id, payload.summary())
            return self._intermediate_output(state, payload)

    def execute_denoise_stage(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Denoise stage: restore state from payload, run the DiT denoise loop.

        Creates a fresh runner-local state from the request (so request-level
        sampling/generator/session plumbing is preserved), unpacks the incoming
        stage payload into it (``unpack_stage_state`` mutates the existing state —
        it never fabricates one), runs the pipeline's whole-request ``diffuse``
        atom, and packs a denoise->decode payload. Any live session/KV state the
        denoise stage owns (AR-Diffusion KV) is attached by the runner subclass
        before this call — never via the payload. Never re-runs encode.
        """
        self._require_disaggregated_pipeline()
        with self._grad_context():
            is_primary = self._reset_peak_memory()
            clear_pipeline_stage_durations(self.pipeline)
            payload = self._extract_incoming_payload(req)
            payload.expect_boundary(StageBoundary.ENCODE_TO_DIT)
            state = self._create_state_from_request(req)
            state = self.pipeline.unpack_stage_state(payload, state)
            self.state_cache[req.request_id] = state
            try:
                with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
                    with record_function("pipeline_denoise_stage"):
                        state = self.pipeline.diffuse(state)
                out_payload = self.pipeline.pack_stage_state(state, StageBoundary.DIT_TO_DECODE)
                merge_stage_durations(state, consume_pipeline_stage_durations(self.pipeline))
            finally:
                # For the first (non-streaming) implementation the whole request
                # denoises in one call, so its state is releasable now. Streaming
                # session retention is handled by the pipeline via its own session
                # map, not this cache.
                self.state_cache.pop(req.request_id, None)
            if is_primary:
                state.peak_memory_mb = max(state.peak_memory_mb, self._sample_peak_memory_mb())
            logger.debug("denoise stage: request %s payload exported (%s)", req.request_id, out_payload.summary())
            return self._intermediate_output(state, out_payload)

    def execute_decode_stage(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Decode stage: restore decode state, run VAE/postprocess, return output.

        Restores only what decode needs, runs decode + postprocess, and returns
        the normal user-visible ``DiffusionOutput``. Never instantiates or runs
        the DiT, advances a scheduler, or re-runs encoders.
        """
        self._require_disaggregated_pipeline()
        with self._grad_context():
            is_primary = self._reset_peak_memory()
            clear_pipeline_stage_durations(self.pipeline)
            payload = self._extract_incoming_payload(req)
            payload.expect_boundary(StageBoundary.DIT_TO_DECODE)
            state = self._create_state_from_request(req)
            state = self.pipeline.unpack_stage_state(payload, state)
            self.state_cache[req.request_id] = state
            try:
                with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
                    with record_function("pipeline_decode_stage"):
                        state = self.pipeline.decode(state)
                        output = self.pipeline.postprocess(state)
                if output is None:
                    raise RuntimeError(f"Decode stage produced no output for request {req.request_id!r}.")
                self._attach_stepwise_metrics(state, output)
            finally:
                self.state_cache.pop(req.request_id, None)
            if is_primary:
                peak = self._sample_peak_memory_mb()
                state.peak_memory_mb = max(state.peak_memory_mb, peak)
                output.peak_memory_mb = max(output.peak_memory_mb, state.peak_memory_mb)
            logger.debug("decode stage: request %s produced final output", req.request_id)
            return output

    def execute_model_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        od_config: OmniDiffusionConfig,
    ) -> BatchRunnerOutput:
        """Execute scheduled request-mode requests through the batch forward path.

        Builds a ``DiffusionRequestBatch`` from scheduled new requests, runs
        per-request setup, and calls ``pipeline.forward(batch)``. The pipeline
        must declare ``supports_request_batch = True``.
        """
        # Stage-aware batch guard (RFC #4590 §B.2): the batch path runs the
        # monolithic whole-request ``pipeline.forward(batch)``, which composes
        # encode+denoise+decode in one process. A disaggregated encode/denoise/
        # decode worker must NEVER take that path — it would silently re-run the
        # phases this stage does not own (and, for denoise, bypass the per-stage
        # payload/session plumbing). Stage-aware continuous batching is not yet
        # implemented, so reject the batch path outright for disaggregated roles
        # rather than let a role fall through to the monolithic forward.
        if self.is_disaggregated_stage:
            raise RuntimeError(
                f"Disaggregated stage {self.model_stage!r} cannot use the monolithic "
                "request-batch path (pipeline.forward(batch)); it must run through the "
                "stage-specific execute_model dispatch. Set max_num_seqs=1 for "
                "disaggregated stages (stage-aware continuous batching is not implemented)."
            )
        reqs = [nr.req for nr in scheduler_output.scheduled_new_reqs]
        return self._execute_request_list(
            reqs,
            od_config=od_config,
            allow_single_output=False,
            require_request_batch_support=True,
            record_name="pipeline_forward_batch",
        )

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def _supports_step_mode(self) -> bool:
        """Return whether current pipeline supports step execution."""
        return self.pipeline is not None and supports_step_execution(self.pipeline)

    def _update_states(self, scheduler_output: DiffusionSchedulerOutput) -> tuple[list[DiffusionRequestState], list[str]]:
        """Step-before update: cleanup finished requests and get/create one running state."""
        for request_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(request_id, None)

        resolved: list[DiffusionRequestState] = []
        new_request_ids: list[str] = []
        try:
            # process new requests
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                new_request_ids.append(request_id)
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
                new_state = DiffusionRequestState(
                    request_id=request_id,
                    sampling=copy.deepcopy(sched_new_req.req.sampling_params),
                    prompt=sched_new_req.req.prompt,
                    kv_sender_info=sched_new_req.req.kv_sender_info,
                )
                state_req = copy.copy(sched_new_req.req)
                state_req.sampling_params = new_state.sampling
                self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                    state_req,
                    cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                    target_device=self._target_device,
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
                self._initialize_generator(state.sampling)
                clear_pipeline_stage_durations(self.pipeline)
                # encode
                self.pipeline.prepare_encode(state)
                merge_stage_durations(
                    state,
                    consume_pipeline_stage_durations(self.pipeline),
                )

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
    ) -> None:
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
            if interrupted or state.request_denoise_completed:
                self.state_cache.pop(state.request_id, None)

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one step for one scheduled request and return runner output."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self._supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        # Stepwise mode only supports the basic state-driven denoise path for now.
        # Request-mode extras such as cache backends, editing inputs, and
        # similar features are not supported here yet.
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            had_active_states = bool(self.state_cache)
            states, new_request_ids = self._update_states(scheduler_output)
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if new_request_ids and not had_active_states and is_primary and current_omni_platform.is_available():
                current_omni_platform.reset_peak_memory_stats()
            input_batch = self._prepare_batch_inputs(states, new_request_ids)
            attn_metadata = {}

            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
            ):
                clear_pipeline_stage_durations(self.pipeline)
                noise_pred = self.pipeline.denoise_step(input_batch, states=states)
                denoise_stage_durations = consume_pipeline_stage_durations(self.pipeline)
                for state in states:
                    merge_stage_durations(
                        state,
                        denoise_stage_durations,
                    )

                runner_output_list = []
                pipeline_interrupted = getattr(self.pipeline, "interrupt", False)
                if noise_pred is None and pipeline_interrupted:
                    for state in states:
                        runner_output_list.append(
                            RunnerOutput(
                                request_id=state.request_id,
                                step_index=state.step_index,
                                finished=True,
                                result=DiffusionOutput(error="stepwise denoise interrupted"),
                            )
                        )

                else:
                    offset = 0
                    for req in states:
                        row_num = req.latents.shape[0]
                        try:
                            self.pipeline.step_scheduler(
                                req, noise_pred[offset : offset + row_num] if noise_pred is not None else None
                            )
                            if self.od_config.streaming_output:
                                should_decode = req.chunk_denoise_completed
                            else:
                                should_decode = req.denoise_completed

                            if should_decode:
                                clear_pipeline_stage_durations(self.pipeline)
                                result = self.pipeline.post_decode(req)
                                if result is not None:
                                    self._attach_stepwise_metrics(
                                        req,
                                        result,
                                    )
                            else:
                                result = None
                            # finished should be computed after post_decode() advanced chunk_index
                            finished = (
                                req.request_denoise_completed
                                if self.od_config.streaming_output
                                else req.denoise_completed
                            )
                            runner_output_list.append(
                                RunnerOutput(
                                    request_id=req.request_id,
                                    step_index=req.step_index,
                                    finished=finished,
                                    result=result,
                                )
                            )
                            offset = offset + row_num
                        except Exception as per_req_exc:
                            offset = offset + row_num
                            logger.error(
                                "Stepwise per-request error for %s: %s",
                                req.request_id,
                                per_req_exc,
                                exc_info=True,
                            )
                            runner_output_list.append(
                                RunnerOutput(
                                    request_id=req.request_id,
                                    step_index=req.step_index,
                                    finished=True,
                                    result=DiffusionOutput(error=str(per_req_exc)),
                                )
                            )

                    if noise_pred is not None and offset != noise_pred.shape[0]:
                        raise ValueError(
                            f"Stepwise noise_pred consumed {offset} rows, "
                            f"but batched noise_pred has {noise_pred.shape[0]} rows."
                        )

                if is_primary:
                    batch_peak_memory_mb = self._sample_peak_memory_mb()
                    states_by_id = {state.request_id: state for state in states}
                    for state in states:
                        state.peak_memory_mb = max(state.peak_memory_mb, batch_peak_memory_mb)
                    for runner_output in runner_output_list:
                        if runner_output.result is None:
                            continue
                        state = states_by_id.get(runner_output.request_id)
                        if state is None:
                            continue
                        runner_output.result.peak_memory_mb = max(
                            runner_output.result.peak_memory_mb,
                            state.peak_memory_mb,
                        )

                self._update_states_after(states, input_batch, pipeline_interrupted)

                return BatchRunnerOutput.from_list(runner_output_list)
