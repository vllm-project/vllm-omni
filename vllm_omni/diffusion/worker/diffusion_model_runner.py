# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import copy
import gc
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import cache_summary
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionMetadata,
    DiffusionPagedAttentionRow,
)
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_rank
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportsPromptUpdate,
    adopt_request_scoped_cache_dit,
    is_request_scoped_cache_dit_enabled,
    supports_prompt_update,
    supports_step_execution,
)
from vllm_omni.diffusion.offloader import enable_offload_backend
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    KVPrefetchJob,
    NewRequestData,
    validate_new_request_data_identity,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    RunnerOutput,
    StepRequestState,
    attach_stage_durations,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniInteractionPrompt

logger = init_logger(__name__)


def _dit_any_rank_failed(local_failed: bool) -> bool:
    """All-reduce a per-request failure flag across the DiT process group.

    Every DiT rank must reach this point in lockstep; the caller wraps
    ``pipeline.prepare_encode`` so both the success and the exception paths
    call the helper. In single-rank execution this collapses to the local
    flag with no collectives issued.
    """
    if not torch.distributed.is_initialized():
        return local_failed
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_dit_group

        group = get_dit_group()
    except (AssertionError, ImportError):
        group = None
    if group is None:
        return local_failed
    signal = torch.tensor(1 if local_failed else 0, dtype=torch.int32)
    if current_omni_platform.is_available():
        signal = signal.to(device=current_omni_platform.device_type)
    torch.distributed.all_reduce(signal, op=torch.distributed.ReduceOp.MAX, group=group)
    return bool(signal.item())


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
        self.model_memory_usage = 0
        self.diffusion_kv_backend = DiffusionKVModelRunnerBackend(
            vllm_config=vllm_config,
            od_config=od_config,
            device=device,
        )
        # Compatibility view for callers that inspect the installed config;
        # physical ownership remains in ``diffusion_kv_backend``.
        self.kv_cache_config: KVCacheConfig | None = None

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, StepRequestState] = {}

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

    def _validate_diffusion_kv_metadata(
        self,
        *,
        request_id: str,
        metadata: DiffusionKVMetadata | None,
    ) -> None:
        cache_mode = getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        if cache_mode is DiffusionKVCacheMode.PAGED_SCHEDULER and metadata is None:
            raise ValueError(f"paged_scheduler request {request_id!r} requires Diffusion KV metadata")
        if cache_mode is not DiffusionKVCacheMode.PAGED_SCHEDULER and metadata is not None:
            raise ValueError(f"{cache_mode.value} request {request_id!r} must not carry Diffusion KV metadata")

        if metadata is not None and metadata.request_id != request_id:
            raise ValueError(
                f"Diffusion KV metadata request mismatch: expected={request_id!r}, got={metadata.request_id!r}"
            )

    def _compile_transformer(self, attr_name: str) -> None:
        """Compile a transformer attribute on the pipeline with torch.compile."""
        model = getattr(self.pipeline, attr_name, None)
        if model is None:
            return

        compile_granularity = self.od_config.diffusion_compile_granularity
        compile_dynamic = self.od_config.diffusion_compile_dynamic
        try:
            if compile_granularity == "full":
                model.compile(dynamic=compile_dynamic)
                compiled_model = model
            else:
                compiled_model = regionally_compile(model, dynamic=compile_dynamic)
            setattr(self.pipeline, attr_name, compiled_model)
        except Exception as e:
            logger.warning(
                "Model runner: %s torch.compile setup for %s failed before activation: %s. "
                "Continuing with the uncompiled model; lazy compilation errors can still "
                "surface on the first request.",
                compile_granularity,
                attr_name,
                e,
            )
            return

        logger.info(
            "Model runner: %s configured for lazy %s torch.compile with dynamic=%s; "
            "compilation errors may surface on the first request.",
            attr_name,
            compile_granularity,
            compile_dynamic,
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

        current_omni_platform.init_diffusion_model_runner_runtime(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )

        load_device = (
            "cpu"
            if self.od_config.enable_cpu_offload
            or self.od_config.enable_layerwise_offload
            or getattr(self.od_config, "enable_distributed_layerwise_offload", False)
            else str(self.device)
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
        self.model_memory_usage = int(m.consumed_memory)
        logger.info("Model runner: Model loaded successfully.")

        if self.od_config.streaming_output and not getattr(self.od_config, "step_execution", False):
            logger.warning("streaming_output=True requires step_execution=True; enabling step execution.")
            self.od_config.step_execution = True

        if getattr(self.od_config, "step_execution", False) and not self._supports_step_mode():
            raise ValueError(
                "step_execution=True requires a pipeline implementing "
                "prepare_encode(), denoise_step(), step_scheduler(), and post_decode(); "
                f"{self.od_config.model_class_name} does not support that contract."
            )
        if self.od_config.streaming_output and not self._supports_step_mode():
            raise ValueError(
                "streaming_output=True requires step execution support; "
                f"{self.od_config.model_class_name} does not support that contract."
            )

        # The offloader owns loader-plan handoff and startup recovery. The
        # runner only receives the pipeline/backend pair that is ready to use.
        self.pipeline, self.offload_backend = enable_offload_backend(
            self.od_config,
            self.pipeline,
            device=self.device,
        )

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
                    transformer_attrs = getattr(self.pipeline, "_dit_modules", None)
                    if not transformer_attrs:
                        transformer_attrs = ("transformer", "transformer_2")
                    for attr_name in transformer_attrs:
                        self._compile_transformer(attr_name)
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
                # Install configured cache capability once at startup. A model
                # may explicitly adopt the enabled Cache-DiT backend and then
                # own all later request-boundary enable/disable transitions.
                self.cache_backend.enable(self.pipeline)
                if str(self.od_config.cache_backend).lower() == "cache_dit" and adopt_request_scoped_cache_dit(
                    self.pipeline,
                    self.cache_backend,
                ):
                    logger.info(
                        "Pipeline %s owns request-scoped Cache-DiT transitions.",
                        type(self.pipeline).__name__,
                    )
                    self.cache_backend = None

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

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Collect native specs from cache-enabled loaded attention modules."""

        if self.od_config.diffusion_kv_mode is not DiffusionKVCacheMode.PAGED_SCHEDULER:
            return {}
        if self.pipeline is None:
            raise RuntimeError("Model must be loaded before collecting Diffusion KV cache specs")

        cache_layers: dict[str, tuple[Attention, KVCacheSpec]] = {}
        for module_path, module in self.pipeline.named_modules():
            if not isinstance(module, Attention):
                continue
            spec = module.get_kv_cache_spec(self.vllm_config)
            if spec is not None:
                layer_name = module.prefix
                if not isinstance(layer_name, str) or not layer_name:
                    raise RuntimeError(
                        "Paged Diffusion Attention must expose a non-empty canonical prefix; "
                        f"module_path={module_path!r}"
                    )
                if layer_name in cache_layers:
                    raise RuntimeError(f"Duplicate canonical paged Diffusion Attention prefix {layer_name!r}")
                cache_layers[layer_name] = (module, spec)
        if not cache_layers:
            raise RuntimeError(
                "paged_scheduler Diffusion KV found no cache-enabled Attention modules "
                f"in {type(self.pipeline).__name__}"
            )
        return self.diffusion_kv_backend.register_kv_cache_layers(cache_layers)

    def set_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Physically initialize the Engine-generated rank-local config."""

        self.diffusion_kv_backend.initialize_kv_cache(kv_cache_config)
        self.kv_cache_config = self.diffusion_kv_backend.kv_cache_config

    def install_diffusion_kv_metadata(self, metadata: DiffusionKVMetadata) -> bool:
        return self.diffusion_kv_backend.install_diffusion_kv_metadata(metadata)

    def get_diffusion_kv_row(
        self,
        request_id: str,
        sequence_id: int | None,
        context_id: str | None = None,
    ) -> int:
        return self.diffusion_kv_backend.get_diffusion_kv_row(request_id, sequence_id, context_id)

    def remove_diffusion_kv_requests(self, request_ids: list[str]) -> int:
        return self.diffusion_kv_backend.remove_diffusion_kv_requests(request_ids)

    def refresh_diffusion_kv_block_table_layout(self) -> None:
        self.diffusion_kv_backend.refresh_block_table_layout()

    def _build_paged_attention_metadata(
        self,
        metadata: list[DiffusionKVMetadata],
    ) -> DiffusionPagedAttentionMetadata:
        """Translate Scheduler sequence state into ordered request-level rows."""

        cfg_size = int(getattr(self.od_config.parallel_config, "cfg_parallel_size", 1) or 1)
        cfg_rank = get_classifier_free_guidance_rank() if cfg_size > 1 else None
        prefill_rows: list[DiffusionPagedAttentionRow] = []
        denoise_rows: list[DiffusionPagedAttentionRow] = []
        for request_metadata in metadata:
            sequences = request_metadata.sequences
            if cfg_rank is not None and len(sequences) > 1:
                if len(sequences) != cfg_size:
                    raise ValueError(
                        "Paged CFG parallel execution requires one Scheduler sequence per CFG rank: "
                        f"request={request_metadata.request_id!r}, cfg_size={cfg_size}, rows={len(sequences)}"
                    )
                sequences = (sequences[cfg_rank],)
            for sequence in sequences:
                active_seq_len = sequence.prefix_len + sequence.target_len
                if active_seq_len > sequence.seq_len:
                    raise ValueError(
                        "Paged denoise span exceeds its Scheduler allocation: "
                        f"request={request_metadata.request_id!r}, sequence={sequence.sequence_id}, "
                        f"active={active_seq_len}, allocated={sequence.seq_len}"
                    )
                prefill_rows.append(
                    DiffusionPagedAttentionRow(
                        request_id=request_metadata.request_id,
                        sequence_id=sequence.sequence_id,
                        query_len=sequence.seq_len,
                        seq_len=sequence.seq_len,
                    )
                )
                denoise_rows.append(
                    DiffusionPagedAttentionRow(
                        request_id=request_metadata.request_id,
                        sequence_id=sequence.sequence_id,
                        query_len=sequence.target_len,
                        seq_len=active_seq_len,
                        kv_start_pos=sequence.prefix_len,
                    )
                )
        return DiffusionPagedAttentionMetadata(tuple(prefill_rows), tuple(denoise_rows))

    def clear_prompt_embed_cache(self) -> None:
        """Evict all cached text-encoder outputs (e.g. between training epochs).

        Kept primarily for extension purposes.
        """
        if self.prompt_embed_cache is not None:
            self.prompt_embed_cache.clear()

    def release_captured_graphs(self) -> None:
        """Drop every CUDA graph held for this model, wherever it is kept.

        Sleep level 2 discards the memory a capture was recorded against, so a
        graph that outlives it replays over freed storage. The runner owns
        compilation and execution resources, so it owns the release: a pipeline
        that captures graphs of its own implements ``release_captured_graphs``
        and is collected here, instead of every caller having to know which
        pipelines have one.
        """
        runners = getattr(self, "graph_runners", None)
        if runners is not None:
            runners.clear()
        release = getattr(self.pipeline, "release_captured_graphs", None)
        if callable(release):
            release()

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
        req.kv_recv_ms = kv_recv_ms
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
        if num_inference_steps is None and first_req.sampling_params.timesteps is not None:
            num_inference_steps = len(first_req.sampling_params.timesteps)
        if num_inference_steps is None and first_req.sampling_params.sigmas is not None:
            num_inference_steps = len(first_req.sampling_params.sigmas)
        if num_inference_steps is None:
            num_inference_steps = getattr(self.pipeline, "default_num_inference_steps", None)
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
        for i in range(len(reqs)):
            # Carry the runner-measured KV-recv timing onto the output so the
            # engine's step_streaming can surface it as diffusion_kv_load_s.
            outputs[i].kv_recv_ms = reqs[i].kv_recv_ms
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
        record_output_peak_memory: bool = True,
        in_diffusion_kv_memory_profile: bool = False,
        diffusion_kv_metadata: list[DiffusionKVMetadata] | None = None,
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
        use_distributed_offload = getattr(self.od_config, "enable_distributed_layerwise_offload", False)
        grad_context = torch.no_grad() if (use_hsdp or use_distributed_offload) else torch.inference_mode()
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
            if is_primary and record_output_peak_memory:
                current_omni_platform.reset_peak_memory_stats()

            paged_kv_runtime = None
            paged_kv_context: AbstractContextManager[Any] = nullcontext()
            if diffusion_kv_metadata is not None:
                if len(diffusion_kv_metadata) != len(reqs):
                    raise ValueError(
                        "Diffusion KV metadata count must match the request batch: "
                        f"metadata={len(diffusion_kv_metadata)}, requests={len(reqs)}"
                    )
                paged_metadata = self._build_paged_attention_metadata(diffusion_kv_metadata)
                paged_kv_runtime, paged_kv_context = self.diffusion_kv_backend.activate_paged_attention_metadata(
                    paged_metadata
                )
            with (
                set_forward_context(
                    vllm_config=self.vllm_config,
                    omni_diffusion_config=od_config,
                    paged_kv_runtime=paged_kv_runtime,
                    in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
                ),
                paged_kv_context,
            ):
                with record_function(record_name):
                    raw_outputs = self.pipeline.forward(batch)
                    outputs = _normalize_pipeline_outputs(
                        raw_outputs,
                        expected_count=len(reqs),
                        allow_single_output=allow_single_output,
                        pipeline_name=type(self.pipeline).__name__,
                    )

            if is_primary and outputs and record_output_peak_memory:
                batch_peak_memory_mb = self._sample_peak_memory_mb()
                for output in outputs:
                    output.peak_memory_mb = max(output.peak_memory_mb, batch_peak_memory_mb)

            # Log prompt-embed cache activity; hits/misses accumulate across requests.
            prompt_embed_cache = getattr(self, "prompt_embed_cache", None)
            if is_primary and prompt_embed_cache is not None:
                logger.debug("prompt-embed cache: %s", prompt_embed_cache.stats())

            runner_cache_dit_enabled = self.cache_backend is not None and self.cache_backend.is_enabled()
            if (
                od_config.cache_backend == "cache_dit"
                and od_config.enable_cache_dit_summary
                and (runner_cache_dit_enabled or is_request_scoped_cache_dit_enabled(self.pipeline))
            ):
                cache_summary(self.pipeline, details=True)

        return self._runner_output_from_outputs(reqs, outputs)

    def _attach_stepwise_metrics(
        self,
        state: StepRequestState,
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
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
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
        self._validate_diffusion_kv_metadata(
            request_id=req.request_id,
            metadata=diffusion_kv_metadata,
        )
        installed_request = False
        if diffusion_kv_metadata is not None:
            self.install_diffusion_kv_metadata(diffusion_kv_metadata)
            installed_request = True
        try:
            runner_output = self._execute_request_list(
                [req],
                od_config=self.od_config,
                allow_single_output=True,
                require_request_batch_support=False,
                kv_prefetch_job=kv_prefetch_job,
                record_name="pipeline_forward",
                diffusion_kv_metadata=[diffusion_kv_metadata] if diffusion_kv_metadata is not None else None,
            )
            output = runner_output.runner_outputs[0].result
            assert output is not None
            return output
        finally:
            if installed_request:
                self.remove_diffusion_kv_requests([req.request_id])

    def profile_run(self, requests: list[OmniDiffusionRequest]) -> None:
        """Run the maximum per-rank request batch for memory profiling.

        This deliberately bypasses Scheduler admission and Diffusion KV
        metadata validation because cache capacity has not been sized yet.
        It otherwise uses the normal execution-mode path so model inputs,
        collective communication, backend workspaces, denoising, and decode
        allocations contribute to the observed peak. Step execution profiles
        one fused ``InputBatch`` instead of sequential single-request forwards.
        """

        if not requests:
            raise ValueError("Diffusion memory profiling requires at least one request.")

        runner_output: BatchRunnerOutput | None = None
        request_ids = [request.request_id for request in requests]
        try:
            if getattr(self.od_config, "step_execution", False):
                scheduler_output = DiffusionSchedulerOutput(
                    step_id=0,
                    scheduled_new_reqs=[
                        NewRequestData(request_id=request.request_id, req=request) for request in requests
                    ],
                    scheduled_cached_reqs=CachedRequestData.make_empty(),
                    finished_req_ids=set(),
                    num_running_reqs=len(requests),
                    num_waiting_reqs=0,
                )
                runner_output = self._execute_stepwise(
                    scheduler_output,
                    validate_kv_metadata=False,
                    record_output_peak_memory=False,
                    in_diffusion_kv_memory_profile=True,
                )
            else:
                runner_output = self._execute_request_list(
                    requests,
                    od_config=self.od_config,
                    allow_single_output=len(requests) == 1,
                    require_request_batch_support=len(requests) > 1,
                    record_name="pipeline_memory_profile",
                    # The enclosing native memory_profiling context owns the
                    # peak counters. Resetting them here would discard request
                    # preparation allocations and understate the budget.
                    record_output_peak_memory=False,
                    in_diffusion_kv_memory_profile=True,
                )
            current_omni_platform.synchronize()
        finally:
            for request_id in request_ids:
                self.state_cache.pop(request_id, None)
            self.input_batch = None
            del runner_output
            gc.collect()

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
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
            self._validate_diffusion_kv_metadata(
                request_id=new_req.req.request_id,
                metadata=new_req.diffusion_kv_metadata,
            )
        installed_request_ids: list[str] = []
        try:
            for new_req in scheduler_output.scheduled_new_reqs:
                if new_req.diffusion_kv_metadata is not None:
                    self.install_diffusion_kv_metadata(new_req.diffusion_kv_metadata)
                    installed_request_ids.append(new_req.request_id)
            reqs = [nr.req for nr in scheduler_output.scheduled_new_reqs]
            return self._execute_request_list(
                reqs,
                od_config=od_config,
                allow_single_output=False,
                require_request_batch_support=True,
                record_name="pipeline_forward_batch",
                diffusion_kv_metadata=[
                    new_req.diffusion_kv_metadata
                    for new_req in scheduler_output.scheduled_new_reqs
                    if new_req.diffusion_kv_metadata is not None
                ]
                or None,
            )
        finally:
            if installed_request_ids:
                self.remove_diffusion_kv_requests(installed_request_ids)

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def _supports_step_mode(self) -> bool:
        """Return whether current pipeline supports step execution."""
        return self.pipeline is not None and supports_step_execution(self.pipeline)

    def _update_states(self, scheduler_output: DiffusionSchedulerOutput) -> tuple[list[StepRequestState], list[str]]:
        """Step-before update: cleanup finished requests and get/create one running state."""
        for request_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(request_id, None)

        resolved: list[StepRequestState] = []
        new_request_ids: list[str] = []
        try:
            # process new requests
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                new_request_ids.append(request_id)
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
                new_state = StepRequestState(
                    request_id=request_id,
                    sampling=copy.deepcopy(sched_new_req.req.sampling_params),
                    prompt=sched_new_req.req.prompt,
                    kv_sender_info=sched_new_req.req.kv_sender_info,
                    prepared_layout=getattr(sched_new_req.req, "prepared_layout", None),
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

    def _prepare_batch_inputs(
        self,
        states: list[StepRequestState],
        new_request_ids: list[str],
    ) -> tuple[list[StepRequestState], InputBatch | None, list[RunnerOutput]]:
        # process new reqs
        prepared_states: list[StepRequestState] = []
        error_outputs: list[RunnerOutput] = []
        for state in states:
            if state.request_id in new_request_ids:
                # Everything that runs before ``_dit_any_rank_failed`` must be
                # inside the try: an exception in ``_initialize_generator`` or
                # ``clear_pipeline_stage_durations`` on one rank would skip the
                # all-reduce here while every peer proceeds into it, and the
                # peers then hang on the NCCL collective until timeout.
                per_req_exc: BaseException | None = None
                try:
                    self._initialize_generator(state.sampling)
                    clear_pipeline_stage_durations(self.pipeline)
                    # encode
                    self.pipeline.prepare_encode(state)
                    merge_stage_durations(
                        state,
                        consume_pipeline_stage_durations(self.pipeline),
                    )
                except Exception as exc:
                    per_req_exc = exc
                # Pipelines that do rank-0-only work (e.g. MiniMax H3
                # reference-video prep) must broadcast per-request failures
                # internally so downstream collectives stay in step; even so,
                # cross-check that every DiT rank agrees so a rank-local error
                # (or a future pipeline that omits the guard) does not leave
                # the process group half-way through a new request.
                if _dit_any_rank_failed(per_req_exc is not None):
                    self.state_cache.pop(state.request_id, None)
                    if per_req_exc is None:
                        per_req_exc = RuntimeError(
                            f"Stepwise preparation failed on another DiT rank for {state.request_id}"
                        )
                    logger.error(
                        "Stepwise request preparation failed for %s: %s",
                        state.request_id,
                        per_req_exc,
                        exc_info=isinstance(per_req_exc, Exception),
                    )
                    error_outputs.append(
                        RunnerOutput(
                            request_id=state.request_id,
                            step_index=state.step_index,
                            finished=True,
                            result=DiffusionOutput.from_exception(per_req_exc),
                        )
                    )
                    continue
            prepared_states.append(state)

        if not prepared_states:
            return prepared_states, None, error_outputs
        input_batch = InputBatch.make_batch(
            prepared_states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self.input_batch = input_batch
        return prepared_states, input_batch, error_outputs

    def _update_states_after(
        self,
        states: list[StepRequestState],
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
        return self._execute_stepwise(
            scheduler_output,
            validate_kv_metadata=True,
            record_output_peak_memory=True,
        )

    def _execute_stepwise(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        *,
        validate_kv_metadata: bool,
        record_output_peak_memory: bool,
        in_diffusion_kv_memory_profile: bool = False,
    ) -> BatchRunnerOutput:
        """Execute one step with explicit validation and profiling policy."""

        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
            if validate_kv_metadata:
                self._validate_diffusion_kv_metadata(
                    request_id=new_req.req.request_id,
                    metadata=new_req.diffusion_kv_metadata,
                )
        if not self._supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        # Stepwise mode only supports the basic state-driven denoise path for now.
        # Request-mode extras such as cache backends, editing inputs, and
        # similar features are not supported here yet.
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        # Scheduler metadata is installed only for newly admitted requests;
        # cached requests continue to use the row installed on their first
        # step. Paged rows are retired here for scheduler-finished requests so
        # a subsequent wave can reuse the native table slots. Dense execution
        # must not issue Worker KV cleanup side effects.
        if (
            getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
            and scheduler_output.finished_req_ids
        ):
            self.remove_diffusion_kv_requests(list(scheduler_output.finished_req_ids))
        installed_request_ids: list[str] = []
        try:
            for new_req in scheduler_output.scheduled_new_reqs:
                if new_req.diffusion_kv_metadata is not None:
                    self.install_diffusion_kv_metadata(new_req.diffusion_kv_metadata)
                    installed_request_ids.append(new_req.request_id)
            return self._execute_stepwise_core(
                scheduler_output,
                record_output_peak_memory=record_output_peak_memory,
                in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
            )
        except Exception:
            if installed_request_ids:
                self.remove_diffusion_kv_requests(installed_request_ids)
            raise

    def _execute_stepwise_core(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        *,
        record_output_peak_memory: bool,
        in_diffusion_kv_memory_profile: bool = False,
    ) -> BatchRunnerOutput:
        """Run the denoise step after metadata admission and row installation."""

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            had_active_states = bool(self.state_cache)
            states, new_request_ids = self._update_states(scheduler_output)
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if (
                record_output_peak_memory
                and new_request_ids
                and not had_active_states
                and is_primary
                and current_omni_platform.is_available()
            ):
                current_omni_platform.reset_peak_memory_stats()
            states, input_batch, runner_output_list = self._prepare_batch_inputs(states, new_request_ids)
            if input_batch is None:
                return BatchRunnerOutput.from_list(runner_output_list)
            attn_metadata = {}

            kv_backend = getattr(self, "diffusion_kv_backend", None)
            paged_kv_runtime = kv_backend if getattr(kv_backend, "paged_attention_adapter", None) is not None else None
            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
                paged_kv_runtime=paged_kv_runtime,
                in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
            ):
                clear_pipeline_stage_durations(self.pipeline)
                noise_pred = self.pipeline.denoise_step(input_batch, states=states)
                denoise_stage_durations = consume_pipeline_stage_durations(self.pipeline)
                for state in states:
                    merge_stage_durations(
                        state,
                        denoise_stage_durations,
                    )

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
                                should_decode = req.chunk_denoise_completed or req.request_denoise_completed
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
                            self.state_cache.pop(req.request_id, None)
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
                                    result=DiffusionOutput.from_exception(per_req_exc),
                                )
                            )

                    if noise_pred is not None and offset != noise_pred.shape[0]:
                        raise ValueError(
                            f"Stepwise noise_pred consumed {offset} rows, "
                            f"but batched noise_pred has {noise_pred.shape[0]} rows."
                        )

                if is_primary and record_output_peak_memory:
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

                terminal_request_ids = [
                    runner_output.request_id for runner_output in runner_output_list if runner_output.finished
                ]
                self._update_states_after(states, input_batch, pipeline_interrupted)
                if (
                    getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
                    is DiffusionKVCacheMode.PAGED_SCHEDULER
                    and terminal_request_ids
                ):
                    self.remove_diffusion_kv_requests(terminal_request_ids)

                return BatchRunnerOutput.from_list(runner_output_list)

    def submit_interaction(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
    ) -> None:
        """Route a midway interaction to the matching active stepwise request feature."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.od_config.streaming_output:
            raise ValueError("submit_interaction requires streaming_output=True")
        if not self._supports_step_mode():
            raise ValueError("submit_interaction requires step execution support")

        event = interaction.get("event")
        if isinstance(event, dict) and "prompt" in event and "multi_modal_data" not in event:
            # Is a prompt update interaction.
            self._submit_prompt_update_interaction(request_id, interaction)
            return

        raise NotImplementedError(
            "Only text-only prompt update interactions with 'event.prompt' and optional "
            "'transition_chunks' are supported in this release"
        )

    def _submit_prompt_update_interaction(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
    ) -> None:
        """Queue a prompt-update interaction for an active stepwise request."""
        if not supports_prompt_update(self.pipeline):
            raise ValueError(f"prompt_update is not supported by pipeline {self.od_config.model_class_name!r}")

        state = self.state_cache.get(request_id)
        if state is None:
            raise ValueError(f"No active request state for prompt_update: {request_id!r}")

        event = cast(dict[str, Any], interaction.get("event"))
        prompt = event["prompt"]
        transition_chunks = interaction.get("transition_chunks")
        event_id = interaction.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty")
        pipeline = cast(SupportsPromptUpdate, self.pipeline)
        pipeline.prepare_prompt_update(state, prompt, event_id, transition_chunks)
