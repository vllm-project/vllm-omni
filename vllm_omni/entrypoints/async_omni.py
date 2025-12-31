# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, Iterable
from dataclasses import asdict
from pprint import pformat
from typing import Any

from vllm.config import VllmConfig
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.exceptions import EngineDeadError

# Internal imports (our code)
from vllm_omni.config import OmniModelConfig
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import try_close_ray
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.log_utils import (
    OrchestratorMetrics,
)
from vllm_omni.entrypoints.omni import OmniBase
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_close_cleanup_async(stage_list, stage_in_queues, ray_pg, output_handler):
    """Weak reference cleanup function for AsyncOmni instances."""
    if stage_list:
        for q in stage_in_queues:
            try:
                q.put_nowait(None)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to stage input queue: {e}")
        for stage in stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning(f"Failed to stop stage worker: {e}")
    try_close_ray(ray_pg)
    # Cancel output handler
    if output_handler is not None:
        output_handler.cancel()


class AsyncOmni(OmniBase):
    """Asynchronous unified entry point supporting multi-stage pipelines for LLM and Diffusion models.

    Similar to the Omni class, but provides an asynchronous interface supporting
    asynchronous LLM and Diffusion models.

    Args:
        *args: Variable length argument list.
            - args[0]: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
            - model: Model name or path to load (if not in args).
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are loaded from the model.
            - log_stats: Whether to enable statistics logging
              be written to files with stage-specific suffixes.
            - stage_init_timeout: Per-stage init watchdog (seconds). Measured from
              when the previous stage finished (possibly a prior Omni run with GPU
              reuse/overlap) to when the current stage starts to initialize.
            - shm_threshold_bytes: Threshold in bytes for using shared memory
              for IPC. Objects larger than this threshold will use shared memory.
            - worker_backend: Backend for worker processes. Default is "multi_process".
            - ray_address: Address of Ray cluster for Ray backend, if using Ray backend.
            - batch_timeout: Timeout in seconds for batching requests within a stage
            - init_timeout: Timeout in seconds for waiting for all stages to initialize
            - Additional keyword arguments passed to stage engines.

    Example:
        >>> async_llm = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_llm.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        # Pause/resume control attributes
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        super().__init__(*args, **kwargs)

        # Register weak reference cleanup (called on garbage collection)
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_close_cleanup_async,
            self.stage_list,
            self._stage_in_queues,
            self._ray_pg,
            self.output_handler,
        )

    def _create_default_diffusion_stage_cfg(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Create default diffusion stage configuration."""
        # TODO: here is different from the Omni class. We should merge the two in the future.
        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = self._normalize_cache_config(cache_backend, kwargs.get("cache_config", None))

        devices = "0"
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"
        else:
            ulysses_degree = kwargs.get("ulysses_degree") or 1
            ring_degree = kwargs.get("ring_degree") or 1
            sequence_parallel_size = kwargs.get("sequence_parallel_size")
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree
            num_devices = sequence_parallel_size
            for i in range(1, num_devices):
                devices += f",{i}"
            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=1,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                cfg_parallel_size=1,
            )
        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                    "max_batch_size": 1,
                },
                "engine_args": {
                    "parallel_config": parallel_config,
                    "vae_use_slicing": kwargs.get("vae_use_slicing", False),
                    "vae_use_tiling": kwargs.get("vae_use_tiling", False),
                    "cache_backend": cache_backend,
                    "cache_config": cache_config,
                },
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    def _process_stage_ready(self, stage: OmniStage, stage_id: int, result: dict[str, Any]) -> None:
        # Store vllm_config received from worker process (may be None for diffusion stages)
        vllm_config = result.get("vllm_config")
        if vllm_config is not None:
            stage.set_vllm_config(vllm_config)
        tokenizer = result.get("tokenizer")
        if tokenizer is not None:
            stage.set_tokenizer(tokenizer)
        is_tracing_enabled = result.get("is_tracing_enabled")
        if is_tracing_enabled is not None:
            stage.set_is_tracing_enabled(is_tracing_enabled)
        super()._process_stage_ready(stage, stage_id, result)

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness."""
        super()._wait_for_stages_ready(timeout)
        for stage in self.stage_list:
            if stage.vllm_config is not None and stage.tokenizer is not None:
                try:
                    vllm_config = stage.vllm_config
                    tokenizer = stage.tokenizer
                    # Initialize input_processor
                    self.input_processor = OmniInputProcessor(
                        vllm_config=vllm_config,
                        tokenizer=tokenizer,
                    )
                    # Initialize model_config
                    self.model_config = vllm_config.model_config
                    # Initialize io_processor
                    io_processor_plugin = self.model_config.io_processor_plugin
                    self.io_processor = get_io_processor(vllm_config, io_processor_plugin)

                    logger.info(
                        f"[{self._name}] Initialized input_processor, "
                        f"io_processor, and model_config from stage-{stage.stage_id}",
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"[{self._name}] Failed to initialize processors from stage-{stage.stage_id}: {e}",
                    )
        # If no LLM stage found, set processors to None
        if not hasattr(self, "input_processor") or self.input_processor is None:
            logger.warning(
                f"[{self._name}] No LLM stage found, processors will not be available. "
                "This may cause issues with OpenAIServingModels."
            )
            self.input_processor = None
            self.io_processor = None
            self.model_config = None

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC.

        Alias for close() method. Cleans up all stage processes
        and inter-process communication resources.
        """
        if hasattr(self, "_weak_finalizer"):
            self._weak_finalizer()

    async def generate(self, *args: Any, **kwargs: dict[str, Any]) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

        Coordinates multi-stage pipeline through YAML configuration.
        Each stage will use AsyncOmniLLM or AsyncOmniDiffusion based on stage_type.
        Processes the prompt through all stages in the pipeline and yields
        outputs as they become available. Each stage uses its corresponding
        sampling parameters from the sampling_params_list.

        Args:
            *args: Arguments for generation.
                - prompt: Prompt to process. Can be a text string, token IDs,
                    or multimodal prompt.
                - request_id: Unique identifier for this request
                - sampling_params_list: List of SamplingParams, one for each stage.
                    Must have the same length as the number of stages.
                    If None, uses default sampling params for each stage.
            **kwargs: Additional arguments for generation.
                - prompt: Prompt to process. Can be a text string, token IDs,
                    or multimodal prompt.
                - request_id: Unique identifier for this request
                - sampling_params_list: List of SamplingParams, one for each stage.
                    Must have the same length as the number of stages.
                    If None, uses default sampling params for each stage.
                - output_modalities: Optional list of output modalities.

        Yields:
            OmniRequestOutput objects as they are produced by each stage.
            Each output contains the stage_id, final_output_type, and
            the request_output from that stage.

        Raises:
            ValueError: If sampling_params_list has incorrect length.
        """
        # Wait until generation is resumed if the engine is paused.
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        logger.debug(f"[{self._name}] generate() called")

        # Start output handler on the first call to generate()
        self._run_output_handler()

        prompt = args[0] if args else kwargs.get("prompt")
        request_id = args[1] if len(args) > 1 else kwargs.get("request_id")
        sampling_params_list = args[2] if len(args) > 2 else kwargs.get("sampling_params_list")
        output_modalities = kwargs.get("output_modalities", None)
        # TODO: lora_request, trace_headers, priority are not supported yet

        if sampling_params_list is None:
            # For Omni LLM, the params are parsed via the yaml file. For the current version,
            # diffusion params can parsed via the command line.
            omni_params_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["prompt", "request_id", "output_modalities"]
            }

            per_stage_params: list[Any] = []
            for stage_id, stage in enumerate(self.stage_list):
                stage_type = getattr(stage, "stage_type", "llm")
                if stage_type == "diffusion":
                    default_dict = self.default_sampling_params_list[stage_id]
                    # Merge user-provided kwargs
                    merged = {**default_dict, **omni_params_kwargs}
                    # Diffusion only needs to keep diff params, will be used via OmniDiffusionRequest
                    per_stage_params.append(merged)
                else:
                    # LLM directly constructs SamplingParams, don't use the merged params
                    per_stage_params.append(self.default_sampling_params_list[stage_id])

            sampling_params_list = per_stage_params

        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")

        # Orchestrator keeps stage objects for input derivation
        num_stages = len(self.stage_list)
        # Track per-request start time for end-to-end timing
        _req_start_ts: dict[int, float] = {}
        _wall_start_ts: float = time.time()
        # _last_finish_ts: float = _wall_start_ts

        # Determine the final stage for E2E stats (highest stage_id with
        # final_output=True; fallback to last stage)
        final_stage_id_for_e2e = get_final_stage_id_for_e2e(output_modalities, self.output_modalities, self.stage_list)

        # Metrics/aggregation helper
        metrics = OrchestratorMetrics(
            num_stages,
            self._enable_stats,
            _wall_start_ts,
        )
        # Seed stage-0 queue with all requests
        logger.debug(f"[{self._name}] Seeding request into stage-0")
        req_state = ClientRequestState(request_id)
        self.request_states[request_id] = req_state

        # Mark first input time for stage-0
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

        sp0: SamplingParams = sampling_params_list[0]  # type: ignore[index]
        task = {
            "request_id": request_id,
            "engine_inputs": prompt,
            "sampling_params": sp0,
        }
        self.stage_list[0].submit(task)
        _req_start_ts[request_id] = time.time()
        logger.debug(f"[{self._name}] Enqueued request {request_id} to stage-0")

        logger.debug(f"[{self._name}] Entering scheduling loop: stages={num_stages}")
        for stage_id, stage in enumerate(self.stage_list[: final_stage_id_for_e2e + 1]):
            result = await req_state.queue.get()
            assert stage_id == req_state.stage_id

            req_id = result.get("request_id")
            if "error" in result:
                logger.error(
                    f"[{self._name}] Stage {stage_id} error on request {req_id}: {result['error']}",
                )
                raise RuntimeError(result)  # Request Finished due to error

            engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
            # Mark last output time for this stage whenever we receive outputs
            metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
            try:
                _m = asdict(result.get("metrics"))
                if _m is not None:
                    metrics.on_stage_metrics(stage_id, req_id, _m)
            except Exception as e:
                logger.exception(
                    f"[{self._name}] Failed to process metrics for stage {stage_id}, req {req_id}: {e}",
                )
            logger.debug(
                f"[{self._name}] Stage-{stage_id} completed request {req_id}; forwarding or finalizing",
            )
            stage.set_engine_outputs(engine_outputs)

            if getattr(stage, "final_output", False):
                logger.debug(
                    f"[{self._name}] Request {req_id} finalized at stage-{stage_id}",
                )

                # End-to-end timing and time-per-token for final output
                # (only once per request at the designated final stage)
                try:
                    rid_key = str(req_id)
                    if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done:
                        metrics.on_finalize_request(
                            stage_id,
                            req_id,
                            engine_outputs,
                            _req_start_ts.get(req_id, _wall_start_ts),
                        )
                except Exception as e:
                    logger.exception(
                        f"[{self._name}] Finalize request handling error for req {req_id} at stage {stage_id}: {e}",
                    )

                if isinstance(engine_outputs, list):
                    engine_outputs = engine_outputs[0]
                # Handle diffusion outputs that already contain images
                if stage.final_output_type == "image":
                    images = []
                    if isinstance(engine_outputs, OmniRequestOutput) and engine_outputs.images:
                        images = engine_outputs.images
                    elif hasattr(engine_outputs, "images") and engine_outputs.images:
                        images = engine_outputs.images
                    yield OmniRequestOutput(
                        stage_id=stage_id,
                        final_output_type=stage.final_output_type,
                        request_output=engine_outputs,
                        images=images,
                    )
                else:
                    yield OmniRequestOutput(
                        stage_id=stage_id,
                        final_output_type=stage.final_output_type,
                        request_output=engine_outputs,
                    )

            # Forward to next stage if there is one
            next_stage_id = stage_id + 1
            if next_stage_id <= final_stage_id_for_e2e:
                next_stage: OmniStage = self.stage_list[next_stage_id]
                next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                sp_next: SamplingParams = sampling_params_list[next_stage_id]

                # Check if we have a connector for this edge
                connector_key = (str(stage_id), str(next_stage_id))
                connector = self.connectors.get(connector_key)

                sent_via_connector = False
                if connector:
                    sent_via_connector = try_send_via_connector(
                        connector=connector,
                        stage_id=stage_id,
                        next_stage_id=next_stage_id,
                        req_id=req_id,
                        next_inputs=next_inputs,
                        sampling_params=sp_next,
                        original_prompt=prompt,
                        next_stage_queue_submit_fn=self.stage_list[next_stage_id].submit,
                        metrics=metrics,
                    )

                if not sent_via_connector:
                    # Fallback logic removed as we now enforce connector usage.
                    # If no connector is found or send fails, we log an error and raise,
                    # because continuing would cause the request to be silently dropped
                    # and the orchestrator to hang waiting for completion.
                    error_msg = (
                        f"[{self._name}] Failed to send request {req_id} to stage-{next_stage_id} via connector. "
                        "Configure a connector for this edge or inspect connector logs for details."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"[{self._name}] Forwarded request {req_id} to stage-{next_stage_id}")
            else:
                logger.debug(f"[{self._name}] Request {req_id} fully completed")

        logger.debug(f"[{self._name}] All requests completed")

        # Summarize and print stats
        try:
            summary = metrics.build_and_log_summary(final_stage_id_for_e2e)
            logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
        except Exception as e:
            logger.exception(f"[{self._name}] Failed to build/log summary: {e}")
        finally:
            self.request_states.pop(request_id, None)

    def _run_output_handler(self) -> None:
        if self.output_handler is not None:
            return

        stage_list = self.stage_list
        request_states = self.request_states

        async def output_handler():
            try:
                while True:
                    idle = True
                    for stage_id, stage in enumerate(stage_list):
                        result = stage.try_collect()
                        if result is None:
                            continue
                        idle = False
                        if result.get("type") == "stage_ready":
                            # Only happens when stage is initialized slower than expected,
                            # so we wait for a short time and try again
                            await asyncio.sleep(0.05)
                            continue
                        req_id = result.get("request_id")
                        req_state = request_states.get(req_id)
                        if req_state is None:
                            logger.debug(
                                f"[{self._name}] Request may have been aborted; \
                                dropping output for req {req_id} at stage-{stage_id}"
                            )
                            continue
                        await req_state.queue.put(result)
                        req_state.stage_id = stage_id
                    if idle:
                        await asyncio.sleep(0.001)  # Avoid CPU overload when idle
                    else:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.exception("AsyncOmni output_handler failed.")
                for req_state in request_states.values():
                    await req_state.queue.put({"request_id": req_id, "error": str(e)})
                self.output_handler = None  # Make possible for restart

        self.output_handler = asyncio.create_task(output_handler())

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return len(self._stage_in_queues) > 0

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return not self.is_running

    @property
    def _name(self) -> str:
        return "AsyncOrchestrator"

    @property
    def is_async(self) -> bool:
        return True

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    async def abort(self, request_id: str | Iterable[str]) -> None:
        pass

    async def get_vllm_config(self) -> VllmConfig:
        for stage in self.stage_list:
            if stage.is_comprehension:
                # Use the vllm_config received from worker process
                if stage.vllm_config is not None:
                    return stage.vllm_config
        return None

    async def get_model_config(self) -> OmniModelConfig:
        for stage in self.stage_list:
            if stage.is_comprehension:
                # Use the vllm_config received from worker process
                if stage.vllm_config is not None:
                    return stage.vllm_config.model_config
        return None

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return None

    async def get_tokenizer(self) -> TokenizerLike:
        for stage in self.stage_list:
            if stage.is_comprehension:
                return stage.tokenizer
        return None

    async def is_tracing_enabled(self) -> bool:
        for stage in self.stage_list:
            if stage.is_comprehension:
                return stage.is_tracing_enabled
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def reset_mm_cache(self) -> None:
        pass

    async def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
        pass

    async def sleep(self, level: int = 1) -> None:
        pass

    async def wake_up(self, tags: list[str] | None = None) -> None:
        pass

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return False

    async def encode(
        self,
        *args,
        **kwargs,
    ):
        """Generate outputs for a request from a pooling model."""
        raise NotImplementedError("encode() is not implemented for AsyncOmni")

    async def start_profile(self) -> None:
        raise NotImplementedError("start_profile() is not implemented for AsyncOmni")

    async def stop_profile(self) -> None:
        raise NotImplementedError("stop_profile() is not implemented for AsyncOmni")

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """
        Pause generation to allow model weight updates.

        New generation/encoding requests are blocked until resume.

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight
                requests to finish before pausing. When ``False`` (default),
                immediately aborts any in-flight requests.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
                Default is ``True`` (clear caches).
        """

        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        # Note: AsyncOmni uses a stage-based architecture without a central
        # output_processor. For now, we simply set the pause flag and let
        # new requests wait. In-flight requests will complete naturally.
        # TODO: Implement request abortion for stages if needed.

        # Clear cache if requested
        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""

        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()  # Wake up all waiting requests

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""

        async with self._pause_cond:
            return self._paused
