# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import time
import weakref
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config import OmniModelConfig
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length, try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import try_close_ray
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni import OmniBase
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK, OmniStageTaskType
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

# Internal imports (our code)
from vllm_omni.lora.request import LoRARequest
from vllm_omni.metrics import OrchestratorAggregator, StageRequestStats
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_close_cleanup_async(stage_list, stage_in_queues, ray_pg, output_handler):
    """Weak reference cleanup function for AsyncOmni instances."""
    if stage_list:
        for q in stage_in_queues:
            try:
                q.put_nowait(SHUTDOWN_TASK)
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
        model: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
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

    def __init__(self, model: str, **kwargs: dict[str, Any]) -> None:
        # Pause/resume control attributes
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        super().__init__(model, **kwargs)

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
            tensor_parallel_size = kwargs.get("tensor_parallel_size") or 1
            cfg_parallel_size = kwargs.get("cfg_parallel_size") or 1
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree
            num_devices = sequence_parallel_size * tensor_parallel_size * cfg_parallel_size
            for i in range(1, num_devices):
                devices += f",{i}"
            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=tensor_parallel_size,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                cfg_parallel_size=cfg_parallel_size,
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
                    "enable_cache_dit_summary": kwargs.get("enable_cache_dit_summary", False),
                    "enable_cpu_offload": kwargs.get("enable_cpu_offload", False),
                    "enable_layerwise_offload": kwargs.get("enable_layerwise_offload", False),
                    "layerwise_num_gpu_layers": kwargs.get("layerwise_num_gpu_layers", False),
                    "enforce_eager": kwargs.get("enforce_eager", False),
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
                    # Initialize input_processor
                    # OMNI: OmniInputProcessor creates tokenizer internally from vllm_config
                    self.input_processor = OmniInputProcessor(
                        vllm_config=vllm_config,
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

    async def generate(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        *,
        output_modalities: list[str] | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

        Coordinates multi-stage pipeline through YAML configuration.
        Each stage will use AsyncOmniLLM or AsyncOmniDiffusion based on stage_type.
        Processes the prompt through all stages in the pipeline and yields
        outputs as they become available. Each stage uses its corresponding
        sampling parameters from the sampling_params_list.

        Args:
            prompt: Prompt to process. Can be a text string, token IDs,
                or multimodal prompt.
            request_id: Unique identifier for this request
            sampling_params_list: List of SamplingParams, one for each stage.
                Must have the same length as the number of stages.
                If None, uses default sampling params for each stage.
            output_modalities: Optional list of output modalities.

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
        try:
            # Start output handler on the first call to generate()
            self._run_output_handler()

            # TODO: lora_request, trace_headers, priority are not supported yet
            if sampling_params_list is None:
                sampling_params_list = self.default_sampling_params_list

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
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                output_modalities, self.output_modalities, self.stage_list
            )

            # Metrics/aggregation helper
            metrics = OrchestratorAggregator(
                num_stages=num_stages,
                log_stats=self.log_stats,
                wall_start_ts=_wall_start_ts,
                final_stage_id_for_e2e=final_stage_id_for_e2e,
            )
            req_state = ClientRequestState(request_id)
            req_state.metrics = metrics
            self.request_states[request_id] = req_state
            sp0: SamplingParams = sampling_params_list[0]  # type: ignore[index]
            task = {
                "request_id": request_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            }
            self.stage_list[0].submit(task)
            metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()
            _req_start_ts[request_id] = time.time()
            logger.info(
                f"[{self._name}] Entering scheduling loop: stages={num_stages}, final_stage={final_stage_id_for_e2e}"
            )
            if self.async_chunk:
                stage_queues = {stage_id: asyncio.Queue() for stage_id in range(num_stages)}
                req_state.stage_queues = stage_queues
                async for output in self._process_async_results(
                    request_id,
                    prompt,
                    sampling_params_list,
                    req_state,
                    metrics,
                    final_stage_id_for_e2e,
                ):
                    yield output
            else:
                async for output in self._process_sequential_results(
                    request_id,
                    req_state,
                    metrics,
                    final_stage_id_for_e2e,
                    sampling_params_list,
                    prompt,
                ):
                    yield output

            logger.debug(f"[{self._name}] Request {request_id} finalized at stage-{final_stage_id_for_e2e}")
            try:
                # Finalize E2E metrics if not already done
                metrics.on_finalize_request(
                    final_stage_id_for_e2e,
                    request_id,
                    _req_start_ts.get(request_id, _wall_start_ts),
                )

                logger.debug(f"[{self._name}] All requests completed")
                # Summarize and print stats
                metrics.build_and_log_summary()
            except Exception as e:
                logger.exception(f"[{self._name}] Request {request_id} Failed to finalized/build/log summary: {e}")
            finally:
                self.request_states.pop(request_id, None)
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            logger.info("[AsyncOrchestrator] Request %s aborted.", request_id)
            raise

    async def _process_async_results(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_list: list[SamplingParams],
        req_state: ClientRequestState,
        metrics: OrchestratorAggregator,
        final_stage_id_for_e2e: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        all_stages_finished = {stage_id: False for stage_id in range(final_stage_id_for_e2e + 1)}
        submit_flag = True
        while not all(all_stages_finished.values()):
            for stage_id, stage in enumerate(self.stage_list[: final_stage_id_for_e2e + 1]):
                if all_stages_finished[stage_id]:
                    continue
                try:
                    result = req_state.stage_queues[stage_id].get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
                    continue
                engine_outputs, finished, output_to_yield = self._process_single_result(
                    result,
                    stage,
                    stage_id,
                    metrics,
                )
                if submit_flag and stage_id == 0:
                    submit_flag = False
                    prompt_token_ids = engine_outputs.prompt_token_ids
                    engine_input = copy.deepcopy(prompt)
                    engine_input["prompt_token_ids"] = [0] * compute_talker_prompt_ids_length(prompt_token_ids)
                    engine_input["multi_modal_data"] = engine_input["mm_processor_kwargs"] = None
                    for i in range(1, len(self.stage_list)):
                        task = {
                            "request_id": request_id,
                            "engine_inputs": engine_input,
                            "sampling_params": sampling_params_list[i],
                        }
                        self.stage_list[i].submit(task)
                        metrics.stage_first_ts[i] = time.time()
                all_stages_finished[stage_id] = finished

                if output_to_yield:
                    metrics.record_audio_generated_frames(
                        output_to_yield, engine_outputs.finished, stage_id, request_id
                    )
                    yield output_to_yield

    async def _process_sequential_results(
        self,
        request_id: str,
        req_state: ClientRequestState,
        metrics: OrchestratorAggregator,
        final_stage_id_for_e2e: int,
        sampling_params_list: list[SamplingParams],
        prompt: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        for stage_id, stage in enumerate(self.stage_list[: final_stage_id_for_e2e + 1]):
            finished = False
            while not finished:
                result = await req_state.queue.get()
                assert stage_id == req_state.stage_id
                engine_outputs, finished, output_to_yield = self._process_single_result(
                    result,
                    stage,
                    stage_id,
                    metrics,
                )
                if output_to_yield:
                    metrics.record_audio_generated_frames(
                        output_to_yield, engine_outputs.finished, stage_id, request_id
                    )
                    yield output_to_yield
            if not isinstance(engine_outputs, list):
                engine_outputs = [engine_outputs]
            stage.set_engine_outputs(engine_outputs)
            # Forward to next stage if there is one
            next_stage_id = stage_id + 1
            if next_stage_id <= final_stage_id_for_e2e:
                next_stage: OmniStage = self.stage_list[next_stage_id]
                # Derive inputs for the next stage, record postprocess time
                with metrics.stage_postprocess_timer(stage_id, request_id):
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
                        req_id=request_id,
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
                        f"[{self._name}] Failed to send request {request_id} to stage-{next_stage_id} via connector. "
                        "Configure a connector for this edge or inspect connector logs for details."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"[{self._name}] Forwarded request {request_id} to stage-{next_stage_id}")
            else:
                logger.debug(f"[{self._name}] Request {request_id} fully completed")

    def _process_single_result(
        self,
        result: dict[str, Any],
        stage: OmniStage,
        stage_id: int,
        metrics: OrchestratorAggregator,
    ) -> tuple[Any, bool, OmniRequestOutput | None]:
        """
        Process a single result dictionary from a stage.
        Returns:
            engine_outputs: The decoded outputs.
            finished: Whether the stage processing is finished for this request.
            output_to_yield: An OmniRequestOutput to yield, or None.
        """
        req_id = result.get("request_id")
        if "error" in result:
            logger.error(
                f"[{self._name}] Stage {stage_id} error on request {req_id}: {result['error']}",
            )
            raise RuntimeError(result)

        engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
        if isinstance(engine_outputs, list):
            engine_outputs = engine_outputs[0]

        finished = engine_outputs.finished

        # Mark last output time
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())

        try:
            _m: StageRequestStats = result.get("metrics")
            if _m is not None:
                # Accumulate generation time
                metrics.accumulated_gen_time_ms[req_id][stage_id] += _m.stage_gen_time_ms

                # For diffusion stages, we also accumulate diffusion time
                metrics.accumulate_diffusion_metrics(stage.stage_type, req_id, engine_outputs)

                if finished:
                    metrics.on_stage_metrics(stage_id, req_id, _m, stage.final_output_type)
        except Exception as e:
            logger.exception(
                f"[{self._name}] Failed to process metrics for stage {stage_id}, req {req_id}: {e}",
            )

        logger.debug(
            f"[{self._name}] Stage-{stage_id} completed request {req_id}; forwarding or finalizing",
        )

        output_to_yield = None

        if getattr(stage, "final_output", False):
            # Construct output to yield
            images = []
            if stage.final_output_type == "image":
                if isinstance(engine_outputs, OmniRequestOutput) and engine_outputs.images:
                    images = engine_outputs.images
                elif hasattr(engine_outputs, "images") and engine_outputs.images:
                    images = engine_outputs.images

            if stage.final_output_type == "image":
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage.final_output_type,
                    request_output=engine_outputs,
                    images=images,
                )
            else:
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage.final_output_type,
                    request_output=engine_outputs,
                )

        return engine_outputs, finished, output_to_yield

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
                        if hasattr(req_state, "stage_queues") and stage_id in req_state.stage_queues:
                            await req_state.stage_queues[stage_id].put(result)
                        else:
                            # Fallback to old behavior for compatibility
                            await req_state.queue.put(result)
                            req_state.stage_id = stage_id
                    if idle:
                        await asyncio.sleep(0.001)  # Avoid CPU overload when idle
                    else:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.exception("AsyncOmni output_handler failed.")
                for req_state in request_states.values():
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
                    # Send error to all stage queues
                    if hasattr(req_state, "stage_queues"):
                        for queue in req_state.stage_queues.values():
                            await queue.put(error_msg)
                    else:
                        await req_state.queue.put(error_msg)
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
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
        abort_task = {"type": OmniStageTaskType.ABORT, "request_id": request_id}
        for stage in self.stage_list:
            stage.submit(abort_task)
        return None

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

    @property
    def renderer(self):
        """Return the renderer from input_processor if available.

        OMNI: Required by upstream OpenAIServingModels.__init__ which
        accesses engine_client.renderer.
        """
        return self.input_processor.renderer

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

    async def start_profile(self, stages: list[int] | None = None) -> None:
        """Start profiling for specified stages.

        Async wrapper around the base implementation for API consistency.

        Args:
            stages: List of stage IDs to start profiling. If None, starts
                profiling for all stages that have profiling enabled.

        Example:
            >>> await async_omni.start_profile()
            >>> async for output in async_omni.generate(...):
            ...     pass
            >>> await async_omni.stop_profile()
        """
        super().start_profile(stages)

    async def stop_profile(self, stages: list[int] | None = None) -> None:
        """Stop profiling for specified stages.

        Async wrapper around the base implementation for API consistency.

        Args:
            stages: List of stage IDs to stop profiling. If None, stops
                profiling for all stages.

        Example:
            >>> await async_omni.start_profile()
            >>> async for output in async_omni.generate(...):
            ...     pass
            >>> await async_omni.stop_profile()
        """
        super().stop_profile(stages)

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
