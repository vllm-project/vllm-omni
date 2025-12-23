import asyncio
import multiprocessing as mp
import os
import socket
import time
from argparse import Namespace
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

# External library imports (vLLM)
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike, init_tokenizer_from_config
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.func_utils import deprecate_kwargs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager

# Internal imports (our code)
from vllm_omni.config import OmniModelConfig
from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import (
    create_placement_group,
    get_ray_queue_class,
    try_close_ray,
)
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.log_utils import (
    OrchestratorMetrics,
    configure_orchestrator_logger,
    init_stats_paths,
    remove_old_logs,
)
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class AsyncOmni(EngineClient):
    """Async entry point for vLLM-Omni inference.

    This class provides an asynchronous interface for running multi-modal
    comprehension and generation models. It orchestrates multiple
    stages in a pipeline, where each stage runs in a separate process.
    Designed for use with async/await patterns and streaming generation.

    Args:
        model: Model name or path to load
        cli_args: Namespace object containing command-line arguments.
            Expected attributes include:
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are loaded from the model.
            - log_stats: Whether to enable statistics logging
            - log_file: Optional path prefix for log files. If provided, logs will
              be written to files with stage-specific suffixes.
            - init_sleep_seconds: Number of seconds to sleep between starting
              each stage process during initialization
            - shm_threshold_bytes: Threshold in bytes for using shared memory
              for IPC. Objects larger than this threshold will use shared memory.
            - batch_timeout: Timeout in seconds for batching requests within a stage
            - init_timeout: Timeout in seconds for waiting for all stages to initialize
        **kwargs: Additional keyword arguments passed to stage engines

    Example:
        >>> async_llm = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B", cli_args=args)
        >>> async for output in async_llm.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    def __init__(
        self,
        model: str,
        cli_args: Namespace,
        **kwargs: Any,
    ):
        self.worker_backend = getattr(cli_args, "worker_backend", "multi_process")
        self.ray_address = getattr(cli_args, "ray_address", "")
        self._ray_pg = None

        self.batch_timeout = cli_args.batch_timeout
        self._enable_stats: bool = bool(cli_args.log_stats)

        # Pause / resume state for async RL workflows.
        self._pause_cond = asyncio.Condition()
        self._paused = False

        base_engine_args = AsyncOmniEngineArgs.from_cli_args(cli_args).__dict__.copy()

        if cli_args.stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model, base_engine_args)
        else:
            self.config_path = cli_args.stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(cli_args.stage_configs_path, base_engine_args)

        shm_threshold_bytes = cli_args.shm_threshold_bytes
        self.output_handler: asyncio.Task | None = None
        self.request_states: dict[str, ClientRequestState] = {}  # request_id -> state

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=self.worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        self.stage_list: list[OmniStage] = []
        self.default_sampling_params_list: list[SamplingParams] = []
        # Optional file handler for orchestrator
        self._log_file = cli_args.log_file
        if self._log_file:
            remove_old_logs(self._log_file, len(self.stage_configs))
            configure_orchestrator_logger(logger, self._log_file)

        self._stats_file, self._overall_stats_file = init_stats_paths(self._enable_stats, self._log_file)
        self._initialize_stages(model, cli_args.init_sleep_seconds, cli_args.shm_threshold_bytes, cli_args.init_timeout)

        # Here is a duplicated init for the first stage to get the tokenizer,
        # input_processor, and io_processor
        # TODO: Refactor the code to avoid the duplicated init, we need to pass
        # the tokenizer, input_processor, and io_processor back here from the
        # first stage.
        vllm_config = self.stage_list[0].vllm_config
        self.tokenizer = init_tokenizer_from_config(model_config=vllm_config.model_config)
        self.input_processor = OmniInputProcessor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
        )
        self.model_config = vllm_config.model_config
        self.io_processor = get_io_processor(
            vllm_config,
            self.model_config.io_processor_plugin,
        )

    def _initialize_stages(
        self,
        model: str,
        init_sleep_seconds: int,
        shm_threshold_bytes: int,
        init_timeout: int,
    ) -> None:
        self.stage_list: list[OmniStage] = []

        # Build OmniStage instances in parallel, preserve original order
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, OmniStage]:
            idx, cfg = idx_cfg
            return idx, OmniStage(cfg)

        with ThreadPoolExecutor(max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))) as executor:
            futures = [executor.submit(_build_stage, (idx, cfg)) for idx, cfg in enumerate(self.stage_configs)]
            results: list[tuple[int, OmniStage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.default_sampling_params_list = [st.default_sampling_params for st in self.stage_list]
        self.output_modalities = [st.final_output_type for st in self.stage_list]
        logger.debug("[Orchestrator] Loaded %d stages", len(self.stage_list))

        if self.worker_backend == "ray":
            self._queue_cls = get_ray_queue_class()
        else:
            self._ctx = mp.get_context("spawn")
            self._queue_cls = lambda: self._ctx.Queue(maxsize=0)

        self._stage_in_queues: list[mp.Queue] = []
        self._stage_out_queues: list[mp.Queue] = []
        self._init_sleep_seconds = max(0, int(init_sleep_seconds))
        self._shm_threshold_bytes = max(0, int(shm_threshold_bytes))
        self._start_stages(model)
        # Wait for all stages to report readiness before seeding
        self._stages_ready: set[int] = set()
        self._wait_for_stages_ready(timeout=init_timeout)

    def _start_stages(self, model: str) -> None:
        if self.worker_backend == "ray":
            # Initialize Ray Cluster
            self._ray_pg = create_placement_group(
                number_of_stages=len(self.stage_list), address=self.ray_address, strategy="PACK"
            )

        for stage_id, stage in enumerate(self.stage_list):
            # Use unbounded queues to avoid deadlock when seeding many requests
            in_q = self._queue_cls()
            out_q = self._queue_cls()
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)

            # Attach queues and start Stage-owned worker process
            stage.attach_queues(in_q, out_q)

            # Build connectors config for this stage
            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config,
                stage_id,
            )

            stage.init_stage_worker(
                model,
                is_async=True,
                log_file=self._log_file,
                shm_threshold_bytes=self._shm_threshold_bytes,
                ctx=self._ctx if self.worker_backend != "ray" else None,
                batch_timeout=self.batch_timeout,
                connectors_config=stage_connectors_config,
                worker_backend=self.worker_backend,
                ray_placement_group=self._ray_pg,
            )

            logger.debug("[Orchestrator] Stage-%s process started", stage_id)
            time.sleep(self._init_sleep_seconds)

    def close(self) -> None:
        """Close all stage processes and clean up resources.

        Sends shutdown signals to all stage input queues and stops
        all stage worker processes. This method should be called
        when done using the AsyncOmni instance.
        """
        for q in self._stage_in_queues:
            try:
                q.put_nowait(None)
            except Exception as e:
                logger.warning(
                    "[Orchestrator] Failed to send shutdown signal to \
                        stage input queue: %s",
                    e,
                )
        for stage in self.stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning("[Orchestrator] Failed to stop stage worker: %s", e)

        if self.output_handler is not None:
            self.output_handler.cancel()
            self.output_handler = None

        try_close_ray(self._ray_pg)

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception as e:
            logger.debug("[Orchestrator] __del__ close() raised: %s", e, exc_info=True)

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC.

        Alias for close() method. Cleans up all stage processes
        and inter-process communication resources.
        """
        try:
            self.close()
        except Exception as e:
            logger.debug("[Orchestrator] __del__ close() raised: %s", e, exc_info=True)

    async def generate(
        self,
        prompt: PromptType,
        request_id: str,
        sampling_params_list: SamplingParams | Sequence[SamplingParams] | None = None,
        output_modalities: list[str] | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

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
            lora_request: Optional LoRA adapter request for this generation
            trace_headers: Optional tracing headers for observability
            priority: Request priority (higher values processed first)
            data_parallel_rank: Optional data parallel rank for distributed
                inference

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

        logger.debug("[Orchestrator] generate() called")

        # Start output handler on the first call to generate()
        self._run_output_handler()

        if sampling_params_list is None:
            sampling_params_list = self.default_sampling_params_list
        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(
                f"Expected {len(self.stage_list)} sampling params, \
                got {len(sampling_params_list)}"
            )

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
            self._stats_file,
            self._overall_stats_file,
            _wall_start_ts,
        )
        # Seed stage-0 queue with all requests
        logger.debug("[Orchestrator] Seeding request into stage-0")
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
        logger.debug("[Orchestrator] Enqueued request %s to stage-0", request_id)

        logger.debug("[Orchestrator] Entering scheduling loop: stages=%d", num_stages)
        for stage_id, stage in enumerate(self.stage_list[: final_stage_id_for_e2e + 1]):
            result = await req_state.queue.get()
            assert stage_id == req_state.stage_id

            req_id = result.get("request_id")
            if "error" in result:
                logger.error(
                    "Stage %s error on request %s: %s",
                    stage_id,
                    req_id,
                    result["error"],
                )
                raise RuntimeError(result)  # Request Finished due to error

            engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
            # Mark last output time for this stage whenever we receive outputs
            metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
            try:
                _m = result.get("metrics")
                if _m is not None:
                    metrics.on_stage_metrics(stage_id, req_id, _m)
            except Exception as e:
                logger.exception(
                    "[Orchestrator] Failed to process metrics for stage %s, \
                        req %s: %s",
                    stage_id,
                    req_id,
                    e,
                )
            logger.debug(
                "[Orchestrator] Stage-%s completed request %s; \
                    forwarding or finalizing",
                stage_id,
                req_id,
            )
            stage.set_engine_outputs(engine_outputs)

            if getattr(stage, "final_output", False):
                logger.debug(
                    "[Orchestrator] Request %s finalized at stage-%s",
                    req_id,
                    stage_id,
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
                        "[Orchestrator] Finalize request handling error for \
                            req %s at stage %s: %s",
                        req_id,
                        stage_id,
                        e,
                    )

                if isinstance(engine_outputs, list):
                    engine_outputs = engine_outputs[0]
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
                        f"[Orchestrator] Failed to send request {req_id} to stage-{next_stage_id} via connector. "
                        "Configure a connector for this edge or inspect connector logs for details."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(
                    "[Orchestrator] Forwarded request %s to stage-%s",
                    req_id,
                    next_stage_id,
                )
            else:
                logger.debug("[Orchestrator] Request %s fully completed", req_id)

        logger.debug("[Orchestrator] All requests completed")

        # Summarize and print stats
        try:
            summary = metrics.build_and_log_summary(final_stage_id_for_e2e)
            logger.info("[Summary] %s", summary)
        except Exception as e:
            logger.exception("[Orchestrator] Failed to build/log summary: %s", e)
        finally:
            self.request_states.pop(request_id, None)

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        num_stages = len(self.stage_list)
        while len(self._stages_ready) < num_stages:
            progressed = False
            for stage_id, stage in enumerate(self.stage_list):
                if stage_id in self._stages_ready:
                    continue
                result = stage.try_collect()
                if result is None:
                    continue
                progressed = True
                if result.get("type") == "stage_ready":
                    self._stages_ready.add(stage_id)
                    # Store vllm_config received from worker process
                    vllm_config = result.get("vllm_config")
                    if vllm_config is not None:
                        stage.set_vllm_config(vllm_config)
                    tokenizer = result.get("tokenizer")
                    if tokenizer is not None:
                        stage.set_tokenizer(tokenizer)
                    # input_preprocessor = result.get("input_preprocessor")
                    # if input_preprocessor is not None:
                    #     stage.set_input_preprocessor(input_preprocessor)
                    is_tracing_enabled = result.get("is_tracing_enabled")
                    if is_tracing_enabled is not None:
                        stage.set_is_tracing_enabled(is_tracing_enabled)
                    logger.debug("[Orchestrator] Stage-%s reported ready", stage_id)
                else:
                    # No user data should arrive before seeding; ignore other messages
                    pass
            if not progressed:
                time.sleep(0.01)
        if len(self._stages_ready) < num_stages:
            not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
            logger.warning(
                "[Orchestrator] Initialization timeout: only %s/%s stages are \
                    ready; not ready: %s",
                len(self._stages_ready),
                num_stages,
                not_ready,
            )
            # Provide actionable suggestions before shutdown
            try:
                suggestions = [
                    "Verify GPU/device assignment in config (runtime.devices) is \
                        correct.",
                    "Check GPU/host memory availability; reduce model or batch size if needed.",  # noqa: E501
                    "Check model weights path and network reachability (if loading remotely).",  # noqa: E501
                    "Increase initialization wait time (init_sleep_seconds or \
                        call-site timeout).",
                ]
                if getattr(self, "_log_file", None):
                    suggestions.append(
                        f"Inspect per-stage log files for details: \
                            {self._log_file}.stage<id>.log"
                    )
                logger.error(
                    "[Orchestrator] Stage initialization failed, shutting down. \
                        Suggestions:\n- %s",
                    "\n- ".join(suggestions),
                )
            except Exception:
                # Best-effort logging of suggestions
                logger.error(
                    "[Orchestrator] Stage initialization failed and an error \
                        occurred while logging suggestions",
                )

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
                                "[Orchestrator] Request may have been aborted; \
                                    dropping output for req %s at stage-%s  ",
                                req_id,
                                stage_id,
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


class AsyncOmniStageLLM(AsyncLLM):
    """Async single-stage LLM engine for use within a stage worker process.

    This class extends the base vLLM AsyncLLM class with omni-specific
    processors for handling multimodal inputs and outputs. It is used
    internally by AsyncOmniStage workers and should not be instantiated
    directly by users.

    Args:
        engine_args: AsyncOmniEngineArgs containing engine configuration
        vllm_config: Global vLLM configuration
        executor_class: Executor implementation class, e.g. MultiprocExecutor
        log_stats: Whether to log statistics
        usage_context: Usage context of the LLM (default: ENGINE_CONTEXT)
        mm_registry: Multi-modal registry for processing multimodal inputs
        use_cached_outputs: Whether to use cached outputs
        log_requests: Whether to log requests
        start_engine_loop: Whether to start the engine loop automatically
        stat_loggers: Customized stat loggers for the engine.
            If not provided, default stat loggers will be used.
            Note: Stat logger interface may change in V1.
        client_addresses: Optional dictionary mapping client names to addresses
        client_count: Total number of clients (default: 1)
        client_index: Index of this client (default: 0)
    """

    def __init__(
        self,
        engine_args: AsyncOmniEngineArgs,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        self.log_stats = log_stats or (stat_loggers is not None)
        if not log_stats and stat_loggers is not None:
            logger.info(
                "AsyncLLM created with log_stats=False and non-empty custom logger list; "
                "enabling logging without default stat loggers"
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            tokenizer = init_tokenizer_from_config(model_config=vllm_config.model_config)

        # InputProcessor (converts Inputs --> EngineCoreRequests).
        self.input_processor = OmniInputProcessor(
            vllm_config=vllm_config,
            tokenizer=tokenizer,
            mm_registry=mm_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = MultimodalOutputProcessor(
            tokenizer=tokenizer,
            log_stats=self.log_stats,
            engine_core_output_type=engine_args.engine_output_type,
        )

        if self.observability_config.otlp_traces_endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", self.observability_config.otlp_traces_endpoint)
            self.output_processor.tracer = tracer

        # Pause / resume state for async RL workflows.
        self._pause_cond = asyncio.Condition()
        self._paused = False

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
            )
            self.logger_manager.log_engine_initialized()

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        if envs.VLLM_TORCH_PROFILER_DIR:
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                envs.VLLM_TORCH_PROFILER_DIR,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    envs.VLLM_TORCH_PROFILER_DIR, worker_name=worker_name, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    @classmethod
    @deprecate_kwargs(
        "disable_log_requests",
        additional_message=("This argument will have no effect. Use `enable_log_requests` instead."),
    )
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        engine_args: AsyncOmniEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        disable_log_requests: bool = True,  # Deprecated, will be removed
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            engine_args=engine_args,
        )
