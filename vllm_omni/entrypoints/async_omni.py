# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import multiprocessing as mp
import os
import time
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pprint import pformat
from typing import Any

from omegaconf import OmegaConf
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
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.log_utils import (
    OrchestratorMetrics,
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


def _dummy_snapshot_download(model_id):
    return model_id


def omni_snapshot_download(model_id) -> str:
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)
    else:
        return _dummy_snapshot_download(model_id)


class AsyncOmni:
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
            - init_sleep_seconds: Number of seconds to sleep between starting
              each stage process during initialization
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
        model = args[0] if args else kwargs.get("model", "")
        assert model != "", "Null model id detected, please specify a model id."
        model = omni_snapshot_download(model)
        if args:
            args[0] = model
        elif kwargs.get("model", "") != "":
            kwargs["model"] = model

        # Stage management attributes
        self.stage_list: list[OmniStage] = []
        self._stage_in_queues: list[mp.Queue] = []
        self._stage_out_queues: list[mp.Queue] = []
        self._stages_ready: set[int] = set()
        self._ray_pg = None
        self._queue_cls = None
        self._ctx = None

        # Pause/resume control attributes
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        # Initialize stages - each stage will create appropriate instances based on stage_type
        # Stage workers will automatically create AsyncOmniLLM or AsyncOmniDiffusion instances
        # based on stage_type in YAML config (handled in omni_stage.py)
        logger.info(f"Initializing async stages for model: {model}")
        # Use kwargs-based initialization logic to avoid conflicts with old _initialize_stages signature
        self._initialize_stages_from_kwargs(model, kwargs)

    def _initialize_stages_from_kwargs(self, model: str, kwargs: dict[str, Any]) -> None:
        """Initialize stage list management.

        Each stage will create appropriate instances (AsyncOmniLLM or AsyncOmniDiffusion)
        based on stage_type in YAML config.
        """
        init_sleep_seconds = kwargs.get("init_sleep_seconds", 20)
        shm_threshold_bytes = kwargs.get("shm_threshold_bytes", 65536)
        init_timeout = kwargs.get("init_timeout", 300)
        worker_backend = kwargs.get("worker_backend", "multi_process")
        ray_address = kwargs.get("ray_address", None)
        batch_timeout = kwargs.get("batch_timeout", 10)
        stage_configs_path = kwargs.get("stage_configs_path", None)
        log_stats = kwargs.get("log_stats", False)

        # Load stage configs from YAML
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model)
            if not self.stage_configs:
                default_stage_cfg = [
                    {
                        "stage_id": 0,
                        "stage_type": "diffusion",
                        "runtime": {
                            "process": True,
                            "devices": "0",
                            "max_batch_size": 1,
                        },
                        "engine_args": {
                            "parallel_config": DiffusionParallelConfig(
                                pipeline_parallel_size=1,
                                data_parallel_size=1,
                                tensor_parallel_size=1,
                                sequence_parallel_size=1,
                                ulysses_degree=1,
                                ring_degree=1,
                                cfg_parallel_size=1,
                            ),
                            "vae_use_slicing": kwargs.get("vae_use_slicing", False),
                            "vae_use_tiling": kwargs.get("vae_use_tiling", False),
                            "cache_backend": kwargs.get("cache_backend", "none"),
                            "cache_config": kwargs.get("cache_config", None),
                        },
                        "final_output": True,
                        "final_output_type": "image",
                    }
                ]
                default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
                self.stage_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        # Initialize stats paths
        self._enable_stats: bool = bool(log_stats)

        self.worker_backend = worker_backend
        self.ray_address = ray_address
        self.batch_timeout = batch_timeout

        # Build OmniStage instances in parallel, preserving original order
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
        logger.debug("[AsyncOrchestrator] Loaded %d stages", len(self.stage_list))

        if self.worker_backend == "ray":
            self._queue_cls = get_ray_queue_class()
        else:
            self._ctx = mp.get_context("spawn")
            self._queue_cls = lambda: self._ctx.Queue(maxsize=0)

        self._init_sleep_seconds = max(0, int(init_sleep_seconds))
        self._shm_threshold_bytes = max(0, int(shm_threshold_bytes))
        self._start_stages(model)
        # Wait for all stages to report readiness before seeding
        self._wait_for_stages_ready(timeout=init_timeout)

    def _start_stages(self, model: str) -> None:
        """Start all stage processes."""
        if self.worker_backend == "ray":
            # Initialize Ray cluster
            self._ray_pg = create_placement_group(
                number_of_stages=len(self.stage_list), address=self.ray_address, strategy="PACK"
            )

        for stage_id, stage in enumerate(self.stage_list):
            in_q = self._queue_cls()
            out_q = self._queue_cls()
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)
            stage.attach_queues(in_q, out_q)

            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config,
                stage_id,
            )

            stage.init_stage_worker(
                model,
                shm_threshold_bytes=self._shm_threshold_bytes,
                ctx=self._ctx if self.worker_backend != "ray" else None,
                batch_timeout=self.batch_timeout,
                connectors_config=stage_connectors_config,
                worker_backend=self.worker_backend,
                ray_placement_group=self._ray_pg,
            )

            logger.debug("[AsyncOrchestrator] Stage-%s process started", stage_id)
            time.sleep(self._init_sleep_seconds)

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness."""
        deadline = time.time() + max(0, int(timeout))
        num_stages = len(self.stage_list)
        while len(self._stages_ready) < num_stages and time.time() < deadline:
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
                    logger.info("[AsyncOrchestrator] Stage-%s reported ready", stage_id)
                else:
                    # No user data should arrive before seeding; ignore other messages
                    pass
            if not progressed:
                time.sleep(0.01)
        if len(self._stages_ready) < num_stages:
            not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
            logger.warning(
                "[AsyncOrchestrator] Initialization timeout: only %s/%s stages are ready; not ready: %s",
                len(self._stages_ready),
                num_stages,
                not_ready,
            )
            # Provide actionable suggestions before shutdown
            try:
                suggestions = [
                    "Verify GPU/device assignment in config (runtime.devices) is correct.",
                    "Check GPU/host memory availability; reduce model or batch size if needed.",
                    "Check model weights path and network reachability (if loading remotely).",
                    "Increase initialization wait time (init_sleep_seconds or call-site timeout).",
                ]
                logger.error(
                    "[AsyncOrchestrator] Stage initialization failed, shutting down. Suggestions:\n- %s",
                    "\n- ".join(suggestions),
                )
            except Exception:
                # Best-effort logging of suggestions
                logger.error(
                    "[AsyncOrchestrator] Stage initialization failed and an error occurred while logging suggestions",
                )
        elif len(self._stages_ready) == num_stages:
            logger.info("[AsyncOrchestrator] All stages initialized successfully")
            # Initialize input_processor, io_processor, and model_config for API server compatibility
            # Find the first LLM stage (with vllm_config) to get vllm_config and tokenizer
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
                            "[AsyncOrchestrator] Initialized input_processor, "
                            "io_processor, and model_config from stage-%s",
                            stage.stage_id,
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            "[AsyncOrchestrator] Failed to initialize processors from stage-%s: %s",
                            stage.stage_id,
                            e,
                        )
            # If no LLM stage found, set processors to None
            if not hasattr(self, "input_processor") or self.input_processor is None:
                logger.warning(
                    "[AsyncOrchestrator] No LLM stage found, processors will not be available. "
                    "This may cause issues with OpenAIServingModels."
                )
                self.input_processor = None
                self.io_processor = None
                self.model_config = None

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

        logger.debug("[AsyncOrchestrator] generate() called")

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
        logger.debug("[AsyncOrchestrator] Seeding request into stage-0")
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
        logger.debug("[AsyncOrchestrator] Enqueued request %s to stage-0", request_id)

        logger.debug("[AsyncOrchestrator] Entering scheduling loop: stages=%d", num_stages)
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
                _m = asdict(result.get("metrics"))
                if _m is not None:
                    metrics.on_stage_metrics(stage_id, req_id, _m)
            except Exception as e:
                logger.exception(
                    "[AsyncOrchestrator] Failed to process metrics for stage %s, req %s: %s",
                    stage_id,
                    req_id,
                    e,
                )
            logger.debug(
                "[AsyncOrchestrator] Stage-%s completed request %s; forwarding or finalizing",
                stage_id,
                req_id,
            )
            stage.set_engine_outputs(engine_outputs)

            if getattr(stage, "final_output", False):
                logger.debug(
                    "[AsyncOrchestrator] Request %s finalized at stage-%s",
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
                        "[AsyncOrchestrator] Finalize request handling error for req %s at stage %s: %s",
                        req_id,
                        stage_id,
                        e,
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
                        f"[AsyncOrchestrator] Failed to send request {req_id} to stage-{next_stage_id} via connector. "
                        "Configure a connector for this edge or inspect connector logs for details."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(
                    "[AsyncOrchestrator] Forwarded request %s to stage-%s",
                    req_id,
                    next_stage_id,
                )
            else:
                logger.debug("[AsyncOrchestrator] Request %s fully completed", req_id)

        logger.debug("[AsyncOrchestrator] All requests completed")

        # Summarize and print stats
        try:
            summary = metrics.build_and_log_summary(final_stage_id_for_e2e)
            logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
        except Exception as e:
            logger.exception("[AsyncOrchestrator] Failed to build/log summary: %s", e)
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
