import multiprocessing as mp
import os
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import cloudpickle
from pydantic import ValidationError

# External library imports (vLLM)
from vllm.config import CompilationConfig, StructuredOutputsConfig, is_init_field
from vllm.entrypoints.llm import LLM
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)

# Internal imports (our code)
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import (
    create_placement_group,
    get_ray_queue_class,
    try_close_ray,
)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
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


class OmniLLM:
    """Main entry point for vLLM-Omni inference.

    This class provides a high-level interface for running multi-modal
    comprehension and generation models. It orchestrates multiple
    stages in a pipeline, where each stage runs in a separate process
    with copy-based IPC (Queues). Downstream stages start when upstream
    stages finish (window=-1), and each stage processes requests serially
    (max_inflight=1) but pipelines across stages.

    Args:
        model: Model name or path to load
        stage_configs_path: Optional path to YAML file containing stage
            configurations. If None, configurations are loaded from the model.
        log_stats: Whether to enable statistics logging
        log_file: Optional path prefix for log files. If provided, logs will
            be written to files with stage-specific suffixes.
        init_sleep_seconds: Number of seconds to sleep between starting
            each stage process during initialization
        shm_threshold_bytes: Threshold in bytes for using shared memory
            for IPC. Objects larger than this threshold will use shared memory.
        batch_timeout: Timeout in seconds for batching requests within a stage
        init_timeout: Timeout in seconds for waiting for all stages to initialize
        **kwargs: Additional keyword arguments passed to stage engines

    Example:
        >>> llm = OmniLLM(model="Qwen/Qwen2.5-Omni-7B")
        >>> outputs = llm.generate(
        ...     prompts="Hello",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... )
    """

    def __init__(
        self,
        model: str,
        stage_configs_path: str | None = None,
        log_stats: bool = False,
        log_file: str | None = None,
        init_sleep_seconds: int = 20,
        shm_threshold_bytes: int = 65536,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        **kwargs: Any,
    ):
        self.worker_backend = kwargs.get("worker_backend", "multi_process")
        self.ray_address = kwargs.get("ray_address", None)
        self._ray_pg = None
        self.batch_timeout = batch_timeout
        self._enable_stats: bool = bool(log_stats)

        # Do NOT call super().__init__ to avoid creating OmniStageLLM instances in parent.
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=self.worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        # Optional file handler for orchestrator
        self._log_file = log_file

        self._stats_file, self._overall_stats_file = init_stats_paths(self._enable_stats, self._log_file)
        self._initialize_stages(model, init_sleep_seconds, shm_threshold_bytes, init_timeout)
        if self._log_file:
            remove_old_logs(self._log_file, len(self.stage_list))
            configure_orchestrator_logger(logger, self._log_file)

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
        when done using the OmniLLM instance.
        """
        for q in self._stage_in_queues:
            try:
                q.put_nowait(None)
            except Exception as e:
                logger.warning(
                    "[Orchestrator] Failed to send shutdown signal to stage input queue: %s",
                    e,
                )
        for stage in self.stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning("[Orchestrator] Failed to stop stage worker: %s", e)

        try_close_ray(self._ray_pg)

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception as e:
            logger.debug("[Orchestrator] __del__ close() raised: %s", e, exc_info=True)

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params_list: SamplingParams | Sequence[SamplingParams] | None = None,
    ) -> list[OmniRequestOutput]:
        """Generate outputs for the given prompts.

        Processes prompts through all stages in the pipeline and returns
        the final outputs. Each stage uses its corresponding sampling
        parameters from the sampling_params_list.

        Args:
            prompts: Single prompt or sequence of prompts to process.
                Can be text strings, token IDs, or multimodal prompts.
            sampling_params_list: List of SamplingParams, one for each stage.
                Must have the same length as the number of stages.
                Required for pipelined generation.

        Returns:
            List of OmniRequestOutput objects, one for each input prompt.
            Each output contains the stage_id, final_output_type, and
            the request_output from the final stage.

        Raises:
            ValueError: If sampling_params_list is None or has incorrect length.
        """
        try:
            return self._run_generation(prompts, sampling_params_list)
        except Exception as e:
            logger.exception("[Orchestrator] Failed to run generation: %s", e)
            raise e
        finally:
            self.close()

    def _run_generation(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params_list: SamplingParams | Sequence[SamplingParams] | None = None,
    ) -> list[OmniRequestOutput]:
        logger.debug("[Orchestrator] generate() called")
        if sampling_params_list is None:
            raise ValueError("sampling_params_list is required for pipelined generation")
        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")

        # Normalize prompts to a list for per-request iteration
        if not isinstance(prompts, (list, tuple)):
            request_prompts: list[PromptType] = [prompts]
        else:
            request_prompts = list(prompts)

        final_outputs: list[OmniRequestOutput] = []

        # Orchestrator keeps stage objects for input derivation
        num_stages = len(self.stage_list)

        # Generate globally unique request IDs and map them to original prompts
        request_ids: list[str] = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))]
        request_id_to_prompt: dict[str, PromptType] = {rid: p for rid, p in zip(request_ids, request_prompts)}

        # Track per-request start time for end-to-end timing
        _req_start_ts: dict[str, float] = {}
        _wall_start_ts: float = time.time()

        # Determine the final stage for E2E stats (highest stage_id with final_output=True; fallback to last stage)
        final_stage_id_to_prompt = {}
        for rid, prompt in request_id_to_prompt.items():
            if isinstance(prompt, dict):
                prompt_modalities = prompt.get("modalities", None)
            else:
                prompt_modalities = None
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                prompt_modalities, self.output_modalities, self.stage_list
            )
            final_stage_id_to_prompt[rid] = final_stage_id_for_e2e

        # Metrics/aggregation helper
        metrics = OrchestratorMetrics(
            num_stages,
            self._enable_stats,
            self._stats_file,
            self._overall_stats_file,
            _wall_start_ts,
        )

        # Seed stage-0 queue with all requests
        logger.debug("[Orchestrator] Seeding %d requests into stage-0", len(request_prompts))
        # Mark first input time for stage-0
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

        for req_id, prompt in request_id_to_prompt.items():
            sp0: SamplingParams = sampling_params_list[0]  # type: ignore[index]
            task = {
                "request_id": req_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            }
            self.stage_list[0].submit(task)
            _req_start_ts[req_id] = time.time()
            logger.debug("[Orchestrator] Enqueued request %s to stage-0", req_id)

        # For each stage, forward results to next stage; collect finals at the end
        # We pipeline by continually polling output queues in stage order
        remaining_by_stage: list[int] = [len(request_prompts)] + [0] * (num_stages - 1)
        completed_requests = 0
        total_requests = len(request_prompts)

        logger.debug(
            "[Orchestrator] Entering scheduling loop: total_requests=%d, stages=%d",
            total_requests,
            num_stages,
        )
        while completed_requests < total_requests:
            made_progress = False
            for stage_id, stage in enumerate(self.stage_list):
                result = stage.try_collect()
                if result is None:
                    continue

                made_progress = True
                req_id = result.get("request_id")
                if "error" in result:
                    error_msg = result.get("error", "Unknown error")
                    error_tb = result.get("error_tb", "")
                    logger.error(
                        "Stage %s error on request %s: %s\n%s",
                        stage_id,
                        req_id,
                        error_msg,
                        error_tb,
                    )
                    continue

                if result.get("type") == "stage_ready":
                    # Only happens when stage is initialized slower than expected,
                    # so we wait for a short time and try again
                    time.sleep(0.05)
                    continue

                engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                # Mark last output time for this stage whenever we receive outputs
                metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
                try:
                    _m = result.get("metrics")
                    if _m is not None:
                        metrics.on_stage_metrics(stage_id, req_id, _m)
                except Exception as e:
                    logger.exception(
                        "[Orchestrator] Failed to process metrics for stage %s, req %s: %s",
                        stage_id,
                        req_id,
                        e,
                    )
                logger.debug(
                    "[Orchestrator] Stage-%s completed request %s; forwarding or finalizing",
                    stage_id,
                    req_id,
                )
                stage.set_engine_outputs(engine_outputs)

                if getattr(stage, "final_output", False):
                    final_outputs.append(
                        OmniRequestOutput(
                            stage_id=stage_id,
                            final_output_type=stage.final_output_type,  # type: ignore[attr-defined]
                            request_output=engine_outputs,
                        )
                    )
                    logger.debug(
                        "[Orchestrator] Request %s finalized at stage-%s",
                        req_id,
                        stage_id,
                    )

                    # End-to-end timing and time-per-token for final output
                    # (only once per request at the designated final stage)
                    try:
                        rid_key = str(req_id)
                        if stage_id == final_stage_id_to_prompt[req_id] and rid_key not in metrics.e2e_done:
                            metrics.on_finalize_request(
                                stage_id,
                                req_id,
                                engine_outputs,
                                _req_start_ts.get(req_id, _wall_start_ts),
                            )
                    except Exception as e:
                        logger.exception(
                            "[Orchestrator] Finalize request handling error for req %s at stage %s: %s",
                            req_id,
                            stage_id,
                            e,
                        )

                next_stage_id = stage_id + 1
                if next_stage_id <= final_stage_id_to_prompt[req_id]:
                    next_stage: OmniStage = self.stage_list[next_stage_id]
                    try:
                        next_inputs = next_stage.process_engine_inputs(self.stage_list, [request_id_to_prompt[req_id]])
                    except Exception as e:
                        logger.exception(
                            "[Orchestrator] Process engine inputs error for req %s at stage %s: %s",
                            req_id,
                            next_stage_id,
                            e,
                        )
                        continue
                    sp_next: SamplingParams = sampling_params_list[next_stage_id]  # type: ignore[index]

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
                            original_prompt=request_id_to_prompt[req_id],
                            next_stage_queue_submit_fn=self.stage_list[next_stage_id].submit,
                            metrics=metrics,
                        )

                    if not sent_via_connector:
                        raise RuntimeError(
                            f"[Orchestrator] Failed to send request {req_id} to stage-{next_stage_id} via connector. "
                            "Configure a connector for this edge or inspect connector logs for details."
                        )
                    logger.debug(
                        "[Orchestrator] Forwarded request %s to stage-%s",
                        req_id,
                        next_stage_id,
                    )
                    remaining_by_stage[next_stage_id] += 1
                else:
                    completed_requests += 1
                    logger.debug(
                        "[Orchestrator] Request %s fully completed (%d/%d)",
                        req_id,
                        completed_requests,
                        total_requests,
                    )

            if not made_progress:
                time.sleep(0.005)
        logger.debug("[Orchestrator] All requests completed")

        # Summarize and print stats
        try:
            summary = metrics.build_and_log_summary(final_stage_id_to_prompt)
            logger.info("[Summary] %s", summary)
        except Exception as e:
            logger.exception("[Orchestrator] Failed to build/log summary: %s", e)

        return final_outputs

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
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
                    logger.info("[Orchestrator] Stage-%s reported ready", stage_id)
                else:
                    # No user data should arrive before seeding; ignore other messages
                    pass
            if not progressed:
                time.sleep(0.01)
        if len(self._stages_ready) < num_stages:
            not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
            logger.warning(
                "[Orchestrator] Initialization timeout: only %s/%s stages are ready; not ready: %s",
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
                if getattr(self, "_log_file", None):
                    suggestions.append(f"Inspect per-stage log files for details: {self._log_file}.stage<id>.log")
                logger.error(
                    "[Orchestrator] Stage initialization failed, shutting down. Suggestions:\n- %s",
                    "\n- ".join(suggestions),
                )
            except Exception:
                # Best-effort logging of suggestions
                logger.error(
                    "[Orchestrator] Stage initialization failed and an error occurred while logging suggestions",
                )
        elif len(self._stages_ready) == num_stages:
            logger.info("[Orchestrator] All stages initialized successfully")


class OmniStageLLM(LLM):
    """Single-stage LLM engine for use within a stage worker process.

    This class extends the base vLLM LLM class with omni-specific
    processors for handling multimodal inputs and outputs. It is used
    internally by OmniStage workers and should not be instantiated directly
    by users.

    Args:
        model: Model name or path to load
        compilation_config: Optional compilation configuration. Can be an
            integer (compilation level), dict, or CompilationConfig instance.
        hf_overrides: Optional HuggingFace model configuration overrides
        structured_outputs_config: Optional structured outputs configuration.
            Can be a dict or StructuredOutputsConfig instance.
        **kwargs: Additional keyword arguments passed to the base LLM class
            and engine
    """

    def __init__(
        self,
        model: str,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        hf_overrides: dict[str, Any] | None = None,
        structured_outputs_config: dict[str, Any] | StructuredOutputsConfig | None = None,
        **kwargs: Any,
    ):
        """LLM constructor."""
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(level=compilation_config)
            elif isinstance(compilation_config, dict):
                compilation_config_instance = CompilationConfig(
                    **{k: v for k, v in compilation_config.items() if is_init_field(CompilationConfig, k)}
                )
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        if structured_outputs_config is not None:
            if isinstance(structured_outputs_config, dict):
                structured_outputs_instance = StructuredOutputsConfig(
                    **{k: v for k, v in structured_outputs_config.items() if is_init_field(StructuredOutputsConfig, k)}
                )
            else:
                structured_outputs_instance = structured_outputs_config
        else:
            structured_outputs_instance = StructuredOutputsConfig()

        engine_args = OmniEngineArgs(
            model=model,
            compilation_config=compilation_config_instance,
            structured_outputs_config=structured_outputs_instance,
            **kwargs,
        )

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.llm_engine.output_processor = MultimodalOutputProcessor(
            tokenizer=self.llm_engine.tokenizer,
            log_stats=self.llm_engine.log_stats,
            engine_core_output_type=engine_args.engine_output_type,
        )
        self.llm_engine.input_processor = OmniInputProcessor(
            vllm_config=self.llm_engine.vllm_config, tokenizer=self.llm_engine.tokenizer
        )
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()  # type: ignore

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks

        # Load the Input/Output processor plugin if any
        io_processor_plugin = self.llm_engine.model_config.io_processor_plugin
        self.io_processor = get_io_processor(self.llm_engine.vllm_config, io_processor_plugin)
        self.model_config = self.llm_engine.model_config
        self.input_processor = self.llm_engine.input_processor
