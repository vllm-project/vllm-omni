"""
Stage manager for orchestrating multiple engines in vLLM-Omni.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.
"""

import asyncio
import fcntl
import importlib
import multiprocessing as mp
import os
import queue
import sys
import traceback
from dataclasses import fields
from typing import Any

from vllm.inputs import TextPrompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.distributed.omni_connectors import build_stage_connectors
from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
from vllm_omni.distributed.ray_utils.utils import kill_ray_actor, start_ray_actor
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.entrypoints.log_utils import count_tokens_from_outputs
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.entrypoints.omni_llm import OmniLLM
from vllm_omni.entrypoints.stage_utils import (
    SHUTDOWN_TASK,
    OmniStageTaskType,
    _to_dict,
    is_profiler_task,
    maybe_dump_to_shm,
    set_stage_devices,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.utils import detect_device_type

logger = init_logger(__name__)


def _build_od_config(engine_args: dict[str, Any], model: str) -> dict[str, Any]:
    """Build OmniDiffusionConfig kwargs from engine args."""
    od_config = engine_args.get("od_config", {})
    if not od_config:
        od_config = {"model": model}
        od_field_names = {f.name for f in fields(OmniDiffusionConfig)}
        for key, value in engine_args.items():
            if key in od_field_names:
                od_config[key] = value
    return od_config


def prepare_sampling_params(sampling_params: Any, stage_type: str) -> Any:
    """Prepare sampling parameters for the given stage type.

    Args:
        sampling_params: Raw sampling parameters (dict or SamplingParams)
        stage_type: Either "llm" or "diffusion"

    Returns:
        Processed sampling parameters ready for engine consumption
    """
    if stage_type == "diffusion":
        # For diffusion stages: extract kwargs, handling different input types
        if isinstance(sampling_params, dict):
            diffusion_kwargs = dict(sampling_params)
        else:
            diffusion_kwargs = getattr(sampling_params, "__dict__", {}) or {}

        # Remove 'prompt' and 'request_id' to avoid conflict with explicit arguments
        diffusion_kwargs.pop("prompt", None)
        diffusion_kwargs.pop("request_id", None)
        return diffusion_kwargs

    else:  # stage_type == "llm"
        # For LLM stages: ensure we have a SamplingParams object
        if isinstance(sampling_params, dict):
            return SamplingParams(**sampling_params)
        return sampling_params


class OmniStage:
    """Stage manager for orchestrating a single stage in the omni pipeline.

    Encapsulates per-stage process lifecycle and worker logic, including
    device setup, LLM initialization, batching, and shared-memory IPC.
    Preserves input processing utilities for cross-stage data wiring.

    Args:
        stage_config: Stage configuration object containing engine arguments,
            runtime settings, and stage-specific parameters
    """

    def __init__(self, stage_config: Any, stage_init_timeout: int = 300):
        logger.info(f"[OmniStage] stage_config: {stage_config}")
        self.stage_config = stage_config
        self.engine = None
        self.async_engine = None
        self.vllm_config = None
        self.tokenizer = None
        self.input_preprocessor = None
        self.is_tracing_enabled = False
        self.stage_id = stage_config.stage_id
        self.engine_args = stage_config.engine_args
        self.model_stage = stage_config.engine_args.model_stage
        self.requires_multimodal_data = getattr(stage_config.runtime, "requires_multimodal_data", False)
        self.engine_input_source = getattr(stage_config, "engine_input_source", [])
        self.engine_output_type = getattr(stage_config.engine_args, "engine_output_type", None)
        self.engine_outputs = None
        self.is_comprehension = getattr(stage_config, "is_comprehension", False)
        # Support for different stage types: "llm" (default) or "diffusion"
        self.stage_type = getattr(stage_config, "stage_type", "llm")
        if hasattr(stage_config, "custom_process_input_func"):
            # Import the module specified in the config (already a full module path)
            module_path, func_name = stage_config.custom_process_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_input_func = getattr(module, func_name)
        else:
            self.custom_process_input_func = None

        self.final_output = getattr(stage_config, "final_output", False)
        self.final_output_type = getattr(stage_config, "final_output_type", None)
        default_sampling_params = getattr(stage_config, "default_sampling_params", {})
        # For LLM stage, this can directly be a SamplingParams-compatible dict;
        # For diffusion stage, this only serves as default values for diffusion kwargs.
        self.default_sampling_params = _to_dict(default_sampling_params)
        # Runtime orchestration state (added)
        self._in_q: mp.Queue | None = None
        self._out_q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._shm_threshold_bytes: int = 65536
        self._stage_init_timeout: int = stage_init_timeout

    def set_engine(self, engine: LLMEngine) -> None:
        """Set the LLM engine for this stage.

        Args:
            engine: LLMEngine instance to use for this stage
        """
        self.engine = engine

    def set_async_engine(self, async_engine: AsyncLLM) -> None:
        """Set the async LLM engine for this stage.

        Args:
            async_engine: AsyncLLM instance to use for this stage
        """
        self.async_engine = async_engine

    def set_vllm_config(self, vllm_config: Any) -> None:
        """Set the vLLM configuration for this stage.

        Args:
            vllm_config: VllmConfig instance received from worker process
        """
        self.vllm_config = vllm_config

    def set_tokenizer(self, tokenizer: TokenizerLike) -> None:
        """Set the tokenizer for this stage.

        Args:
            tokenizer: Tokenizer instance received from worker process
        """
        self.tokenizer = tokenizer

    def set_input_preprocessor(self, input_preprocessor: InputPreprocessor) -> None:
        """Set the input preprocessor for this stage.

        Args:
            input_preprocessor: InputPreprocessor instance received from worker process
        """
        self.input_preprocessor = input_preprocessor

    def set_is_tracing_enabled(self, is_tracing_enabled: bool) -> None:
        """Set whether tracing is enabled for this stage.

        Args:
            is_tracing_enabled: Boolean indicating if tracing is enabled
        """
        self.is_tracing_enabled = is_tracing_enabled

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
        """Set the engine outputs for this stage.

        Args:
            engine_outputs: EngineCoreOutput from this stage's processing
        """
        self.engine_outputs = engine_outputs

    # ----------------- New Orchestration APIs -----------------
    def attach_queues(self, in_q: mp.Queue, out_q: mp.Queue) -> None:
        """Attach input and output queues for IPC communication.

        Args:
            in_q: Input queue for receiving tasks from orchestrator
            out_q: Output queue for sending results to orchestrator
        """
        self._in_q = in_q
        self._out_q = out_q

    def init_stage_worker(
        self,
        model: str,
        *,
        is_async: bool = False,
        shm_threshold_bytes: int = 65536,
        ctx: mp.context.BaseContext | None = None,
        batch_timeout: int = 10,
        connectors_config: dict | None = None,
        worker_backend: str = "multi_process",
        **kwargs: Any,
    ) -> None:
        """Initialize and start the stage worker process.

        Creates a worker process that runs the LLM engine for this stage.
        The worker handles batching, generation, and IPC communication.

        Args:
            model: Model name or path to load
            is_async: Whether to use async engine (default: False)
            shm_threshold_bytes: Threshold for using shared memory for IPC
            ctx: Optional multiprocessing context (default: spawn)
            batch_timeout: Timeout in seconds for batching requests
            connectors_config: Configuration for stage connectors
            worker_backend: Backend type ("multi_process" or "ray")
            **kwargs: Additional arguments (e.g. ray_placement_group)

        Raises:
            AssertionError: If queues are not attached before calling this method
        """
        assert self._in_q is not None and self._out_q is not None, "Queues must be attached before start_process"

        if worker_backend == "ray":
            ray_placement_group = kwargs.get("ray_placement_group", None)
            assert ray_placement_group is not None, "Ray placement group must be provided"
            self._shm_threshold_bytes = sys.maxsize
        else:
            self._shm_threshold_bytes = shm_threshold_bytes

        ctx = ctx or mp.get_context("spawn")
        # Prepare lightweight dict config for worker
        engine_args = _to_dict(self.engine_args)
        runtime_cfg = _to_dict(getattr(self.stage_config, "runtime", {}))
        stage_payload: dict[str, Any] = {
            "stage_id": self.stage_id,
            "engine_args": engine_args,
            "runtime": runtime_cfg,
            "shm_threshold_bytes": self._shm_threshold_bytes,
            "connectors_config": connectors_config or {},
            "stage_type": self.stage_type,
        }
        try:
            old_env = os.environ.get("VLLM_LOGGING_PREFIX")
            new_env = f"[Stage-{self.stage_id}] {'' if old_env is None else old_env}"
            os.environ["VLLM_LOGGING_PREFIX"] = new_env
            if worker_backend == "ray":
                if is_async:
                    self._ray_actor = start_ray_actor(
                        _stage_worker_async_entry,
                        ray_placement_group,
                        self.stage_id,
                        self,
                        model=model,
                        stage_payload=stage_payload,
                        batch_timeout=batch_timeout,
                        stage_init_timeout=self._stage_init_timeout,
                    )
                else:
                    self._ray_actor = start_ray_actor(
                        _stage_worker,
                        ray_placement_group,
                        self.stage_id,
                        model=model,
                        stage_payload=stage_payload,
                        in_q=self._in_q,
                        out_q=self._out_q,
                        batch_timeout=batch_timeout,
                        stage_init_timeout=self._stage_init_timeout,
                    )
            else:
                if is_async:
                    self._proc = ctx.Process(
                        target=_stage_worker_async_entry,
                        args=(
                            self,
                            model,
                            stage_payload,
                            batch_timeout,
                            self._stage_init_timeout,
                        ),
                    )
                else:
                    self._proc = ctx.Process(
                        target=_stage_worker,
                        args=(
                            model,
                            stage_payload,
                            self._in_q,
                            self._out_q,
                            batch_timeout,
                            self._stage_init_timeout,
                        ),
                    )
                self._proc.start()
        finally:
            if old_env is None:
                os.environ.pop("VLLM_LOGGING_PREFIX", None)
            else:
                os.environ["VLLM_LOGGING_PREFIX"] = old_env

    def stop_stage_worker(self) -> None:
        """Stop the stage worker process gracefully.

        Sends shutdown signal to the worker and waits for it to terminate.
        If graceful shutdown fails, forcefully terminates the process.
        Handles both multiprocessing Process and Ray Actor.
        """
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(SHUTDOWN_TASK)
            except Exception as e:
                logger.warning("Failed to send shutdown to in_q: %s", e)

        if hasattr(self, "_ray_actor") and self._ray_actor:
            kill_ray_actor(self._ray_actor)
            self._ray_actor = None
        elif self._proc is not None:
            try:
                self._proc.join(timeout=5)
            except Exception as e:
                logger.debug("join() failed: %s", e)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception as e:
                    logger.warning("terminate() failed: %s", e)

    def submit(self, payload: dict[str, Any]) -> None:
        """Submit a task to the stage worker.

        Args:
            payload: Dictionary containing task data (request_id, engine_inputs,
                sampling_params, etc.)
        """
        assert self._in_q is not None
        self._in_q.put(payload)

    def try_collect(self) -> dict[str, Any] | None:
        """Try to collect a result from the stage worker without blocking.

        Returns:
            Result dictionary if available, None otherwise. Result contains
            request_id, engine_outputs (or engine_outputs_shm), and metrics.
        """
        assert self._out_q is not None
        try:
            return self._out_q.get_nowait()
        except Exception:
            return None

    def process_engine_inputs(
        self, stage_list: list[Any], prompt: OmniTokensPrompt | TextPrompt = None
    ) -> list[OmniTokensPrompt | TextPrompt]:
        """Process engine inputs for this stage from upstream stage outputs.

        Derives inputs for this stage from outputs of upstream stages.
        Uses engine_input_source configuration to determine which upstream
        stage outputs to use. Supports custom processing functions.

        Args:
            stage_list: List of all stages in the pipeline
            prompt: Optional original prompt (for multimodal data preservation)

        Returns:
            List of processed engine inputs ready for this stage

        Raises:
            ValueError: If engine_input_source is empty or invalid
        """
        if self.custom_process_input_func is None:
            engine_inputs = []
            if len(self.engine_input_source) == 0:
                raise ValueError("engine_input_source is empty")
            source_stage_id = self.engine_input_source[0]
            source_outputs = stage_list[source_stage_id].engine_outputs
            if not isinstance(prompt, list):
                prompt = [prompt]
            multi_modal_data = {
                source_output.request_id: p.get("multi_modal_data", None)
                for source_output, p in zip(source_outputs, prompt)
            }

            for source_output in source_outputs:
                engine_input = OmniTokensPrompt(
                    prompt_token_ids=source_output.outputs[0].token_ids,
                    multi_modal_data=(
                        multi_modal_data[source_output.request_id]
                        if self.requires_multimodal_data and multi_modal_data
                        else None
                    ),
                )
                engine_inputs.append(engine_input)
            return engine_inputs

        else:
            engine_input_source = self.engine_input_source
            return self.custom_process_input_func(
                stage_list, engine_input_source, prompt, self.requires_multimodal_data
            )


def _stage_worker(
    model: str,
    stage_payload: dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    batch_timeout: int = 10,
    stage_init_timeout: int = 300,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""
    # Use local aliases to avoid conflicts with global imports in worker process
    logger.info(f"Starting stage worker with model: {model}")
    import os as _os
    import time as _time

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
    connectors_config = stage_payload.get("connectors_config", {})
    stage_type = stage_payload.get("stage_type", "llm")

    # Aggregates for running average
    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    # Monotonic batch id per stage process for orchestrator dedup on time aggregation
    _batch_seq = 0

    # Device mapping
    device_type = None
    try:
        device_type = detect_device_type()
        set_stage_devices(stage_id, runtime_cfg.get("devices"), device_type=device_type)
    except Exception as e:
        logger.warning("Device setup failed: %s", e)

    # Sequential initialization on the same device to avoid memory calculation errors
    # when multiple instances start simultaneously
    # For TP/PP/DP/SP, we need to lock ALL devices that will be used by this stage
    lock_files = []
    if device_type == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                # Get all parallel sizes from engine_args or parallel_config (defaults to 1)
                if "parallel_config" in engine_args:
                    parallel_config = engine_args["parallel_config"]
                    tensor_parallel_size = parallel_config.get("tensor_parallel_size", 1)
                    pipeline_parallel_size = parallel_config.get("pipeline_parallel_size", 1)
                    data_parallel_size = parallel_config.get("data_parallel_size", 1)
                    prefill_context_parallel_size = 1  # not used for diffusion
                    sequence_parallel_size = parallel_config.get("sequence_parallel_size", 1)
                else:
                    tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
                    pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
                    data_parallel_size = engine_args.get("data_parallel_size", 1)
                    prefill_context_parallel_size = engine_args.get("prefill_context_parallel_size", 1)
                    sequence_parallel_size = 1  # not use in omni model

                # Calculate total number of devices needed for this stage
                # For a single stage worker:
                # - TP: splits model across GPUs (always needed)
                # - PP: splits layers across pipelinestages, but each stage uses TP devices
                # - DP: replicates model, but each replica uses TP devices
                # - PCP: context parallelism, typically uses TP devices
                # - SP: sequence parallelism, typically uses TP devices
                # The number of devices per stage is determined by TP * PP * DP * PCP * SP size
                # (PP/DP/PCP are higher-level parallelism that don't add devices per stage)
                num_devices_per_stage = (
                    tensor_parallel_size
                    * pipeline_parallel_size
                    * data_parallel_size
                    * prefill_context_parallel_size
                    * sequence_parallel_size
                )

                # Get physical device IDs from CUDA_VISIBLE_DEVICES
                # After set_stage_devices, CUDA_VISIBLE_DEVICES is set to physical device(s)
                cuda_visible_devices = _os.environ.get("CUDA_VISIBLE_DEVICES")
                physical_devices = []

                if cuda_visible_devices:
                    try:
                        physical_devices = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
                    except (ValueError, IndexError):
                        pass

                if not physical_devices:
                    # Fallback: use logical device count if CUDA_VISIBLE_DEVICES not set
                    num_devices = torch.cuda.device_count()
                    physical_devices = list(range(num_devices))

                # Determine which devices will be used (min of devices per stage and available devices)
                num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
                devices_to_lock = physical_devices[:num_devices_to_lock]

                # Sort devices_to_lock to prevent deadlock (all processes acquire locks in same order)
                devices_to_lock = sorted(devices_to_lock)

                logger.debug(
                    "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d; will lock %d devices: %s",
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    data_parallel_size,
                    prefill_context_parallel_size,
                    sequence_parallel_size,
                    num_devices_to_lock,
                    devices_to_lock,
                )

                # Acquire exclusive locks for all devices using fcntl.flock
                # Locks are automatically released when process dies
                wait_start = _time.time()
                acquired_lock_fds = []  # Store file descriptors to keep locks alive

                for device_id in devices_to_lock:
                    lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                    lock_acquired = False

                    while not lock_acquired:
                        try:
                            # Open or create the lock file
                            lock_fd = _os.open(lock_file, _os.O_CREAT | _os.O_RDWR, 0o644)

                            # Try to acquire exclusive lock (non-blocking first)
                            try:
                                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                # Successfully acquired lock - write PID
                                _os.ftruncate(lock_fd, 0)  # Clear file
                                _os.write(lock_fd, f"{_os.getpid()}\n".encode())
                                _os.fsync(lock_fd)  # Ensure written to disk
                                lock_acquired = True
                                acquired_lock_fds.append(lock_fd)
                                logger.debug("Acquired exclusive lock for device %s", device_id)
                            except BlockingIOError:
                                # Lock is held by another process
                                _os.close(lock_fd)

                                # Check if we've been waiting too long
                                if _time.time() - wait_start > stage_init_timeout:
                                    logger.warning(
                                        "Timeout waiting for device %s initialization lock, proceeding anyway",
                                        device_id,
                                    )
                                    break

                                # Wait a bit before retrying
                                _time.sleep(0.1)
                        except OSError as e:
                            # Other error - log and continue without lock
                            logger.debug(
                                "Failed to acquire lock for device %s: %s, continuing anyway",
                                device_id,
                                e,
                            )
                            try:
                                _os.close(lock_fd)
                            except (OSError, NameError):
                                pass
                            break

                lock_files = acquired_lock_fds
        except Exception as e:
            logger.debug(
                "[Stage-%s] Failed to set up sequential initialization lock: %s",
                stage_id,
                e,
            )
    # Init engine based on stage_type
    logger.debug(
        "[Stage-%s] Initializing %s engine with args keys=%s",
        stage_id,
        stage_type,
        list(engine_args.keys()),
    )
    try:
        if stage_type == "diffusion":
            engine_args.pop("model_stage")
            stage_engine = OmniDiffusion(**engine_args)
        else:
            # Default to LLM engine
            stage_engine = OmniLLM(model=model, **engine_args)
    finally:
        # Release all locks by closing file descriptors
        # Locks are automatically released when file descriptors are closed
        # or when process dies
        for lock_fd in lock_files:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                _os.close(lock_fd)
                logger.debug("Released initialization lock (fd=%s)", lock_fd)
            except (OSError, ValueError):
                pass
    logger.debug("Engine initialized")

    # Check if stage engine supports profiling (via vLLM's built-in profiler)
    has_profiler = hasattr(stage_engine, "start_profile") and hasattr(stage_engine, "stop_profile")
    if has_profiler:
        logger.info(
            "[Stage-%s] vLLM profiler support detected (model_stage=%s)",
            stage_id,
            engine_args.get("model_stage", None),
        )

    def handle_profiler_task(task_type: OmniStageTaskType) -> None:
        """Handle profiler task."""
        if task_type == OmniStageTaskType.PROFILER_START:
            if has_profiler:
                try:
                    stage_engine.start_profile()
                    logger.info("[Stage-%s] Profiler started via vLLM engine", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start profiler: %s", stage_id, e)
        elif task_type == OmniStageTaskType.PROFILER_STOP:
            if has_profiler:
                try:
                    stage_engine.stop_profile()
                    logger.info("[Stage-%s] Profiler stopped via vLLM engine", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop profiler: %s", stage_id, e)

    # Initialize OmniConnectors if configured
    connectors = {}
    if connectors_config:
        built_connectors = build_stage_connectors(
            stage_id=stage_id,
            connectors_config=connectors_config,
        )
        if built_connectors is None:
            return
        connectors = built_connectors

    # Signal readiness to orchestrator
    try:
        out_q.put({"type": "stage_ready", "stage_id": stage_id})
    except Exception:
        pass

    max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
    logger.info(f"Max batch size: {max_batch_size}")

    # Batch processing loop
    while True:
        task = in_q.get()

        _recv_dequeue_ts = _time.time()
        task_type = task.get("type", OmniStageTaskType.GENERATE)
        if task_type == OmniStageTaskType.SHUTDOWN:
            logger.info("Received shutdown signal")
            break

        # Handle profiler control commands
        if is_profiler_task(task_type):
            handle_profiler_task(task_type)
            continue

        batch_tasks: list[dict[str, Any]] = [task]
        start_time = _time.time()
        if max_batch_size > 1:
            while len(batch_tasks) < max_batch_size:
                if not in_q.empty():
                    extra = in_q.get_nowait()
                    if extra == SHUTDOWN_TASK:
                        in_q.put(SHUTDOWN_TASK)
                        break
                    # Handle profiler commands that arrive during batching
                    extra_type = extra.get("type") if isinstance(extra, dict) else None
                    if is_profiler_task(extra_type):
                        handle_profiler_task(extra_type)
                        continue  # Don't add to batch_tasks
                    batch_tasks.append(extra)
                    end_time = _time.time()
                    duration = end_time - start_time
                    if duration > batch_timeout:
                        break
                    else:
                        continue
                else:
                    end_time = _time.time()
                    duration = end_time - start_time
                    _time.sleep(0.05)
                    if duration > batch_timeout:
                        break
                    else:
                        continue

        batch_request_ids: list[Any] = []
        batch_engine_inputs: list[Any] = []
        _rx_bytes_by_rid: dict[Any, int] = {}
        _rx_decode_ms_by_rid: dict[Any, float] = {}
        _in_flight_ms_by_rid: dict[Any, float] = {}
        for t in batch_tasks:
            rid = t["request_id"]
            try:
                sent_ts = float(t.get("sent_ts", None)) if isinstance(t, dict) else None
                if sent_ts is not None:
                    _in_flight_ms_by_rid[rid] = (_recv_dequeue_ts - sent_ts) * 1000.0
                else:
                    _in_flight_ms_by_rid[rid] = 0.0
            except Exception:
                _in_flight_ms_by_rid[rid] = 0.0

            # Resolve input data strictly via connectors if payload
            # is larger than shm_threshold_bytes or using other connectors
            ein, _rx_metrics = try_recv_via_connector(
                task=t,
                connectors=connectors,
                stage_id=stage_id,
            )

            if ein is None or _rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{stage_id}] Missing connector payload for request {rid}. "
                    "Ensure connectors are configured for all incoming edges."
                )

            _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
            _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

            batch_request_ids.append(rid)
            if isinstance(ein, list):
                batch_engine_inputs.extend(ein)
            elif isinstance(ein, dict):
                batch_engine_inputs.append(ein)
            elif isinstance(ein, str):
                # For diffusion stage-0, ein might be a string prompt directly
                batch_engine_inputs.append(ein)
            else:
                # For other types (e.g., OmniTokensPrompt, TextPrompt), append as-is
                batch_engine_inputs.append(ein)
        sampling_params = batch_tasks[0]["sampling_params"]
        logger.debug(
            "Received batch size=%d, request_ids=%s",
            len(batch_tasks),
            batch_request_ids,
        )
        try:
            _batch_seq += 1
            gen_outputs: list[Any] = []
            _gen_t0 = _time.time()
            if stage_type == "diffusion":
                # For diffusion, batch_engine_inputs should be prompts (strings)
                # Convert to list of strings if needed
                prompts = []
                for ein in batch_engine_inputs:
                    if isinstance(ein, str):
                        prompts.append(ein)
                    elif isinstance(ein, dict) and "prompt" in ein:
                        prompts.append(ein["prompt"])
                    elif hasattr(ein, "prompt"):
                        prompts.append(ein.prompt)
                    else:
                        prompts.append(str(ein))
                # Prepare diffusion kwargs from sampling parameters
                diffusion_kwargs = prepare_sampling_params(sampling_params, "diffusion")
                # Diffusion generate returns results directly, not an iterator
                diffusion_results = stage_engine.generate(prompts, **diffusion_kwargs)
                # Convert to list format compatible with LLM outputs
                # Ensure each result has a request_id for proper mapping
                if isinstance(diffusion_results, list):
                    gen_outputs = diffusion_results
                    # Assign request_ids if not present
                    for idx, result in enumerate(gen_outputs):
                        if not hasattr(result, "request_id") or result.request_id is None:
                            if idx < len(batch_request_ids):
                                if hasattr(result, "request_id"):
                                    result.request_id = batch_request_ids[idx]
                                else:
                                    # Create a wrapper object if result doesn't support request_id
                                    from types import SimpleNamespace

                                    wrapped = SimpleNamespace()
                                    wrapped.request_id = batch_request_ids[idx]
                                    wrapped.output = result
                                    gen_outputs[idx] = wrapped
                else:
                    gen_outputs = [diffusion_results]
                    # Assign request_id to single result
                    if len(batch_request_ids) > 0:
                        if hasattr(gen_outputs[0], "request_id"):
                            gen_outputs[0].request_id = batch_request_ids[0]
                        else:
                            from types import SimpleNamespace

                            wrapped = SimpleNamespace()
                            wrapped.request_id = batch_request_ids[0]
                            wrapped.output = gen_outputs[0]
                            gen_outputs[0] = wrapped
            else:
                # LLM engine: use vLLM native SamplingParams
                llm_sampling_params = prepare_sampling_params(sampling_params, "llm")
                for ro in stage_engine.generate(batch_engine_inputs, llm_sampling_params, use_tqdm=False):
                    gen_outputs.append(ro)
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
            logger.debug(f"Generate done: batch={len(batch_tasks)}, req_ids={batch_request_ids}, gen_ms={_gen_ms:.1f}")

            # Group outputs per request id with fallback
            req_to_outputs: dict[Any, list[Any]] = {rid: [] for rid in batch_request_ids}
            unmapped: list[Any] = []
            for ro in gen_outputs:
                rid = getattr(ro, "request_id", None)
                if rid in req_to_outputs:
                    req_to_outputs[rid].append(ro)
                else:
                    unmapped.append(ro)
            if unmapped:
                idx = 0
                for ro in unmapped:
                    target_rid = batch_request_ids[idx % len(batch_request_ids)]
                    ro.request_id = target_rid
                    req_to_outputs[target_rid].append(ro)
                    idx += 1

            _agg_total_gen_time_ms += _gen_ms

            # Emit per-request results
            for i, rid in enumerate(batch_request_ids):
                r_outputs = req_to_outputs.get(rid, [])
                _metrics = make_request_stats(
                    r_outputs,
                    _gen_ms,
                    int(_batch_seq),
                    int(len(batch_request_ids)),
                    float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                    int(_rx_bytes_by_rid.get(rid, 0)),
                    float(_in_flight_ms_by_rid.get(rid, 0.0)),
                )
                _agg_total_tokens += _metrics.num_tokens_out
                if i == len(batch_request_ids) - 1:
                    _metrics.stage_stats = make_stage_stats(_agg_total_tokens, _agg_total_gen_time_ms)
                else:
                    _metrics.stage_stats = None
                try:
                    use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                    if use_shm:
                        out_q.put(
                            {
                                "request_id": rid,
                                "stage_id": stage_id,
                                "engine_outputs_shm": payload,
                                "metrics": _metrics,
                            }
                        )
                    else:
                        out_q.put(
                            {
                                "request_id": rid,
                                "stage_id": stage_id,
                                "engine_outputs": payload,
                                "metrics": _metrics,
                            }
                        )
                except Exception:
                    out_q.put(
                        {
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs": r_outputs,
                            "metrics": _metrics,
                        }
                    )
                logger.debug(
                    "Enqueued result for request %s to downstream",
                    rid,
                )
        except Exception as e:
            logger.exception("Failed on batch %s: %s", batch_request_ids, e)
            _tb = traceback.format_exc()
            for rid in batch_request_ids:
                out_q.put(
                    {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "error": str(e),
                        "error_tb": _tb,
                    }
                )


def _stage_worker_async_entry(
    omni_stage: OmniStage,
    model: str,
    stage_payload: dict[str, Any],
    batch_timeout: int = 10,
    stage_init_timeout: int = 300,
) -> None:
    asyncio.run(_stage_worker_async(omni_stage, model, stage_payload, batch_timeout, stage_init_timeout))


async def _stage_worker_async(
    omni_stage: OmniStage,
    model: str,
    stage_payload: dict[str, Any],
    batch_timeout: int = 10,
    stage_init_timeout: int = 300,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""
    # Use local aliases to avoid conflicts with global imports in worker process
    import os as _os
    import time as _time

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
    connectors_config = stage_payload.get("connectors_config", {})
    stage_type = stage_payload.get("stage_type", "llm")

    in_q = omni_stage._in_q
    out_q = omni_stage._out_q

    # Aggregates for running average
    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    # Monotonic batch id per stage process for orchestrator dedup on time
    # aggregation
    _batch_seq = 0

    # Device mapping
    device_type = None
    try:
        from vllm_omni.utils import detect_device_type

        device_type = detect_device_type()
        set_stage_devices(stage_id, runtime_cfg.get("devices"), device_type=device_type)
    except Exception as e:
        logger.warning("Device setup failed: %s", e)

    # Initialize OmniConnectors if configured to match sync worker behavior
    connectors: dict[Any, Any] = {}
    if connectors_config:
        built_connectors = build_stage_connectors(
            stage_id=stage_id,
            connectors_config=connectors_config,
        )
        if built_connectors is None:
            return
        connectors = built_connectors

    # Sequential initialization on the same device to avoid memory calculation errors
    # when multiple instances start simultaneously
    # For TP, we need to lock ALL devices that will be used by this stage
    lock_files = []
    if device_type == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                # Get all parallel sizes from engine_args (defaults to 1)
                tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
                pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
                data_parallel_size = engine_args.get("data_parallel_size", 1)
                prefill_context_parallel_size = engine_args.get("prefill_context_parallel_size", 1)

                # Calculate total number of devices needed for this stage
                # For a single stage worker in omni:
                # - TP: splits model across GPUs (always needed)
                # - PP: splits layers across stages, but each stage uses TP devices
                # - DP: replicates model, but each replica uses TP devices
                # - PCP: context parallelism, typically uses TP devices
                # The number of devices per stage is determined by TP * PP * DP * PCP size
                # (PP/DP/PCP are higher-level parallelism that don't add devices per stage)
                num_devices_per_stage = (
                    tensor_parallel_size * pipeline_parallel_size * data_parallel_size * prefill_context_parallel_size
                )

                # Get physical device IDs from CUDA_VISIBLE_DEVICES
                # After set_stage_devices, CUDA_VISIBLE_DEVICES is set to physical device(s)
                cuda_visible_devices = _os.environ.get("CUDA_VISIBLE_DEVICES")
                physical_devices = []

                if cuda_visible_devices:
                    try:
                        physical_devices = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
                    except (ValueError, IndexError):
                        pass

                if not physical_devices:
                    # Fallback: use logical device count if CUDA_VISIBLE_DEVICES not set
                    num_devices = torch.cuda.device_count()
                    physical_devices = list(range(num_devices))

                # Determine which devices will be used (min of devices per stage and available devices)
                num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
                devices_to_lock = physical_devices[:num_devices_to_lock]

                # Sort devices_to_lock to prevent deadlock (all processes acquire locks in same order)
                devices_to_lock = sorted(devices_to_lock)

                logger.debug(
                    "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d; will lock %d devices: %s",
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    data_parallel_size,
                    prefill_context_parallel_size,
                    num_devices_to_lock,
                    devices_to_lock,
                )

                # Acquire exclusive locks for all devices using fcntl.flock
                # Locks are automatically released when process dies
                wait_start = _time.time()
                acquired_lock_fds = []  # Store file descriptors to keep locks alive

                for device_id in devices_to_lock:
                    lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                    lock_acquired = False

                    while not lock_acquired:
                        try:
                            # Open or create the lock file
                            lock_fd = _os.open(lock_file, _os.O_CREAT | _os.O_RDWR, 0o644)

                            # Try to acquire exclusive lock (non-blocking first)
                            try:
                                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                # Successfully acquired lock - write PID
                                _os.ftruncate(lock_fd, 0)  # Clear file
                                _os.write(lock_fd, f"{_os.getpid()}\n".encode())
                                _os.fsync(lock_fd)  # Ensure written to disk
                                lock_acquired = True
                                acquired_lock_fds.append(lock_fd)
                                logger.debug("Acquired exclusive lock for device %s", device_id)
                            except BlockingIOError:
                                # Lock is held by another process
                                _os.close(lock_fd)

                                # Check if we've been waiting too long
                                if _time.time() - wait_start > stage_init_timeout:
                                    logger.warning(
                                        "Timeout waiting for device %s initialization lock, "
                                        "proceeding anyway with timeout %s",
                                        device_id,
                                        stage_init_timeout,
                                    )
                                    break

                                # Wait a bit before retrying
                                _time.sleep(0.1)
                        except OSError as e:
                            # Other error - log and continue without lock
                            logger.debug(
                                "Failed to acquire lock for device %s: %s, continuing anyway",
                                device_id,
                                e,
                            )
                            try:
                                _os.close(lock_fd)
                            except (OSError, NameError):
                                pass
                            break

                lock_files = acquired_lock_fds
        except Exception as e:
            logger.debug("Failed to set up sequential initialization lock: %s", e)

    # Init engine based on stage_type
    logger.debug(
        "[Stage-%s] Initializing %s engine with args keys=%s",
        stage_id,
        stage_type,
        list(engine_args.keys()),
    )
    try:
        if stage_type == "diffusion":
            # For diffusion, we need to extract diffusion-specific config
            od_config = _build_od_config(engine_args, model)
            logger.debug(f"[Stage-%s] Initializing diffusion engine with config: {od_config}", stage_id)
            stage_engine = AsyncOmniDiffusion(
                model=model,
                od_config=od_config,
                **{k: v for k, v in engine_args.items() if k not in {"od_config", "model"}},
            )
            vllm_config = None  # Diffusion doesn't use vllm_config
        else:
            omni_engine_args = AsyncOmniEngineArgs(model=model, **engine_args)
            usage_context = UsageContext.OPENAI_API_SERVER
            vllm_config = omni_engine_args.create_engine_config(usage_context=usage_context)
            stage_engine = AsyncOmniLLM.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=usage_context,
                engine_args=omni_engine_args,
            )
    finally:
        # Release all locks by closing file descriptors
        # Locks are automatically released when file descriptors are closed
        # or when process dies
        for lock_fd in lock_files:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                _os.close(lock_fd)
                logger.debug("Released initialization lock (fd=%s)", lock_fd)
            except (OSError, ValueError):
                pass
    omni_stage.set_async_engine(stage_engine)
    if hasattr(omni_stage.async_engine, "log_stats") and omni_stage.async_engine.log_stats:

        async def _force_log():
            try:
                while True:
                    await asyncio.sleep(10.0)
                    await omni_stage.async_engine.do_log_stats()
            except asyncio.CancelledError:
                pass

        log_stats_task = asyncio.create_task(_force_log())
    else:
        log_stats_task = None

    # Don't keep the dummy data in memory (only for LLM engines)
    if stage_type != "diffusion":
        await stage_engine.reset_mm_cache()
    logger.debug("[Stage-%s] Engine initialized", stage_id)

    # Check if stage engine supports profiling (via vLLM's built-in profiler)
    has_profiler = hasattr(stage_engine, "start_profile") and hasattr(stage_engine, "stop_profile")
    if has_profiler:
        logger.info(
            "[Stage-%s] vLLM profiler support detected for async stage (model_stage=%s)",
            stage_id,
            engine_args.get("model_stage", None),
        )

    async def handle_profiler_task_async(task_type: OmniStageTaskType) -> None:
        """Handle profiler task asynchronously."""
        if task_type == OmniStageTaskType.PROFILER_START:
            if has_profiler:
                try:
                    await stage_engine.start_profile()
                    logger.info("[Stage-%s] Profiler started via vLLM engine", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start profiler: %s", stage_id, e)
        elif task_type == OmniStageTaskType.PROFILER_STOP:
            if has_profiler:
                try:
                    await stage_engine.stop_profile()
                    logger.info("[Stage-%s] Profiler stopped via vLLM engine", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop profiler: %s", stage_id, e)

    # Signal readiness to orchestrator and send vllm_config back to main process
    try:
        # Send vllm_config back to main process so it can be accessed via
        # get_vllm_config(). This is needed because async_engine is only available
        # in the worker process

        # input_preprocessor = await stage_engine.get_input_preprocessor()
        stage_ready_payload = {
            "type": "stage_ready",
            "stage_id": stage_id,
            "vllm_config": vllm_config,
            "tokenizer": getattr(stage_engine, "tokenizer", None),
        }
        # Only add is_tracing_enabled for LLM engines
        if stage_type != "diffusion":
            stage_ready_payload["is_tracing_enabled"] = await stage_engine.is_tracing_enabled()
        out_q.put(stage_ready_payload)
    except Exception as e:
        logger.warning("Failed to send stage ready signal: %s", e)
    generation_out_q = asyncio.Queue()

    # Batch processing loop
    _rx_bytes_by_rid: dict[Any, int] = {}
    _rx_decode_ms_by_rid: dict[Any, float] = {}
    _in_flight_ms_by_rid: dict[Any, float] = {}

    async def generation_single_request(task: dict[str, Any]):
        _recv_dequeue_ts = _time.time()
        rid = task["request_id"]
        try:
            sent_ts = float(task.get("sent_ts", None)) if isinstance(task, dict) else None
            if sent_ts is not None:
                _in_flight_ms_by_rid[rid] = (_recv_dequeue_ts - sent_ts) * 1000.0
            else:
                _in_flight_ms_by_rid[rid] = 0.0
        except Exception:
            _in_flight_ms_by_rid[rid] = 0.0
        try:
            ein, _rx_metrics = try_recv_via_connector(
                task=task,
                connectors=connectors,
                stage_id=stage_id,
            )
            if ein is None or _rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{stage_id}] Missing connector payload for request {rid}. "
                    "Ensure connectors are configured for all incoming edges."
                )
            _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
            _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

            sampling_params = task["sampling_params"]
            logger.debug("Received batch size=1, request_ids=%s", rid)
            _gen_t0 = _time.time()
            if isinstance(ein, list):
                ein = ein[0]

            if stage_type == "diffusion":
                # For diffusion, ein should be prompts (strings)
                # Convert to string if needed
                if isinstance(ein, str):
                    prompt = ein
                elif isinstance(ein, dict) and "prompt" in ein:
                    prompt = ein["prompt"]
                elif hasattr(ein, "prompt"):
                    prompt = ein.prompt
                else:
                    prompt = str(ein)

                # Prepare diffusion kwargs from sampling parameters
                diffusion_kwargs = prepare_sampling_params(sampling_params, "diffusion")
                # AsyncOmniDiffusion.generate returns a single result, not an async generator
                gen_output = await stage_engine.generate(prompt=prompt, request_id=rid, **diffusion_kwargs)
                _gen_t1 = _time.time()
                _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
                await generation_out_q.put((rid, gen_output, _gen_ms))
            else:
                # LLM stages: ensure using SamplingParams
                llm_sampling_params = prepare_sampling_params(sampling_params, "llm")
                gen_output = None
                async for res in stage_engine.generate(ein, llm_sampling_params, rid):
                    gen_output = res
                    _gen_t1 = _time.time()
                    _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
                    _gen_t0 = _gen_t1
                    await generation_out_q.put((rid, gen_output, _gen_ms))
        except Exception as e:
            logger.exception("Failed on request %s: %s", rid, e)
            out_q.put(
                {
                    "request_id": rid,
                    "stage_id": stage_id,
                    "error": str(e),
                }
            )

    _batch_gen_t0 = _time.time()
    while True:
        try:
            task = in_q.get_nowait()
            task_type = task.get("type", OmniStageTaskType.GENERATE)
            if task_type == OmniStageTaskType.SHUTDOWN:
                logger.debug("Received shutdown signal")
                stage_engine.shutdown()
                break
            elif task_type == OmniStageTaskType.ABORT:
                rid = task["request_id"]
                asyncio.create_task(stage_engine.abort(rid))
            elif is_profiler_task(task_type):
                await handle_profiler_task_async(task_type)
            else:
                asyncio.create_task(generation_single_request(task))

        except queue.Empty:
            await asyncio.sleep(0.001)
        batch_request_outputs: list[Any] = []
        batch_request_ids: list[Any] = []
        _gen_ms_list = []
        batch_metrics: list[Any] = []
        while True:
            try:
                rid, gen_output, _gen_ms = generation_out_q.get_nowait()
                _metrics = make_request_stats(
                    [gen_output],
                    _gen_ms,
                    int(_batch_seq),
                    1,  # temporarily set to 1
                    float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                    int(_rx_bytes_by_rid.get(rid, 0)),
                    float(_in_flight_ms_by_rid.get(rid, 0.0)),
                )
                batch_metrics.append(_metrics)
                batch_request_outputs.append(gen_output)
                _gen_ms_list.append(_gen_ms)
                batch_request_ids.append(rid)
                _agg_total_tokens += _metrics.num_tokens_out
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
                break

        if not batch_request_outputs:
            continue
        _batch_seq += 1

        _batch_gen_t1 = _time.time()
        _agg_total_gen_time_ms += (_batch_gen_t1 - _batch_gen_t0) * 1000
        _batch_gen_t0 = _batch_gen_t1
        for idx, metrics in enumerate(batch_metrics):
            metrics.batch_size = len(batch_metrics)
            if idx == len(batch_metrics) - 1:
                metrics.stage_stats = make_stage_stats(_agg_total_tokens, _agg_total_gen_time_ms)

        logger.debug("Sending outputs to main process")
        for rid, output, _gen_ms, _metrics in zip(
            batch_request_ids, batch_request_outputs, _gen_ms_list, batch_metrics
        ):
            try:
                r_outputs = [output]
                use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                if use_shm:
                    out_q.put(
                        {
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs_shm": payload,
                            "metrics": _metrics,
                        }
                    )
                else:
                    out_q.put(
                        {
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs": payload,
                            "metrics": _metrics,
                        }
                    )
                    logger.debug(f"Enqueued req={rid}, use_shm={use_shm}, tokens_out={_metrics.num_tokens_out}")
            except Exception as e:
                logger.exception(
                    "Failed to enqueue result for request %s: %s",
                    rid,
                    e,
                )
                out_q.put(
                    {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "engine_outputs": r_outputs,
                        "metrics": _metrics,
                    }
                )
            logger.debug("Enqueued result for request %s to downstream", rid)
    if log_stats_task is not None:
        log_stats_task.cancel()
    logger.info("Stage worker exiting")


def count_prompt_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    """Count prompt tokens from engine outputs."""
    total = 0
    for _ro in engine_outputs:
        try:
            prompt_token_ids = getattr(_ro, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                total += len(prompt_token_ids)
        except Exception:
            pass
    return total


def make_request_stats(
    req_output: list[Any],
    stage_gen_time_ms: float,
    batch_id: int,
    batch_size: int,
    rx_decode_time_ms: float,
    rx_transfer_bytes: int,
    rx_in_flight_time_ms: float,
):
    from vllm_omni.entrypoints.log_utils import (
        StageRequestMetrics,
    )

    num_tokens_in = count_prompt_tokens_from_outputs(req_output)
    num_tokens_out = count_tokens_from_outputs(req_output)
    return StageRequestMetrics(
        num_tokens_in=num_tokens_in,
        num_tokens_out=num_tokens_out,
        stage_gen_time_ms=stage_gen_time_ms,
        batch_id=batch_id,
        batch_size=batch_size,
        rx_decode_time_ms=rx_decode_time_ms,
        rx_transfer_bytes=rx_transfer_bytes,
        rx_in_flight_time_ms=rx_in_flight_time_ms,
        stage_stats=None,
    )


def make_stage_stats(_agg_total_tokens: int, _agg_total_gen_time_ms: float):
    from vllm_omni.entrypoints.log_utils import StageStats

    return StageStats(total_token=_agg_total_tokens, total_gen_time=_agg_total_gen_time_ms)
