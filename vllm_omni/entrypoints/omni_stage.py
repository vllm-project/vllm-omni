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
import time
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Literal, cast

from vllm import PromptType, RequestOutput
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
from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.ray_utils.utils import kill_ray_actor, start_ray_actor
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
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
from vllm_omni.entrypoints.utils import detect_pid_host
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniSamplingParams, OmniTokensPrompt
from vllm_omni.metrics import count_tokens_from_outputs
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


@contextmanager
def _sequential_init_lock(engine_args: dict[str, Any], stage_init_timeout: int = 300):
    """Acquire device locks for sequential init if NVML is unavailable.

    If process-scoped memory tracking is available (NVML works), stages can
    safely initialize concurrently — each measures only its own GPU memory.
    Otherwise, fall back to file-based locks to serialize initialization.
    """
    from vllm_omni.worker.gpu_memory_utils import is_process_scoped_memory_available

    if is_process_scoped_memory_available() and detect_pid_host():
        logger.debug(
            "NVML process-scoped memory available and PID host is available — concurrent init is safe, skipping locks"
        )
        yield
        return
    else:
        logger.debug(
            "NVML unavailable or PID host is not available (usually inside a container, "
            "--pid=host is not set in docker run command) — using sequential init locks"
        )

    from vllm_omni.platforms import current_omni_platform

    # Get all parallel sizes from engine_args or parallel_config (defaults to 1)
    if "parallel_config" in engine_args:
        parallel_config = engine_args["parallel_config"]
        tensor_parallel_size = parallel_config.get("tensor_parallel_size", 1)
        pipeline_parallel_size = parallel_config.get("pipeline_parallel_size", 1)
        data_parallel_size = parallel_config.get("data_parallel_size", 1)
        prefill_context_parallel_size = parallel_config.get("prefill_context_parallel_size", 1)
        sequence_parallel_size = parallel_config.get("sequence_parallel_size", 1)
        cfg_parallel_size = parallel_config.get("cfg_parallel_size", 1)
    else:
        tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
        pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
        data_parallel_size = engine_args.get("data_parallel_size", 1)
        prefill_context_parallel_size = engine_args.get("prefill_context_parallel_size", 1)
        sequence_parallel_size = 1
        cfg_parallel_size = 1

    num_devices_per_stage = (
        tensor_parallel_size
        * pipeline_parallel_size
        * data_parallel_size
        * prefill_context_parallel_size
        * sequence_parallel_size
        * cfg_parallel_size
    )

    # Get physical device IDs from device control env var
    device_control_env = current_omni_platform.device_control_env_var
    visible_devices_str = os.environ.get(device_control_env)
    physical_devices = []

    if visible_devices_str:
        try:
            physical_devices = [int(x.strip()) for x in visible_devices_str.split(",") if x.strip()]
        except (ValueError, IndexError):
            pass

    if not physical_devices:
        num_devices = current_omni_platform.get_device_count()
        physical_devices = list(range(num_devices))

    num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
    devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

    logger.debug(
        "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d, CFG=%d; will lock %d devices: %s",
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
        prefill_context_parallel_size,
        sequence_parallel_size,
        cfg_parallel_size,
        num_devices_to_lock,
        devices_to_lock,
    )

    # Acquire exclusive locks for all devices using fcntl.flock
    wait_start = time.time()
    acquired_lock_fds = []

    for device_id in devices_to_lock:
        lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
        lock_acquired = False

        while not lock_acquired:
            try:
                lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)

                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.ftruncate(lock_fd, 0)
                    os.write(lock_fd, f"{os.getpid()}\n".encode())
                    os.fsync(lock_fd)
                    lock_acquired = True
                    acquired_lock_fds.append(lock_fd)
                    logger.debug("Acquired exclusive lock for device %s", device_id)
                except BlockingIOError:
                    os.close(lock_fd)

                    if time.time() - wait_start > stage_init_timeout:
                        logger.warning(
                            "Timeout waiting for device %s initialization lock, proceeding anyway",
                            device_id,
                        )
                        break

                    time.sleep(0.1)
            except OSError as e:
                logger.debug(
                    "Failed to acquire lock for device %s: %s, continuing anyway",
                    device_id,
                    e,
                )
                try:
                    os.close(lock_fd)
                except (OSError, NameError):
                    pass
                break

    # Set FD_CLOEXEC to prevent child processes from inheriting locks
    for lock_fd in acquired_lock_fds:
        try:
            flags = fcntl.fcntl(lock_fd, fcntl.F_GETFD)
            fcntl.fcntl(lock_fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        except (OSError, ValueError):
            pass

    try:
        yield
    finally:
        for lock_fd in acquired_lock_fds:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                logger.debug("Released initialization lock (fd=%s)", lock_fd)
            except (OSError, ValueError):
                pass


def _resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    worker_type = engine_args.pop("worker_type", None)
    if not worker_type:
        return
    if engine_args.get("worker_cls"):
        return
    from vllm_omni.platforms import current_omni_platform

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")


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
        self.stage_type: Literal["llm", "diffusion"] = getattr(stage_config, "stage_type", "llm")
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
        default_sampling_params = _to_dict(default_sampling_params)
        # Further convert it to dataclass to check fields
        try:
            self.default_sampling_params = (
                SamplingParams if self.stage_type == "llm" else OmniDiffusionSamplingParams
            )(**default_sampling_params)
        except TypeError as error:
            raise TypeError(f"Invalid default_sampling_params for stage {self.stage_id}: {error}") from error
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

    def stop_profile(self) -> dict:
        """Stop profiling by sending a signal to worker and waiting for response."""
        if self._in_q is None or self._out_q is None:
            logger.warning(f"[Stage-{self.stage_id}] Queues not initialized, cannot stop profile.")
            return {}

        logger.info(f"[Stage-{self.stage_id}] Sending PROFILER_STOP to worker...")
        self.submit({"type": OmniStageTaskType.PROFILER_STOP})

        # Wait for result from worker
        try:
            # Profiling stop might take time to flush files, give it 600s
            response = self._out_q.get(timeout=600)

            if isinstance(response, dict):
                if response.get("type") == "profiler_result":
                    return response.get("data", {})
                elif "error" in response:
                    logger.error(f"[Stage-{self.stage_id}] Profiler error: {response['error']}")
                    return {}

            # If we got something else (e.g. late generation result), we might lose it here,
            # but usually profiling stop is called when generation is done.
            logger.warning(
                f"[Stage-{self.stage_id}] Received unexpected message while waiting for profiler: {response}"
            )
            return {}

        except queue.Empty:
            logger.error(f"[Stage-{self.stage_id}] Timeout waiting for profiler results.")
            return {}

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
            "engine_input_source": self.engine_input_source,
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

        # [Omni] Inject global request_id into additional_information for cross-stage ID consistency
        # This allows workers (like GPUARModelRunner) to use the global ID for side-channel
        # operations like KV transfer, even if they use internal IDs for execution.
        if "request_id" in payload and "engine_inputs" in payload:
            req_id = payload["request_id"]
            ein = payload["engine_inputs"]

            # Helper to inject into additional_information
            def _inject_global_id(target_ein):
                # OmniTokensPrompt is a TypedDict at runtime, so we treat it as a dict
                if isinstance(target_ein, dict):
                    if "additional_information" not in target_ein:
                        target_ein["additional_information"] = {}

                    # Ensure additional_information is a dict before assignment
                    # (in case it was somehow initialized as None or other type)
                    if target_ein["additional_information"] is None:
                        target_ein["additional_information"] = {}

                    if isinstance(target_ein["additional_information"], dict):
                        # Wrap in list because OmniInputProcessor requires Tensor or list values
                        target_ein["additional_information"]["global_request_id"] = [str(req_id)]

            if isinstance(ein, list):
                for item in ein:
                    _inject_global_id(item)
            else:
                _inject_global_id(ein)

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
    import multiprocessing as _mp
    import os as _os
    import time as _time

    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()
    # IMPORTANT: Ensure vLLM's internal multiprocessing workers (e.g., GPUARWorker /
    # GPUARModelRunner) are spawned with a fork-safe method.
    # Mooncake / gRPC / RDMA and CUDA/NCCL can deadlock under fork-with-threads.
    if _os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        _os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("[Stage] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
    # Best-effort: also force python mp start method in this stage process.
    # This may raise if already set; that's fine.
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
    connectors_config = stage_payload.get("connectors_config", {})
    stage_type: Literal["llm", "diffusion"] = stage_payload.get("stage_type", "llm")

    if stage_type != "diffusion":
        _resolve_worker_cls(engine_args)

    # Aggregates for running average
    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    # Monotonic batch id per stage process for orchestrator dedup on time aggregation
    _batch_seq = 0

    # Device mapping
    device_type = None
    try:
        from vllm_omni.platforms import current_omni_platform

        device_type = current_omni_platform.device_type
        set_stage_devices(stage_id, runtime_cfg.get("devices"), device_type=device_type)
    except Exception as e:
        logger.warning("Device setup failed: %s", e)

    # Use sequential init locks only when NVML is unavailable
    with _sequential_init_lock(engine_args, stage_init_timeout):
        # Init engine based on stage_type
        logger.debug(
            "[Stage-%s] Initializing %s engine with args keys=%s", stage_id, stage_type, list(engine_args.keys())
        )
        if engine_args.get("async_chunk", False):
            logger.debug("[Stage-%s] Async chunk enabled, injecting connectors config", stage_id)
            stage_connector_spec = {}
            for v in connectors_config.values():
                stage_connector_spec = dict(v.get("spec", {}))
                break
            engine_args["stage_connector_spec"] = stage_connector_spec
            engine_args["stage_id"] = stage_id
        if stage_type == "diffusion":
            engine_args.pop("model_stage", None)
            engine_args.pop("model", None)
            stage_engine = OmniDiffusion(
                model=model,
                stage_id=stage_id,
                engine_input_source=stage_payload.get("engine_input_source", []),
                **engine_args,
            )
        else:
            # Default to LLM engine
            stage_engine = OmniLLM(model=model, **engine_args)

    logger.debug("Engine initialized")
    # Initialize OmniConnectors if configured
    connectors: dict[tuple[str, str], OmniConnectorBase] | None = {}
    if connectors_config:
        connectors = build_stage_connectors(
            stage_id=stage_id,
            connectors_config=connectors_config,
        )
        if connectors is None:
            return

    # Signal readiness to orchestrator
    try:
        out_q.put({"type": "stage_ready", "stage_id": stage_id})
    except Exception:
        pass

    max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
    logger.info(f"Max batch size: {max_batch_size}")

    def handle_profiler_task_local(task_type: OmniStageTaskType) -> dict:
        """Handle profiler task locally in the worker process."""
        if task_type == OmniStageTaskType.PROFILER_START:
            if stage_type == "diffusion":
                try:
                    profile_dir = _os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")
                    _os.makedirs(profile_dir, exist_ok=True)
                    trace_filename = f"stage_{stage_id}_diffusion_{int(_time.time())}"
                    stage_engine.start_profile(trace_filename=trace_filename)
                    logger.info("[Stage-%s] Diffusion Torch profiler started", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start diffusion profiler: %s", stage_id, e)
            else:
                try:
                    stage_engine.start_profile()
                    logger.info("[Stage-%s] vLLM profiler started", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start vLLM profiler: %s", stage_id, e)
            return {}

        elif task_type == OmniStageTaskType.PROFILER_STOP:
            if stage_type == "diffusion":
                try:
                    # CRITICAL: Capture return value
                    result_data = stage_engine.stop_profile()
                    logger.info("[Stage-%s] Diffusion Torch profiler stopped", stage_id)
                    return result_data
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop diffusion profiler: %s", stage_id, e)
                    return {}
            else:
                try:
                    stage_engine.stop_profile()
                    logger.info("[Stage-%s] vLLM profiler stopped", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop vLLM profiler: %s", stage_id, e)
                return {}
        return {}

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
            profiler_data = handle_profiler_task_local(task_type)
            # If it was a STOP command, we must reply to the Orchestrator
            if task_type == OmniStageTaskType.PROFILER_STOP:
                out_q.put({"type": "profiler_result", "data": profiler_data})
            continue

        batch_tasks: list[dict[str, Any]] = [task]
        tasks_failed_to_add_to_batch: list[dict[str, Any]] = []
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
                        p_data = handle_profiler_task_local(extra_type)
                        if extra_type == OmniStageTaskType.PROFILER_STOP:
                            out_q.put({"type": "profiler_result", "data": p_data})
                        continue
                    # Ensure that all tasks have the same sampling params
                    # If no, put them in a temporary container and add back to queue
                    # This should be always true, because user only calls omni.generate() once and it blocks
                    # User can only pass one sampling param object, but the list of prompts are separated.
                    if task.get("sampling_params") != extra.get("sampling_params"):
                        logger.warning(
                            """In offline mode, expect all prompts in one `omni.generate()` call to share same sampling params"""  # noqa: E501 # line too long
                            f"""However, prompt {task.get("engine_inputs")} has sampling params {task.get("sampling_params")}, """  # noqa: E501 # line too long
                            f"""whereas the prompt {extra.get("engine_inputs")} has sampling params {extra.get("sampling_params")}."""  # noqa: E501 # line too long
                            """The two tasks cannot be combined in one batch request."""
                        )
                        tasks_failed_to_add_to_batch.append(extra)
                    else:
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
        for task_to_readd in tasks_failed_to_add_to_batch:
            in_q.put(task_to_readd)
        # Ensure that the popped tasks are with identical sampling params. Take one of them.
        batch_engine_sampling_params: OmniSamplingParams = batch_tasks[0]["sampling_params"]

        batch_request_ids: list[Any] = []
        batch_engine_inputs: list[OmniPromptType] = []
        _rx_bytes_by_rid: dict[Any, int] = {}
        _rx_decode_ms_by_rid: dict[Any, float] = {}
        _in_flight_ms_by_rid: dict[Any, float] = {}
        for t in batch_tasks:
            rid = t["request_id"]
            try:
                sent_ts = float(t.get("sent_ts", None)) if isinstance(t, dict) else None
                if sent_ts is not None:
                    _in_flight_ms_by_rid[rid] = max(0.0, (_recv_dequeue_ts - sent_ts) * 1000.0)
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
            # TODO: hack type annotation for now.
            # A better way is to refine type annotation of connection and task/payloads, maybe using template types.
            ein = cast(OmniPromptType | Sequence[OmniPromptType] | None, ein)

            if ein is None or _rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{stage_id}] Missing connector payload for request {rid}. "
                    "Ensure connectors are configured for all incoming edges."
                )

            _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
            _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

            batch_request_ids.append(rid)
            if isinstance(ein, (str, dict)):
                # Types like OmniTextPrompt, TextPrompt are TypedDict, essentially dict and enters this branch
                batch_engine_inputs.append(ein)
            elif isinstance(ein, Sequence):
                batch_engine_inputs.extend(ein)
            else:
                # Other unknown types, append as-is
                batch_engine_inputs.append(ein)
        logger.debug(
            "Received batch size=%d, request_ids=%s",
            len(batch_tasks),
            batch_request_ids,
        )
        try:
            _batch_seq += 1
            gen_outputs: list[OmniRequestOutput | RequestOutput] = []
            _gen_t0 = _time.time()
            if stage_type == "diffusion":
                stage_engine = cast(OmniDiffusion, stage_engine)
                batch_engine_sampling_params = cast(OmniDiffusionSamplingParams, batch_engine_sampling_params)
                # Diffusion generate returns results directly, not an iterator
                diffusion_results = stage_engine.generate(
                    batch_engine_inputs, batch_engine_sampling_params, batch_request_ids
                )
                gen_outputs.extend(diffusion_results)
                # Assign request_ids if not present
                for idx, result in enumerate(gen_outputs):
                    if not hasattr(result, "request_id") or result.request_id is None:
                        if idx < len(batch_request_ids):
                            result.request_id = batch_request_ids[idx]
            else:
                stage_engine = cast(OmniLLM, stage_engine)
                batch_engine_sampling_params = cast(SamplingParams, batch_engine_sampling_params)
                results = stage_engine.generate(
                    batch_engine_inputs,  # type: ignore # silent complaints about list of subclassed TypedDict
                    batch_engine_sampling_params,
                    use_tqdm=False,
                )
                gen_outputs.extend(results)
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
            logger.debug(f"Generate done: batch={len(batch_tasks)}, req_ids={batch_request_ids}, gen_ms={_gen_ms:.1f}")

            # Group outputs per request id with fallback
            req_to_outputs: dict[Any, list[Any]] = {rid: [] for rid in batch_request_ids}
            unmapped: list[Any] = []
            for ro in gen_outputs:
                rid = ro.request_id
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
    import multiprocessing as _mp
    import os as _os
    import time as _time

    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()
    # IMPORTANT: Ensure vLLM's internal multiprocessing workers (e.g., GPUARWorker /
    # GPUARModelRunner) are spawned with a fork-safe method.
    if _os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        _os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("[Stage-async] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
    connectors_config = stage_payload.get("connectors_config", {})
    stage_type = stage_payload.get("stage_type", "llm")

    if stage_type != "diffusion":
        _resolve_worker_cls(engine_args)

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
        from vllm_omni.platforms import current_omni_platform

        device_type = current_omni_platform.device_type
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

    # Use sequential init locks only when NVML is unavailable
    with _sequential_init_lock(engine_args, stage_init_timeout):
        # Init engine based on stage_type
        logger.debug(
            "[Stage-%s] Initializing %s engine with args keys=%s",
            stage_id,
            stage_type,
            list(engine_args.keys()),
        )
        if engine_args.get("async_chunk", False):
            logger.debug("[Stage-%s] Async chunk enabled, injecting connectors config", stage_id)
            stage_connector_spec = {}
            for v in connectors_config.values():
                stage_connector_spec = dict(v.get("spec", {}))
                break
            engine_args["stage_connector_spec"] = stage_connector_spec
            engine_args["stage_id"] = stage_id
        if stage_type == "diffusion":
            # For diffusion, we need to extract diffusion-specific config
            od_config = _build_od_config(engine_args, model)

            # Inject omni config for worker to access stage info
            if "omni_kv_config" not in od_config:
                od_config["omni_kv_config"] = {}
            od_config["omni_kv_config"]["stage_id"] = stage_id
            od_config["omni_kv_config"]["engine_input_source"] = stage_payload.get("engine_input_source", [])

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

    async def handle_profiler_task_async(task_type: OmniStageTaskType) -> None:
        """Handle profiler task asynchronously for both LLM and diffusion stages."""
        if task_type == OmniStageTaskType.PROFILER_START:
            if stage_type == "diffusion":
                try:
                    # Sync call is safe here — diffusion profiling is lightweight
                    profile_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")
                    os.makedirs(profile_dir, exist_ok=True)
                    trace_filename = f"stage_{stage_id}_diffusion_{int(time.time())}"
                    stage_engine.start_profile(trace_filename=trace_filename)
                    logger.info("[Stage-%s] Diffusion Torch profiler started", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start diffusion profiler: %s", stage_id, e)
            else:
                try:
                    await stage_engine.start_profile()
                    logger.info("[Stage-%s] vLLM profiler started", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start vLLM profiler: %s", stage_id, e)

        elif task_type == OmniStageTaskType.PROFILER_STOP:
            if stage_type == "diffusion":
                try:
                    trace_files = stage_engine.stop_profile()
                    logger.info("[Stage-%s] Diffusion Torch profiler stopped", stage_id)
                    if trace_files:
                        logger.info("Diffusion trace files: %s", trace_files)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop diffusion profiler: %s", stage_id, e)
            else:
                try:
                    await stage_engine.stop_profile()
                    logger.info("[Stage-%s] vLLM profiler stopped", stage_id)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop vLLM profiler: %s", stage_id, e)

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
                _in_flight_ms_by_rid[rid] = max(0.0, (_recv_dequeue_ts - sent_ts) * 1000.0)
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
            # TODO: hack type annotation for now.
            # A better way is to refine type annotation of connection and task/payloads, maybe using template types.
            ein = cast(OmniPromptType | Sequence[OmniPromptType] | None, ein)

            if ein is None or _rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{stage_id}] Missing connector payload for request {rid}. "
                    "Ensure connectors are configured for all incoming edges."
                )
            _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
            _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

            logger.debug("Received batch size=1, request_ids=%s", rid)
            _gen_t0 = _time.time()
            if isinstance(ein, Sequence) and not isinstance(ein, str):
                ein = ein[0]

            if stage_type == "diffusion":
                diffusion_sampling_params = cast(OmniDiffusionSamplingParams, task["sampling_params"])
                # AsyncOmniDiffusion.generate returns a single result, not an async generator
                gen_output = await cast(AsyncOmniDiffusion, stage_engine).generate(ein, diffusion_sampling_params, rid)
                _gen_t1 = _time.time()
                _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
                await generation_out_q.put((rid, gen_output, _gen_ms))
            else:
                ein = cast(PromptType, ein)
                llm_sampling_params: SamplingParams = task["sampling_params"]
                gen_output = None
                async for res in cast(AsyncLLM, stage_engine).generate(ein, llm_sampling_params, rid):
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
                r_outputs = [output_strip(output, omni_stage)]
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
    from vllm_omni.metrics import StageRequestStats

    num_tokens_in = count_prompt_tokens_from_outputs(req_output)
    num_tokens_out = count_tokens_from_outputs(req_output)
    return StageRequestStats(
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
    from vllm_omni.metrics import StageStats

    return StageStats(total_token=_agg_total_tokens, total_gen_time_ms=_agg_total_gen_time_ms)


def output_strip(r_output: RequestOutput | OmniRequestOutput, omni_stage: OmniStage):
    """
    Strip unnecessary multimodal outputs from stages results,
    in order to:
    - reduce memory usage
    - reduce transfer & serialization overhead
    """

    # check multimodal data is required by stage output config.
    if omni_stage.final_output and omni_stage.final_output_type != "text":
        return r_output

    # If the request has already finished, should not be altered.
    if getattr(r_output, "finished", False):
        return r_output

    mm_output = getattr(r_output, "multimodal_output", None)
    if mm_output is not None:
        r_output.multimodal_output = {}

    outputs = getattr(r_output, "outputs", None)
    if outputs is not None:
        for out in outputs:
            if getattr(out, "multimodal_output", None):
                out.multimodal_output = {}

    return r_output
