"""
Stage manager for orchestrating multiple engines in vLLM-Omni.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.
"""

import asyncio
import fcntl
import importlib
import logging
import multiprocessing as mp
import os
import sys
import traceback
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

from vllm_omni.distributed.ray_utils.utils import kill_ray_actor, start_ray_actor
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.stage_utils import (
    _to_dict,
    maybe_dump_to_shm,
    set_stage_devices,
)
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class OmniStage:
    """Stage manager for orchestrating a single stage in the omni pipeline.

    Encapsulates per-stage process lifecycle and worker logic, including
    device setup, LLM initialization, batching, and shared-memory IPC.
    Preserves input processing utilities for cross-stage data wiring.

    Args:
        stage_config: Stage configuration object containing engine arguments,
            runtime settings, and stage-specific parameters
    """

    def __init__(self, stage_config: Any):
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
        self.engine_output_type = stage_config.engine_args.engine_output_type
        self.engine_outputs = None
        self.is_comprehension = getattr(stage_config, "is_comprehension", False)
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
        self.default_sampling_params = SamplingParams(**_to_dict(default_sampling_params))
        # Runtime orchestration state (added)
        self._in_q: mp.Queue | None = None
        self._out_q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._log_file: str | None = None
        self._shm_threshold_bytes: int = 65536
        self._logger = logging.getLogger(__name__)

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
        log_file: str | None = None,
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
            log_file: Optional log file path prefix for stage-specific logs
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
        self._log_file = log_file

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
        }

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
                    log_file=self._log_file,
                    batch_timeout=batch_timeout,
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
                        self._log_file,
                        batch_timeout,
                    ),
                )
            self._proc.start()

    def stop_stage_worker(self) -> None:
        """Stop the stage worker process gracefully.

        Sends shutdown signal to the worker and waits for it to terminate.
        If graceful shutdown fails, forcefully terminates the process.
        Handles both multiprocessing Process and Ray Actor.
        """
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(None)
            except Exception as e:
                self._logger.warning("[Stage-%s] Failed to send shutdown to in_q: %s", self.stage_id, e)

        if hasattr(self, "_ray_actor") and self._ray_actor:
            kill_ray_actor(self._ray_actor)
            self._ray_actor = None
        elif self._proc is not None:
            try:
                self._proc.join(timeout=5)
            except Exception as e:
                self._logger.debug("[Stage-%s] join() failed: %s", self.stage_id, e, exc_info=True)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception as e:
                    self._logger.warning("[Stage-%s] terminate() failed: %s", self.stage_id, e)

        # Cleanup temporary stage log if we created one (only when no log_file provided)
        try:
            if not self._log_file:
                tmp_path = f"/tmp/omni_stage{self.stage_id}.log"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception:
            pass

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
    log_file: str | None = None,
    batch_timeout: int = 10,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""
    import logging as _logging
    import os as _os
    import time as _time

    from vllm_omni.distributed.omni_connectors import build_stage_connectors
    from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
    from vllm_omni.entrypoints.log_utils import (
        compute_and_log_stage_request_stats,
        count_tokens_from_outputs,
        log_stage_batch_stats,
        log_stage_running_avg,
    )
    from vllm_omni.entrypoints.omni_llm import OmniStageLLM
    # no inline JSONL/serialization imports; logging handled by utilities

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
    connectors_config = stage_payload.get("connectors_config", {})

    # Per-stage logger: clear inherited handlers to avoid broken parent streams
    try:
        stage_log = _logging.getLogger(__name__)
        stage_log.setLevel(_logging.DEBUG)
        for _h in list(stage_log.handlers):
            try:
                stage_log.removeHandler(_h)
            except Exception:
                pass
        stage_log.propagate = False

        class _StageFilter(_logging.Filter):
            def filter(self, record: _logging.LogRecord) -> bool:
                setattr(record, "stage", stage_id)
                return True

        # Prefer file logging even when log_file is not provided, to avoid invalid stdio in child procs
        try:
            if log_file:
                _path = f"{log_file}.stage{stage_id}.log"
            else:
                _path = f"/tmp/omni_stage{stage_id}.log"
            # Ensure dir exists
            _os.makedirs(_os.path.dirname(_path), exist_ok=True)
            fh = _logging.FileHandler(_path)
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(
                _logging.Formatter("%(asctime)s [PID:%(process)d] [Stage-%(stage)s] %(levelname)s: %(message)s")
            )

            class _StageFilter(_logging.Filter):
                def filter(self, record: _logging.LogRecord) -> bool:
                    setattr(record, "stage", stage_id)
                    return True

            fh.addFilter(_StageFilter())
            stage_log.addHandler(fh)
            try:
                # Also route vLLM internal logs to the same handler for this stage
                _vllm_logger = _logging.getLogger("vllm")
                _vllm_logger.setLevel(_logging.DEBUG)
                for _vh in list(_vllm_logger.handlers):
                    try:
                        _vllm_logger.removeHandler(_vh)
                    except Exception:
                        pass
                _vllm_logger.propagate = False
                _vllm_logger.addHandler(fh)
            except Exception:
                pass
        except Exception:
            # Final fallback: attach NullHandler to avoid logging errors
            stage_log.addHandler(_logging.NullHandler())
    except Exception:
        pass

    try:
        _logging.raiseExceptions = False
        _root_logger = _logging.getLogger()
        for _h in list(_root_logger.handlers):
            try:
                _root_logger.removeHandler(_h)
            except Exception:
                pass
        _root_logger.addHandler(_logging.NullHandler())
    except Exception:
        pass

    # Stage stats JSONL file
    _stats_file = f"{log_file}.stage{stage_id}.stats.jsonl" if log_file else None

    # Aggregates for running average
    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    # Monotonic batch id per stage process for orchestrator dedup on time aggregation
    _batch_seq = 0

    # Device mapping
    device_type = None
    try:
        from vllm_omni.utils import detect_device_type

        device_type = detect_device_type()
        set_stage_devices(stage_id, runtime_cfg.get("devices"), device_type=device_type)
    except Exception as e:
        _logging.getLogger(__name__).warning("[Stage-%s] Device setup failed: %s", stage_id, e)

    # Sequential initialization on the same device to avoid memory calculation errors
    # when multiple instances start simultaneously
    # For TP/PP/DP/SP, we need to lock ALL devices that will be used by this stage
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
                # For a single stage worker:
                # - TP: splits model across GPUs (always needed)
                # - PP: splits layers across pipelinestages, but each stage uses TP devices
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

                _logging.getLogger(__name__).debug(
                    "[Stage-%s] Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d; will lock %d devices: %s",
                    stage_id,
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    data_parallel_size,
                    prefill_context_parallel_size,
                    num_devices_to_lock,
                    devices_to_lock,
                )

                # Acquire exclusive locks for all devices using fcntl.flock
                # Locks are automatically released when process dies
                max_wait_time = 300  # 5 minutes max wait
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
                                _logging.getLogger(__name__).debug(
                                    "[Stage-%s] Acquired exclusive lock for device %s", stage_id, device_id
                                )
                            except BlockingIOError:
                                # Lock is held by another process
                                _os.close(lock_fd)

                                # Check if we've been waiting too long
                                if _time.time() - wait_start > max_wait_time:
                                    _logging.getLogger(__name__).warning(
                                        "[Stage-%s] Timeout waiting for device %s "
                                        "initialization lock, proceeding anyway",
                                        stage_id,
                                        device_id,
                                    )
                                    break

                                # Wait a bit before retrying
                                _time.sleep(0.1)
                        except OSError as e:
                            # Other error - log and continue without lock
                            _logging.getLogger(__name__).debug(
                                "[Stage-%s] Failed to acquire lock for device %s: %s, continuing anyway",
                                stage_id,
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
            _logging.getLogger(__name__).debug(
                "[Stage-%s] Failed to set up sequential initialization lock: %s", stage_id, e
            )

    # Init LLM
    _logging.getLogger(__name__).debug(
        "[Stage-%s] Initializing engine with args keys=%s", stage_id, list(engine_args.keys())
    )
    try:
        stage_engine = OmniStageLLM(model=model, **engine_args)
    finally:
        # Release all locks by closing file descriptors
        # Locks are automatically released when file descriptors are closed
        # or when process dies
        for lock_fd in lock_files:
            try:
                _os.close(lock_fd)
                _logging.getLogger(__name__).debug("[Stage-%s] Released initialization lock (fd=%s)", stage_id, lock_fd)
            except (OSError, ValueError):
                pass
    _logging.getLogger(__name__).debug("[Stage-%s] Engine initialized", stage_id)

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

    # Batch processing loop
    while True:
        task = in_q.get()
        _recv_dequeue_ts = _time.time()
        if task is None:
            _logging.getLogger(__name__).error("[Stage-%s] Received shutdown signal", stage_id)
            break

        max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
        print(f"[Stage-{stage_id}] Max batch size: {max_batch_size}")
        batch_tasks: list[dict[str, Any]] = [task]
        start_time = _time.time()
        if max_batch_size > 1:
            while len(batch_tasks) < max_batch_size:
                if not in_q.empty():
                    extra = in_q.get_nowait()
                    if extra is None:
                        in_q.put(None)
                        break
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

            if _rx_metrics:
                _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
                _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

            batch_request_ids.append(rid)
            if isinstance(ein, list):
                batch_engine_inputs.extend(ein)
            elif isinstance(ein, dict):
                batch_engine_inputs.append(ein)
            else:
                _logging.getLogger(__name__).error("[Stage-%s] Invalid engine input type: %s", stage_id, type(ein))
        sampling_params = batch_tasks[0]["sampling_params"]
        _logging.getLogger(__name__).debug(
            "[Stage-%s] Received batch size=%d, request_ids=%s",
            stage_id,
            len(batch_tasks),
            batch_request_ids,
        )
        print("--------------------------------", flush=True)
        print(
            f"[Stage-{stage_id}] Received batch size={len(batch_tasks)}, request_ids={batch_request_ids}",
            flush=True,
        )
        print("--------------------------------", flush=True)
        try:
            _batch_seq += 1
            gen_outputs: list[Any] = []
            _gen_t0 = _time.time()
            for ro in stage_engine.generate(batch_engine_inputs, sampling_params, use_tqdm=False):
                gen_outputs.append(ro)
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
            try:
                print(
                    f"[Stage-{stage_id}] Generate done: batch={len(batch_tasks)}, "
                    f"req_ids={batch_request_ids}, gen_ms={_gen_ms:.1f}",
                    flush=True,
                )
            except Exception:
                pass

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

            # Per-request stats logging and aggregates
            for rid in batch_request_ids:
                _r_outputs = req_to_outputs.get(rid, [])
                _num_tokens = count_tokens_from_outputs(_r_outputs)
                _agg_total_tokens += _num_tokens
                _agg_total_gen_time_ms += _gen_ms

            if _stats_file:
                _avg_tokens_per_s = (
                    (_agg_total_tokens * 1000.0 / _agg_total_gen_time_ms) if _agg_total_gen_time_ms > 0 else 0.0
                )
                log_stage_running_avg(
                    _stats_file,
                    stage_id,
                    int(_agg_total_tokens),
                    float(_agg_total_gen_time_ms),
                    float(_avg_tokens_per_s),
                )
                log_stage_batch_stats(
                    _stats_file,
                    stage_id,
                    len(batch_tasks),
                    float(_gen_ms),
                    list(batch_request_ids),
                )

            # Emit per-request results
            for rid in batch_request_ids:
                r_outputs = req_to_outputs.get(rid, [])
                try:
                    use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                    _metrics = {
                        "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                        "stage_gen_time_ms": _gen_ms,
                        "batch_id": int(_batch_seq),
                        "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                        "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                        "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                    }
                    if _stats_file:
                        compute_and_log_stage_request_stats(
                            _stats_file,
                            stage_id,
                            rid,
                            len(batch_tasks),
                            r_outputs,
                            float(_gen_ms),
                            int(_metrics["rx_transfer_bytes"]),  # type: ignore[index]
                            float(_metrics["rx_decode_time_ms"]),  # type: ignore[index]
                        )
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
                            "metrics": {
                                "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                                "stage_gen_time_ms": _gen_ms,
                                "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                                "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                                "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                            },
                        }
                    )
                _logging.getLogger(__name__).debug(
                    "[Stage-%s] Enqueued result for request %s to downstream",
                    stage_id,
                    rid,
                )
        except Exception as e:
            _logging.getLogger(__name__).exception("[Stage-%s] Failed on batch %s: %s", stage_id, batch_request_ids, e)
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
) -> None:
    asyncio.run(_stage_worker_async(omni_stage, model, stage_payload, batch_timeout))


async def _stage_worker_async(
    omni_stage: OmniStage,
    model: str,
    stage_payload: dict[str, Any],
    batch_timeout: int = 10,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""
    import logging as _logging
    import os as _os
    import time as _time

    from vllm_omni.distributed.omni_connectors import build_stage_connectors
    from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
    from vllm_omni.entrypoints.async_omni import AsyncOmniStageLLM
    from vllm_omni.entrypoints.log_utils import (
        compute_and_log_stage_request_stats,
        count_tokens_from_outputs,
        log_stage_batch_stats,
        log_stage_running_avg,
    )

    # no inline JSONL/serialization imports; logging handled by utilities

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
    connectors_config = stage_payload.get("connectors_config", {})

    log_file = omni_stage._log_file
    in_q = omni_stage._in_q
    out_q = omni_stage._out_q
    # Per-stage file logger (optional)
    try:
        if log_file:
            stage_log = _logging.getLogger(__name__)
            stage_log.setLevel(_logging.DEBUG)
            fh = _logging.FileHandler(f"{log_file}.stage{stage_id}.log")
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(
                _logging.Formatter("%(asctime)s [PID:%(process)d] [Stage-%(stage)s] %(levelname)s: %(message)s")
            )  # noqa: E501

            class _StageFilter(_logging.Filter):
                def filter(self, record: _logging.LogRecord) -> bool:
                    setattr(record, "stage", stage_id)
                    return True

            fh.addFilter(_StageFilter())
            stage_log.addHandler(fh)
    except Exception:
        pass

    # Stage stats JSONL file
    _stats_file = f"{log_file}.stage{stage_id}.stats.jsonl" if log_file else None

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
        _logging.getLogger(__name__).warning("[Stage-%s] Device setup failed: %s", stage_id, e)

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

                _logging.getLogger(__name__).debug(
                    "[Stage-%s] Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d; will lock %d devices: %s",
                    stage_id,
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    data_parallel_size,
                    prefill_context_parallel_size,
                    num_devices_to_lock,
                    devices_to_lock,
                )

                # Acquire exclusive locks for all devices using fcntl.flock
                # Locks are automatically released when process dies
                max_wait_time = 300  # 5 minutes max wait
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
                                _logging.getLogger(__name__).debug(
                                    "[Stage-%s] Acquired exclusive lock for device %s", stage_id, device_id
                                )
                            except BlockingIOError:
                                # Lock is held by another process
                                _os.close(lock_fd)

                                # Check if we've been waiting too long
                                if _time.time() - wait_start > max_wait_time:
                                    _logging.getLogger(__name__).warning(
                                        "[Stage-%s] Timeout waiting for device %s "
                                        "initialization lock, proceeding anyway",
                                        stage_id,
                                        device_id,
                                    )
                                    break

                                # Wait a bit before retrying
                                _time.sleep(0.1)
                        except OSError as e:
                            # Other error - log and continue without lock
                            _logging.getLogger(__name__).debug(
                                "[Stage-%s] Failed to acquire lock for device %s: %s, continuing anyway",
                                stage_id,
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
            _logging.getLogger(__name__).debug(
                "[Stage-%s] Failed to set up sequential initialization lock: %s", stage_id, e
            )

    # Init LLM
    _logging.getLogger(__name__).debug(
        "[Stage-%s] Initializing engine with args keys=%s",
        stage_id,
        list(engine_args.keys()),
    )
    try:
        omni_engine_args = AsyncOmniEngineArgs(**engine_args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = omni_engine_args.create_engine_config(usage_context=usage_context)
        stage_engine = AsyncOmniStageLLM.from_vllm_config(
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
                _os.close(lock_fd)
                _logging.getLogger(__name__).debug("[Stage-%s] Released initialization lock (fd=%s)", stage_id, lock_fd)
            except (OSError, ValueError):
                pass
    omni_stage.set_async_engine(stage_engine)
    # Don't keep the dummy data in memory
    await stage_engine.reset_mm_cache()
    _logging.getLogger(__name__).debug("[Stage-%s] Engine initialized", stage_id)
    # Signal readiness to orchestrator and send vllm_config back to main process
    try:
        # Send vllm_config back to main process so it can be accessed via
        # get_vllm_config(). This is needed because async_engine is only available
        # in the worker process

        # input_preprocessor = await stage_engine.get_input_preprocessor()
        out_q.put(
            {
                "type": "stage_ready",
                "stage_id": stage_id,
                "vllm_config": vllm_config,
                "tokenizer": getattr(stage_engine, "tokenizer", None),
                "is_tracing_enabled": await stage_engine.is_tracing_enabled(),
                # "input_preprocessor": input_preprocessor,
            }
        )
    except Exception as e:
        _logging.getLogger(__name__).warning("[Stage-%s] Failed to send stage ready signal: %s", stage_id, e)

    # Batch processing loop
    while True:
        task = in_q.get()
        _recv_dequeue_ts = _time.time()
        if task is None:
            _logging.getLogger(__name__).debug("[Stage-%s] Received shutdown signal", stage_id)
            break

        _rx_bytes_by_rid: dict[Any, int] = {}
        _rx_decode_ms_by_rid: dict[Any, float] = {}
        _in_flight_ms_by_rid: dict[Any, float] = {}

        rid = task["request_id"]
        try:
            sent_ts = float(task.get("sent_ts", None)) if isinstance(task, dict) else None
            if sent_ts is not None:
                _in_flight_ms_by_rid[rid] = (_recv_dequeue_ts - sent_ts) * 1000.0
            else:
                _in_flight_ms_by_rid[rid] = 0.0
        except Exception:
            _in_flight_ms_by_rid[rid] = 0.0
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
        _logging.getLogger(__name__).debug("[Stage-%s] Received batch size=1, request_ids=%s", stage_id, rid)
        print("--------------------------------", flush=True)
        print(f"[Stage-{stage_id}] Received batch size=1, request_ids={rid}", flush=True)
        print("--------------------------------", flush=True)
        try:
            _batch_seq += 1
            _gen_t0 = _time.time()
            if isinstance(ein, list):
                ein = ein[0]

            async for res in stage_engine.generate(ein, sampling_params, rid):
                gen_output = res
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0

            r_outputs = [gen_output]
            _num_tokens = count_tokens_from_outputs(r_outputs)
            _agg_total_tokens += _num_tokens
            _agg_total_gen_time_ms += _gen_ms

            if _stats_file:
                _avg_tokens_per_s = (
                    (_agg_total_tokens * 1000.0 / _agg_total_gen_time_ms) if _agg_total_gen_time_ms > 0 else 0.0
                )
                log_stage_running_avg(
                    _stats_file,
                    stage_id,
                    int(_agg_total_tokens),
                    float(_agg_total_gen_time_ms),
                    float(_avg_tokens_per_s),
                )
                log_stage_batch_stats(_stats_file, stage_id, 1, float(_gen_ms), [rid])

            try:
                use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                _metrics = {
                    "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                    "stage_gen_time_ms": _gen_ms,
                    "batch_id": int(_batch_seq),
                    "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                    "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                    "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                }
                if _stats_file:
                    compute_and_log_stage_request_stats(
                        _stats_file,
                        stage_id,
                        rid,
                        1,
                        r_outputs,
                        float(_gen_ms),
                        int(_metrics["rx_transfer_bytes"]),  # type: ignore[index]
                        float(_metrics["rx_decode_time_ms"]),  # type: ignore[index]
                    )
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
                    try:
                        print(
                            f"[Stage-{stage_id}] Enqueued req={rid}, use_shm={use_shm}, "
                            f"tokens_out={_metrics['num_tokens_out']}",
                            flush=True,
                        )
                    except Exception:
                        pass
            except Exception as e:
                _logging.getLogger(__name__).exception(
                    "[Stage-%s] Failed to enqueue result for request %s: %s",
                    stage_id,
                    rid,
                    e,
                )
                out_q.put(
                    {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "engine_outputs": r_outputs,
                        "metrics": {
                            "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                            "stage_gen_time_ms": _gen_ms,
                            "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                            "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                            "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                        },
                    }
                )
            _logging.getLogger(__name__).debug("[Stage-%s] Enqueued result for request %s to downstream", stage_id, rid)

        except Exception as e:
            _logging.getLogger(__name__).exception("[Stage-%s] Failed on request %s: %s", stage_id, rid, e)
            out_q.put(
                {
                    "request_id": rid,
                    "stage_id": stage_id,
                    "error": str(e),
                }
            )
    print("--------------------------------", flush=True)
    print(f"[Stage-{stage_id}] Stage worker exiting", flush=True)
    print("--------------------------------", flush=True)
