# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""runtime_v2 MultiprocWorkerPool.

Owns its own worker subprocesses (one per topology rank) and reuses the
existing ``DiffusionWorker`` / ``DiffusionModelRunner`` for model load and
execution. Each worker process:

* boots a ``DiffusionWorker`` (device + distributed env + model parallel +
  model load are all done inside ``DiffusionWorker.__init__``),
* builds per-``TaskKind`` ``WorkerExecutor`` objects from the loaded pipeline
  via the runtime_v2 registry adapter (imported lazily so importing this
  module does not require the registry/adapter to exist yet),
* runs a command loop over a multiprocessing pipe: dispatch a task, fetch
  artifacts, evict a request, or shut down.

The controller side (:class:`MultiprocWorkerPool`) spawns the processes, owns
the command/event/result pipes, forwards worker events into a queue the
scheduler polls, aggregates per-rank task events into one group-level event,
and exposes the artifact-fetch API the scheduler uses to materialize a
request's terminal output.

This is the PR1 single-group port. Cross-group artifact migration is out of
scope: the worker has no migrate command phase.
"""

from __future__ import annotations

import contextlib
import copy
import multiprocessing as mp
import pickle
import queue
import signal
import threading
import time
import traceback
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection, wait
from typing import Any

import torch
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm.logger import init_logger

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.ipc import (
    diffusion_output_has_shm_handles,
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)
from vllm_omni.diffusion.runtime_v2.interfaces import WorkerExecutor
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactKind,
    ArtifactValue,
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


@dataclass(frozen=True)
class SerializedArtifactValue:
    handle: Any
    payload: bytes
    transport: str = "pickle"
    payload_nbytes: int = 0

    @property
    def value(self) -> Any:
        return _decode_serialized_payload(self)


@dataclass(frozen=True)
class ProcessDispatchTaskCommand:
    task: InferenceTask
    inline_inputs: tuple[SerializedArtifactValue, ...]
    result_owner_rank: int
    release_after_exec_artifact_ids: tuple[str, ...] = ()
    group_spec: ExecutionGroupSpec | None = None


@dataclass(frozen=True)
class ShutdownWorkerCommand:
    reason: str = ""


@dataclass(frozen=True)
class FetchArtifactsCommand:
    request_id: str
    group_id: str
    artifact_ids: tuple[str, ...]
    fetch_id: str = ""


@dataclass(frozen=True)
class EvictRequestCommand:
    request_id: str


@dataclass(frozen=True)
class FetchArtifactsResult:
    request_id: str
    worker_rank: int
    artifacts: tuple[SerializedArtifactValue | ArtifactValue, ...] = ()
    error: str | None = None
    fetch_id: str = ""


@dataclass(frozen=True)
class WorkerReadyMessage:
    worker_rank: int
    status: str = "ready"
    message: str = ""


@dataclass(frozen=True)
class _FixedParallelSession:
    parallel_spec: ParallelSpec
    world: Any
    dp: Any
    cfg: Any
    pp: Any
    sp: Any
    tp: Any
    fs: Any
    dit: Any
    ep: Any

    @classmethod
    def capture_current(cls, parallel_spec: ParallelSpec) -> _FixedParallelSession:
        session = cls(
            parallel_spec=parallel_spec,
            world=omni_parallel_state._WORLD,
            dp=omni_parallel_state._DP,
            cfg=omni_parallel_state._CFG,
            pp=omni_parallel_state._PP,
            sp=omni_parallel_state._SP,
            tp=vllm_parallel_state._TP,
            fs=omni_parallel_state._FS,
            dit=omni_parallel_state._DIT,
            ep=getattr(vllm_parallel_state, "_EP", None),
        )
        session.validate()
        return session

    def validate(self) -> None:
        if self.world is None:
            raise RuntimeError("runtime_v2 worker session requires initialized distributed world group")
        if self.dp is None or self.cfg is None or self.pp is None or self.sp is None or self.tp is None:
            raise RuntimeError("runtime_v2 worker session requires initialized legacy parallel groups")
        if self.tp.world_size != int(self.parallel_spec.tp):
            raise RuntimeError(
                f"legacy TP world_size={self.tp.world_size} does not match runtime_v2 tp={self.parallel_spec.tp}"
            )
        if self.sp.world_size != int(self.parallel_spec.sp):
            raise RuntimeError(
                f"legacy SP world_size={self.sp.world_size} does not match runtime_v2 sp={self.parallel_spec.sp}"
            )
        if self.cfg.world_size != int(self.parallel_spec.cfg):
            raise RuntimeError(
                f"legacy CFG world_size={self.cfg.world_size} does not match runtime_v2 cfg={self.parallel_spec.cfg}"
            )

    def activate(self) -> None:
        omni_parallel_state._WORLD = self.world
        omni_parallel_state._DP = self.dp
        omni_parallel_state._CFG = self.cfg
        omni_parallel_state._PP = self.pp
        omni_parallel_state._SP = self.sp
        omni_parallel_state._FS = self.fs
        omni_parallel_state._DIT = self.dit
        vllm_parallel_state._DP = self.dp
        vllm_parallel_state._PP = self.pp
        vllm_parallel_state._TP = self.tp
        if hasattr(vllm_parallel_state, "_EP"):
            vllm_parallel_state._EP = self.ep


def _clone_diffusion_output_for_transport(output: DiffusionOutput) -> DiffusionOutput:
    """Shallow-copy a ``DiffusionOutput`` so SHM packing can swap tensor fields.

    ``pack_diffusion_output_shm`` REASSIGNS the tensor attributes (``output``,
    ``trajectory_latents``, ``trajectory_timesteps``, ``trajectory_log_probs``)
    to SHM-handle dicts. We pack a copy so the worker's original stays intact.
    ``dataclasses.replace`` carries EVERY field of ``DiffusionOutput`` forward
    (only the copied instance is later mutated), so this transport path stays
    field-for-field identical to the pickle path -- and adding a new field to
    ``DiffusionOutput`` can never silently drop it here. Do NOT switch this back
    to enumerating a subset of fields.
    """
    return replace(output)


def _decode_serialized_payload(artifact_value: SerializedArtifactValue, *, unpack_shm: bool = True) -> Any:
    value = pickle.loads(artifact_value.payload)
    if unpack_shm and artifact_value.transport == "shm" and isinstance(value, DiffusionOutput):
        unpack_diffusion_output_shm(value)
    return value


def _serialize_artifact_value(
    artifact_value: ArtifactValue,
    *,
    prefer_shm_output: bool = False,
) -> SerializedArtifactValue:
    if (
        prefer_shm_output
        and artifact_value.handle.kind == ArtifactKind.OUTPUT
        and isinstance(artifact_value.value, DiffusionOutput)
    ):
        try:
            output_for_transport = _clone_diffusion_output_for_transport(artifact_value.value)
            pack_diffusion_output_shm(output_for_transport)
            if diffusion_output_has_shm_handles(output_for_transport):
                payload = pickle.dumps(output_for_transport, protocol=pickle.HIGHEST_PROTOCOL)
                return SerializedArtifactValue(
                    handle=artifact_value.handle,
                    payload=payload,
                    transport="shm",
                    payload_nbytes=len(payload),
                )
        except Exception as exc:
            try:
                unpack_diffusion_output_shm(output_for_transport)
            except Exception:
                pass
            logger.warning(
                "runtime_v2 fetch_artifacts shm serialize fallback to pickle: request_id=%s artifact_id=%s error=%s",
                artifact_value.handle.request_id,
                artifact_value.handle.artifact_id,
                exc,
            )

    payload = pickle.dumps(artifact_value.value, protocol=pickle.HIGHEST_PROTOCOL)
    return SerializedArtifactValue(
        handle=artifact_value.handle,
        payload=payload,
        transport="pickle",
        payload_nbytes=len(payload),
    )


def _deserialize_artifact_value(
    artifact_value: SerializedArtifactValue,
    *,
    unpack_shm: bool = True,
) -> ArtifactValue:
    return ArtifactValue(
        handle=artifact_value.handle,
        value=_decode_serialized_payload(artifact_value, unpack_shm=unpack_shm),
    )


def _discard_artifacts_shm(artifacts: Iterable[SerializedArtifactValue | ArtifactValue]) -> None:
    """Unlink any POSIX-SHM segment backing artifacts that are being dropped.

    A dropped artifact (a stale/late fetch result, or the ones already packed
    when a later artifact in the same fetch turns out missing) is never relayed
    downstream, so its packed handles would otherwise leak until worker exit.
    Deserializing with ``unpack_shm=True`` reads + unlinks the segment; the
    reconstructed value is discarded. Best-effort per artifact (a segment may
    already be gone).
    """
    for artifact in artifacts:
        if isinstance(artifact, SerializedArtifactValue) and artifact.transport == "shm":
            with contextlib.suppress(Exception):
                _deserialize_artifact_value(artifact, unpack_shm=True)


def _clone_runtime_worker_config(od_config: OmniDiffusionConfig) -> OmniDiffusionConfig:
    worker_config = copy.deepcopy(od_config)
    worker_config.enable_runtime_v2 = False
    return worker_config


class _WorkerProcessRuntime:
    """Single worker subprocess: boots a DiffusionWorker, builds executors, and
    runs the command loop (dispatch / fetch / evict / shutdown)."""

    def __init__(
        self,
        *,
        worker_rank: int,
        device_id: int | None,
        od_config: OmniDiffusionConfig,
        parallel_spec: ParallelSpec,
        dist_rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        master_port: int | None = None,
        group_id: str = "g0",
        command_pipe_r: Connection,
        event_pipe_w: Connection,
        result_pipe_w: Connection,
    ) -> None:
        self.worker_rank = worker_rank
        self.device_id = device_id
        self.od_config = od_config
        self.parallel_spec = parallel_spec
        self.dist_rank = int(worker_rank if dist_rank is None else dist_rank)
        self.local_rank = int(self.dist_rank if local_rank is None else local_rank)
        configured_world_size = getattr(od_config, "num_gpus", None) if world_size is None else world_size
        configured_master_port = getattr(od_config, "master_port", None) if master_port is None else master_port
        self.world_size = self._safe_positive_int(configured_world_size, default=1)
        self.master_port = self._safe_positive_int(configured_master_port, default=30005)
        self.group_id = group_id
        self.command_pipe_r = command_pipe_r
        self.event_pipe_w = event_pipe_w
        self.result_pipe_w = result_pipe_w
        # Owns the DiffusionWorker (device + distributed env + model
        # parallel + model load happen inside its __init__).
        self._worker: Any | None = None
        self.executors: dict[Any, WorkerExecutor] = {}
        self.local_artifacts: dict[tuple[str, str, str], ArtifactValue] = {}
        # PR1 single-group session: captured after model load and re-activated
        # before each task so global parallel-state pointers stay correct.
        self.fixed_session: _FixedParallelSession | None = None
        self._artifacts_lock = threading.RLock()
        self._result_pipe_lock = threading.Lock()
        self._fetch_queue: queue.Queue[FetchArtifactsCommand | None] = queue.Queue()
        self._fetch_thread: threading.Thread | None = None
        self._fetch_stop = threading.Event()
        self._fetch_copy_stream: Any | None = None

    @staticmethod
    def _safe_positive_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed <= 0:
            return default
        return parsed

    @staticmethod
    def _command_name(command: Any) -> str:
        if isinstance(command, ShutdownWorkerCommand):
            return "shutdown"
        if isinstance(command, ProcessDispatchTaskCommand):
            return "dispatch_task"
        if isinstance(command, FetchArtifactsCommand):
            return "fetch_artifacts"
        if isinstance(command, EvictRequestCommand):
            return "evict_request"
        return type(command).__name__

    @staticmethod
    def _artifact_key(request_id: str, group_id: str, artifact_id: str) -> tuple[str, str, str]:
        return (request_id, group_id, artifact_id)

    @staticmethod
    def _require_task_group_id(task: InferenceTask) -> str:
        if task.group_id is None:
            raise ValueError(f"task {task.task_id} has no group_id")
        return task.group_id

    @staticmethod
    def _validate_task_parallel_spec(task: InferenceTask, spec: ExecutionGroupSpec) -> None:
        if (
            int(task.parallel_spec.tp) != int(spec.parallel_spec.tp)
            or int(task.parallel_spec.sp) != int(spec.parallel_spec.sp)
            or int(task.parallel_spec.cfg) != int(spec.parallel_spec.cfg)
            or bool(task.parallel_spec.cfg_parallel) != bool(spec.parallel_spec.cfg_parallel)
        ):
            raise ValueError(
                f"task {task.task_id} parallel_spec does not match group {spec.group_id!r}: "
                f"task={task.parallel_spec!r} group={spec.parallel_spec!r}"
            )

    def _activate_group_session(self) -> None:
        # Reactivating the fixed single-group session is a cheap global pointer
        # swap before each task.
        if self.fixed_session is None:
            raise RuntimeError("runtime_v2 worker session is not initialized")
        self.fixed_session.activate()

    def _install_parent_death_signal(self) -> None:
        """Tie this GPU worker's lifetime to its parent process.

        Without this, if the owning diffusion process exits while the
        worker is mid-task or mid-collective, the child keeps holding GPU memory
        (or hangs) until it next returns to ``command_pipe_r.recv()`` and observes
        the closed pipe. Mirror ``DiffusionWorker.worker_main``: install a SIGTERM
        handler that exits cleanly (runs ``finally: self._shutdown()``) and arm
        ``PR_SET_PDEATHSIG`` so the OS delivers SIGTERM the instant the parent
        dies. Best-effort / Linux-only; runs on the worker subprocess main thread.
        """
        from vllm_omni.engine.stage_init_utils import set_death_signal

        def _handler(signum, _frame):
            raise SystemExit(128 + signum)

        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, _handler)
        set_death_signal(signal.SIGTERM)

    def run(self) -> None:
        self._install_parent_death_signal()
        try:
            self._initialize()
            self._start_fetch_thread()
            self.event_pipe_w.send(WorkerReadyMessage(worker_rank=self.worker_rank))
            while True:
                command = self.command_pipe_r.recv()
                logger.debug(
                    "runtime_v2 worker command recv: rank=%s cmd=%s",
                    self.worker_rank,
                    self._command_name(command),
                )
                if isinstance(command, ShutdownWorkerCommand):
                    break
                if isinstance(command, ProcessDispatchTaskCommand):
                    self._execute_task(command)
                    continue
                if isinstance(command, FetchArtifactsCommand):
                    self._prepare_fetch_copy_stream_dependency(command)
                    self._fetch_queue.put(command)
                    continue
                if isinstance(command, EvictRequestCommand):
                    self._handle_evict_request(command)
                    continue
                raise TypeError(f"unsupported runtime_v2 worker command: {type(command)!r}")
        except Exception:
            self.event_pipe_w.send(
                WorkerReadyMessage(
                    worker_rank=self.worker_rank,
                    status="failed",
                    message=traceback.format_exc(),
                )
            )
            raise
        finally:
            self._shutdown()

    def _initialize(self) -> None:
        worker_config = _clone_runtime_worker_config(self.od_config)
        # Bootstrap params consumed by DiffusionWorker via od_config.
        worker_config.num_gpus = int(self.world_size)
        worker_config.master_port = int(self.master_port)

        # Worker bootstrap: DiffusionWorker.__init__ runs init_device()
        # (init_distributed_environment + initialize_model_parallel) and
        # load_model(), so the pipeline is ready right after construction.
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        self._worker = DiffusionWorker(
            local_rank=self.local_rank,
            rank=self.dist_rank,
            od_config=worker_config,
        )
        pipeline = getattr(getattr(self._worker, "model_runner", None), "pipeline", None)
        if pipeline is None:
            raise RuntimeError(f"worker {self.worker_rank} failed to initialize diffusion pipeline")

        # Lazy registry import: importing this module must not require the
        # registry / adapter to exist yet.
        from vllm_omni.diffusion.runtime_v2.registry import get_runtime_v2_adapter

        adapter = get_runtime_v2_adapter(getattr(self.od_config, "model_class_name", None))
        adapter.validate_pipeline(pipeline, self.od_config)
        self.executors = adapter.build_executors(pipeline)

        # PR1 single group: capture the parallel session DiffusionWorker just
        # initialized as the fixed "g0" session.
        self.fixed_session = _FixedParallelSession.capture_current(self.parallel_spec)
        self._activate_group_session()

        if torch.cuda.is_available():
            self._fetch_copy_stream = torch.cuda.Stream()
        else:
            self._fetch_copy_stream = None
        logger.info(
            "runtime_v2 multiproc worker initialized: worker_rank=%s group=%s dist_rank=%s world_size=%s "
            "master_port=%s tp=%s sp=%s cfg=%s",
            self.worker_rank,
            self.group_id,
            self.dist_rank,
            self.world_size,
            self.master_port,
            self.parallel_spec.tp,
            self.parallel_spec.sp,
            self.parallel_spec.cfg,
        )

    def _shutdown(self) -> None:
        self._stop_fetch_thread()
        if self._worker is not None:
            try:
                self._worker.shutdown()
            except Exception as exc:
                logger.warning("runtime_v2 worker shutdown failed: rank=%s error=%s", self.worker_rank, exc)

    def _start_fetch_thread(self) -> None:
        if self._fetch_thread is not None and self._fetch_thread.is_alive():
            return
        self._fetch_stop.clear()
        self._fetch_thread = threading.Thread(
            target=self._fetch_loop,
            name=f"runtime-v2-fetch-worker-{self.worker_rank}",
            daemon=True,
        )
        self._fetch_thread.start()

    def _stop_fetch_thread(self) -> None:
        self._fetch_stop.set()
        try:
            self._fetch_queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - unbounded queue
            pass
        thread = self._fetch_thread
        if thread is not None:
            thread.join(timeout=5.0)
            self._fetch_thread = None

    def _fetch_loop(self) -> None:
        while not self._fetch_stop.is_set():
            try:
                command = self._fetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command is None:
                return
            try:
                self._handle_fetch_artifacts(command)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "runtime_v2 artifact fetch failed: request_id=%s fetch_id=%s",
                    command.request_id,
                    command.fetch_id,
                )
                self._send_result(
                    FetchArtifactsResult(
                        request_id=command.request_id,
                        worker_rank=self.worker_rank,
                        error=f"{exc}\n{traceback.format_exc()}",
                        fetch_id=command.fetch_id,
                    )
                )

    def _send_result(self, payload: Any) -> None:
        with self._result_pipe_lock:
            self.result_pipe_w.send(payload)

    def _prepare_fetch_copy_stream_dependency(self, command: FetchArtifactsCommand) -> None:
        if self._fetch_copy_stream is None or not torch.cuda.is_available():
            return
        producer_stream = torch.cuda.current_stream()
        self._fetch_copy_stream.wait_stream(producer_stream)

    def _resolve_inputs(
        self,
        task: InferenceTask,
        inline_inputs: tuple[SerializedArtifactValue, ...],
    ) -> dict[str, Any]:
        inline_by_id = {
            artifact_value.handle.artifact_id: _deserialize_artifact_value(artifact_value)
            for artifact_value in inline_inputs
        }
        resolved_inputs: dict[str, Any] = {}
        group_id = self._require_task_group_id(task)
        for artifact in task.inputs:
            key = self._artifact_key(artifact.request_id, group_id, artifact.artifact_id)
            with self._artifacts_lock:
                local_value = self.local_artifacts.get(key)
            if local_value is not None:
                resolved_inputs[artifact.artifact_id] = local_value.value
                continue
            if artifact.artifact_id in inline_by_id:
                value = inline_by_id[artifact.artifact_id]
                with self._artifacts_lock:
                    self.local_artifacts[key] = value
                resolved_inputs[artifact.artifact_id] = value.value
                continue
            raise KeyError(f"worker {self.worker_rank} cannot resolve input artifact {artifact.artifact_id}")
        return resolved_inputs

    def _make_event(
        self,
        task: InferenceTask,
        kind: WorkerEventKind,
        *,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> WorkerEvent:
        return WorkerEvent(
            event_id=str(uuid.uuid4()),
            task_id=task.task_id,
            request_id=task.request_id,
            group_id=self._require_task_group_id(task),
            worker_rank=self.worker_rank,
            kind=kind,
            timestamp_ns=time.monotonic_ns(),
            message=message,
            metadata=metadata or {},
        )

    def _execute_task(self, command: ProcessDispatchTaskCommand) -> None:
        if self.fixed_session is None:
            raise RuntimeError("runtime_v2 worker session is not initialized")
        if self._worker is None:
            raise RuntimeError("runtime_v2 worker is not initialized")
        task = command.task
        # Local, rank-symmetric setup (group/spec validation, session
        # activation, executor lookup) is safe to downgrade to a per-task
        # TASK_FAILED: every rank of the group runs the same task, so a failure
        # here happens identically on all ranks and none is left waiting on a
        # peer. This keeps one malformed request from killing the worker process
        # (the worker-init checks above stay fatal -- a missing session/worker is
        # not per-task recoverable).
        try:
            group_id = self._require_task_group_id(task)
            group_spec = command.group_spec
            if group_spec is not None:
                self._validate_task_parallel_spec(task, group_spec)
            self._activate_group_session()
            executor = self.executors.get(task.kind)
            if executor is None:
                self.event_pipe_w.send(
                    self._make_event(
                        task,
                        WorkerEventKind.TASK_FAILED,
                        message=f"worker does not support task kind {task.kind}",
                    )
                )
                return
        except Exception as exc:
            self.event_pipe_w.send(
                self._make_event(
                    task,
                    WorkerEventKind.TASK_FAILED,
                    message=f"task setup failed: {exc}\n{traceback.format_exc()}",
                )
            )
            return

        # Input artifacts are rank-symmetric for the fixed execution group.
        resolved_inputs = self._resolve_inputs(task, command.inline_inputs)
        self.event_pipe_w.send(self._make_event(task, WorkerEventKind.TASK_EXEC_BEGIN))
        try:
            worker = self._worker
            use_hsdp = bool(
                getattr(getattr(worker, "od_config", self.od_config), "parallel_config", None) is not None
                and getattr(getattr(worker, "od_config", self.od_config).parallel_config, "use_hsdp", False)
            )
            grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
            with set_forward_context(
                vllm_config=getattr(worker, "vllm_config", None),
                omni_diffusion_config=getattr(worker, "od_config", self.od_config),
            ):
                with grad_context:
                    outputs = executor.execute(task, resolved_inputs)
            published_outputs: list[WorkerLocalArtifactRef] = []
            with self._artifacts_lock:
                for artifact_value in outputs:
                    key = self._artifact_key(
                        artifact_value.handle.request_id,
                        group_id,
                        artifact_value.handle.artifact_id,
                    )
                    self.local_artifacts[key] = artifact_value
                    if self.worker_rank == command.result_owner_rank:
                        published_outputs.append(
                            WorkerLocalArtifactRef(
                                handle=artifact_value.handle,
                                group_id=group_id,
                                worker_rank=self.worker_rank,
                            )
                        )
            self.event_pipe_w.send(
                self._make_event(
                    task,
                    WorkerEventKind.TASK_LAUNCH_END,
                    metadata={"published_outputs": tuple(published_outputs)},
                )
            )
            with self._artifacts_lock:
                for artifact_id in command.release_after_exec_artifact_ids:
                    self.local_artifacts.pop(self._artifact_key(task.request_id, group_id, artifact_id), None)
            self.event_pipe_w.send(self._make_event(task, WorkerEventKind.TASK_EXEC_END))
        except Exception as exc:
            self.event_pipe_w.send(
                self._make_event(
                    task,
                    WorkerEventKind.TASK_FAILED,
                    message=f"{exc}\n{traceback.format_exc()}",
                )
            )

    def _handle_fetch_artifacts(self, command: FetchArtifactsCommand) -> None:
        # Fetch runs on a background thread. It must not mutate the process-wide
        # runtime_v2 parallel session while the main worker thread may be inside
        # model execution.
        artifacts: list[SerializedArtifactValue] = []
        stream_context = (
            torch.cuda.stream(self._fetch_copy_stream)
            if self._fetch_copy_stream is not None
            else contextlib.nullcontext()
        )
        with stream_context:
            for artifact_id in command.artifact_ids:
                key = self._artifact_key(command.request_id, command.group_id, artifact_id)
                with self._artifacts_lock:
                    value = self.local_artifacts.get(key)
                if value is None:
                    # Unlink any SHM segment already packed for an EARLIER
                    # artifact in this same fetch: the error result below carries
                    # no artifacts, so nothing downstream reclaims them and they
                    # would leak until worker exit. (Latent in PR1, which fetches
                    # a single artifact, but the loop is written for N.)
                    _discard_artifacts_shm(artifacts)
                    self._send_result(
                        FetchArtifactsResult(
                            fetch_id=command.fetch_id,
                            request_id=command.request_id,
                            worker_rank=self.worker_rank,
                            error=(
                                "artifact not found: "
                                f"request_id={command.request_id}, group_id={command.group_id}, "
                                f"artifact_id={artifact_id}"
                            ),
                        )
                    )
                    return
                serialized = _serialize_artifact_value(value, prefer_shm_output=True)
                artifacts.append(serialized)
        self._send_result(
            FetchArtifactsResult(
                fetch_id=command.fetch_id,
                request_id=command.request_id,
                worker_rank=self.worker_rank,
                artifacts=tuple(artifacts),
            )
        )

    def _handle_evict_request(self, command: EvictRequestCommand) -> None:
        with self._artifacts_lock:
            keys = [key for key in self.local_artifacts if key[0] == command.request_id]
            for key in keys:
                self.local_artifacts.pop(key, None)


def _worker_process_entrypoint(
    *,
    worker_rank: int,
    device_id: int | None,
    od_config: OmniDiffusionConfig,
    parallel_spec: ParallelSpec,
    dist_rank: int | None,
    local_rank: int | None,
    world_size: int | None,
    master_port: int | None,
    group_id: str,
    command_pipe_r: Connection,
    event_pipe_w: Connection,
    result_pipe_w: Connection,
) -> None:
    runtime = _WorkerProcessRuntime(
        worker_rank=worker_rank,
        device_id=device_id,
        od_config=od_config,
        parallel_spec=parallel_spec,
        dist_rank=dist_rank,
        local_rank=local_rank,
        world_size=world_size,
        master_port=master_port,
        group_id=group_id,
        command_pipe_r=command_pipe_r,
        event_pipe_w=event_pipe_w,
        result_pipe_w=result_pipe_w,
    )
    runtime.run()


@dataclass(frozen=True)
class WorkerProcessHandle:
    process: mp.Process
    worker_rank: int
    command_pipe_w: Connection
    event_pipe_r: Connection
    result_pipe_r: Connection


@dataclass
class _TaskDispatchState:
    group_id: str
    expected_ranks: frozenset[int]
    result_owner_rank: int
    events_by_kind: dict[WorkerEventKind, dict[int, WorkerEvent]] = field(default_factory=dict)


class MultiprocWorkerPool:
    """Centralized multiprocess worker pool for diffusion runtime_v2."""

    def __init__(
        self,
        topology: RuntimeTopology,
        od_config: OmniDiffusionConfig,
    ) -> None:
        self.topology = topology
        self.od_config = od_config
        self._mp_ctx = mp.get_context("spawn")
        self.worker_handles: dict[int, WorkerProcessHandle] = {}
        self._event_queue: queue.Queue[WorkerEvent | WorkerReadyMessage] = queue.Queue()
        # Result-channel demux: the reader thread is the single producer but
        # there can be multiple consumers (fetch drain from the API thread).
        # Split into per-rank queues so each consumer only sees its own rank.
        self._result_queues: dict[int, queue.Queue[Any]] = {}
        self._task_dispatch_state: dict[str, _TaskDispatchState] = {}
        self._state_lock = threading.RLock()
        self._reader_stop = threading.Event()
        self._reader_error: BaseException | None = None
        self._reader_thread: threading.Thread | None = None
        self._fetch_lock = threading.RLock()
        self._inflight_fetches: dict[str, int] = {}
        self._completed_fetches: dict[str, FetchArtifactsResult] = {}

    def start(self, timeout_s: float = 600.0) -> None:
        if self.worker_handles:
            raise RuntimeError("runtime_v2 worker pool is already started")
        # Shared-world mode: all workers join one global torch.distributed world.
        global_world_size = len(self.topology.workers)
        shared_master_port = int(getattr(self.od_config, "master_port", 30005) or 30005)
        for worker in self.topology.workers:
            group = max(
                self.topology.get_groups_for_worker(worker.worker_rank),
                key=lambda candidate: (
                    int(candidate.parallel_spec.sp),
                    int(candidate.parallel_spec.tp),
                    len(candidate.ranks),
                ),
            )
            group_world_size = len(group.ranks)
            dist_rank = worker.worker_rank
            local_rank = int(worker.device_id if worker.device_id is not None else worker.worker_rank)
            worker_od_config = self._build_worker_od_config(
                group_spec=group,
                group_world_size=group_world_size,
                global_world_size=global_world_size,
                shared_master_port=shared_master_port,
            )
            command_pipe_r, command_pipe_w = self._mp_ctx.Pipe(duplex=False)
            event_pipe_r, event_pipe_w = self._mp_ctx.Pipe(duplex=False)
            result_pipe_r, result_pipe_w = self._mp_ctx.Pipe(duplex=False)
            process = self._mp_ctx.Process(
                target=_worker_process_entrypoint,
                kwargs={
                    "worker_rank": worker.worker_rank,
                    "device_id": worker.device_id,
                    "od_config": worker_od_config,
                    "parallel_spec": group.parallel_spec,
                    "dist_rank": dist_rank,
                    "local_rank": local_rank,
                    "world_size": global_world_size,
                    "master_port": shared_master_port,
                    "group_id": group.group_id,
                    "command_pipe_r": command_pipe_r,
                    "event_pipe_w": event_pipe_w,
                    "result_pipe_w": result_pipe_w,
                },
                name=f"runtime-v2-worker-{worker.worker_rank}",
                daemon=True,
            )
            process.start()
            command_pipe_r.close()
            event_pipe_w.close()
            result_pipe_w.close()
            self.worker_handles[worker.worker_rank] = WorkerProcessHandle(
                process=process,
                worker_rank=worker.worker_rank,
                command_pipe_w=command_pipe_w,
                event_pipe_r=event_pipe_r,
                result_pipe_r=result_pipe_r,
            )
            self._result_queues[worker.worker_rank] = queue.Queue()

        ready_workers: set[int] = set()
        deadline = time.monotonic() + timeout_s
        while len(ready_workers) < len(self.worker_handles):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                dead = {
                    rank: handle.process.exitcode
                    for rank, handle in self.worker_handles.items()
                    if not handle.process.is_alive()
                }
                self.shutdown()
                raise TimeoutError(f"timed out waiting for runtime_v2 workers to start; dead={dead}")
            ready = wait([handle.event_pipe_r for handle in self.worker_handles.values()], timeout=min(0.1, remaining))
            for reader in ready:
                try:
                    event = reader.recv()
                except EOFError:
                    # A worker died during startup (OOM / segfault): its event
                    # pipe closed, so wait() flags it readable and recv() raises.
                    # Tear the whole pool down -- surviving ranks are stuck in the
                    # shared-world rendezvous -- instead of letting EOFError escape
                    # uncaught and orphan them (RuntimeV2Runner.__init__ would then
                    # unwind with _runner=None, so close() could not reach them).
                    self.shutdown()
                    raise RuntimeError("runtime_v2 worker died during startup before signaling ready")
                if isinstance(event, WorkerReadyMessage):
                    if event.status != "ready":
                        self.shutdown()
                        raise RuntimeError(f"runtime_v2 worker {event.worker_rank} failed to start: {event.message}")
                    ready_workers.add(event.worker_rank)
                else:
                    self._enqueue_event(event)
        self._reader_stop.clear()
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="runtime-v2-multiproc-forwarder",
            daemon=True,
        )
        self._reader_thread.start()
        logger.info("runtime_v2 multiproc worker pool started: workers=%s", sorted(self.worker_handles))

    def _build_worker_od_config(
        self,
        *,
        group_spec: ExecutionGroupSpec,
        group_world_size: int,
        global_world_size: int,
        shared_master_port: int,
    ) -> OmniDiffusionConfig:
        parallel_spec = group_spec.parallel_spec
        worker_config = _clone_runtime_worker_config(od_config=self.od_config)
        base_parallel = worker_config.parallel_config
        # Keep per-group parallel semantics for execution while still joining a
        # shared global world. This config is consumed by DiffusionWorker.
        worker_config.parallel_config = DiffusionParallelConfig(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=int(parallel_spec.tp),
            enable_expert_parallel=bool(getattr(base_parallel, "enable_expert_parallel", False)),
            sequence_parallel_size=int(parallel_spec.sp),
            ulysses_degree=int(group_spec.ulysses_degree),
            ring_degree=int(group_spec.ring_degree),
            cfg_parallel_size=int(parallel_spec.cfg),
            vae_patch_parallel_size=int(getattr(base_parallel, "vae_patch_parallel_size", 1)),
            use_hsdp=False,
            hsdp_shard_size=-1,
            hsdp_replicate_size=1,
        )
        expected_world_size = int(worker_config.parallel_config.world_size)
        if expected_world_size != int(group_world_size):
            raise ValueError(
                f"invalid runtime_v2 group parallel spec: tp*sp*cfg={expected_world_size} "
                f"!= group_world_size={group_world_size}"
            )
        # Distributed bootstrap params are global in shared-world mode.
        worker_config.num_gpus = int(global_world_size)
        worker_config.master_port = int(shared_master_port)
        return worker_config

    def shutdown(self) -> None:
        self._reader_stop.set()
        for handle in self.worker_handles.values():
            try:
                handle.command_pipe_w.send(ShutdownWorkerCommand())
            except Exception:
                pass
        reader_thread = self._reader_thread
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)
        for handle in self.worker_handles.values():
            try:
                handle.process.join(timeout=5.0)
            except Exception:
                pass
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=5.0)
                if handle.process.is_alive():
                    # SIGTERM can't be serviced by a worker wedged in a C-level
                    # collective (e.g. NCCL); SIGKILL it so it can't linger
                    # holding GPU memory after the pool reports stopped.
                    handle.process.kill()
                    handle.process.join(timeout=5.0)
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        self._reader_thread = None
        # The reader stops before worker shutdown so it cannot race cleanup.
        # A worker fetch thread may still have written a final SHM-backed result
        # directly to its pipe in that window; drain those raw pipes after the
        # processes exit and before dropping the handles.
        stranded_in_pipes: list[FetchArtifactsResult] = []
        for handle in self.worker_handles.values():
            try:
                while handle.result_pipe_r.poll():
                    payload = handle.result_pipe_r.recv()
                    if isinstance(payload, FetchArtifactsResult):
                        stranded_in_pipes.append(payload)
            except Exception:
                pass
        self.worker_handles.clear()
        with self._state_lock:
            self._task_dispatch_state.clear()
        # Drain any results the reader received but no poll ever retrieved (e.g.
        # the last request was aborted, so its rank is never polled again) before
        # clearing. The reader thread was joined above, so there are no concurrent
        # puts. These are reclaimed alongside _completed_fetches below.
        stranded_in_queues: list[FetchArtifactsResult] = []
        for result_queue in self._result_queues.values():
            while True:
                try:
                    payload = result_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(payload, FetchArtifactsResult):
                    stranded_in_queues.append(payload)
        self._result_queues.clear()
        with self._fetch_lock:
            self._inflight_fetches.clear()
            stranded = list(self._completed_fetches.values())
            self._completed_fetches.clear()
        # Reclaim packed POSIX-SHM from any completed/queued-but-never-drained
        # results: the segments were created by the workers and outlive them, so
        # unlink here rather than leak /dev/shm.
        for result in (*stranded_in_pipes, *stranded_in_queues, *stranded):
            self._discard_fetch_result_shm(result)
        logger.info("runtime_v2 multiproc worker pool stopped")

    def dispatch(
        self,
        task: InferenceTask,
        inline_inputs: tuple[ArtifactValue, ...],
        release_after_exec_artifact_ids: tuple[str, ...] = (),
    ) -> None:
        if task.group_id is None:
            raise ValueError("task must have an assigned group before dispatch")
        group = self.topology.get_group(task.group_id)
        with self._state_lock:
            self._register_task_dispatch(task)
        command = ProcessDispatchTaskCommand(
            task=task,
            inline_inputs=tuple(_serialize_artifact_value(artifact_value) for artifact_value in inline_inputs),
            result_owner_rank=self.topology.get_group_leader(task.group_id),
            release_after_exec_artifact_ids=release_after_exec_artifact_ids,
            group_spec=group,
        )
        for worker_rank in group.ranks:
            self.worker_handles[worker_rank].command_pipe_w.send(command)

    def poll(self, timeout_s: float = 0.0) -> list[WorkerEvent | WorkerReadyMessage]:
        self._raise_reader_error()
        events: list[WorkerEvent | WorkerReadyMessage] = []
        try:
            if timeout_s > 0:
                first = self._event_queue.get(timeout=timeout_s)
            else:
                first = self._event_queue.get_nowait()
        except queue.Empty:
            self._raise_reader_error()
            return events

        events.append(first)
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                self._raise_reader_error()
                return events

    def fetch_artifacts(self, request_id: str, group_id: str, artifact_ids: tuple[str, ...]) -> FetchArtifactsResult:
        fetch_id = self.start_fetch_artifacts(request_id=request_id, group_id=group_id, artifact_ids=artifact_ids)
        deadline = time.monotonic() + 30.0
        while True:
            result = self.poll_fetch_artifacts(fetch_id)
            if result is not None:
                return self._normalize_fetch_result(result)
            if time.monotonic() >= deadline:
                self.discard_fetch(fetch_id)
                raise TimeoutError(f"timed out waiting for FetchArtifactsResult fetch_id={fetch_id}")
            time.sleep(0.001)

    def start_fetch_artifacts(self, request_id: str, group_id: str, artifact_ids: tuple[str, ...]) -> str:
        leader_rank = self.topology.get_group_leader(group_id)
        handle = self.worker_handles[leader_rank]
        fetch_id = str(uuid.uuid4())
        with self._fetch_lock:
            self._inflight_fetches[fetch_id] = leader_rank
        handle.command_pipe_w.send(
            FetchArtifactsCommand(
                fetch_id=fetch_id,
                request_id=request_id,
                group_id=group_id,
                artifact_ids=artifact_ids,
            )
        )
        return fetch_id

    def poll_fetch_artifacts(self, fetch_id: str) -> FetchArtifactsResult | None:
        with self._fetch_lock:
            completed = self._completed_fetches.pop(fetch_id, None)
            if completed is not None:
                self._inflight_fetches.pop(fetch_id, None)
                return self._normalize_fetch_result(completed)
            leader_rank = self._inflight_fetches.get(fetch_id)
            if leader_rank is None:
                return None

        self._drain_fetch_results_for_rank(leader_rank)
        with self._fetch_lock:
            completed = self._completed_fetches.pop(fetch_id, None)
            if completed is None:
                return None
            self._inflight_fetches.pop(fetch_id, None)
            return self._normalize_fetch_result(completed)

    def discard_fetch(self, fetch_id: str) -> None:
        with self._fetch_lock:
            leader_rank = self._inflight_fetches.pop(fetch_id, None)
            completed = self._completed_fetches.pop(fetch_id, None)
        # A discarded fetch (abort/cleanup) never reaches the downstream unpack
        # site, so its terminal output's packed POSIX-SHM segment would leak until
        # worker exit. (1) Unlink an already-completed result popped above.
        if completed is not None:
            self._discard_fetch_result_shm(completed)
        # (2) The worker may have the result already queued in the pipe. Drain the
        # leader rank now: the fetch is untracked, so the reader routes it to
        # _discard_fetch_result_shm (is_tracked=False) instead of stranding it. A
        # result the worker sends LATER is reclaimed by the next poll on this rank
        # or by shutdown cleanup. SHM I/O is done outside the fetch lock.
        if leader_rank is not None:
            with contextlib.suppress(Exception):
                self._drain_fetch_results_for_rank(leader_rank)

    def _drain_fetch_results_for_rank(self, leader_rank: int) -> None:
        result_queue = self._result_queues.get(leader_rank)
        if result_queue is None:
            return
        while True:
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(payload, FetchArtifactsResult):
                raise RuntimeError(
                    f"unexpected result type from worker {leader_rank}: expected FetchArtifactsResult, "
                    f"got {type(payload).__name__}"
                )
            fetch_id = payload.fetch_id
            if not fetch_id:
                raise RuntimeError(f"received fetch result without fetch_id from worker {leader_rank}")
            with self._fetch_lock:
                is_tracked = fetch_id in self._inflight_fetches or fetch_id in self._completed_fetches
                if is_tracked:
                    self._completed_fetches[fetch_id] = payload
            if not is_tracked:
                # The request was aborted/cleaned up between start_fetch_artifacts
                # and now, so this result is never drained downstream. Its
                # artifacts may be SerializedArtifactValue(transport="shm"): the
                # normal path keeps them packed and unlinks the POSIX-SHM segment
                # LATER at the final postprocess site, so simply dropping the
                # payload would leak the segment until worker exit. Unlink here
                # (outside the fetch lock -- SHM I/O must not block dispatch).
                self._discard_fetch_result_shm(payload)
                logger.debug(
                    "runtime_v2 drop stale fetch result: fetch_id=%s leader_rank=%s",
                    fetch_id,
                    leader_rank,
                )

    @staticmethod
    def _discard_fetch_result_shm(result: FetchArtifactsResult) -> None:
        """Unlink any POSIX-SHM segment backing a dropped (stale) fetch result."""
        _discard_artifacts_shm(result.artifacts)

    @staticmethod
    def _normalize_fetch_result(result: FetchArtifactsResult) -> FetchArtifactsResult:
        artifacts: list[ArtifactValue] = []
        for artifact in result.artifacts:
            if isinstance(artifact, ArtifactValue):
                artifacts.append(artifact)
            else:
                # Keep SHM handles packed through scheduler/control path; unpack
                # at the final postprocess site to avoid extra host copies.
                artifacts.append(_deserialize_artifact_value(artifact, unpack_shm=False))
        return FetchArtifactsResult(
            fetch_id=result.fetch_id,
            request_id=result.request_id,
            worker_rank=result.worker_rank,
            artifacts=tuple(artifacts),
            error=result.error,
        )

    def evict_request(self, request_id: str) -> None:
        command = EvictRequestCommand(request_id=request_id)
        for handle in self.worker_handles.values():
            handle.command_pipe_w.send(command)

    def check_health(self) -> None:
        self._raise_reader_error()
        for handle in self.worker_handles.values():
            if not handle.process.is_alive():
                raise RuntimeError(
                    f"runtime_v2 worker {handle.worker_rank} died unexpectedly with exit code {handle.process.exitcode}"
                )

    def _raise_reader_error(self) -> None:
        if self._reader_error is not None:
            raise RuntimeError("runtime_v2 multiproc pipe forwarder failed") from self._reader_error

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                readers = [handle.event_pipe_r for handle in self.worker_handles.values()] + [
                    handle.result_pipe_r for handle in self.worker_handles.values()
                ]
                if not readers:
                    return
                ready = wait(readers, timeout=0.1)
                if not ready:
                    continue
                for reader in ready:
                    worker_rank = self._find_reader_owner_rank(reader)
                    if worker_rank is None:
                        continue
                    handle = self.worker_handles.get(worker_rank)
                    if handle is None:
                        continue
                    try:
                        payload = reader.recv()
                    except EOFError as exc:
                        if self._reader_stop.is_set():
                            return
                        self._reader_error = exc
                        return
                    if reader is handle.event_pipe_r:
                        self._enqueue_event(payload)
                    else:
                        self._result_queues[worker_rank].put(payload)
        except BaseException as exc:  # pragma: no cover - defensive path
            self._reader_error = exc
            logger.exception("runtime_v2 multiproc pipe forwarder failed")

    def _find_reader_owner_rank(self, reader: Connection) -> int | None:
        for worker_rank, handle in self.worker_handles.items():
            if reader is handle.event_pipe_r or reader is handle.result_pipe_r:
                return worker_rank
        return None

    def _register_task_dispatch(self, task: InferenceTask) -> None:
        if task.group_id is None:
            raise ValueError("task must have an assigned group before registration")
        group = self.topology.get_group(task.group_id)
        self._task_dispatch_state[task.task_id] = _TaskDispatchState(
            group_id=task.group_id,
            expected_ranks=frozenset(group.ranks),
            result_owner_rank=self.topology.get_group_leader(task.group_id),
        )

    def _consume_event(self, event: WorkerEvent | WorkerReadyMessage) -> list[WorkerEvent | WorkerReadyMessage]:
        if not isinstance(event, WorkerEvent):
            return [event]
        if event.kind in (
            WorkerEventKind.TASK_LAUNCH_END,
            WorkerEventKind.TASK_EXEC_BEGIN,
            WorkerEventKind.TASK_EXEC_END,
            WorkerEventKind.TASK_FAILED,
        ):
            with self._state_lock:
                return self._aggregate_task_event(event)
        return [event]

    def _enqueue_event(self, event: WorkerEvent | WorkerReadyMessage) -> None:
        for aggregated_event in self._consume_event(event):
            if isinstance(aggregated_event, WorkerEvent):
                logger.debug(
                    "runtime_v2 event enqueued: request_id=%s task_id=%s kind=%s group=%s worker_rank=%s metadata=%s",
                    aggregated_event.request_id,
                    aggregated_event.task_id,
                    aggregated_event.kind,
                    aggregated_event.group_id,
                    aggregated_event.worker_rank,
                    dict(aggregated_event.metadata),
                )
            self._event_queue.put(aggregated_event)

    def _aggregate_task_event(self, event: WorkerEvent) -> list[WorkerEvent]:
        state = self._task_dispatch_state.get(event.task_id)
        if state is None:
            return []

        if event.kind == WorkerEventKind.TASK_FAILED:
            self._task_dispatch_state.pop(event.task_id, None)
            return [
                WorkerEvent(
                    event_id=f"aggregate:{event.task_id}:{event.kind.value}",
                    task_id=event.task_id,
                    request_id=event.request_id,
                    group_id=state.group_id,
                    worker_rank=event.worker_rank,
                    kind=event.kind,
                    timestamp_ns=event.timestamp_ns,
                    message=event.message,
                    metadata={"failed_rank": event.worker_rank},
                )
            ]

        seen_by_rank = state.events_by_kind.setdefault(event.kind, {})
        if event.worker_rank in seen_by_rank:
            return []
        seen_by_rank[event.worker_rank] = event
        if frozenset(seen_by_rank) != state.expected_ranks:
            logger.debug(
                "runtime_v2 event pending aggregation: request_id=%s task_id=%s "
                "kind=%s seen_ranks=%s expected_ranks=%s",
                event.request_id,
                event.task_id,
                event.kind,
                tuple(sorted(seen_by_rank)),
                tuple(sorted(state.expected_ranks)),
            )
            return []

        aggregated_event = self._build_aggregated_event(
            state=state,
            kind=event.kind,
            events_by_rank=seen_by_rank,
        )
        logger.debug(
            "runtime_v2 event aggregated: request_id=%s task_id=%s kind=%s ranks=%s metadata=%s",
            aggregated_event.request_id,
            aggregated_event.task_id,
            aggregated_event.kind,
            tuple(sorted(seen_by_rank)),
            dict(aggregated_event.metadata),
        )
        if event.kind == WorkerEventKind.TASK_EXEC_END:
            self._task_dispatch_state.pop(event.task_id, None)
        return [aggregated_event]

    def _build_aggregated_event(
        self,
        *,
        state: _TaskDispatchState,
        kind: WorkerEventKind,
        events_by_rank: dict[int, WorkerEvent],
    ) -> WorkerEvent:
        owner_event = events_by_rank.get(state.result_owner_rank)
        representative = owner_event or min(
            events_by_rank.values(),
            key=lambda event: (event.timestamp_ns, event.worker_rank),
        )
        metadata = dict(owner_event.metadata) if owner_event is not None else dict(representative.metadata)
        metadata["completed_ranks"] = tuple(sorted(events_by_rank))
        return WorkerEvent(
            event_id=f"aggregate:{representative.task_id}:{kind.value}",
            task_id=representative.task_id,
            request_id=representative.request_id,
            group_id=state.group_id,
            worker_rank=state.result_owner_rank,
            kind=kind,
            timestamp_ns=max(event.timestamp_ns for event in events_by_rank.values()),
            message=representative.message,
            metadata=metadata,
        )
