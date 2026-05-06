from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker import WorkerProc

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)

_DEQUEUE_TIMEOUT_S = 5.0


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    broadcast_mq: MessageQueue | None = None
    result_mq: MessageQueue | None = None
    num_workers: int = 0
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if hasattr(self, "wake_events") and self.wake_events:
            for ev in self.wake_events:
                ev.set()

        if self.broadcast_mq is not None:
            try:
                for _ in range(self.num_workers):
                    self.broadcast_mq.enqueue(SHUTDOWN_MESSAGE, timeout=1.0)

                self.broadcast_mq = None
                self.result_mq = None
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)

        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(30)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(30)


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self.is_failed = False
        self._failure_callbacks: list[Callable[[], None]] = []

        num_workers = self.od_config.num_gpus
        self.wake_events = [mp.Event() for _ in range(num_workers)]

        self._broadcast_mq = self._init_broadcast_queue(num_workers)
        broadcast_handle = self._broadcast_mq.export_handle()

        # Launch workers
        processes, result_handle = self._launch_workers(broadcast_handle, self.wake_events)
        self._result_mq = self._init_result_queue(result_handle)
        self._processes = processes

        self.resources = BackgroundResources(
            broadcast_mq=self._broadcast_mq,
            result_mq=self._result_mq,
            num_workers=num_workers,
            processes=self._processes,
        )
        self._finalizer = weakref.finalize(self, self.resources)

        self.start_worker_monitor()

    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        return MessageQueue(
            n_reader=num_workers,
            n_local_reader=num_workers,
            local_reader_ranks=list(range(num_workers)),
        )

    def _init_result_queue(self, result_handle) -> MessageQueue | None:
        if result_handle is None:
            logger.error("Failed to get result queue handle from workers")
            return None
        return MessageQueue.create_from_handle(result_handle, 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if self._result_mq is None:
            raise RuntimeError("Result queue not initialized")

    def _dequeue_one_with_failure_polling(self, deadline: float | None, method: str) -> Any:
        """Block until one result message, polling ``is_failed`` between chunk timeouts."""
        while True:
            if deadline is None:
                chunk_timeout = _DEQUEUE_TIMEOUT_S
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RPC call to {method} timed out.")
                chunk_timeout = min(_DEQUEUE_TIMEOUT_S, remaining)
            try:
                return self._result_mq.dequeue(timeout=chunk_timeout)
            except (TimeoutError, zmq.error.Again):
                if self.is_failed:
                    raise EngineDeadError()
                continue

    def _launch_workers(self, broadcast_handle, wake_events):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Extract worker_extension_cls and custom_pipeline_args from od_config
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = mp.Process(
                target=WorkerProc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                    wake_events[i],
                    worker_extension_cls,
                    custom_pipeline_args,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handle = None
        for writer in scheduler_pipe_writers:
            writer.close()

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
                processes[i].join()
                logger.error(f"Exit code: {processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError("Initialization failed. Please see the error messages above.")

            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)
            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handle

    def start_worker_monitor(self) -> None:
        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        sentinels = [p.sentinel for p in self._processes]
        if not sentinels:
            return

        def _monitor() -> None:
            try:
                finished = multiprocessing.connection.wait(sentinels)
            except OSError:
                return

            if self._closed:
                return

            dead = [p.name for p in self._processes if p.sentinel in finished]
            if dead:
                logger.error(
                    "Diffusion worker(s) died unexpectedly: %s",
                    dead,
                )
                self.is_failed = True

            self.shutdown()

            for cb in self._failure_callbacks:
                try:
                    cb()
                except Exception:
                    logger.exception("failure_callback raised")

        t = threading.Thread(target=_monitor, daemon=True, name="diffusion-worker-monitor")
        t.start()

    def register_failure_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Register a callback invoked when a worker process dies."""
        self._failure_callbacks.append(callback)

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        self._ensure_open()
        rpc_request = {
            "type": "rpc",
            "method": "generate",
            "args": (request,),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
        }

        try:
            self._broadcast_mq.enqueue(rpc_request)
            response = self._result_mq.dequeue()

            try:
                unpack_diffusion_output_shm(response)
            except Exception as e:
                logger.warning("SHM unpack failed (data may already be inline): %s", e)

            if isinstance(response, dict) and response.get("status") == "error":
                raise RuntimeError(
                    f"Worker failed with error '{response.get('error')}', "
                    "please check the stack trace above for the root cause"
                )
            if not isinstance(response, DiffusionOutput):
                raise RuntimeError(f"Unexpected response type for generate: {type(response)!r}")
            return response
        except Exception as e:
            logger.error(f"Generate call failed: {e}")
            raise

    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Adapt request-mode scheduler output to worker execute_model RPC."""
        from vllm_omni.diffusion.worker.utils import RunnerOutput

        self._ensure_open()
        if scheduler_output.num_scheduled_reqs != 1:
            raise ValueError(
                f"Request mode currently supports batch_size=1, "
                f"but got {scheduler_output.num_scheduled_reqs} scheduled requests."
            )

        new_req = scheduler_output.scheduled_new_reqs[0]
        result = self.collective_rpc(
            "execute_model",
            args=(new_req.req, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, DiffusionOutput):
            raise RuntimeError(f"Unexpected response type for execute_request: {type(result)!r}")

        return RunnerOutput(
            req_id=new_req.sched_req_id,
            step_index=None,
            finished=True,
            result=result,
        )

    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Forward step-mode scheduler output to worker execute_stepwise RPC."""
        from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, RunnerOutput

        self._ensure_open()
        result = self.collective_rpc(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

        if isinstance(result, BaseRunnerOutput):
            return result
        # TODO: Remove this fallback; DiffusionOutput cannot faithfully represent
        # failed multi-request step batches.
        if isinstance(result, DiffusionOutput):
            req_id = scheduler_output.scheduled_req_ids[0] if scheduler_output.scheduled_req_ids else ""
            return RunnerOutput(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=result,
            )
        else:
            raise RuntimeError(f"Unexpected response type for execute_step: {type(result)!r}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        self._ensure_open()

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        # When unique_reply_rank is None, all workers must execute the RPC
        # but only rank 0 can reply (it's the only one with a result_mq).
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank if unique_reply_rank is not None else 0,
            "exec_all_ranks": unique_reply_rank is None or exec_all_ranks,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)

            # Only rank 0 has a result_mq, so we always expect exactly 1 response
            num_responses = 1

            responses = []
            for _ in range(num_responses):
                response = self._dequeue_one_with_failure_polling(deadline, method)

                try:
                    unpack_diffusion_output_shm(response)
                except Exception as e:
                    logger.warning("SHM unpack failed (data may already be inline): %s", e)

                # Check if response indicates an error
                if isinstance(response, dict) and response.get("status") == "error":
                    raise RuntimeError(
                        f"Worker failed with error '{response.get('error')}', "
                        "please check the stack trace above for the root cause"
                    )

                responses.append(response)

            return responses[0] if unique_reply_rank is not None else responses
        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        if self.is_failed:
            raise EngineDeadError()
        self._ensure_open()
        for p in self._processes:
            if not p.is_alive():
                self.is_failed = True
                raise EngineDeadError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        try:
            self._finalizer()
        finally:
            self._broadcast_mq = None
            self._result_mq = None
            self.resources = None
            self._processes = []
