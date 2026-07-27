from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Any, cast

import zmq
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.ipc import DIFFUSION_RPC_RESULT_ENVELOPE, unpack_diffusion_output_shm
from vllm_omni.diffusion.worker import WorkerProc

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)

_DEQUEUE_TIMEOUT_S = 5.0


@dataclass
class _ExecutorShutdownCleaner:
    """Finalizer that shuts down executor worker processes."""

    broadcast_mq: MessageQueue | None = None
    num_workers: int = 0
    processes: list[mp.Process] | None = None

    def __call__(self) -> None:
        """Clean up background resources."""
        if self.broadcast_mq is not None:
            try:
                for _ in range(self.num_workers):
                    self.broadcast_mq.enqueue(SHUTDOWN_MESSAGE, timeout=1.0)

                self.broadcast_mq = None
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)

        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(5)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(5)


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self._is_failed = False
        self._failure_callbacks: list[Callable[[], None]] = []
        self._result_mq: MessageQueue | None = None

        num_workers = cast(int, self.od_config.num_gpus)
        self.wake_events = [mp.Event() for _ in range(num_workers)]

        self._broadcast_mq = self._init_broadcast_queue(num_workers)
        broadcast_handle = self._broadcast_mq.export_handle()

        # Launch workers
        processes, result_handle = self._launch_workers(broadcast_handle, self.wake_events)
        self._processes = processes

        shutdown_cleaner = _ExecutorShutdownCleaner(
            broadcast_mq=self._broadcast_mq,
            num_workers=num_workers,
            processes=self._processes,
        )
        self._shutdown_cleaner: _ExecutorShutdownCleaner | None = shutdown_cleaner
        self._finalizer = weakref.finalize(self, shutdown_cleaner)

        try:
            self._result_mq = self._init_result_queue(result_handle)
        except Exception:
            self.shutdown()
            raise

        self._start_worker_monitor()

    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        return MessageQueue(
            n_reader=num_workers,
            n_local_reader=num_workers,
            local_reader_ranks=list(range(num_workers)),
        )

    def _init_result_queue(self, result_handle: Handle | None) -> MessageQueue:
        if result_handle is None:
            raise RuntimeError("Failed to get result queue handle from workers")
        return MessageQueue.create_from_handle(result_handle, 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if self._result_mq is None:
            raise RuntimeError("Result queue is closed")
        if self._broadcast_mq is None:
            raise RuntimeError("Broadcast queue is closed")

    def _dequeue_one_with_failure_polling(self, deadline: float | None, method: str) -> Any:
        """Block until one result message, polling ``_is_failed`` between chunk timeouts."""
        while True:
            if deadline is None:
                chunk_timeout = _DEQUEUE_TIMEOUT_S
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RPC call to {method} timed out.")
                chunk_timeout = min(_DEQUEUE_TIMEOUT_S, remaining)
            try:
                return self._result_mq.dequeue(timeout=chunk_timeout)  # pyright: ignore[reportOptionalMemberAccess] MQ is not None before shutdown
            except (TimeoutError, zmq.error.Again):
                if self._is_failed:
                    raise EngineDeadError()
                continue

    @staticmethod
    def _raise_for_rpc_error_dict(response: Any) -> None:
        if isinstance(response, dict) and response.get("status") == "error":
            raise RuntimeError(
                f"Worker failed with error '{response.get('error')}', "
                "please check the stack trace above for the root cause"
            )

    @staticmethod
    def _unwrap_rpc_result_envelope(response: Any) -> Any:
        if not (isinstance(response, dict) and response.get("type") == DIFFUSION_RPC_RESULT_ENVELOPE):
            return response

        rank_statuses = response.get("rank_statuses") or []
        failed = [status for status in rank_statuses if not status.get("ok", False)]
        if failed:
            details = "; ".join(
                f"rank {status.get('rank')}: {status.get('error_type') or 'Error'}: {status.get('error')}"
                for status in failed
            )
            tracebacks = "\n\n".join(
                f"rank {status.get('rank')} traceback:\n{status['traceback']}"
                for status in failed
                if status.get("traceback")
            )
            if tracebacks:
                details = f"{details}\n\n{tracebacks}"
            method = response.get("method", "<unknown>")
            raise RuntimeError(f"RPC '{method}' failed on worker rank(s): {details}")

        result = response.get("result")
        if isinstance(result, bool):
            # Only bool-returning RPCs participate in the all-rank AND.
            # Non-bool results leave bool_result unset and are ignored here.
            bool_results = [
                status.get("bool_result") for status in rank_statuses if status.get("bool_result") is not None
            ]
            if bool_results and not all(bool_results):
                return False
        return result

    @staticmethod
    def _handle_rpc_response(response: Any) -> Any:
        MultiprocDiffusionExecutor._raise_for_rpc_error_dict(response)
        response = MultiprocDiffusionExecutor._unwrap_rpc_result_envelope(response)
        # After unwrapping, a worker method result may itself be the same
        # {"status": "error"} shape produced by worker_busy_loop transport
        # failures. Preserve the pre-envelope error handling for that case.
        MultiprocDiffusionExecutor._raise_for_rpc_error_dict(response)
        return response

    def _launch_workers(
        self,
        broadcast_handle: Handle,
        wake_events: list[Event],
    ) -> tuple[list[mp.Process], Handle | None]:
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = cast(int, od_config.num_gpus)
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

            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handle

    @property
    def is_dead(self) -> bool:
        """Whether the executor is shut down or a worker has failed fatally."""
        return self._closed or self._is_failed

    def _start_worker_monitor(self) -> None:
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

            dead = [p for p in self._processes if p.sentinel in finished]
            if dead:
                details = []
                for p in dead:
                    code = p.exitcode
                    # Negative exitcode == killed by signal N (-9 = SIGKILL/OOM,
                    # -11 = SIGSEGV). Surface this so callers don't only see
                    # "died unexpectedly" with no root cause.
                    if code is not None and code < 0:
                        try:
                            import signal as _signal

                            sig = _signal.Signals(-code).name
                        except (ValueError, ImportError):
                            sig = f"signal {-code}"
                        details.append(f"{p.name}(exitcode={code}, {sig})")
                    else:
                        details.append(f"{p.name}(exitcode={code})")
                logger.error(
                    "Diffusion worker(s) died unexpectedly: %s",
                    details,
                )
                self._is_failed = True

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

    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Adapt request-mode scheduler output to worker execute_model RPCs.

        Returns a BatchRunnerOutput with one RunnerOutput per scheduled request.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        runner_outputs: list[RunnerOutput] = []

        for new_req in scheduler_output.scheduled_new_reqs:
            req = new_req.req
            try:
                result = self.collective_rpc(
                    "execute_model",
                    args=(req, self.od_config, scheduler_output.kv_prefetch_job),
                    unique_reply_rank=0,
                    exec_all_ranks=True,
                )
                if not isinstance(result, DiffusionOutput):
                    raise RuntimeError(f"Unexpected response type: {type(result)!r}")
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=result,
                    )
                )
            except Exception as exc:
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )
                )

        return BatchRunnerOutput.from_list(runner_outputs)

    def execute_batch(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute request-mode work through the unified request-batch path.

        A scheduler wave with one request is the conservative serial case and
        uses the single-request worker RPC. Waves with multiple requests use the
        fused request-batch RPC and require pipeline request-batch support.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

        self._ensure_open()
        if len(scheduler_output.scheduled_new_reqs) <= 1:
            return self.execute_request(scheduler_output)

        result = self.collective_rpc(
            "execute_model_batch",
            args=(scheduler_output, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, BatchRunnerOutput):
            raise RuntimeError(f"Unexpected response type for execute_batch: {type(result)!r}")
        return result

    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Forward step-mode scheduler output to worker execute_stepwise RPC."""
        from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

        self._ensure_open()
        result = self.collective_rpc(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

        if isinstance(result, BaseRunnerOutput):
            return result
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

        # Prepare RPC request message. When unique_reply_rank is None, all
        # workers must execute the RPC but only rank 0 can reply (it's the
        # only one with a result_mq). Collect detailed rank statuses only for
        # this control-plane all-rank path; forward-path exec_all_ranks RPCs
        # avoid the per-step host object gather.
        execute_all_ranks = unique_reply_rank is None or exec_all_ranks
        collect_rank_status = unique_reply_rank is None
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank if unique_reply_rank is not None else 0,
            "exec_all_ranks": execute_all_ranks,
            "collect_rank_status": collect_rank_status,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)  # pyright: ignore[reportOptionalMemberAccess] MQ is not None before shutdown

            # Only rank 0 has a result_mq, so we always expect exactly 1 response
            num_responses = 1

            responses = []
            for _ in range(num_responses):
                response = self._dequeue_one_with_failure_polling(deadline, method)

                try:
                    unpack_diffusion_output_shm(response)
                except Exception as e:
                    logger.warning("SHM unpack failed (data may already be inline): %s", e)

                response = MultiprocDiffusionExecutor._handle_rpc_response(response)

                responses.append(response)

            return responses[0] if unique_reply_rank is not None else responses
        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        if self._is_failed:
            raise EngineDeadError()
        self._ensure_open()
        for p in self._processes:
            if not p.is_alive():
                self._is_failed = True
                raise EngineDeadError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        try:
            self._finalizer()
        finally:
            self._broadcast_mq = None
            self._result_mq = None
            self._shutdown_cleaner = None
            self._processes = []
