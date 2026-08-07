from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import multiprocessing.connection
import queue
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

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
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

    # Class-level defaults so tests using object.__new__ (without _init_executor)
    # don't hit AttributeError when collective_rpc accesses these.
    _rpc_wave_id: int = 0

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self._is_failed = False
        self._failure_callbacks: list[Callable[[], None]] = []
        self._result_mq: MessageQueue | None = None
        self._rpc_wave_id: int = 0

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

        # Async output: result pump thread for dispatching AsyncDiffusionOutput
        # messages in request-mode (step_execution=False).
        self._rpc_id_counter = 0
        self._rpc_id_lock = threading.Lock()
        self._rpc_futures: dict[str, concurrent.futures.Future[AsyncDiffusionOutput]] = {}
        self._output_futures: dict[str, concurrent.futures.Future[DiffusionOutput]] = {}
        self._completed_outputs: dict[str, concurrent.futures.Future[DiffusionOutput]] = {}
        self._batch_split_map: dict[str, dict[str, str]] = {}  # batch_id -> {per_req_id: request_id}
        self._futures_lock = threading.RLock()
        self._pump_running = False
        self._pump_stop = threading.Event()
        # When pump is active it is the sole reader of result_mq; non-async
        # messages are placed here for collective_rpc() to consume.
        self._sync_result_buffer: queue.Queue = queue.Queue()
        if not self.od_config.step_execution:
            self._start_result_pump()

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
        """Block until one result message, polling ``_is_failed`` between chunk timeouts.

        When async output is enabled, pump is the sole reader of result_mq;
        non-async messages are placed in _sync_result_buffer, so this method
        reads from there instead of result_mq directly.
        """
        while True:
            if deadline is None:
                chunk_timeout = _DEQUEUE_TIMEOUT_S
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RPC call to {method} timed out.")
                chunk_timeout = min(_DEQUEUE_TIMEOUT_S, remaining)
            if not self.od_config.step_execution:
                try:
                    return self._sync_result_buffer.get(timeout=chunk_timeout)
                except queue.Empty:
                    if self._is_failed or self._closed:
                        raise EngineDeadError()
                    continue
            else:
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

    _MAX_STALE_DISCARDS = 16

    def _validate_wave_id(self, response: Any, expected_wave_id: int, deadline: float | None, method: str) -> Any:
        """Discard stale RPC responses from a previous wave."""
        discards = 0
        while discards < self._MAX_STALE_DISCARDS:
            if not isinstance(response, dict):
                return response
            resp_wave_id = response.get("wave_id")
            if resp_wave_id is None or resp_wave_id == expected_wave_id:
                return response
            logger.warning(
                "Discarding stale RPC response (wave_id=%s, expected=%s, method=%s)",
                resp_wave_id,
                expected_wave_id,
                method,
            )
            discards += 1
            response = self._dequeue_one_with_failure_polling(deadline, method)
        raise TimeoutError(
            f"Discarded {self._MAX_STALE_DISCARDS} stale RPC responses "
            f"without finding wave_id={expected_wave_id} for method={method}."
        )

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
        In async mode the result may carry async_output_id instead of the final output.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        new_reqs = scheduler_output.scheduled_new_reqs
        runner_outputs: list[RunnerOutput] = []

        # DP multi-concurrency: when DLO+AllGather is active and multiple
        # requests are scheduled, send ALL requests in one broadcast RPC.
        # Each rank picks req[rank % len(reqs)] and computes independently.
        # All ranks reply (unique_reply_rank=None) so we collect dp_size
        # responses and match by dp_rank.
        if (
            len(new_reqs) > 1
            and getattr(self.od_config, "enable_distributed_layerwise_offload", False)
            and getattr(self.od_config, "dlo_use_allgather", True)
        ):
            # Validate: all concurrent requests must share the same
            # num_inference_steps and identical extra_args, because
            # AllGather is a collective that requires every rank to
            # participate at each step.
            step_counts = {
                nr.req.sampling_params.num_inference_steps
                for nr in new_reqs
                if nr.req.sampling_params.num_inference_steps is not None
            }
            has_none = any(nr.req.sampling_params.num_inference_steps is None for nr in new_reqs)
            if (len(step_counts) > 1) or has_none:
                raise ValueError(
                    "DP multi-concurrency requires all concurrent requests to have "
                    "the same explicit num_inference_steps (None is not allowed), got "
                    f"{[nr.req.sampling_params.num_inference_steps for nr in new_reqs]}."
                )
            extra_args_signatures: set = set()
            for nr in new_reqs:
                ea = getattr(nr.req.sampling_params, "extra_args", None)
                extra_args_signatures.add(json.dumps(ea, sort_keys=True, default=repr) if ea is not None else None)
            if len(extra_args_signatures) > 1:
                raise ValueError(
                    "DP multi-concurrency requires all concurrent requests to "
                    "share identical extra_args. Different extra_args can change "
                    "the forward schedule and cause AllGather deadlock."
                )

            reqs_list = [nr.req for nr in new_reqs]
            try:
                results = self.collective_rpc(
                    "execute_model",
                    args=(reqs_list, self.od_config, scheduler_output.kv_prefetch_job),
                    unique_reply_rank=None,
                    exec_all_ranks=True,
                )
                results = results if isinstance(results, list) else [results]
                for i, new_req in enumerate(new_reqs):
                    res = results[i] if i < len(results) else results[0]
                    if isinstance(res, DiffusionOutput):
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=res,
                            )
                        )
                    else:
                        raise RuntimeError(f"Unexpected response type [{i}]: {type(res)!r}")
            except Exception as exc:
                for new_req in new_reqs:
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=DiffusionOutput(error=str(exc)),
                        )
                    )
            return BatchRunnerOutput.from_list(runner_outputs)

        for new_req in new_reqs:
            req = new_req.req
            try:
                result = self.collective_rpc(
                    "execute_model",
                    args=(req, self.od_config, scheduler_output.kv_prefetch_job),
                    unique_reply_rank=0,
                    exec_all_ranks=True,
                )
                if isinstance(result, AsyncDiffusionOutput) and result.kind == AsyncOutputKind.COMPUTE_DONE:
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=None,
                            async_output_id=result.async_output_id,
                        )
                    )
                elif isinstance(result, DiffusionOutput):
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=result,
                        )
                    )
                else:
                    raise RuntimeError(f"Unexpected response type: {type(result)!r}")
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

        When async output is enabled, per-request async_output_ids are
        propagated through RunnerOutput so the engine waits in
        step_streaming() instead of blocking the busy loop here.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        if len(scheduler_output.scheduled_new_reqs) <= 1:
            return self.execute_request(scheduler_output)

        parallel_config = getattr(self.od_config, "parallel_config", None)
        dp_size = getattr(parallel_config, "data_parallel_size", 1)
        if (
            dp_size > 1
            and getattr(self.od_config, "enable_distributed_layerwise_offload", False)
            and getattr(self.od_config, "dlo_use_allgather", True)
        ):
            # DLO DP uses one independent request per DP replica.  It is not a
            # fused pipeline request batch, so models such as MiniMax-H3 do not
            # need to advertise supports_request_batch=True.
            return self.execute_request(scheduler_output)

        result = self.collective_rpc(
            "execute_model_batch",
            args=(scheduler_output, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if isinstance(result, AsyncDiffusionOutput) and result.kind == AsyncOutputKind.COMPUTE_DONE:
            # Propagate async_output_id to per-request RunnerOutputs so the
            # engine waits in step_streaming() instead of blocking here.
            batch_id = result.async_output_id
            per_req_map: dict[str, str] = {}
            runner_outputs: list[RunnerOutput] = []
            for new_req in scheduler_output.scheduled_new_reqs:
                per_req_id = f"{batch_id}/{new_req.request_id}"
                per_req_map[per_req_id] = new_req.request_id
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=None,
                        async_output_id=per_req_id,
                    )
                )
            with self._futures_lock:
                self._batch_split_map[batch_id] = per_req_map
            return BatchRunnerOutput.from_list(runner_outputs)
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

        # ── Path 1: async execute_model / execute_model_batch ──
        # Request-mode methods use the compute_done/output_ready split
        # so the D2H copy runs on a side stream in the worker background
        # thread while the default stream is free for the next forward.
        # pump routes the result back to a Future via rpc_id.
        multi_rank_reply = unique_reply_rank is None and exec_all_ranks
        if (
            not self.od_config.step_execution
            and method in ("execute_model", "execute_model_batch")
            and not multi_rank_reply
        ):
            rpc_id = self._next_rpc_id()
            rpc_request["rpc_id"] = rpc_id
            fut: concurrent.futures.Future = concurrent.futures.Future()
            with self._futures_lock:
                self._rpc_futures[rpc_id] = fut
            self._broadcast_mq.enqueue(rpc_request)  # pyright: ignore[reportOptionalMemberAccess] MQ is not None before shutdown
            try:
                return fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                with self._futures_lock:
                    self._rpc_futures.pop(rpc_id, None)
                raise TimeoutError(f"RPC call to {method} timed out.")

        # ── Path 2: step-execution or other non-execute_model RPCs ──
        # _dequeue_one_with_failure_polling reads from _sync_result_buffer
        # when pump is active (request-mode), or result_mq directly otherwise.
        self._rpc_wave_id += 1
        wave_id = self._rpc_wave_id
        rpc_request["wave_id"] = wave_id

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)  # pyright: ignore[reportOptionalMemberAccess] MQ is not None before shutdown

            # Determine number of responses to collect:
            # - unique_reply_rank=None + exec_all_ranks=True: all DP ranks reply
            #   (N responses, one per DP worker).
            # - Otherwise: 1 response (only rank 0 or specified rank)
            if unique_reply_rank is None and exec_all_ranks:
                dp_size = getattr(self.od_config.parallel_config, "data_parallel_size", 1)
                num_responses = max(1, dp_size)
            else:
                num_responses = 1

            responses: list = []
            if unique_reply_rank is None and exec_all_ranks and num_responses > 1:
                # DP multi-concurrency: collect num_responses replies, sort by dp_rank.
                tagged: list[tuple[int, Any]] = []
                collected_errors: list[str] = []
                for _ in range(num_responses):
                    response = self._dequeue_one_with_failure_polling(deadline, method)
                    response = self._validate_wave_id(response, wave_id, deadline, method)
                    try:
                        unpack_diffusion_output_shm(response)
                    except Exception as e:
                        logger.warning("SHM unpack failed (data may already be inline): %s", e)
                    if isinstance(response, dict) and response.get("status") == "error":
                        collected_errors.append(str(response.get("error", "unknown")))
                    else:
                        response = MultiprocDiffusionExecutor._handle_rpc_response(response)
                        if isinstance(response, dict) and "dp_rank" in response:
                            tagged.append((response["dp_rank"], response["output"]))
                        else:
                            tagged.append((len(tagged), response))
                if collected_errors:
                    raise RuntimeError(f"Worker error: {collected_errors[0]}")
                tagged.sort(key=lambda x: x[0])
                responses = [r for _, r in tagged]
            else:
                for _ in range(num_responses):
                    response = self._dequeue_one_with_failure_polling(deadline, method)
                    response = self._validate_wave_id(response, wave_id, deadline, method)

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

    # ------------------------------------------------------------------
    # Async output: result pump
    # ------------------------------------------------------------------

    def _start_result_pump(self) -> None:
        self._pump_running = True
        self._pump_stop.clear()
        self._result_pump_thread = threading.Thread(target=self._result_pump, daemon=True, name="DiffusionResultPump")
        self._result_pump_thread.start()
        logger.info("Async result pump started")

    def _result_pump(self) -> None:
        """Sole reader of result_mq when async output is enabled.

        Dispatches AsyncDiffusionOutput messages to the appropriate future:
        * RPC_RESULT / COMPUTE_DONE → _rpc_futures[rpc_id]
        * OUTPUT_READY → _output_futures[async_output_id]
        """
        while not self._pump_stop.is_set():
            try:
                msg = self._result_mq.dequeue(timeout=1.0)
            except TimeoutError:
                if self._is_failed:
                    break
                continue
            except Exception:
                logger.exception("Result pump dequeue failed")
                if self._is_failed:
                    break
                continue

            if not isinstance(msg, AsyncDiffusionOutput):
                # Non-async message: place into the sync buffer for
                # collective_rpc() to consume via Path 2.
                self._sync_result_buffer.put(msg)
                continue

            if msg.kind in (AsyncOutputKind.RPC_RESULT, AsyncOutputKind.COMPUTE_DONE):
                with self._futures_lock:
                    fut = self._rpc_futures.pop(msg.rpc_id, None) if msg.rpc_id else None
                if fut is not None and not fut.done():
                    if msg.error:
                        fut.set_exception(RuntimeError(msg.error))
                    else:
                        fut.set_result(msg)
            elif msg.kind == AsyncOutputKind.OUTPUT_READY:
                batch_id = msg.async_output_id
                with self._futures_lock:
                    per_req_map = self._batch_split_map.pop(batch_id, None) if batch_id else None
                if per_req_map is not None:
                    # Batch result: split into per-request DiffusionOutputs.
                    try:
                        unpack_diffusion_output_shm(msg.output)
                    except Exception:
                        logger.exception("SHM unpack failed for batch %s", batch_id)
                    batch_output = msg.output
                    for per_req_id, req_id in per_req_map.items():
                        req_output = batch_output.get_request_output(req_id)
                        per_req_result: DiffusionOutput
                        if req_output is not None and req_output.result is not None:
                            per_req_result = req_output.result
                        elif msg.error:
                            per_req_result = DiffusionOutput(error=msg.error)
                        else:
                            per_req_result = DiffusionOutput(error="No output result for batch request")
                        fut: concurrent.futures.Future = concurrent.futures.Future()
                        fut.set_result(per_req_result)
                        with self._futures_lock:
                            pending = self._output_futures.pop(per_req_id, None)
                            if pending is not None and not pending.done():
                                pending.set_result(per_req_result)
                            else:
                                self._completed_outputs[per_req_id] = fut
                else:
                    with self._futures_lock:
                        fut = self._output_futures.pop(batch_id, None) if batch_id else None
                    if fut is not None and not fut.done():
                        if msg.error:
                            fut.set_exception(RuntimeError(msg.error))
                        else:
                            try:
                                unpack_diffusion_output_shm(msg.output)
                            except Exception as e:
                                logger.exception("SHM unpack failed in result pump")
                                fut.set_exception(e)
                                continue
                            fut.set_result(msg.output)
                    elif batch_id:
                        fut = concurrent.futures.Future()
                        if msg.error:
                            fut.set_exception(RuntimeError(msg.error))
                        else:
                            try:
                                unpack_diffusion_output_shm(msg.output)
                            except Exception as e:
                                logger.exception("SHM unpack failed in result pump (cached)")
                                fut.set_exception(e)
                            else:
                                fut.set_result(msg.output)
                        with self._futures_lock:
                            self._completed_outputs[batch_id] = fut

    def _next_rpc_id(self) -> str:
        with self._rpc_id_lock:
            self._rpc_id_counter += 1
            return str(self._rpc_id_counter)

    def wait_output_ready(self, async_output_id: str) -> concurrent.futures.Future[DiffusionOutput]:
        """Return a Future that resolves when the async output is ready."""
        with self._futures_lock:
            cached = self._completed_outputs.pop(async_output_id, None)
            if cached is not None:
                return cached
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self._output_futures[async_output_id] = fut
        return fut

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
        self._pump_stop.set()
        try:
            self._finalizer()
        finally:
            self._broadcast_mq = None
            self._result_mq = None
            with self._futures_lock:
                for fut in self._rpc_futures.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError("Executor shut down"))
                for fut in self._output_futures.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError("Executor shut down"))
                self._rpc_futures.clear()
                self._output_futures.clear()
                self._batch_split_map.clear()
            self._shutdown_cleaner = None
            self._processes = []
