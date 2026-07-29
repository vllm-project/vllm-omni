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
    result_mq: MessageQueue | None = None
    num_workers: int = 0
    processes: list[mp.Process] | None = None

    def __call__(self) -> None:
        """Clean up background resources."""
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
        processes, result_handles = self._launch_workers(broadcast_handle, self.wake_events)
        self._result_mqs = self._init_result_queues(result_handles)
        self._result_mq = self._result_mqs[0] if self._result_mqs else None
        self._processes = processes

        shutdown_cleaner = _ExecutorShutdownCleaner(
            broadcast_mq=self._broadcast_mq,
            result_mq=self._result_mqs[0] if self._result_mqs else None,
            num_workers=num_workers,
            processes=self._processes,
        )
        self._shutdown_cleaner: _ExecutorShutdownCleaner | None = shutdown_cleaner
        self._finalizer = weakref.finalize(self, shutdown_cleaner)

        self._start_worker_monitor()

    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        return MessageQueue(
            n_reader=num_workers,
            n_local_reader=num_workers,
            local_reader_ranks=list(range(num_workers)),
        )

    def _init_result_queues(self, result_handles: list) -> list[MessageQueue]:
        """Create one reader per worker result queue."""
        queues: list[MessageQueue] = []
        for i, handle in enumerate(result_handles):
            if handle is None:
                logger.error(f"Failed to get result queue handle from worker {i}")
                queues.append(None)  # type: ignore
            else:
                queues.append(MessageQueue.create_from_handle(handle, 0))
        return queues

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if not hasattr(self, "_result_mqs") or not self._result_mqs:
            raise RuntimeError("Result queues not initialized")
        if self._broadcast_mq is None:
            raise RuntimeError("Broadcast queue is closed")

    def _dequeue_one_with_failure_polling(self, deadline: float | None, method: str) -> Any:
        """Block until one result message, polling ``_is_failed`` between chunk timeouts.

        When multiple result queues exist (one per worker), polls all of them
        round-robin to collect responses from any worker.
        """
        if hasattr(self, "_result_mqs") and self._result_mqs:
            mqs = [mq for mq in self._result_mqs if mq is not None]
        else:
            mqs = [self._result_mq] if self._result_mq else []

        if not mqs:
            raise RuntimeError("No result queue available")

        while True:
            if deadline is None:
                chunk_timeout = _DEQUEUE_TIMEOUT_S
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RPC call to {method} timed out.")
                chunk_timeout = min(_DEQUEUE_TIMEOUT_S, remaining)

            per_q_timeout = max(0.05, chunk_timeout / max(len(mqs), 1))
            start_idx = getattr(self, "_next_queue_idx", 0)
            for offset in range(len(mqs)):
                idx = (start_idx + offset) % len(mqs)
                try:
                    result = mqs[idx].dequeue(timeout=per_q_timeout)
                    self._next_queue_idx = (idx + 1) % len(mqs)
                    return result
                except (TimeoutError, zmq.error.Again):
                    continue

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
    ) -> tuple[list[mp.Process], list[Handle | None]]:
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
        result_handles: list = []
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

            result_handles.append(data.get("result_handle"))

            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handles

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

        For dist_offload_dp with multiple scheduled requests, sends ALL
        requests in a single RPC.  Each worker picks one based on its DP
        rank (AllGather only gathers weight shards, so ranks compute
        different requests in parallel).  Returns a BatchRunnerOutput with
        one RunnerOutput per scheduled request.

        All concurrent requests must have the same number of inference
        steps, because AllGather is a collective that requires every rank
        to participate at each step.  Mismatched step counts cause one
        rank to exit early while others hang.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        new_reqs = scheduler_output.scheduled_new_reqs
        runner_outputs: list[RunnerOutput] = []

        if len(new_reqs) > 1:
            # AllGather is a collective — every rank must participate at each
            # step.  If concurrent requests resolve to different step counts
            # (e.g. T2V=35 vs action_mode=30), one rank exits early and the
            # other hangs.  Reject this here.
            #
            # num_inference_steps=None means "pipeline default", which may
            # resolve differently per request mode (e.g. Cosmos3 action_mode
            # resolves to 30 while T2V resolves to 35).  Since
            # RequestBatchSamplingParamsKey does not include extra_args
            # (which carries action_mode), the scheduler cannot guarantee
            # that all-None requests share a mode.  Reject all-None to be
            # safe — callers must specify num_inference_steps explicitly
            # when using DP multi-concurrency.
            step_counts = {
                nr.req.sampling_params.num_inference_steps
                for nr in new_reqs
                if nr.req.sampling_params.num_inference_steps is not None
            }
            has_none = any(nr.req.sampling_params.num_inference_steps is None for nr in new_reqs)
            if (len(step_counts) > 1) or has_none:
                raise ValueError(
                    "DP multi-concurrency requires all concurrent requests to have "
                    "the same explicit num_inference_steps (None is not allowed "
                    "because it may resolve differently per request mode), got "
                    f"{[nr.req.sampling_params.num_inference_steps for nr in new_reqs]}. "
                    "AllGather is a collective that requires every rank to participate "
                    "at each step."
                )

            # 2. Validate action_mode: different modes execute different forward schedules
            action_modes = set()
            for nr in new_reqs:
                ea = getattr(nr.req, "extra_args", None)
                if ea and isinstance(ea, dict):
                    action_modes.add(ea.get("action_mode"))
            if len(action_modes) > 1:
                raise ValueError(
                    "DP multi-concurrency requires all concurrent requests to "
                    f"share the same action_mode, got {action_modes}. Different "
                    "modes execute different forward schedules, causing AllGather "
                    "deadlock."
                )

        if len(new_reqs) > 1:
            # DP multi-concurrency: send all requests in one broadcast RPC.
            # Each rank picks req[rank % len(reqs)] and computes independently.
            # All ranks reply via shared result_mq (unique_reply_rank=None),
            # executor collects N responses — no gather, no OOM.
            reqs_list = [nr.req for nr in new_reqs]
            try:
                results = self.collective_rpc(
                    "execute_model",
                    args=(reqs_list, self.od_config, scheduler_output.kv_prefetch_job),
                    unique_reply_rank=None,
                    exec_all_ranks=True,
                )
                # results is a list of N DiffusionOutputs (one per rank)
                results = results if isinstance(results, list) else [results]
                for i, new_req in enumerate(new_reqs):
                    res = results[i] if i < len(results) else results[0]
                    if not isinstance(res, DiffusionOutput):
                        raise RuntimeError(f"Unexpected response type [{i}]: {type(res)!r}")
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=res,
                        )
                    )
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
        else:
            # Single request — original path
            for new_req in new_reqs:
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

        When dp_concurrent is active, always use execute_request instead —
        each rank picks one request from the list and computes independently.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

        self._ensure_open()
        if len(scheduler_output.scheduled_new_reqs) <= 1:
            return self.execute_request(scheduler_output)

        if getattr(self.od_config, "enable_distributed_layerwise_offload", False):
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

        # Prepare RPC request message. When unique_reply_rank is None:
        # - All workers execute the RPC
        # - All workers reply via shared result_mq (each rank has result_mq)
        # - Executor collects N responses (one per rank)
        # When unique_reply_rank is set (e.g. 0):
        # - All workers execute (if exec_all_ranks=True)
        # - Only the specified rank replies
        # - Executor collects 1 response
        execute_all_ranks = unique_reply_rank is None or exec_all_ranks
        # For DP multi-concurrency (unique_reply_rank=None, exec_all_ranks=True),
        # we want all ranks to reply independently — set output_rank to None
        # so should_reply is True for all ranks.
        if unique_reply_rank is None and exec_all_ranks:
            output_rank_for_rpc = None  # all ranks reply
            collect_rank_status = False
        else:
            output_rank_for_rpc = unique_reply_rank if unique_reply_rank is not None else 0
            collect_rank_status = unique_reply_rank is None
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": output_rank_for_rpc,
            "exec_all_ranks": execute_all_ranks,
            "collect_rank_status": collect_rank_status,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)  # pyright: ignore[reportOptionalMemberAccess] MQ is not None before shutdown

            # Determine number of responses to collect:
            # - unique_reply_rank=None + exec_all_ranks=True: all DP ranks reply
            #   (N responses, one per DP worker).  Use data_parallel_size, not
            #   num_gpus, because num_gpus includes TP/SP ranks that should not
            #   reply in DP multi-concurrency mode.
            # - Otherwise: 1 response (only rank 0 or specified rank)
            if unique_reply_rank is None and exec_all_ranks:
                dp_size = getattr(self.od_config.parallel_config, "data_parallel_size", 1)
                num_responses = max(1, dp_size)
            else:
                num_responses = 1

            responses = []
            if unique_reply_rank is None and exec_all_ranks and num_responses > 1:
                # DP multi-concurrency: only dp_size primary ranks (one per
                # DP replica) reply, but there may be more queues (world_size
                # with SP/TP).  Use round-robin polling to collect exactly
                # num_responses replies, then sort by dp_rank tag to match
                # results to requests.
                tagged: list[tuple[int, Any]] = []
                collected_errors: list[str] = []
                for _ in range(num_responses):
                    response = self._dequeue_one_with_failure_polling(deadline, method)
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
            self._result_mqs = []
            self._result_mq = None
            self._shutdown_cleaner = None
            self._processes = []
