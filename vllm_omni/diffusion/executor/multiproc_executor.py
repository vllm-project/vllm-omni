import multiprocessing as mp
import time
import weakref
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.scheduler is not None:
            try:
                for _ in range(self.scheduler.num_workers):
                    self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
                self.scheduler.close()
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

        # Initialize scheduler
        self.scheduler = Scheduler()
        self.scheduler.initialize(self.od_config)
        broadcast_handle = self.scheduler.get_broadcast_handle()

        # Launch workers
        processes, result_handle = self._launch_workers(broadcast_handle)

        if result_handle is not None:
            self.scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")

        self._processes = processes

        self.resources = BackgroundResources(scheduler=self.scheduler, processes=self._processes)
        self._finalizer = weakref.finalize(self, self.resources)

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Get the appropriate worker class for current device
        worker_proc = get_diffusion_worker_class()

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = mp.Process(
                target=worker_proc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
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

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        return self.scheduler.add_req(request)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self.scheduler.mq.enqueue(rpc_request)

            # Determine which workers we expect responses from
            num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

            responses = []
            for _ in range(num_responses):
                dequeue_timeout = None if deadline is None else (deadline - time.monotonic())
                try:
                    if self.scheduler.result_mq is None:
                        raise RuntimeError("Result queue not initialized")

                    response = self.scheduler.result_mq.dequeue(timeout=dequeue_timeout)

                    # Check if response indicates an error
                    if isinstance(response, dict) and response.get("status") == "error":
                        raise RuntimeError(
                            f"Worker failed with error '{response.get('error')}', "
                            "please check the stack trace above for the root cause"
                        )

                    responses.append(response)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e

            return responses[0] if unique_reply_rank is not None else responses

        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        self._finalizer()
