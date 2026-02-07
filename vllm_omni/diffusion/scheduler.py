# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from threading import Lock

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_context = getattr(self, "context", None)
        if existing_context is not None and not existing_context.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self.context = zmq.Context()  # Standard synchronous context

        # Initialize single MessageQueue for all message types (generation & RPC)
        # Assuming all readers are local for now as per current launch_engine implementation
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )

        self.result_mq = None
        self._result_lock = Lock()
        self._result_cv = threading.Condition(self._result_lock)
        self._pending_results: dict[str, DiffusionOutput] = {}
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = False
        self._reader_error: Exception | None = None

    def initialize_result_queue(self, handle):
        # Initialize MessageQueue for receiving results
        # We act as rank 0 reader for this queue
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized result MessageQueue")
        self._stop_reader = False
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._result_reader_loop,
            name="diffusion-result-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _result_reader_loop(self):
        while not self._stop_reader:
            try:
                message = self.result_mq.dequeue(timeout=0.1, indefinite=False)
            except TimeoutError:
                continue
            except Exception as e:
                with self._result_cv:
                    self._reader_error = RuntimeError(f"Result reader failed: {e}")
                    self._result_cv.notify_all()
                return

            if isinstance(message, dict) and message.get("status") == "error":
                with self._result_cv:
                    self._reader_error = RuntimeError("worker error")
                    self._result_cv.notify_all()
                return

            if isinstance(message, dict) and message.get("type") == "generation_result":
                key = message.get("request_key")
                output = message.get("output")
                if not isinstance(key, str) or not isinstance(output, DiffusionOutput):
                    with self._result_cv:
                        self._reader_error = RuntimeError("Invalid generation result from worker")
                        self._result_cv.notify_all()
                    return
                with self._result_cv:
                    self._pending_results[key] = output
                    self._result_cv.notify_all()
                continue

            if isinstance(message, DiffusionOutput):
                # Legacy fallback: generation results should carry request_key.
                key = message.request_key
                if not key:
                    with self._result_cv:
                        self._reader_error = RuntimeError("Legacy DiffusionOutput missing request_key")
                        self._result_cv.notify_all()
                    return
                with self._result_cv:
                    self._pending_results[key] = message
                    self._result_cv.notify_all()
                continue

            with self._result_cv:
                self._reader_error = RuntimeError("Unexpected response type from worker")
                self._result_cv.notify_all()
            return

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Sends a request to the scheduler and waits for the response."""
        try:
            request_key = request.request_key

            if self.result_mq is None:
                raise RuntimeError("Result queue not initialized")

            # Broadcast generation request to all workers.
            self.mq.enqueue(request)

            with self._result_cv:
                while True:
                    if self._reader_error is not None:
                        raise self._reader_error
                    if request_key in self._pending_results:
                        return self._pending_results.pop(request_key)
                    self._result_cv.wait(timeout=0.1)
        except zmq.error.Again:
            logger.error("Timeout waiting for response from scheduler.")
            raise TimeoutError("Scheduler did not respond in time.")

    def close(self):
        """Closes the socket and terminates the context."""
        self._stop_reader = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if hasattr(self, "context"):
            self.context.term()
        self.context = None
        self.mq = None
        self.result_mq = None
        self._pending_results = {}
        self._reader_error = None
