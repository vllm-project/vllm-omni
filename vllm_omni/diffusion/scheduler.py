# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        self._pending_results: dict[str, DiffusionOutput] = {}

    def initialize_result_queue(self, handle):
        # Initialize MessageQueue for receiving results
        # We act as rank 0 reader for this queue
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized result MessageQueue")

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

            while True:
                with self._result_lock:
                    if request_key in self._pending_results:
                        return self._pending_results.pop(request_key)

                    message = self.result_mq.dequeue()

                    if isinstance(message, dict) and message.get("status") == "error":
                        raise RuntimeError("worker error")

                    if isinstance(message, dict) and message.get("type") == "generation_result":
                        key = message.get("request_key")
                        output = message.get("output")
                        if not isinstance(output, DiffusionOutput):
                            raise RuntimeError("Invalid generation result from worker")
                        if key == request_key:
                            return output
                        if isinstance(key, str):
                            self._pending_results[key] = output
                            continue
                        raise RuntimeError("Generation result missing request key")

                    if isinstance(message, DiffusionOutput):
                        return message
                    raise RuntimeError("Unexpected response type from worker")
        except zmq.error.Again:
            logger.error("Timeout waiting for response from scheduler.")
            raise TimeoutError("Scheduler did not respond in time.")

    def close(self):
        """Closes the socket and terminates the context."""
        if hasattr(self, "context"):
            self.context.term()
        self.context = None
        self.mq = None
        self.result_mq = None
        self._pending_results = {}
