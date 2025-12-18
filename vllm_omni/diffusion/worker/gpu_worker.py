# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import cloudpickle
import multiprocessing as mp
import os
import pickle
import time

import torch
import zmq
from vllm.config import LoadConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import (
    SHUTDOWN_MESSAGE,
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class GPUWorker:
    """
    A worker that executes the model on a single GPU.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.pipeline = None

        self.init_device_and_model()

    def init_device_and_model(self) -> None:
        """Initialize the device and load the model."""
        world_size = self.od_config.num_gpus
        rank = self.rank
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        # hack
        vllm_config = VllmConfig()
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.num_gpus
        set_current_vllm_config(vllm_config)

        init_distributed_environment(world_size=world_size, rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")

        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config)
        time_before_load = time.perf_counter()
        with DeviceMemoryProfiler() as m:
            self.pipeline = model_loader.load_model(
                od_config=self.od_config,
                load_device=f"cuda:{rank}",
            )
        time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info(f"Worker {self.rank}: Model loaded successfully.")

        # Setup cache backend based on type (both backends use enable()/reset() interface)
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)

    def print_message(self, message: str) -> str:
        """
        Print a message from the worker.
        
        Args:
            message: The message to print
            
        Returns:
            A confirmation string with the worker rank
        """
        print(f"[Worker {self.rank}] {message}", flush=True)
        logger.info(f"[Worker {self.rank}] {message}")
        return f"Worker {self.rank} printed: {message}"

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Generate output for the given requests.
        
        Args:
            requests: List of diffusion requests
            
        Returns:
            DiffusionOutput with generated results
        """
        return self.execute_model(requests, self.od_config)

    def do_shutdown(self) -> str:
        """
        Shutdown the worker gracefully.
        
        Returns:
            Confirmation message
        """
        self.shutdown()
        return f"Worker {self.rank} shutdown complete"

    @torch.inference_mode()
    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        # TODO: dealing with first req for now
        req = reqs[0]

        # Refresh cache context if needed
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.num_inference_steps)

        output = self.pipeline.forward(req)
        return output

    def shutdown(self) -> None:
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
                logger.info("Worker %s: Destroyed process group", self.rank)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Worker %s: Failed to destroy process group: %s", self.rank, exc)


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        broadcast_handle,
    ):
        self.od_config = od_config

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)

        # Initialize MessageQueue reader from handle (unified for generation & RPC)
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        self.result_mq = None
        self.result_mq_handle = None

        # Setup result sender (only for rank 0 for now, or whoever needs to reply)
        # Assuming only rank 0 replies to scheduler as per original logic
        if gpu_id == 0:
            # Create MessageQueue for results (1 writer -> 1 reader)
            # We assume the reader (SyncScheduler) will act as rank 0
            self.result_mq = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
            self.result_mq_handle = self.result_mq.export_handle()
            logger.info(f"Worker {gpu_id} created result MessageQueue")

        assert od_config.master_port is not None
        worker = GPUWorker(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )
        self.worker = worker
        self.gpu_id = gpu_id
        self._running = True

    def return_result(self, output: DiffusionOutput):
        """
        replies to client, only on rank 0
        """
        if self.result_mq is not None:
            self.result_mq.enqueue(output)

    def recv_message(self):
        """
        Receive unified messages (RPC requests, shutdown) from broadcast queue.
        Uses indefinite=True to block until a message arrives.
        """
        return self.mq.dequeue(indefinite=True)

    def execute_rpc(self, rpc_request: dict):
        """Execute an RPC request and return the result."""
        try:
            method = rpc_request["method"]
            args = rpc_request.get("args", ())
            kwargs = rpc_request.get("kwargs", {})
            output_rank = rpc_request.get("output_rank")

            # Only execute if we should reply (either output_rank is None or matches our rank)
            if output_rank is not None and output_rank != self.gpu_id:
                return None

            # Deserialize method if it's a callable
            if isinstance(method, bytes):
                method = cloudpickle.loads(method)

            # Execute the method
            if isinstance(method, str):
                # Method is a string, call it on the worker
                func = getattr(self.worker, method)
                result = func(*args, **kwargs)
            else:
                # Method is a callable
                result = method(self.worker, *args, **kwargs)

            return result
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    # TODO: queueing, cancellation
    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers"""

        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            # Receive unified message (generation request, RPC request, or shutdown)
            msg = None
            try:
                msg = self.recv_message()
            except Exception as e:
                logger.error(
                    f"Error receiving message in worker loop: {e}",
                    exc_info=True,
                )
                continue

            if msg is None:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            # Route message based on type
            if isinstance(msg, dict) and msg.get("type") == "rpc":
                # Handle RPC request
                try:
                    result = self.execute_rpc(msg)
                    if result is not None and self.gpu_id == 0:
                        self.return_result(result)
                except Exception as e:
                    logger.error(f"Error processing RPC: {e}", exc_info=True)
                    if self.gpu_id == 0:
                        self.return_result({"status": "error", "error": str(e)})

            elif isinstance(msg, dict) and msg.get("type") == "shutdown":
                # Handle shutdown message
                logger.info("Worker %s: Received shutdown message", self.gpu_id)
                self._running = False
                continue

            else:
                # Handle generation request (OmniDiffusionRequest list)
                try:
                    output = self.worker.execute_model(msg, self.od_config)
                except Exception as e:
                    logger.error(
                        f"Error executing forward in event loop: {e}",
                        exc_info=True,
                    )
                    output = DiffusionOutput(error=str(e))

                try:
                    self.return_result(output)
                except zmq.ZMQError as e:
                    # Reply failed; log and keep loop alive to accept future requests
                    logger.error(f"ZMQ error sending reply: {e}")
                    continue

        logger.info("event loop terminated.")
        try:
            self.worker.shutdown()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Worker %s: Shutdown encountered an error: %s", self.gpu_id, exc)
        # if self.result_sender is not None:
        #     self.result_sender.close()
        self.context.term()

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
    ) -> None:
        """Worker initialization and execution loops."""

        worker_proc = WorkerProc(
            od_config,
            gpu_id=rank,
            broadcast_handle=broadcast_handle,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
                "result_handle": worker_proc.result_mq_handle if rank == 0 else None,
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")
