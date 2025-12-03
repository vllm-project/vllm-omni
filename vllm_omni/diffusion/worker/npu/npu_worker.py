# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import os

import torch
import zmq
from transformers import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.registry import initialize_model
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class NPUWorker:
    """
    A worker that executes the model on a single NPU.
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

        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(device)

        # hack
        # set hf_config to a fake one to avolid get attr error
        class _FakePretrainedConfig(PretrainedConfig):
            def __getattr__(self, name):
                return "fake"

        vllm_config = VllmConfig(model_config=ModelConfig(hf_config=_FakePretrainedConfig()))
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.num_gpus
        set_current_vllm_config(vllm_config)

        init_distributed_environment(world_size=world_size, rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        with device:
            with set_default_torch_dtype(self.od_config.dtype):
                self.pipeline = initialize_model(self.od_config)
                self.pipeline.load_weights()
                self.pipeline.eval()
        logger.info(f"Worker {self.rank}: Initialized device, model, and distributed environment.")
        logger.info(f"Worker {self.rank}: Model loaded successfully.")

    @torch.inference_mode()
    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        # TODO: dealing with first req for now
        req = reqs[0]
        output = self.pipeline.forward(req)
        return output


class NPUWorkerProc:
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

        # Initialize MessageQueue reader from handle
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
        worker = NPUWorker(
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

    def recv_reqs(self):
        """
        Receive requests from broadcast queue
        """
        return self.mq.dequeue(indefinite=True)

    # TODO: queueing, cancellation
    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers"""

        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            reqs = None
            # 1: receive requests
            try:
                reqs = self.recv_reqs()
            except Exception as e:
                logger.error(
                    f"Error receiving requests in scheduler event loop: {e}",
                    exc_info=True,
                )
                continue

            # 2: execute, make sure a reply is always sent
            try:
                output = self.worker.execute_model(reqs, self.od_config)
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

        worker_proc = NPUWorkerProc(
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
