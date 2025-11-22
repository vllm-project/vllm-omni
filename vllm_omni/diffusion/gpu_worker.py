# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
import os
from typing import List
import zmq
from .utils import get_zmq_socket

import torch
from setproctitle import setproctitle

from .req import OmniDiffusionRequest
from .utils import is_port_available
from .data import PortArgs, OutputBatch
from .model import TestModel
from vllm_omni.diffusion.ditrubuted.parallel_state import init_worker_distributed_environment

from vllm.logger import init_logger
from vllm_omni.diffusion.data import EngineArgs

logger = init_logger(__name__)

CYAN = "\033[1;36m"
RESET = "\033[0;0m"


class GPUWorker:
    """
    A worker that executes the model on a single GPU.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        master_port: int,
        engine_args: EngineArgs,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.master_port = master_port
        # FIXME: should we use tcp as distribute init method?
        self.engine_args = engine_args
        self.pipeline = None

        self.init_device_and_model()

    def init_device_and_model(self) -> None:
        """Initialize the device and load the model."""
        setproctitle(f"sgl_diffusion::scheduler:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.engine_args.num_gpus)

        init_worker_distributed_environment(world_size=self.engine_args.num_gpus, rank=self.rank, local_rank=self.local_rank)

        # self.pipeline = build_pipeline(self.engine_args)
        self.pipeline = TestModel(3, 6)
        logger.info(
            f"Worker {self.rank}: Initialized device, model, and distributed environment."
        )
        print(f"{CYAN}Worker {self.rank}: Model loaded successfully.{RESET}")

    @torch.inference_mode()
    def execute_model(
        self, batch: List[OmniDiffusionRequest], engine_args: EngineArgs
    ) -> OutputBatch:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        # TODO: dealing with first req for now
        req = batch[0]
        output_batch = self.pipeline.forward(req, engine_args)
        print("output_batch ", output_batch)
        return output_batch


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        engine_args: EngineArgs,
        gpu_id: int,
        port_args: PortArgs,
    ):
        self.engine_args = engine_args
        self.port_args = port_args

        # set_global_server_args(server_args=server_args)

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)
        endpoint = engine_args.scheduler_endpoint()
        if gpu_id == 0:
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.REP, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None
        assert port_args.master_port is not None
        worker = GPUWorker(
            local_rank=gpu_id,
            master_port=port_args.master_port,
            rank=gpu_id,
            engine_args=engine_args,
        )
        self.worker = worker
        self.gpu_id = gpu_id
        self._running = True

    def return_result(self, output_batch: OutputBatch):
        """
        replies to client, only on rank 0
        """
        if self.receiver is not None:
            self.receiver.send_pyobj(output_batch)

    def recv_reqs(self):
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            recv_reqs = self.receiver.recv_pyobj()
            assert isinstance(recv_reqs, list)
        else:
            recv_reqs = None

        assert recv_reqs is not None

        return recv_reqs

    # TODO: queueing, cancellation
    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers"""

        logger.info(
            f"Rank 0 scheduler listening on tcp://*:{self.engine_args.scheduler_port}"
        )

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
                output_batch = self.worker.execute_model(reqs, self.engine_args)
            except Exception as e:
                logger.error(
                    f"Error executing forward in scheduler event loop: {e}",
                    exc_info=True,
                )
                output_batch = OutputBatch(error=str(e))

            try:
                self.return_result(output_batch)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        logger.info("Scheduler event loop terminated.")
        if self.receiver is not None:
            self.receiver.close()
        self.context.term()

    @staticmethod
    def worker_main(
        rank: int,
        engine_args: EngineArgs,
        pipe_writer: mp.connection.Connection,
    ) -> None:
        """Worker initialization and execution loops."""

        port_args = PortArgs.from_engine_args(engine_args)

        worker_proc = WorkerProc(
            engine_args,
            gpu_id=rank,
            port_args=port_args,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")
