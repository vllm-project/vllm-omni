# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import time

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, OmniDiffusionConfig
from vllm_omni.diffusion.registry import get_diffusion_post_process_func, get_diffusion_pre_process_func
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import scheduler
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(self, od_config: OmniDiffusionConfig):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        self._processes: list[mp.Process] = []
        self._closed = False
        self._make_client()

    def step(self, requests: list[OmniDiffusionRequest]):
        try:
            # Apply pre-processing if available
            if self.pre_process_func is not None:
                preprocess_start_time = time.time()
                requests = self.pre_process_func(requests)
                preprocess_time = time.time() - preprocess_start_time
                logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

            output = self.add_req_and_wait_for_response(requests)
            if output.error:
                raise Exception(f"{output.error}")
            logger.info("Generation completed successfully.")

            postprocess_start_time = time.time()
            result = self.post_process_func(output.output) if self.post_process_func is not None else output.output
            postprocess_time = time.time() - postprocess_start_time
            logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

            return result
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    @staticmethod
    def make_engine(config: OmniDiffusionConfig) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config)

    def _make_client(self):
        # TODO rename it
        scheduler.initialize(self.od_config)

        # Get the broadcast handle from the initialized scheduler
        broadcast_handle = scheduler.get_broadcast_handle()

        processes, result_handle = self._launch_workers(
            broadcast_handle=broadcast_handle,
        )

        if result_handle is not None:
            scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")

        self._processes = processes

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

    def add_req_and_wait_for_response(self, requests: list[OmniDiffusionRequest]):
        return scheduler.add_req(requests)

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        prompt = "dummy run"
        num_inference_steps = 1
        height = 1024
        width = 1024
        req = OmniDiffusionRequest(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_outputs_per_prompt=1,
        )
        logger.info("dummy run to warm up the model")
        self.add_req_and_wait_for_response([req])

    def close(self, *, timeout_s: float = 30.0) -> None:
        if self._closed:
            return
        self._closed = True

        # Send shutdown signal to worker processes via broadcast queue
        try:
            if getattr(scheduler, "mq", None) is not None:
                for _ in range(self.od_config.num_gpus or 1):
                    scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to send shutdown signal: %s", exc)

        # Join all worker processes, terminate if they refuse to exit
        for proc in self._processes:
            if not proc.is_alive():
                continue
            proc.join(timeout_s)
            if proc.is_alive():
                logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                proc.terminate()
                proc.join(timeout_s)

        scheduler.close()
        self._processes = []

    def __del__(self):  # pragma: no cover - best effort cleanup
        self.close()
