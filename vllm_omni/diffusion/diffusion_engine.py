# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import time
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import PIL.Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, OmniDiffusionConfig
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler, scheduler
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    Create a BackgroundResources instance to encapsulate all background resources
    (e.g., the scheduler and worker processes) that need explicit cleanup.
    This object holds references to external system resources that are not managed
    by Python's garbage collector (like OS processes, message queues, etc.),
    so they must be cleaned up manually to avoid resource leaks or zombie processes.
    """

    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if scheduler is not None:
            try:
                for _ in range(scheduler.num_workers):
                    scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
                scheduler.close()
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)
        for proc in self.processes:
            if not proc.is_alive():
                continue
            proc.join(30)
            if proc.is_alive():
                logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                proc.terminate()
                proc.join(30)


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
        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

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

            if output.output is None:
                logger.warning("Output is None, returning empty OmniRequestOutput")
                # Return empty output for the first request
                if len(requests) > 0:
                    request = requests[0]
                    request_id = request.request_id or ""
                    prompt = request.prompt
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else None
                    return OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics={},
                        latents=None,
                    )
                return None

            postprocess_start_time = time.time()
            images = self.post_process_func(output.output) if self.post_process_func is not None else output.output
            postprocess_time = time.time() - postprocess_start_time
            logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

            # Convert to OmniRequestOutput format
            # Ensure images is a list
            if not isinstance(images, list):
                images = [images] if images is not None else []

            # Handle single request or multiple requests
            if len(requests) == 1:
                # Single request: return single OmniRequestOutput
                request = requests[0]
                request_id = request.request_id or ""
                prompt = request.prompt
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else None

                metrics = {}
                if output.trajectory_timesteps is not None:
                    metrics["trajectory_timesteps"] = output.trajectory_timesteps

                return OmniRequestOutput.from_diffusion(
                    request_id=request_id,
                    images=images,
                    prompt=prompt,
                    metrics=metrics,
                    latents=output.trajectory_latents,
                )
            else:
                # Multiple requests: return list of OmniRequestOutput
                # Split images based on num_outputs_per_prompt for each request
                results = []
                image_idx = 0

                for request in requests:
                    request_id = request.request_id or ""
                    prompt = request.prompt
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else None

                    # Get images for this request
                    num_outputs = request.num_outputs_per_prompt
                    request_images = images[image_idx : image_idx + num_outputs] if image_idx < len(images) else []
                    image_idx += num_outputs

                    metrics = {}
                    if output.trajectory_timesteps is not None:
                        metrics["trajectory_timesteps"] = output.trajectory_timesteps

                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_images,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                        )
                    )

                return results
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

        self.resources = BackgroundResources(scheduler=scheduler, processes=self._processes)
        # Use weakref.finalize instead of __del__ or relying on self.close() at shutdown.
        # During interpreter shutdown, global state (e.g., modules, built-ins) may already
        # be cleared (set to None), so calling normal cleanup methods can fail with
        # AttributeError: 'NoneType' object has no attribute '...'.
        # weakref.finalize schedules cleanup *before* such destruction begins,
        # ensuring resources are released while the runtime environment is still intact.
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

    def add_req_and_wait_for_response(self, requests: list[OmniDiffusionRequest]):
        return scheduler.add_req(requests)

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        prompt = "dummy run"
        # note that num_inference_steps=1 will cause timestep and temb None in the pipeline
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it

            dummy_image = PIL.Image.new("RGB", (width, height), color=(0, 0, 0))
        else:
            dummy_image = None
        req = OmniDiffusionRequest(
            prompt=prompt,
            height=height,
            width=width,
            pil_image=dummy_image,
            num_inference_steps=num_inference_steps,
            num_outputs_per_prompt=1,
        )
        logger.info("dummy run to warm up the model")
        requests = self.pre_process_func([req]) if self.pre_process_func is not None else [req]
        self.add_req_and_wait_for_response(requests)

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) or callable to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        if self._closed:
            raise RuntimeError("DiffusionEngine is closed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        assert isinstance(method, str)
        send_method = method

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": send_method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            scheduler.mq.enqueue(rpc_request)

            # Determine which workers we expect responses from
            num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

            responses = []
            for _ in range(num_responses):
                dequeue_timeout = None if deadline is None else (deadline - time.monotonic())
                try:
                    if scheduler.result_mq is None:
                        raise RuntimeError("Result queue not initialized")

                    response = scheduler.result_mq.dequeue(timeout=dequeue_timeout)

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

    def close(self) -> None:
        self._finalizer()

    def abort(self, request_id: str | Iterable[str]) -> None:
        # TODO implement it
        logger.warning("DiffusionEngine abort is not implemented yet")
        pass
