# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import os
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext

import torch
import zmq
from vllm.config import LoadConfig, VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
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
        self.connector = None
        self.device = None

        self.init_device_and_model()

        # Initialize OmniConnector after vllm_config is available (via init_device_and_model)
        self._init_omni_connector()

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

        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        # hack
        vllm_config = VllmConfig()
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        self.vllm_config = vllm_config
        with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=self.od_config):
            init_distributed_environment(world_size=world_size, rank=rank)
            logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")
            parallel_config = self.od_config.parallel_config
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            )

            load_config = LoadConfig()
            model_loader = DiffusersPipelineLoader(load_config)
            time_before_load = time.perf_counter()
            with self._maybe_get_memory_pool_context(tag="weights"):
                with DeviceMemoryProfiler() as m:
                    self.pipeline = model_loader.load_model(
                        od_config=self.od_config,
                        load_device=str(self.device),
                    )
            time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info(f"Worker {self.rank}: Model loaded successfully.")

        if not self.od_config.enforce_eager:
            try:
                self.pipeline.transformer = regionally_compile(
                    self.pipeline.transformer,
                    dynamic=True,
                )
                logger.info(f"Worker {self.rank}: Model compiled with torch.compile.")
            except Exception as e:
                logger.warning(f"Worker {self.rank}: torch.compile failed with error: {e}. Using eager mode.")

        # Setup cache backend based on type (both backends use enable()/reset() interface)
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Generate output for the given requests.

        Args:
            requests: List of diffusion requests

        Returns:
            DiffusionOutput with generated results
        """
        return self.execute_model(requests, self.od_config)

    @torch.inference_mode()
    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        if not reqs or len(reqs) == 0:
            raise ValueError("Cannot execute model with empty request list")
        # TODO: dealing with first req for now
        req = reqs[0]

        # [Omni] KV Cache Receiving Logic
        if getattr(req, "need_kv_receive", False) and self.connector is not None:
            self._receive_kv_cache_for_request(req)

        if req.generator is None and req.seed is not None:
            req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # Refresh cache context if needed
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.num_inference_steps)
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            output = self.pipeline.forward(req)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.pipeline.load_weights(weights)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache.
                Currently only support level 1.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def shutdown(self) -> None:
        destroy_distributed_env()

    def _init_omni_connector(self) -> None:
        # TODO(wzliu)! get real connector from yaml file instead of hardcode
        """Initialize OmniConnector for KV cache transfer."""
        try:
            from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
            from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

            connector_config = None

            # 1. Try to get from omni_kv_config (injected from YAML)
            # Use self.od_config because self.vllm_config is a dummy VllmConfig without model_config
            if self.od_config.omni_kv_config:
                connector_config = self.od_config.omni_kv_config.get("connector_config")

            if not connector_config:
                logger.warning("No OmniConnector config found, skipping initialization")
                return

            logger.info(f"Initializing OmniConnector with config: {connector_config}")

            c_type = connector_config.get("type")
            if not c_type:
                logger.error("Connector config missing 'type'")
                return

            c_extra = {k: v for k, v in connector_config.items() if k != "type"}
            connector_spec = ConnectorSpec(name=c_type, extra=c_extra)

            self.connector = OmniConnectorFactory.create_connector(connector_spec)

        except Exception as e:
            logger.error(f"Failed to initialize OmniConnector: {e}")
            import traceback

            traceback.print_exc()

    def _receive_kv_cache_for_request(self, req: OmniDiffusionRequest) -> None:
        """Receive KV cache for a request via OmniConnector."""
        # TODO(wzliu)! must get control info from stage queue instead of hardcode
        if not req.request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return

        try:
            logger.info(f"Attempting to receive KV cache for request {req.request_id}")

            # TODO: Key used for transfer (must match sender side)
            # key = f"kv_cache_{req.request_id}"

            # Get data from connector
            # Determine from_stage and to_stage dynamically
            omni_kv_config = self.od_config.omni_kv_config
            stage_id = omni_kv_config.get("stage_id")
            engine_input_source = omni_kv_config.get("engine_input_source", [])

            to_stage = stage_id
            # Default to stage_id - 1 if input source is not explicit
            if engine_input_source:
                from_stage = engine_input_source[0]
            elif isinstance(stage_id, int):
                from_stage = stage_id - 1
            else:
                raise ValueError("Invalid stage id")
            logger.info(f"Wait for KV cache for request {req.request_id} from stage {from_stage} to {to_stage}...")

            # Check if we should receive KV cache based on config
            need_recv_cache = omni_kv_config.get("need_recv_cache", False)
            if need_recv_cache:
                while True:
                    result = self.connector.get(
                        from_stage=from_stage,
                        to_stage=to_stage,
                        request_id=f"kv_cache_{req.request_id}",
                    )
                    if result:
                        break
                    # loop forever for testing
                    time.sleep(0.5)
            else:
                logger.info(f"Skip receiving KV cache for {req.request_id} (need_recv_cache=False)")
                result = None

            if result:
                data, size = result
                logger.info(f"Successfully received KV cache for {req.request_id}")

                # Assume data structure matches KVCacheTransferData.to_dict()
                if isinstance(data, dict) and "layer_blocks" in data:
                    # Get layer blocks and ensure they are on the correct device
                    layer_blocks = data["layer_blocks"]

                    # Move tensors to GPU if needed (OmniSerializer should handle tensor reconstruction)
                    for cache_list in [layer_blocks["key_cache"], layer_blocks["value_cache"]]:
                        for i, tensor in enumerate(cache_list):
                            if isinstance(tensor, torch.Tensor) and tensor.device != self.pipeline.device:
                                cache_list[i] = tensor.to(self.pipeline.device).contiguous()
                    from types import SimpleNamespace

                    req.past_key_values = SimpleNamespace(**layer_blocks)

                if "metadata" in data:
                    req.kv_metadata = data["metadata"]

            else:
                logger.warning(f"No KV cache received for {req.request_id} (timeout or empty)")

        except Exception as e:
            logger.error(f"Error receiving KV cache for {req.request_id}: {e}")
            import traceback

            traceback.print_exc()


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
        self.worker = self._create_worker(gpu_id, od_config)
        self.gpu_id = gpu_id
        self._running = True

    def _create_worker(self, gpu_id: int, od_config: OmniDiffusionConfig) -> GPUWorker:
        """Create a worker instance. Override in subclasses for different worker types."""
        return GPUWorker(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )

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

    def execute_rpc(self, rpc_request: dict) -> tuple[object | None, bool]:
        """Execute an RPC request and indicate whether to reply."""

        method = rpc_request["method"]
        args = rpc_request.get("args", ())
        kwargs = rpc_request.get("kwargs", {})
        output_rank = rpc_request.get("output_rank")
        exec_all_ranks = rpc_request.get("exec_all_ranks", False)

        should_execute = exec_all_ranks or output_rank is None or output_rank == self.gpu_id
        should_reply = (output_rank is None or output_rank == self.gpu_id) and self.result_mq is not None

        if not should_execute:
            return None, False

        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
                result = func(*args, **kwargs)
            else:
                result = method(self.worker, *args, **kwargs)
            return result, should_reply
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}, should_reply

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

            if msg is None or len(msg) == 0:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            # Route message based on type
            if isinstance(msg, dict) and msg.get("type") == "rpc":
                # Handle RPC request
                try:
                    result, should_reply = self.execute_rpc(msg)
                    if should_reply:
                        self.return_result(result)
                except Exception as e:
                    logger.error(f"Error processing RPC: {e}", exc_info=True)
                    if self.result_mq is not None:
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
