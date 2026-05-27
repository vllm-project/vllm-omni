# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray-based distributed executor for diffusion models."""

import os
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)

PG_WAIT_TIMEOUT = 1800
INIT_WORKER_TIMEOUT = 600
EXECUTE_MODEL_TIMEOUT = 600

try:
    import ray
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError:
    ray = None  # type: ignore
    ActorHandle = None
    PlacementGroup = None
    PlacementGroupSchedulingStrategy = None


@dataclass
class RayWorkerMetaData:
    """Metadata for a Ray worker (rank is assigned after sorting)."""

    worker: ActorHandle
    rank: int = -1
    ip: str = ""


@dataclass
class BackgroundResources:
    """Used as a weakref.finalize callback for clean shutdown."""

    workers: list[RayWorkerMetaData] | None = None

    def __call__(self):
        if not self.workers:
            return

        # Graceful shutdown
        graceful_futures = []
        for meta in self.workers:
            try:
                graceful_futures.append(meta.worker.shutdown.remote())
            except Exception:
                pass  # actor may already be dead
        if graceful_futures:
            try:
                ray.get(graceful_futures, timeout=30)
            except Exception:
                logger.warning("Some workers did not shut down gracefully, force-killing")

        for meta in self.workers:
            try:
                ray.kill(meta.worker)
            except Exception as e:
                logger.warning(f"Failed to kill worker rank {meta.rank}: {e}")
        self.workers.clear()


class RayDiffusionWorkerWrapper:
    """Ray actor wrapper that lazily initializes a DiffusionWorker."""

    def __init__(self, rpc_rank: int):
        self.rpc_rank = rpc_rank
        self.worker = None
        self.od_config = None

    def get_node_ip(self) -> str:
        return get_ip()

    def get_open_port(self) -> str:
        return str(get_open_port())

    def update_environment_variables(self, env_vars: dict[str, str]) -> None:
        for k, v in env_vars.items():
            if k in os.environ and os.environ[k] != v:
                logger.warning(f"Overwriting environment variable {k} from '{os.environ[k]}' to '{v}'")
            os.environ[k] = v

    def init_worker(self, od_config: OmniDiffusionConfig) -> None:
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

        from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

        self.rpc_rank = int(os.environ["RANK"])
        wrapper = WorkerWrapperBase(
            gpu_id=int(os.environ["LOCAL_RANK"]),
            od_config=od_config,
            worker_extension_cls=od_config.worker_extension_cls,
            custom_pipeline_args=getattr(od_config, "custom_pipeline_args", None),
            rank=self.rpc_rank,
        )
        self.worker = wrapper.worker
        self.od_config = od_config

    def execute_model(self, request: OmniDiffusionRequest) -> DiffusionOutput | None:
        if self.worker is None:
            raise RuntimeError("Worker is not initialized")
        output = self.worker.execute_model(request, self.od_config)
        # Only rank 0 returns outputs to avoid redundant device-to-host copies and Ray transfers
        if self.rpc_rank == 0:
            return output.to_cpu()
        return None

    def _move_result_to_cpu(self, result: Any) -> Any:
        if isinstance(result, DiffusionOutput):
            return result.to_cpu()

        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        if isinstance(result, RunnerOutput):
            if isinstance(result.result, DiffusionOutput):
                result.result = result.result.to_cpu()
            return result

        if isinstance(result, BatchRunnerOutput):
            for runner_output in result.runner_outputs:
                if isinstance(runner_output.result, DiffusionOutput):
                    runner_output.result = runner_output.result.to_cpu()
            return result

        return result

    def execute_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        output_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        kwargs = kwargs or {}
        should_execute = exec_all_ranks or output_rank is None or output_rank == self.rpc_rank
        should_reply = output_rank is None or output_rank == self.rpc_rank

        if not should_execute:
            return None

        result = self.execute_method(method, *args, **kwargs)
        if not should_reply:
            return None
        return self._move_result_to_cpu(result)

    def execute_method(self, method: str, *args, **kwargs) -> Any:
        if self.worker is None:
            raise RuntimeError("Worker is not initialized")
        try:
            func = getattr(self.worker, method)
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error executing method {method!r}")
            raise e

    def check_alive(self) -> bool:
        return self.worker is not None

    def shutdown(self) -> None:
        if self.worker is not None:
            try:
                self.worker.shutdown()
            except Exception as e:
                logger.warning(f"Rank {self.rpc_rank}: Error during shutdown: {e}")
            self.worker = None


class RayDiffusionExecutor(DiffusionExecutor):
    def _init_executor(self) -> None:
        if ray is None:
            raise ImportError("Ray is required for the 'ray' distributed executor backend.")

        self._closed = False
        self.workers: list[RayWorkerMetaData] = []

        if not ray.is_initialized():
            ray_address = getattr(self.od_config, "ray_address", None)
            if ray_address:
                logger.info(f"Connecting to Ray cluster at {ray_address}")
                ray.init(address=ray_address)
            else:
                logger.info("Initializing local Ray instance")
                ray.init()

        # Finalizer must exist before _init_workers_ray so shutdown()
        # works if init fails partway through.
        self._resources = BackgroundResources(workers=self.workers)
        self._finalizer = weakref.finalize(self, self._resources)

        placement_group = self._create_placement_group()
        self._init_workers_ray(placement_group)

    def _create_placement_group(self) -> "PlacementGroup":
        num_gpus = self.od_config.num_gpus

        current_pg = ray.util.get_current_placement_group()
        if current_pg is not None:
            logger.info("Using existing placement group")
            return current_pg

        bundles = [{"GPU": 1} for _ in range(num_gpus)]
        placement_group = ray.util.placement_group(bundles, strategy="PACK")
        logger.info(f"Waiting for placement group with {num_gpus} GPU bundles...")
        try:
            ray.get(placement_group.ready(), timeout=PG_WAIT_TIMEOUT)
        except ray.exceptions.GetTimeoutError:
            raise ValueError(
                f"Cannot create placement group with {num_gpus} GPUs within "
                f"{PG_WAIT_TIMEOUT}s. Check available resources with `ray status`."
            ) from None
        return placement_group

    def _init_workers_ray(self, placement_group: "PlacementGroup") -> None:
        num_gpus = self.od_config.num_gpus
        driver_ip = get_ip()

        worker_cls = ray.remote(
            num_cpus=0,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            ),
        )(RayDiffusionWorkerWrapper)

        worker_metadata: list[RayWorkerMetaData] = []
        for rank in range(num_gpus):
            actor = worker_cls.remote(rpc_rank=rank)
            worker_metadata.append(RayWorkerMetaData(worker=actor))

        worker_ips = ray.get([w.worker.get_node_ip.remote() for w in worker_metadata])
        for meta, ip in zip(worker_metadata, worker_ips):
            meta.ip = ip

        ip_counts: dict[str, int] = defaultdict(int)
        for meta in worker_metadata:
            ip_counts[meta.ip] += 1

        def sort_key(meta: RayWorkerMetaData):
            # Driver node first, then nodes with fewer workers, then by IP
            return (0 if meta.ip == driver_ip else 1, ip_counts[meta.ip], meta.ip)

        sorted_metadata = sorted(worker_metadata, key=sort_key)
        for i, meta in enumerate(sorted_metadata):
            meta.rank = i
        self.workers = sorted_metadata
        self._resources.workers = self.workers

        unique_ips = set(meta.ip for meta in self.workers)
        master_addr = "127.0.0.1" if len(unique_ips) == 1 else self.workers[0].ip
        master_port = ray.get(self.workers[0].worker.get_open_port.remote())

        env_futures = []
        for meta in self.workers:
            env_vars = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "RANK": str(meta.rank),
                # Ray remaps CUDA_VISIBLE_DEVICES per actor, so local device is always 0
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(num_gpus),
            }
            env_futures.append(meta.worker.update_environment_variables.remote(env_vars))
        ray.get(env_futures)

        init_futures = []
        for meta in self.workers:
            future = meta.worker.init_worker.remote(od_config=self.od_config)
            init_futures.append(future)

        try:
            ray.get(init_futures, timeout=INIT_WORKER_TIMEOUT)
            logger.info(f"All {len(self.workers)} workers initialized successfully")
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            self.shutdown()
            raise

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        # All workers must execute (distributed computation), but only rank 0
        # returns the full output — others return None to avoid redundant
        # device-to-host copies and object transfers.
        futures = [meta.worker.execute_model.remote(request) for meta in self.workers]
        try:
            rank0_result = ray.get(futures[0], timeout=EXECUTE_MODEL_TIMEOUT)
            # Wait for remaining workers to finish their side of the
            # distributed computation, but discard their None results.
            ray.get(futures[1:], timeout=EXECUTE_MODEL_TIMEOUT)
            if not isinstance(rank0_result, DiffusionOutput):
                raise RuntimeError(f"Unexpected response type for generate: {type(rank0_result)!r}")
            return rank0_result
        except ray.exceptions.RayTaskError as e:
            logger.error(f"Worker execution failed: {e}")
            raise RuntimeError(f"Diffusion generation failed: {e}") from e
        except ray.exceptions.GetTimeoutError as e:
            logger.error("Worker execution timed out")
            raise TimeoutError("Diffusion generation timed out") from e

    def execute_request(self, scheduler_output: "DiffusionSchedulerOutput") -> "BaseRunnerOutput":
        """Adapt request-mode scheduler output to worker execute_model RPC."""
        from vllm_omni.diffusion.worker.utils import RunnerOutput

        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")
        if scheduler_output.num_scheduled_reqs != 1:
            raise ValueError(
                f"Request mode currently supports batch_size=1, "
                f"but got {scheduler_output.num_scheduled_reqs} scheduled requests."
            )

        new_req = scheduler_output.scheduled_new_reqs[0]
        result = self.collective_rpc(
            "execute_model",
            args=(new_req.req, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, DiffusionOutput):
            raise RuntimeError(f"Unexpected response type for execute_request: {type(result)!r}")

        return RunnerOutput(
            req_id=new_req.sched_req_id,
            step_index=None,
            finished=True,
            result=result,
        )

    def execute_step(self, scheduler_output: "DiffusionSchedulerOutput") -> "BaseRunnerOutput":
        """Forward step-mode scheduler output to worker execute_stepwise RPC."""
        from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, RunnerOutput

        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")
        result = self.collective_rpc(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

        if isinstance(result, BaseRunnerOutput):
            return result
        # TODO: Remove this fallback with MultiprocDiffusionExecutor's matching
        # compatibility path; DiffusionOutput cannot represent failed batches.
        if isinstance(result, DiffusionOutput):
            req_id = scheduler_output.scheduled_req_ids[0] if scheduler_output.scheduled_req_ids else ""
            return RunnerOutput(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=result,
            )
        else:
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
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        kwargs = kwargs or {}
        output_rank = unique_reply_rank
        futures = [
            meta.worker.execute_rpc.remote(
                method,
                args,
                kwargs,
                output_rank,
                output_rank is None or exec_all_ranks,
            )
            for meta in self.workers
        ]

        try:
            responses = ray.get(futures, timeout=timeout)
            if unique_reply_rank is not None:
                return responses[unique_reply_rank]
            return responses
        except ray.exceptions.GetTimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except ray.exceptions.RayTaskError as e:
            logger.error(f"RPC call failed: {e}")
            raise RuntimeError(f"RPC call to {method} failed: {e}") from e

    def check_health(self) -> None:
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        for meta in self.workers:
            try:
                alive = ray.get(meta.worker.check_alive.remote(), timeout=10)
                if not alive:
                    raise RuntimeError(f"Worker rank {meta.rank} is not healthy")
            except Exception as e:
                raise RuntimeError(f"Worker rank {meta.rank} health check failed: {e}") from e

    def shutdown(self) -> None:
        self._closed = True
        self._finalizer()
