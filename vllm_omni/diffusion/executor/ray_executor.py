# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray-based distributed executor for diffusion models."""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

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


class RayDiffusionWorkerWrapper:
    """Ray actor wrapper that lazily initializes a DiffusionWorker."""

    def __init__(self, rpc_rank: int):
        self.rpc_rank = rpc_rank
        self.worker = None
        self.od_config = None

    def get_node_ip(self) -> str:
        return get_ip()

    def update_environment_variables(self, env_vars: dict[str, str]) -> None:
        for k, v in env_vars.items():
            if k in os.environ and os.environ[k] != v:
                logger.warning(f"Overwriting environment variable {k} from '{os.environ[k]}' to '{v}'")
            os.environ[k] = v

    def init_worker(self, od_config: OmniDiffusionConfig) -> None:
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

        from vllm_omni.diffusion.worker import DiffusionWorker

        self.rpc_rank = int(os.environ["RANK"])
        self.worker = DiffusionWorker(
            local_rank=int(os.environ["LOCAL_RANK"]),
            rank=self.rpc_rank,
            od_config=od_config,
        )
        self.od_config = od_config

    def execute_model(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if self.worker is None:
            raise RuntimeError("Worker is not initialized")
        return self.worker.execute_model(request, self.od_config).to_cpu()

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

        placement_group = self._create_placement_group()
        self._init_workers_ray(placement_group)

    def _create_placement_group(self) -> "PlacementGroup":
        num_gpus = self.od_config.num_gpus

        current_pg = ray.util.get_current_placement_group()
        if current_pg is not None:
            logger.info("Using existing placement group")
            return current_pg

        # Pin at least one bundle to the driver node
        bundles = [{"GPU": 1} for _ in range(num_gpus)]
        bundles[0][f"node:{get_ip()}"] = 0.001

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

        unique_ips = set(meta.ip for meta in self.workers)
        master_addr = "127.0.0.1" if len(unique_ips) == 1 else driver_ip
        master_port = str(get_open_port())

        env_futures = []
        for meta in self.workers:
            env_vars = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "RANK": str(meta.rank),
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

        futures = [meta.worker.execute_model.remote(request) for meta in self.workers]
        try:
            results = ray.get(futures, timeout=EXECUTE_MODEL_TIMEOUT)
            return results[0]
        except ray.exceptions.RayTaskError as e:
            logger.error(f"Worker execution failed: {e}")
            raise RuntimeError(f"Diffusion generation failed: {e}") from e
        except ray.exceptions.GetTimeoutError as e:
            logger.error("Worker execution timed out")
            raise TimeoutError("Diffusion generation timed out") from e

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed.")

        kwargs = kwargs or {}
        futures = [meta.worker.execute_method.remote(method, *args, **kwargs) for meta in self.workers]

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
        if self._closed:
            return

        self._closed = True
        for meta in self.workers:
            try:
                ray.kill(meta.worker)
            except Exception as e:
                logger.warning(f"Failed to kill worker rank {meta.rank}: {e}")
        self.workers.clear()

    def __del__(self):
        self.shutdown()
