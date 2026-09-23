# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray-based distributed executor for diffusion models.

Ray owns worker placement and process lifecycle. PyTorch distributed still
owns the model-parallel data plane: the executor gives every actor a global
rank and a common rendezvous address before the actor constructs its worker.
"""

from __future__ import annotations

import json
import os
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.media import DiffusionMediaOutput
from vllm_omni.diffusion.offloader.config import (
    TEXT_ENCODER_COMPONENT,
    any_selected_component_uses_allgather,
    resolve_offload,
)
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup

    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)

_PLACEMENT_GROUP_WAIT_TIMEOUT_S = 1800
_WORKER_INIT_TIMEOUT_S = 7200
_HEALTH_CHECK_TIMEOUT_S = 10
_DLO_DP_WAVE_TIMEOUT_S = float(os.environ.get("VLLM_OMNI_DLO_DP_WAVE_TIMEOUT", 600.0))

try:
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError:
    ray = None  # type: ignore[assignment]
    PlacementGroupSchedulingStrategy = None  # type: ignore[assignment,misc]


def _worker_env(od_config: OmniDiffusionConfig) -> dict[str, str]:
    """Copy inference settings, keeping worker identity local to each actor."""
    prefixes = (
        "VLLM_",
        "OMNI_",
        "DIFFUSION_",
        "NCCL_",
        "TORCH_NCCL_",
        "UCX_",
        "HF_",
        "HUGGINGFACE_",
        "HUGGING_FACE_HUB_",
    )
    names = {"PYTHONPATH", "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS"}
    env = {key: value for key, value in os.environ.items() if key.startswith(prefixes) or key in names}
    # Explicit stage settings may include plugin-specific variables outside
    # the standard prefixes, and take precedence over the driver's defaults.
    env.update(getattr(od_config, "ray_worker_env", {}))
    worker_specific = {
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "VLLM_HOST_IP",
        "VLLM_HOST_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "WORLD_SIZE",
    }
    return {key: value for key, value in env.items() if key not in worker_specific and not key.startswith("RAY_")}


def _uses_dlo_dp_concurrency(od_config: OmniDiffusionConfig) -> bool:
    parallel_config = getattr(od_config, "parallel_config", None)
    return bool(
        (getattr(parallel_config, "data_parallel_size", 1) or 1) > 1
        and any_selected_component_uses_allgather(od_config)
    )


def _is_empty_dp_prompt(prompt: object) -> bool:
    """Return whether a DP request has no usable text prompt."""
    if prompt is None:
        return True
    if isinstance(prompt, (str, list, tuple)):
        return not prompt
    if isinstance(prompt, dict):
        return (
            not prompt.get("prompt")
            and not prompt.get("prompt_token_ids")
            and not prompt.get("prompt_ids")
            and prompt.get("prompt_embeds") is None
        )
    return False


def _text_encoder_input_signature(prompt: object) -> tuple[bool, bool]:
    """Describe precomputed embeddings that change encoder forward counts."""
    if not isinstance(prompt, dict):
        return False, False
    return prompt.get("prompt_embeds") is not None, prompt.get("negative_prompt_embeds") is not None


def _uses_text_encoder_allgather(config: object) -> bool:
    resolved = resolve_offload(config)
    return resolved.offloads(TEXT_ENCODER_COMPONENT) and resolved.uses_allgather(TEXT_ENCODER_COMPONENT)


def _move_to_cpu(value: Any) -> Any:
    """Recursively detach tensors before Ray serializes a worker response."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, DiffusionMediaOutput):
        return value.to_cpu()
    if isinstance(value, DiffusionOutput):
        for output_field in fields(value):
            setattr(value, output_field.name, _move_to_cpu(getattr(value, output_field.name)))
        return value
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)

    # RunnerOutput and BatchRunnerOutput are kept out of the module imports to
    # avoid pulling the complete worker stack into the driver at import time.
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    if isinstance(value, RunnerOutput):
        value.result = _move_to_cpu(value.result)
        return value
    if isinstance(value, BatchRunnerOutput):
        value.runner_outputs = [_move_to_cpu(output) for output in value.runner_outputs]
        return value
    return value


@dataclass
class _RayWorkerMetadata:
    worker: ActorHandle
    rank: int = -1
    ip: str = ""


@dataclass
class _RayExecutorResources:
    """Resources owned by the executor's finalizer."""

    workers: list[_RayWorkerMetadata]
    placement_group: PlacementGroup | None = None
    owns_placement_group: bool = False

    def __call__(self) -> None:
        if ray is None:
            return

        # Actor shutdown RPCs queue behind inference; kill actors to interrupt it.
        for metadata in self.workers:
            try:
                ray.kill(metadata.worker, no_restart=True)
            except Exception as exc:
                logger.warning("Failed to kill Ray diffusion worker rank %d: %s", metadata.rank, exc)
        self.workers.clear()

        if self.owns_placement_group and self.placement_group is not None:
            try:
                ray.util.remove_placement_group(self.placement_group)
            except Exception as exc:
                logger.warning("Failed to remove Ray diffusion placement group: %s", exc)
            self.placement_group = None


class RayDiffusionWorkerWrapper:
    """Ray actor that constructs one rank of the existing DiffusionWorker."""

    def __init__(self, provisional_rank: int) -> None:
        self.rank = provisional_rank
        self.worker = None

    def get_node_ip(self) -> str:
        return get_ip()

    def get_open_port(self) -> int:
        return get_open_port()

    def init_worker(self, od_config: OmniDiffusionConfig, rank: int, distributed_init_method: str) -> None:
        from vllm_omni.platforms import current_omni_platform
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

        from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

        worker_cls_path = current_omni_platform.get_diffusion_worker_cls()
        self.rank = rank
        self.worker = WorkerWrapperBase(
            gpu_id=0,
            rank=self.rank,
            distributed_init_method=distributed_init_method,
            od_config=od_config,
            worker_extension_cls=od_config.worker_extension_cls,
            custom_pipeline_args=getattr(od_config, "custom_pipeline_args", None),
            base_worker_class=resolve_obj_by_qualname(worker_cls_path),
        )

    def _is_primary_in_replica(self) -> bool:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_classifier_free_guidance_rank,
            get_pipeline_parallel_rank,
            get_sequence_parallel_rank,
        )

        return (
            get_sequence_parallel_rank() == 0
            and get_classifier_free_guidance_rank() == 0
            and get_tensor_model_parallel_rank() == 0
            and get_pipeline_parallel_rank() == 0
        )

    def execute_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        output_rank: int | None = None,
        exec_all_ranks: bool = False,
        primary_replies_only: bool = False,
    ) -> dict[str, Any]:
        if self.worker is None:
            raise RuntimeError("Ray diffusion worker is not initialized")

        should_execute = exec_all_ranks or output_rank is None or output_rank == self.rank
        should_reply = output_rank is None or output_rank == self.rank
        if primary_replies_only:
            should_reply = self._is_primary_in_replica()

        result = None
        if should_execute:
            result = self.worker.execute_method(method, *(args or ()), **(kwargs or {}))
        if should_reply:
            result = _move_to_cpu(result)
        else:
            result = None
        return {"rank": self.rank, "replied": should_reply, "result": result}

    def check_alive(self) -> bool:
        return self.worker is not None


class RayDiffusionExecutor(DiffusionExecutor):
    """Execute one diffusion worker per Ray GPU actor."""

    def _init_executor(self) -> None:
        if ray is None:
            raise ImportError(
                "Ray is required for distributed_executor_backend='ray'. "
                'Install the project dependencies or `pip install "ray[default]"`.'
            )

        self._closed = False
        self._is_failed = False
        self._failure_callbacks: list[Callable[[], None]] = []
        self.workers: list[_RayWorkerMetadata] = []

        if not ray.is_initialized():
            logger.info("Initializing Ray for the diffusion executor")
            ray.init()

        placement_group, owns_placement_group = self._get_or_create_placement_group()
        self._resources = _RayExecutorResources(
            workers=self.workers,
            placement_group=placement_group,
            owns_placement_group=owns_placement_group,
        )
        self._finalizer = weakref.finalize(self, self._resources)

        try:
            self._init_workers(placement_group)
        except Exception:
            self.shutdown()
            raise

    @property
    def is_dead(self) -> bool:
        return self._closed or self._is_failed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RayDiffusionExecutor is closed")
        if self._is_failed:
            raise EngineDeadError()

    def _get_or_create_placement_group(self) -> tuple[PlacementGroup, bool]:
        num_gpus = int(self.od_config.num_gpus or 1)
        current_group = ray.util.get_current_placement_group()
        if current_group is not None:
            # A worker needs one full GPU within a single bundle.
            gpu_capacity = sum(int(bundle.get("GPU", 0)) for bundle in current_group.bundle_specs)
            if gpu_capacity < num_gpus:
                raise ValueError(
                    f"Ray diffusion requires {num_gpus} full GPUs, "
                    f"but the placement group can accommodate only {gpu_capacity} one-GPU workers."
                )
            logger.info("Using the current Ray placement group for diffusion workers")
            return current_group, False

        placement_group = ray.util.placement_group([{"GPU": 1} for _ in range(num_gpus)], strategy="PACK")
        logger.info("Waiting for a Ray placement group with %d GPU bundle(s)", num_gpus)
        try:
            ray.get(placement_group.ready(), timeout=_PLACEMENT_GROUP_WAIT_TIMEOUT_S)
        except BaseException as exc:
            # The executor finalizer is not registered until this method returns.
            try:
                ray.util.remove_placement_group(placement_group)
            except Exception:
                logger.warning("Failed to remove placement group after startup failure", exc_info=True)
            if isinstance(exc, ray.exceptions.GetTimeoutError):
                raise ValueError(
                    f"Cannot reserve {num_gpus} Ray GPU bundle(s) within "
                    f"{_PLACEMENT_GROUP_WAIT_TIMEOUT_S}s; check `ray status`."
                ) from exc
            raise
        return placement_group, True

    def _init_workers(self, placement_group: PlacementGroup) -> None:
        num_gpus = int(self.od_config.num_gpus or 1)
        driver_ip = get_ip()
        actor_cls = ray.remote(
            num_cpus=0,
            num_gpus=1,
            runtime_env={"env_vars": _worker_env(self.od_config)},
            max_concurrency=1,
            concurrency_groups={"health": 1},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            ),
        )(RayDiffusionWorkerWrapper)

        # Register actors immediately so partial startup failures can clean them up.
        for rank in range(num_gpus):
            actor = actor_cls.remote(provisional_rank=rank)
            self.workers.append(_RayWorkerMetadata(worker=actor, rank=rank))

        worker_ips = ray.get(
            [item.worker.get_node_ip.remote() for item in self.workers],
            timeout=_WORKER_INIT_TIMEOUT_S,
        )
        for item, worker_ip in zip(self.workers, worker_ips):
            item.ip = worker_ip

        ip_counts: dict[str, int] = defaultdict(int)
        for item in self.workers:
            ip_counts[item.ip] += 1

        # Keep ranks from the same host contiguous. Prefer the driver host for
        # rank 0 when it owns a GPU; otherwise choose deterministically by IP.
        self.workers.sort(key=lambda item: (0 if item.ip == driver_ip else 1, ip_counts[item.ip], item.ip))
        for rank, item in enumerate(self.workers):
            item.rank = rank

        unique_ips = {item.ip for item in self.workers}
        master_addr = "127.0.0.1" if len(unique_ips) == 1 else self.workers[0].ip
        master_port = ray.get(self.workers[0].worker.get_open_port.remote(), timeout=_WORKER_INIT_TIMEOUT_S)
        distributed_init_method = get_distributed_init_method(master_addr, master_port)

        logger.info(
            "Initializing %d Ray diffusion worker(s) across %d node(s)",
            num_gpus,
            len(unique_ips),
        )
        futures = [
            item.worker.init_worker.remote(self.od_config, item.rank, distributed_init_method) for item in self.workers
        ]
        ray.get(futures, timeout=_WORKER_INIT_TIMEOUT_S)
        logger.info("All %d Ray diffusion workers initialized", num_gpus)

    def _mark_failed(self, exc: BaseException) -> None:
        if self._closed or self._is_failed:
            return
        self._is_failed = True
        logger.error("Ray diffusion executor failed: %s", exc)
        for callback in self._failure_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Diffusion executor failure callback raised")

    def register_failure_callback(self, callback: Callable[[], None]) -> None:
        self._failure_callbacks.append(callback)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        self._ensure_open()

        # ``None + exec_all_ranks`` is the multiprocessing executor's
        # rank-local-DP response mode: all ranks compute, but only the primary
        # rank of each DP replica returns an output.
        primary_replies_only = unique_reply_rank is None and exec_all_ranks
        execute_all_ranks = unique_reply_rank is None or exec_all_ranks
        try:
            futures = [
                item.worker.execute_rpc.remote(
                    method=method,
                    args=args,
                    kwargs=kwargs or {},
                    output_rank=unique_reply_rank,
                    exec_all_ranks=execute_all_ranks,
                    primary_replies_only=primary_replies_only,
                )
                for item in self.workers
            ]
            envelopes = ray.get(futures, timeout=timeout)
        except ray.exceptions.GetTimeoutError as exc:
            self._mark_failed(exc)
            self.shutdown()
            raise TimeoutError(f"RPC call to {method} timed out") from exc
        except ray.exceptions.RayActorError as exc:
            self._mark_failed(exc)
            self.shutdown()
            raise EngineDeadError() from exc
        except ray.exceptions.RayTaskError as exc:
            # ray.get raises on the first failed rank without waiting for its
            # peers. They may still be blocked in a collective, so no worker
            # can safely accept another RPC from this executor.
            self._mark_failed(exc)
            self.shutdown()
            raise EngineDeadError() from exc

        replies = sorted(
            (envelope for envelope in envelopes if envelope["replied"]),
            key=lambda envelope: envelope["rank"],
        )
        if unique_reply_rank is not None:
            for envelope in replies:
                if envelope["rank"] == unique_reply_rank:
                    return envelope["result"]
            raise RuntimeError(f"Ray diffusion rank {unique_reply_rank} did not return a response for {method}")
        return [envelope["result"] for envelope in replies]

    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        from vllm_omni.diffusion.sched.interface import validate_new_request_data_identity
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        new_reqs = scheduler_output.scheduled_new_reqs
        for new_req in new_reqs:
            validate_new_request_data_identity(new_req)

        if len(new_reqs) > 1 and _uses_dlo_dp_concurrency(self.od_config):
            compatibility_keys = [build_request_batch_sampling_params_key(item.req) for item in new_reqs]
            if any(key != compatibility_keys[0] for key in compatibility_keys[1:]):
                raise ValueError(
                    "Rank-local DP concurrency requires compatible shape, CFG, denoise schedule, "
                    "output count, and LoRA settings for every request in a collective wave."
                )
            extra_args_signatures = {
                json.dumps(getattr(item.req.sampling_params, "extra_args", None), sort_keys=True, default=repr)
                for item in new_reqs
            }
            if len(extra_args_signatures) > 1:
                raise ValueError("DP multi-concurrency requires identical extra_args for every request")
            if _uses_text_encoder_allgather(self.od_config):
                encoder_signatures = {_text_encoder_input_signature(item.req.prompt) for item in new_reqs}
                if len(encoder_signatures) > 1:
                    raise ValueError(
                        "DLO text_encoder AllGather requires every concurrent request "
                        "to provide the same positive/negative prompt embedding fields."
                    )
            empty_prompt_ids = [item.request_id for item in new_reqs if _is_empty_dp_prompt(item.req.prompt)]
            if empty_prompt_ids:
                raise ValueError(
                    "DP multi-concurrency requires a non-empty prompt for every request; "
                    f"empty prompt request IDs: {empty_prompt_ids}."
                )

            try:
                results = self.collective_rpc(
                    "execute_model",
                    timeout=_DLO_DP_WAVE_TIMEOUT_S,
                    args=(new_reqs, self.od_config, scheduler_output.kv_prefetch_job),
                    unique_reply_rank=None,
                    exec_all_ranks=True,
                )
                tagged_results = sorted(results, key=lambda item: item["dp_rank"])
                outputs = []
                for index, new_req in enumerate(new_reqs):
                    result = tagged_results[index]["output"]
                    if not isinstance(result, DiffusionOutput):
                        raise RuntimeError(f"Unexpected DP response type [{index}]: {type(result)!r}")
                    outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=result,
                        )
                    )
                return BatchRunnerOutput.from_list(outputs)
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    self._mark_failed(exc)
                    self.shutdown()
                return BatchRunnerOutput.from_list(
                    [
                        RunnerOutput(
                            request_id=item.request_id,
                            step_index=None,
                            finished=True,
                            result=DiffusionOutput.from_exception(exc),
                        )
                        for item in new_reqs
                    ]
                )

        outputs = []
        for new_req in new_reqs:
            args: tuple = (new_req.req, self.od_config, scheduler_output.kv_prefetch_job)
            if new_req.diffusion_kv_metadata is not None:
                args += (new_req.diffusion_kv_metadata,)
            try:
                timeout_options: dict[str, Any] = {}
                if any_selected_component_uses_allgather(self.od_config):
                    timeout_options["timeout"] = _DLO_DP_WAVE_TIMEOUT_S
                result = self.collective_rpc(
                    "execute_model",
                    args=args,
                    unique_reply_rank=0,
                    exec_all_ranks=True,
                    **timeout_options,
                )
                if not isinstance(result, DiffusionOutput):
                    raise RuntimeError(f"Unexpected response type: {type(result)!r}")
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    self._mark_failed(exc)
                    self.shutdown()
                result = DiffusionOutput.from_exception(exc)
            outputs.append(
                RunnerOutput(
                    request_id=new_req.request_id,
                    step_index=None,
                    finished=True,
                    result=result,
                )
            )
        return BatchRunnerOutput.from_list(outputs)

    def execute_batch(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

        self._ensure_open()
        if len(scheduler_output.scheduled_new_reqs) <= 1:
            return self.execute_request(scheduler_output)
        if _uses_dlo_dp_concurrency(self.od_config):
            return self.execute_request(scheduler_output)

        result = self.collective_rpc(
            "execute_model_batch",
            args=(scheduler_output, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, BatchRunnerOutput):
            raise RuntimeError(f"Unexpected response type for execute_batch: {type(result)!r}")
        return result

    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

        self._ensure_open()
        result = self.collective_rpc(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, BaseRunnerOutput):
            raise RuntimeError(f"Unexpected response type for execute_step: {type(result)!r}")
        return result

    def check_health(self) -> None:
        self._ensure_open()
        try:
            alive = ray.get(
                [item.worker.check_alive.options(concurrency_group="health").remote() for item in self.workers],
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
        except Exception as exc:
            self._mark_failed(exc)
            raise EngineDeadError() from exc
        dead_ranks = [item.rank for item, is_alive in zip(self.workers, alive) if not is_alive]
        if dead_ranks:
            exc = RuntimeError(f"Ray diffusion worker ranks are not healthy: {dead_ranks}")
            self._mark_failed(exc)
            raise EngineDeadError() from exc

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finalizer()
