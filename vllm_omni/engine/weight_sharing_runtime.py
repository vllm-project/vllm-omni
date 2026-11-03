# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""StageRuntime adapter for the model-independent sharing lifecycle."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import msgspec
import psutil

from vllm_omni.config.omni_config import VllmOmniARStageConfig, VllmOmniDiffusionStageConfig
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.weight_sharing.runtime import (
    CONFIG_KEY,
    CONTROL_METHOD,
    RANK_RESULTS_KEY,
    Bootstrap,
    RankResult,
    SharingPlan,
)
from vllm_omni.engine.cuda_mps import physical_gpu_uuid
from vllm_omni.engine.stage_client import StagePoolClient
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan

if TYPE_CHECKING:
    from vllm_omni.engine.stage_runtime import StageRuntime


class StageSharingHost:
    """Single-node, single-API, two-stage host; all calls use the existing loop.

    Bootstrap travels as a JSON-safe dict in additional_config before loading.
    Runtime bytes travel via collective_rpc_async (never tensors or storages).
    """

    def __init__(self, runtime: StageRuntime, plan: SharingPlan, stages: list[LogicalStageInitPlan]):
        self.runtime = runtime
        self.plan = plan
        self.stages = {stage.stage_id: stage for stage in stages}
        self.clients: dict[int, StagePoolClient] = {}
        self.processes: dict[int, dict[int, psutil.Process]] = {}
        self._consumer_launch_uncertain = False
        self._validate()

    def _validate(self) -> None:
        if set(self.stages) != {self.plan.provider_stage, self.plan.consumer_stage}:
            raise ValueError("sharing currently requires exactly one AR stage and one DiT stage")
        if self.runtime._client_count != 1 or self.runtime._external_stage_addresses is not None:
            raise ValueError("sharing requires one locally owning API process")
        if self.runtime._async_chunk or self.runtime._parallel_stage_init:
            raise ValueError("sharing requires non-streaming, dependency-ordered stage initialization")
        tp = len(self.plan.device_uuids)
        for stage_id, stage in self.stages.items():
            if len(stage.replicas) != 1 or stage.replicas[0].launch_mode != "local":
                raise ValueError("sharing requires one local replica per stage")
            replica = stage.replicas[0]
            devices = self.runtime._resolve_replica_physical_devices(stage_id, replica.metadata.runtime_cfg)
            if (
                devices is None
                or tuple(physical_gpu_uuid(device.strip()) for device in devices.split(",")) != self.plan.device_uuids
            ):
                raise ValueError("sharing stages must use the same ordered physical GPUs")
            if stage_id == self.plan.provider_stage:
                config = replica.stage_vllm_config
                if not isinstance(replica.stage_cfg, VllmOmniARStageConfig) or config is None:
                    raise ValueError("sharing provider must be an AR EngineCore stage")
                parallel = config.parallel_config
                if parallel.distributed_executor_backend != "mp":
                    raise ValueError("sharing requires the AR multiprocessing executor")
                if (
                    config.model_config.enable_sleep_mode
                    or config.lora_config is not None
                    or (replica.engine_args_dict or {}).get("cpu_offload_gb", 0)
                    or parallel.enable_eplb
                ):
                    raise ValueError("sharing does not support sleep/offload or LoRA")
            else:
                if not isinstance(replica.stage_cfg, VllmOmniDiffusionStageConfig):
                    raise ValueError("sharing requires a typed DiT stage config")
                config = replica.stage_cfg.diffusion_config
                parallel = replica.stage_cfg.parallel_config
                if (
                    config.distributed_executor_backend != "mp"
                    or replica.stage_cfg.stage_pipeline_config.inline_diffusion
                ):
                    raise ValueError("sharing requires a separate StageDiffusionProc and multiprocessing workers")
                if any(
                    (
                        config.enable_cpu_offload,
                        config.enable_layerwise_offload,
                        config.enable_distributed_layerwise_offload,
                        config.diffusion_offload_config,
                        config.lora_path,
                        config.custom_pipeline_args,
                        parallel.use_hsdp,
                        replica.stage_cfg.model_config.enable_sleep_mode,
                    )
                ):
                    raise ValueError("sharing does not support DiT offload, LoRA, HSDP or custom/inline pipelines")
            if parallel.tensor_parallel_size != tp or parallel.enable_expert_parallel:
                raise ValueError("sharing requires equal TP sizes with EP disabled")
            if any(
                getattr(parallel, name, 1) != 1
                for name in (
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "sequence_parallel_size",
                    "ulysses_degree",
                    "ring_degree",
                    "cfg_parallel_size",
                    "allgather_degree",
                    "vae_patch_parallel_size",
                    "text_encoder_tp_size",
                )
            ):
                raise ValueError("sharing supports tensor parallelism only")

    async def launch(self, stage: int, bootstrap: Bootstrap) -> None:
        if self.runtime._init_cancelled.is_set():
            raise RuntimeError("sharing initialization cancelled")
        replica = self.stages[stage].replicas[0]
        encoded = msgspec.to_builtins(bootstrap)
        if stage == self.plan.provider_stage:
            config = replica.stage_vllm_config
            assert config is not None
            config.additional_config = {**config.additional_config, CONFIG_KEY: encoded}
            config.cache_config.kv_cache_memory_bytes = self.plan.budgets[0].kv_cache_bytes
        else:
            config = replica.stage_cfg.diffusion_config
            config.additional_config = {**config.additional_config, CONFIG_KEY: encoded}
        # Preserve diffusion's required orchestrator-thread launch and existing
        # per-device locks, spawn cancellation, readiness deadline and rollback.
        if stage == self.plan.consumer_stage:
            self._consumer_launch_uncertain = True
        self.clients[stage] = self.runtime._initialize_replica(replica, self.runtime._stage_init_timeout)

    async def rpc(self, stage: int, operation: str, messages: tuple[bytes, ...]) -> object:
        client = self.clients[stage]
        client.check_health()
        result = await client.collective_rpc_async(
            CONTROL_METHOD, timeout=self.plan.timeout, args=(operation, messages)
        )
        if stage == self.plan.consumer_stage:
            # DiT's existing RPC path returns one envelope from global rank 0.
            # Its status gather carries every TP rank's sharing result.
            if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
                raise RuntimeError("invalid diffusion sharing RPC envelope")
            result = result[0][RANK_RESULTS_KEY]
        # Capture process identity (including creation time) for confirmed
        # teardown. Do not rely on the EngineCore's exit proving worker exit.
        ranks = msgspec.convert(result, type=list[RankResult])
        tp = len(self.plan.device_uuids)
        if (
            sorted(rank.identity.rank for rank in ranks) != list(range(tp))
            or len({rank.process_id for rank in ranks}) != tp
        ):
            raise RuntimeError("sharing RPC did not identify every worker process")
        for rank in ranks:
            owned = self.processes.setdefault(stage, {})
            if rank.process_id not in owned:
                if rank.process_id <= 0 or rank.process_id == psutil.Process().pid:
                    raise RuntimeError("sharing worker is not a separate process")
                process = psutil.Process(rank.process_id)
                if not any(parent.pid == psutil.Process().pid for parent in process.parents()):
                    raise RuntimeError("sharing worker is not owned by this runtime")
                owned[rank.process_id] = process
        if stage == self.plan.consumer_stage:
            self._consumer_launch_uncertain = False
        return result

    async def quiesce(self, stage: int) -> None:
        client = self.clients[stage]
        if stage == self.plan.provider_stage:
            await cast(StageEngineCoreClient, client).pause_scheduler_async(mode="abort")
        else:
            await client.collective_rpc_async("weight_sharing_quiesce", timeout=self.plan.timeout)

    def check_health(self, stage: int) -> None:
        self.clients[stage].check_health()
        for process in self.processes.get(stage, {}).values():
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError(f"stage {stage}: sharing worker {process.pid} exited")

    async def terminate(self, stage: int) -> None:
        client = self.clients.get(stage)
        if client is None:
            if stage == self.plan.consumer_stage and self._consumer_launch_uncertain:
                # The existing launcher rolls back the manager, but does not
                # return an all-worker exit acknowledgement after failed startup.
                raise RuntimeError("consumer startup cleanup is unconfirmed; provider must remain alive")
            return
        processes = self.processes.setdefault(stage, {})
        # Also clean up the local DiT process tree. This alone cannot prove all
        # workers exited: a dead manager's surviving workers may be reparented.
        if stage == self.plan.consumer_stage:
            manager = cast(StageDiffusionClient, client)._proc_manager
            if manager is not None and manager.proc.is_alive():
                try:
                    parent = psutil.Process(manager.proc.pid)
                    for process in [parent, *parent.children(recursive=True)]:
                        processes[process.pid] = process
                except psutil.NoSuchProcess:
                    pass
        client.shutdown()
        # Existing process managers handle cooperative termination first.
        # Terminate remaining local workers before permitting provider teardown.
        for process in processes.values():
            try:
                if process.is_running():
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = await asyncio.to_thread(psutil.wait_procs, list(processes.values()), timeout=self.plan.timeout)
        if alive:
            raise RuntimeError(f"stage {stage}: worker exit unconfirmed; retaining provider resources")
        if stage == self.plan.consumer_stage and self._consumer_launch_uncertain:
            raise RuntimeError("consumer startup cleanup is unconfirmed; not all worker PIDs were reported")
        del self.clients[stage]
