# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""StageRuntime adapter for the model-independent sharing lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import msgspec
from vllm.utils.system_utils import get_mp_context

from vllm_omni.config.omni_config import VllmOmniARStageConfig, VllmOmniDiffusionStageConfig
from vllm_omni.config.weight_sharing import WeightSharingConfig
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.weight_sharing.runtime import (
    CONFIG_KEY,
    CONTROL_METHOD,
    RANK_RESULTS_KEY,
    Bootstrap,
    DeviceBudget,
    SharingPlan,
)
from vllm_omni.distributed.weight_sharing.worker_registry import ConsumerStartupError, ConsumerWorkerRegistry
from vllm_omni.engine.cuda_mps import physical_gpu_uuid
from vllm_omni.engine.stage_client import StagePoolClient
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan

if TYPE_CHECKING:
    from vllm_omni.engine.stage_runtime import StageRuntime


def resolve_sharing_plan(
    runtime: StageRuntime, config: WeightSharingConfig, stages: list[LogicalStageInitPlan]
) -> SharingPlan:
    """Freeze model/layout agreement and budgets before either stage allocates."""
    from vllm.utils.import_utils import import_pynvml

    by_id = {stage.stage_id: stage for stage in stages}
    if set(by_id) != {config.provider_stage, config.consumer_stage}:
        raise ValueError("weight sharing requires exactly the configured AR and DiT stages")
    provider = by_id[config.provider_stage].replicas[0]
    consumer = by_id[config.consumer_stage].replicas[0]
    if (
        not isinstance(provider.stage_cfg, VllmOmniARStageConfig)
        or not isinstance(consumer.stage_cfg, VllmOmniDiffusionStageConfig)
        or provider.stage_vllm_config is None
    ):
        raise ValueError("weight sharing requires an AR provider and DiT consumer")
    ar = provider.stage_vllm_config.model_config
    dit = consumer.stage_cfg.model_config
    if provider.stage_cfg.stage_pipeline_config.model_arch != "HunyuanImage3ForCausalMM":
        raise ValueError("the shared deployment profile currently supports HunyuanImage3 only")
    model = str(ar.model)
    if (dit.model or runtime._model) != model or dit.revision != ar.revision:
        raise ValueError("shared stages must load the same model and checkpoint revision")
    if Path(model).is_dir():
        root = Path(model).resolve()
        # Fingerprint metadata without reading another full model into CPU RAM.
        # Both workers use this same local snapshot; files must stay immutable.
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".safetensors", ".bin"))
        metadata = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        revision = "local-" + hashlib.sha256(json.dumps(metadata).encode()).hexdigest()
        model = str(root)
    else:
        resolved_revision = getattr(ar.hf_config, "_commit_hash", None)
        if not resolved_revision:
            raise ValueError("weight sharing requires a resolved HF commit or a local checkpoint directory")
        revision = str(resolved_revision)
        # Pin the download performed later by DiT to the commit AR resolved.
        ar.revision = revision
        dit.revision = revision
        consumer.stage_cfg.diffusion_config.revision = revision
    dtype = str(ar.dtype).removeprefix("torch.")
    dit_dtype = str(dit.dtype).removeprefix("torch.")
    if dit_dtype not in ("auto", dtype):
        raise ValueError("shared stages must use identical weight dtypes")
    dit.dtype = dtype
    consumer.stage_cfg.diffusion_config.dtype = dtype
    moe = provider.stage_cfg.model_config.moe_backend
    if moe == "auto" or moe != dit.moe_backend:
        raise ValueError("weight sharing requires the same explicit moe_backend on both stages")
    devices = runtime._resolve_replica_physical_devices(config.provider_stage, provider.metadata.runtime_cfg)
    if not devices:
        raise ValueError("weight sharing requires explicit stage devices")
    uuids = tuple(physical_gpu_uuid(device.strip()) for device in devices.split(","))
    ar_kv = provider.stage_vllm_config.cache_config.kv_cache_memory_bytes
    dit_kv = consumer.stage_cfg.cache_config.kv_cache_memory_bytes
    if not ar_kv or not dit_kv:
        raise ValueError("weight sharing requires explicit positive kv_cache_memory_bytes on both stages")
    nvml = import_pynvml()
    nvml.nvmlInit()
    try:
        budgets = []
        for uuid in uuids:
            info = nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByUUID(uuid))
            capacity = int(info.total * config.gpu_memory_utilization)
            budget = DeviceBudget(capacity, info.total - capacity, ar_kv, dit_kv)
            budget.validate()
            required = budget.reserve_bytes + ar_kv + dit_kv
            if info.free < required:
                raise ValueError(
                    f"{uuid}: insufficient free GPU memory before shared loading "
                    f"(free={info.free / 1024**3:.3f}, KV + reserve={required / 1024**3:.3f} GiB)"
                )
            budgets.append(budget)
    finally:
        nvml.nvmlShutdown()
    layout = f"hunyuan-image3-v1:{dtype}:tp{len(uuids)}:ep0:moe={moe}"
    return SharingPlan(
        config.provider_stage,
        config.consumer_stage,
        uuids,
        tuple(budgets),
        model,
        revision,
        layout,
        timeout=config.timeout,
    )


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
        self._consumer_started = False
        self._consumer_registry: ConsumerWorkerRegistry | None = None
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
                if not config.model_config.use_v2_model_runner:
                    raise ValueError("weight sharing requires model_runner: v2")
                if parallel.distributed_executor_backend != "mp":
                    raise ValueError("sharing requires the AR multiprocessing executor")
                if (
                    config.model_config.enable_sleep_mode
                    or config.lora_config is not None
                    or (replica.engine_args_dict or {}).get("cpu_offload_gb", 0)
                    or parallel.enable_eplb
                    or config.quant_config is not None
                    or config.weight_transfer_config is not None
                    or not config.model_config.enforce_eager
                ):
                    raise ValueError(
                        "sharing requires eager, unquantized AR weights without sleep/offload, LoRA or updates"
                    )
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
                        config.quantization_config,
                        replica.stage_cfg.weight_transfer_config,
                        not replica.stage_cfg.model_config.enforce_eager,
                    )
                ):
                    raise ValueError(
                        "sharing requires eager, unquantized DiT weights without offload, LoRA, "
                        "HSDP, updates or custom pipelines"
                    )
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
                    "nnodes_within_dp",
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
            replica.stage_cfg.cache_config.kv_cache_memory_bytes = self.plan.budgets[0].consumer_kv_cache_bytes
        # Preserve diffusion's required orchestrator-thread launch and existing
        # per-device locks, spawn cancellation, readiness deadline and rollback.
        if stage == self.plan.consumer_stage:
            # Own the registry before configuration or process creation can fail.
            self._consumer_registry = ConsumerWorkerRegistry(get_mp_context(), len(self.plan.device_uuids))
            setattr(replica.stage_cfg, "_sharing_worker_registry", self._consumer_registry)
            self._consumer_started = True
        try:
            self.clients[stage] = self.runtime._initialize_replica(replica, self.runtime._stage_init_timeout)
        except ConsumerStartupError as exc:
            self._consumer_registry = exc.registry
            raise

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
        return result

    async def quiesce(self, stage: int) -> None:
        client = self.clients[stage]
        if stage == self.plan.provider_stage:
            await cast(StageEngineCoreClient, client).pause_scheduler_async(mode="abort")
        else:
            await client.collective_rpc_async("weight_sharing_quiesce", timeout=self.plan.timeout)

    async def drain_requests(self) -> None:
        """Abort late submissions and wait for stage execution to drain."""
        provider = cast(StageEngineCoreClient, self.clients[self.plan.provider_stage])
        await provider.pause_scheduler_async(mode="abort", clear_cache=False)
        consumer = self.clients[self.plan.consumer_stage]
        result = await consumer.collective_rpc_async("weight_sharing_drain", timeout=self.plan.timeout)
        if result is not True:
            raise RuntimeError("consumer request drain was not confirmed")

    async def resume_provider(self) -> None:
        """Resume AR after the runtime has synchronized every shared rank."""
        await cast(StageEngineCoreClient, self.clients[self.plan.provider_stage]).resume_scheduler_async()

    def check_health(self, stage: int) -> None:
        self.clients[stage].check_health()

    async def terminate(self, stage: int) -> None:
        client = self.clients.get(stage)
        if client is not None:
            if stage == self.plan.consumer_stage:
                manager = cast(StageDiffusionClient, client)._proc_manager
                self._consumer_registry = getattr(manager, "sharing_workers", None)
            # Reuse existing managers for cooperative/process-tree shutdown.
            client.shutdown()
        if stage == self.plan.consumer_stage and self._consumer_started:
            # The registry predates CUDA init and survives a dead manager. It
            # seals late workers and confirms exit, including partial startup.
            # A stage-process exit alone is not permission to free the provider.
            if self._consumer_registry is None:
                raise RuntimeError("consumer startup cleanup is unconfirmed; provider must remain alive")
            await asyncio.to_thread(self._consumer_registry.terminate, self.plan.timeout)
        self.clients.pop(stage, None)
