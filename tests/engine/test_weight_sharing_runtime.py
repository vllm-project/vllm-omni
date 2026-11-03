# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving integration seams; CPU tests in the normal vLLM test environment."""

import asyncio
import json

import msgspec
import psutil
import pytest
from vllm.config import VllmConfig
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config.omni_config import VllmOmniARStageConfig, VllmOmniDiffusionStageConfig
from vllm_omni.config.stage_config import StagePipelineConfig
from vllm_omni.distributed.weight_sharing.contracts import SessionState
from vllm_omni.distributed.weight_sharing.runtime import (
    CONFIG_KEY,
    CONTROL_METHOD,
    RANK_RESULTS_KEY,
    Bootstrap,
    DeviceBudget,
    RankResult,
    SharingPlan,
    SharingRuntime,
)
from vllm_omni.engine import weight_sharing_runtime as module
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_client import StagePoolClient
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan, ReplicaInitPlan, StageMetadata
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def host(mocker):
    runtime = StageRuntime([], "dummy", None, stage_init_timeout=5, async_chunk=False)
    plan = SharingPlan(0, 1, ("GPU-0", "GPU-1"), (DeviceBudget(1024, 256, 128, 128, 128, 128),) * 2)
    configs = [
        VllmOmniARStageConfig(stage_pipeline_config=StagePipelineConfig(stage_id=0, model_stage="ar")),
        VllmOmniDiffusionStageConfig(stage_pipeline_config=StagePipelineConfig(stage_id=1, model_stage="dit")),
    ]
    stages = []
    for stage, cfg in enumerate(configs):
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.distributed_executor_backend = "mp"
        if stage == 1:
            cfg.diffusion_config.distributed_executor_backend = "mp"
        vllm_config = mocker.Mock(spec=VllmConfig)
        vllm_config.parallel_config = cfg.parallel_config
        vllm_config.model_config = cfg.model_config
        vllm_config.lora_config = None
        vllm_config.additional_config = {"existing": 1}
        vllm_config.cache_config = cfg.cache_config
        metadata = mocker.Mock(spec=StageMetadata, stage_id=stage, stage_type="llm" if stage == 0 else "diffusion")
        metadata.runtime_cfg = {"devices": "0,1"}
        replica = ReplicaInitPlan(0, 1, "local", cfg, metadata, {}, (None, None, None), vllm_config, object, {})
        stages.append(LogicalStageInitPlan(stage, stage, [replica]))
    mocker.patch.object(runtime, "_resolve_replica_physical_devices", return_value="0,1")
    mocker.patch.object(module, "physical_gpu_uuid", side_effect=lambda device: f"GPU-{device}")
    return module.StageSharingHost(runtime, plan, stages)


@pytest.mark.parametrize("bad", ["ep", "tp", "replica", "inline", "offload", "devices", "parallel_init"])
def test_unsupported_topology_is_rejected_before_launch(host, bad, mocker):
    consumer = host.stages[1].replicas[0]
    cfg = consumer.stage_cfg
    if bad == "ep":
        cfg.parallel_config.enable_expert_parallel = True
    elif bad == "tp":
        cfg.parallel_config.tensor_parallel_size = 1
    elif bad == "replica":
        host.stages[1].replicas.append(consumer)
    elif bad == "inline":
        cfg.diffusion_config.distributed_executor_backend = "uni"
    elif bad == "offload":
        cfg.diffusion_config.enable_cpu_offload = True
    elif bad == "devices":
        mocker.patch.object(host.runtime, "_resolve_replica_physical_devices", return_value="1,0")
    else:
        host.runtime._parallel_stage_init = True
    launch = mocker.patch.object(host.runtime, "_initialize_replica")
    with pytest.raises(ValueError):
        host._validate()
    launch.assert_not_called()


def test_bootstrap_and_kv_budget_arrive_before_model_loading(host, mocker):
    group = SharingRuntime(host.plan, host)
    launch = mocker.patch.object(host.runtime, "_initialize_replica", return_value=mocker.Mock(spec=StagePoolClient))

    async def run():
        await host.launch(0, Bootstrap("provider", group.identities, host.plan.budgets))
        await host.launch(1, Bootstrap("consumer", group.identities, host.plan.budgets, (b"rank0", b"rank1")))

    asyncio.run(run())
    provider = host.stages[0].replicas[0].stage_vllm_config
    consumer = host.stages[1].replicas[0].stage_cfg.diffusion_config
    assert provider.cache_config.kv_cache_memory_bytes == 128
    assert provider.additional_config["existing"] == 1
    restored = msgspec.convert(json.loads(json.dumps(consumer.additional_config[CONFIG_KEY])), type=Bootstrap)
    assert restored.bundles == (b"rank0", b"rank1")
    assert launch.call_count == 2


def test_diffusion_rpc_unwraps_all_rank_metadata(host, mocker):
    group = SharingRuntime(host.plan, host)
    results = [
        msgspec.to_builtins(RankResult(identity, SessionState.MAPPED, 64, 64, 10 + rank))
        for rank, identity in enumerate(group.identities)
    ]
    client = mocker.Mock(spec=StagePoolClient)
    client.collective_rpc_async = mocker.AsyncMock(return_value=[{RANK_RESULTS_KEY: results}])
    host.clients[1] = client
    host.processes[1] = {10: mocker.Mock(spec=psutil.Process), 11: mocker.Mock(spec=psutil.Process)}
    assert asyncio.run(host.rpc(1, "status", ())) == results
    client.collective_rpc_async.assert_awaited_once_with(CONTROL_METHOD, timeout=60.0, args=("status", ()))


def test_runtime_shutdown_does_not_bypass_unconfirmed_consumer_exit(host, mocker):
    sharing = mocker.Mock(spec=SharingRuntime)
    sharing.termination_complete = False
    host.runtime.sharing = sharing
    client = mocker.Mock(spec=StagePoolClient)
    host.runtime.stage_pools = [mocker.Mock(clients=[client])]
    host.runtime.shutdown()
    client.shutdown.assert_not_called()


def test_partial_consumer_launch_cannot_confirm_termination(host, mocker):
    group = SharingRuntime(host.plan, host)
    mocker.patch.object(host.runtime, "_initialize_replica", side_effect=RuntimeError("startup failed"))

    async def run():
        with pytest.raises(RuntimeError, match="startup failed"):
            await host.launch(1, Bootstrap("consumer", group.identities, host.plan.budgets, (b"a", b"b")))
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            await host.terminate(1)

    asyncio.run(run())


def test_live_consumer_worker_prevents_termination_acknowledgement(host, mocker):
    client = mocker.Mock(spec=module.StageDiffusionClient)
    client._proc_manager = None
    host.clients[1] = client
    process = mocker.Mock(spec=psutil.Process)
    host.processes[1] = {10: process}
    mocker.patch.object(module.psutil, "wait_procs", return_value=([], [process]))
    with pytest.raises(RuntimeError, match="worker exit unconfirmed"):
        asyncio.run(host.terminate(1))
    client.shutdown.assert_called_once()
    process.kill.assert_called_once()
    assert host.clients[1] is client


def test_dead_manager_before_rank_report_does_not_prove_worker_exit(host, mocker):
    client = mocker.Mock(spec=module.StageDiffusionClient)
    client._proc_manager = mocker.Mock()
    client._proc_manager.proc.is_alive.return_value = False
    host.clients[1] = client
    host._consumer_launch_uncertain = True
    mocker.patch.object(module.psutil, "wait_procs", return_value=([], []))
    with pytest.raises(RuntimeError, match="not all worker PIDs"):
        asyncio.run(host.terminate(1))
    assert host.clients[1] is client


def test_dead_shared_rank_escalates_instead_of_independent_eviction(host, mocker):
    orchestrator = object.__new__(Orchestrator)
    group = SharingRuntime(host.plan, host)
    group.state = "READY"
    orchestrator.weight_sharing = group
    pool = mocker.Mock()
    orchestrator.stage_pools = [pool]
    with pytest.raises(RuntimeError, match="sharing group"):
        asyncio.run(orchestrator._handle_dead_replica(0, 0, EngineDeadError()))
    assert group.state == "FAILED"
    pool.evict_replica.assert_not_called()
