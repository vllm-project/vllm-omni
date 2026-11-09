# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving integration seams; CPU tests in the normal vLLM test environment."""

import asyncio
import json
from types import SimpleNamespace

import msgspec
import pytest
from vllm.config import VllmConfig
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config.omni_config import VllmOmniARStageConfig, VllmOmniDiffusionStageConfig
from vllm_omni.config.stage_config import StagePipelineConfig
from vllm_omni.config.weight_sharing import WeightSharingConfig
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
    plan = SharingPlan(0, 1, ("GPU-0", "GPU-1"), (DeviceBudget(1024, 128, 128, 64),) * 2, "model", "revision", "layout")
    configs = [
        VllmOmniARStageConfig(
            stage_pipeline_config=StagePipelineConfig(
                stage_id=0, model_stage="ar", model_arch="HunyuanImage3ForCausalMM"
            )
        ),
        VllmOmniDiffusionStageConfig(stage_pipeline_config=StagePipelineConfig(stage_id=1, model_stage="dit")),
    ]
    stages = []
    for stage, cfg in enumerate(configs):
        cfg.model_config.use_v2_model_runner = True
        cfg.model_config.enforce_eager = True
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.distributed_executor_backend = "mp"
        if stage == 1:
            cfg.diffusion_config.distributed_executor_backend = "mp"
        vllm_config = mocker.Mock(spec=VllmConfig)
        vllm_config.parallel_config = cfg.parallel_config
        vllm_config.model_config = cfg.model_config
        vllm_config.lora_config = None
        vllm_config.quant_config = None
        vllm_config.weight_transfer_config = None
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
        await host.launch(0, group.bootstrap("provider"))
        await host.launch(1, group.bootstrap("consumer", (b"rank0", b"rank1")))

    asyncio.run(run())
    provider = host.stages[0].replicas[0].stage_vllm_config
    consumer = host.stages[1].replicas[0].stage_cfg.diffusion_config
    assert provider.cache_config.kv_cache_memory_bytes == 128
    assert provider.additional_config["existing"] == 1
    restored = msgspec.convert(json.loads(json.dumps(consumer.additional_config[CONFIG_KEY])), type=Bootstrap)
    assert tuple(group.payload for group in restored.groups) == (b"rank0", b"rank1")
    assert launch.call_count == 2


@pytest.mark.parametrize("free_bytes", [512, 1024])
def test_resolved_plan_uses_total_memory_ratio_and_checks_physical_free(host, mocker, tmp_path, free_bytes):
    provider = host.stages[0].replicas[0]
    consumer = host.stages[1].replicas[0]
    for replica in (provider, consumer):
        replica.stage_cfg.model_config.model = str(tmp_path)
        replica.stage_cfg.model_config.revision = None
        replica.stage_cfg.model_config.dtype = "bfloat16"
        replica.stage_cfg.model_config.moe_backend = "triton"
    provider.stage_vllm_config.cache_config.kv_cache_memory_bytes = 128
    consumer.stage_cfg.cache_config.kv_cache_memory_bytes = 64
    (tmp_path / "config.json").write_text("{}")
    nvml = mocker.Mock()
    nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(free=free_bytes, total=2048)
    mocker.patch("vllm.utils.import_utils.import_pynvml", return_value=nvml)
    config = WeightSharingConfig(0, 1, gpu_memory_utilization=0.75)
    if free_bytes == 512:
        with pytest.raises(ValueError, match="insufficient free GPU memory"):
            module.resolve_sharing_plan(host.runtime, config, list(host.stages.values()))
        nvml.nvmlShutdown.assert_called_once()
        return
    resolved = module.resolve_sharing_plan(host.runtime, config, list(host.stages.values()))
    assert resolved.checkpoint_revision.startswith("local-")
    assert resolved.layout == "hunyuan-image3-v1:bfloat16:tp2:ep0:moe=triton"
    assert all(
        b.capacity_bytes == 1536 and b.reserve_bytes == 512 and b.consumer_kv_cache_bytes == 64
        for b in resolved.budgets
    )
    assert all(b.kv_cache_bytes == 128 for b in resolved.budgets)
    nvml.nvmlShutdown.assert_called_once()


def test_diffusion_rpc_unwraps_all_rank_metadata(host, mocker):
    group = SharingRuntime(host.plan, host)
    results = [msgspec.to_builtins(RankResult(identity, SessionState.MAPPED, 64, 64)) for identity in group.identities]
    client = mocker.Mock(spec=StagePoolClient)
    client.collective_rpc_async = mocker.AsyncMock(return_value=[{RANK_RESULTS_KEY: results}])
    host.clients[1] = client
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


@pytest.mark.parametrize("rank", [0, 1])
def test_shared_vllm_config_supports_upstream_replace(rank, tmp_path):
    """Warmup copies VllmConfig; sharing must not add undeclared attributes."""
    from transformers import LlamaConfig
    from vllm.config import DeviceConfig, ModelConfig
    from vllm.config.utils import replace

    from vllm_omni.distributed.weight_sharing.worker_session import resolve_worker_sharing

    plan = SharingPlan(0, 1, ("GPU-0", "GPU-1"), (DeviceBudget(2048, 64, 64, 64),) * 2, "model", "revision", "layout")
    bootstrap = SharingRuntime(plan, None).bootstrap("provider")
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True),
        device_config=DeviceConfig(device="cpu"),
        additional_config={CONFIG_KEY: msgspec.to_builtins(bootstrap)},
    )
    fields_before = set(vars(config))
    resolved = resolve_worker_sharing(config, role="provider", rank=rank)
    assert resolved == (bootstrap.groups[rank], bootstrap.budgets[rank])
    assert set(vars(config)) == fields_before
    copied = replace(config, cache_config=replace(config.cache_config))
    assert resolve_worker_sharing(copied, role="provider", rank=rank) == resolved
    assert copied.additional_config == config.additional_config

    # Reproduce the original failure using the actual upstream field checker.
    config.weight_sharing = bootstrap.groups[rank]
    with pytest.raises(ValueError, match="Field 'weight_sharing' not found"):
        replace(config, cache_config=config.cache_config)


@pytest.mark.parametrize("confirmed", [True, False])
def test_abort_drain_uses_existing_stage_controls(host, mocker, confirmed):
    provider = mocker.Mock()
    provider.pause_scheduler_async = mocker.AsyncMock()
    provider.resume_scheduler_async = mocker.AsyncMock()
    consumer = mocker.Mock(spec=StagePoolClient)
    consumer.collective_rpc_async = mocker.AsyncMock(return_value=confirmed)
    host.clients = {0: provider, 1: consumer}

    async def run():
        if confirmed:
            await host.drain_requests()
            await host.resume_provider()
        else:
            with pytest.raises(RuntimeError, match="not confirmed"):
                await host.drain_requests()
            provider.resume_scheduler_async.assert_not_called()

    asyncio.run(run())
    provider.pause_scheduler_async.assert_awaited_once_with(mode="abort", clear_cache=False)
    consumer.collective_rpc_async.assert_awaited_once_with("weight_sharing_drain", timeout=host.plan.timeout)
