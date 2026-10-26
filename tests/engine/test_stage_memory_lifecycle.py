# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.distributed.omni_coordinator.messages import ReplicaStatus
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.stat_logger import OmniPrometheusStatLogger, _RelabelGauge

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MemoryClient:
    stage_type = "llm"

    def __init__(self, address, allocated=11, reserved=13, *, supported=True):
        self.client_addresses = {"input_address": address}
        self.snapshot = {"allocated_bytes": allocated, "reserved_bytes": reserved} if supported else None
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def collective_rpc_async(self, *, method, **kwargs):
        assert method == "get_stage_post_warmup_memory_stats"
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return [self.snapshot]

    def shutdown(self):
        pass


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=2)


@pytest.fixture
def lifecycle(monkeypatch):
    import vllm_omni.engine.membership_controller as membership_mod
    import vllm_omni.metrics.stat_logger as logger_mod

    registry = CollectorRegistry()
    clients = [MemoryClient("tcp://old-0"), MemoryClient("tcp://old-1", 17, 19)]
    config = SimpleNamespace(model_config=SimpleNamespace(served_model_name="m"))
    pool = StagePool(0, clients, stage_vllm_config=config)
    replacements: dict[int, MemoryClient] = {}

    class Hub:
        def get_replica_list(self):
            return SimpleNamespace(
                replicas=[
                    SimpleNamespace(
                        stage_id=0,
                        input_addr=client.client_addresses["input_address"],
                        status=ReplicaStatus.UP,
                    )
                    for client in pool.clients
                    if client is not None
                ]
            )

        def close(self):
            pass

    monkeypatch.setattr(membership_mod, "OmniCoordClientForHub", lambda _: Hub())
    controller = MembershipController(
        stage_pools=[pool],
        coordinator_pub_address="tcp://hub",
        load_balancer_factory=object,
        remote_replica_factory=lambda stage_id, replica_id: replacements[replica_id],
    )
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=[pool],
        membership_controller=controller,
    )

    # Initialize the real logger's memory and replica-registration state without
    # registering unrelated upstream collectors in the process-wide registry.
    logger = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
    logger.vllm_config = config
    logger.engine_indexes = [0, 1]
    logger._stage_replica_map = {0: ("0", "0"), 1: ("0", "1")}
    logger._omni_per_engine_labelvalues = {0: ["m", "0", "0"], 1: ["m", "0", "1"]}
    monkeypatch.setattr(logger_mod, "_ENGINE_INDEX_MAP", dict(logger._stage_replica_map))
    sleep_gauge = _RelabelGauge("test_engine_awake", "Awake", labelnames=["model_name", "engine"], registry=registry)
    logger.gauge_engine_sleep_state = {
        "awake": {idx: sleep_gauge.labels(*labels) for idx, labels in logger.per_engine_labelvalues.items()}
    }
    logger._init_stage_post_warmup_memory_families(registry)
    orchestrator._stat_logger = logger

    def values(replica_id):
        labels = {"model_name": "m", "stage": "0", "replica": str(replica_id)}
        return tuple(
            registry.get_sample_value(name, labels)
            for name in (defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, defs.STAGE_POST_WARMUP_MEMORY_RESERVED_BYTES)
        )

    async def run(scenario):
        async def requests():
            await scenario()
            raise EngineDeadError("lifecycle scenario complete")

        async def outputs():
            await asyncio.Event().wait()

        monkeypatch.setattr(orchestrator, "_request_handler", requests)
        monkeypatch.setattr(orchestrator, "_orchestration_output_handler", outputs)
        await orchestrator.run()

    return SimpleNamespace(
        orchestrator=orchestrator,
        controller=controller,
        pool=pool,
        clients=clients,
        replacements=replacements,
        logger=logger,
        registry=registry,
        values=values,
        run=run,
    )


@pytest.mark.asyncio
async def test_unregister_and_rejoin_refreshes_snapshot_through_run(lifecycle):
    env = lifecycle

    async def scenario():
        await env.orchestrator.publish_stage_post_warmup_memory()
        assert env.values(0) == (11, 13)
        assert env.values(1) == (17, 19)
        env.orchestrator._update_stage_replica_waiting(0, 0, 3)
        env.orchestrator._update_stage_replica_waiting(0, 1, 5)

        await env.controller.handle_unregister(0, "tcp://old-0")
        assert env.values(0) == (None, None)
        assert env.values(1) == (17, 19)
        assert env.orchestrator._stage_replica_waiting == {(0, 1): 5}

        replacement = MemoryClient("tcp://new-0", 101, 103)
        env.replacements[0] = replacement
        await env.controller.handle_register(0, 0)
        await _wait_until(lambda: env.values(0) == (101, 103))
        assert replacement.calls == 1
        assert env.values(1) == (17, 19)
        assert env.orchestrator._stage_replica_to_engine_idx[(0, 0)] == 0

    await env.run(scenario)


@pytest.mark.asyncio
async def test_dead_replica_eviction_preserves_live_memory_samples(lifecycle):
    env = lifecycle
    await env.orchestrator.publish_stage_post_warmup_memory()
    assert env.values(0) == (11, 13)
    assert env.values(1) == (17, 19)

    await env.orchestrator._handle_dead_replica(0, 0, EngineDeadError("engine failure"))

    assert env.pool.live_replica_ids() == [1]
    assert env.values(0) == (None, None)
    assert env.values(1) == (17, 19)


@pytest.mark.asyncio
async def test_public_exposition_absent_then_zero(lifecycle):
    env = lifecycle
    names = (
        "vllm_omni:stage_post_warmup_memory_allocated_bytes",
        "vllm_omni:stage_post_warmup_memory_reserved_bytes",
    )

    def families():
        return {
            family.name: family for family in text_string_to_metric_families(generate_latest(env.registry).decode())
        }

    before = families()
    for name in names:
        assert before[name].type == "gauge"
        assert before[name].samples == []

    env.clients[0].snapshot = {"allocated_bytes": 0, "reserved_bytes": 0}
    await env.orchestrator.publish_stage_post_warmup_memory()

    after = families()
    for name, other_value in zip(names, (17, 19)):
        assert after[name].type == "gauge"
        samples = after[name].samples
        assert len(samples) == 2
        assert {(sample.name, tuple(sorted(sample.labels.items())), sample.value) for sample in samples} == {
            (name, (("model_name", "m"), ("replica", "0"), ("stage", "0")), 0),
            (name, (("model_name", "m"), ("replica", "1"), ("stage", "0")), other_value),
        }


@pytest.mark.asyncio
async def test_new_replica_registers_metric_slot(lifecycle):
    env = lifecycle

    async def scenario():
        await env.orchestrator.publish_stage_post_warmup_memory()
        env.replacements[4] = MemoryClient("tcp://new-4", 101, 103)
        await env.controller.handle_register(0, 4)
        await _wait_until(lambda: env.values(4) == (101, 103))
        engine_idx = env.orchestrator._stage_replica_to_engine_idx[(0, 4)]
        assert env.logger.stage_replica_map[engine_idx] == ("0", "4")
        assert env.values(0) == (11, 13)
        assert env.values(1) == (17, 19)

    await env.run(scenario)


@pytest.mark.asyncio
async def test_disabled_stats_skip_startup_and_join_rpcs(lifecycle):
    env = lifecycle
    env.orchestrator._stat_logger = None

    async def scenario():
        await env.orchestrator.publish_stage_post_warmup_memory()
        new_client = MemoryClient("tcp://new-4")
        env.replacements[4] = new_client
        await env.controller.handle_register(0, 4)
        await _wait_until(lambda: (0, 4) in env.controller._attached_remote_replicas)
        await env.controller.drain_tasks(timeout=2)
        assert all(client.calls == 0 for client in [*env.clients, new_client])
        assert env.values(4) == (None, None)

    await env.run(scenario)


@pytest.mark.asyncio
async def test_unsupported_replacement_leaves_snapshot_absent(lifecycle):
    env = lifecycle

    async def scenario():
        await env.orchestrator.publish_stage_post_warmup_memory()
        await env.controller.handle_unregister(0, "tcp://old-0")
        replacement = MemoryClient("tcp://unsupported", supported=False)
        env.replacements[0] = replacement
        await env.controller.handle_register(0, 0)
        await _wait_until(lambda: replacement.calls == 1)
        await env.controller.drain_tasks(timeout=2)
        assert env.values(0) == (None, None)
        assert env.values(1) == (17, 19)

    await env.run(scenario)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["startup", "join"])
async def test_detached_rpc_cannot_overwrite_replacement_snapshot(lifecycle, source):
    env = lifecycle

    async def scenario():
        old = env.clients[0] if source == "startup" else MemoryClient("tcp://slow-0")
        old.release.clear()
        startup_task = None
        try:
            if source == "startup":
                startup_task = asyncio.create_task(env.orchestrator.publish_stage_post_warmup_memory())
            else:
                await env.controller.handle_unregister(0, "tcp://old-0")
                env.replacements[0] = old
                await env.controller.handle_register(0, 0)
            await asyncio.wait_for(old.started.wait(), timeout=2)
            if source == "join":
                # The snapshot probe must not hold registration open.
                await _wait_until(lambda: (0, 0) in env.controller._attached_remote_replicas)
            await env.controller.handle_unregister(0, old.client_addresses["input_address"])
            assert env.values(0) == (None, None)

            replacement = MemoryClient("tcp://replacement-0", 101, 103)
            env.replacements[0] = replacement
            await env.controller.handle_register(0, 0)
            await _wait_until(lambda: env.values(0) == (101, 103))
            old.release.set()
            if startup_task is not None:
                await startup_task
            await env.controller.drain_tasks(timeout=2)
            assert env.values(0) == (101, 103)
            assert env.pool.clients[0] is replacement
        finally:
            old.release.set()
            if startup_task is not None:
                await startup_task

    await env.run(scenario)


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_error", [False, True])
async def test_shutdown_removes_memory_snapshots(lifecycle, monkeypatch, shutdown_error):
    env = lifecycle
    await env.orchestrator.publish_stage_post_warmup_memory()
    assert env.values(0) == (11, 13)
    assert env.values(1) == (17, 19)
    if shutdown_error:

        def fail_shutdown():
            raise RuntimeError("client shutdown failed")

        monkeypatch.setattr(env.clients[0], "shutdown", fail_shutdown)

    env.orchestrator._shutdown_stages()
    assert env.values(0) == (None, None)
    assert env.values(1) == (None, None)
    # A recreated logger reuses these families in the same process. An
    # unsupported replacement must not inherit the previous engine's values.
    replacement_logger = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
    replacement_logger._init_stage_post_warmup_memory_families(env.registry)
    assert (
        replacement_logger._stage_post_warmup_memory_allocated_family
        is env.logger._stage_post_warmup_memory_allocated_family
    )
    assert env.values(0) == (None, None)
    assert env.values(1) == (None, None)
    env.orchestrator._shutdown_stages()
    await env.orchestrator.publish_stage_post_warmup_memory()
    assert [client.calls for client in env.clients] == [1, 1]
    assert env.values(0) == (None, None)


@pytest.mark.asyncio
async def test_shutdown_discards_pending_memory_reply(lifecycle):
    env = lifecycle
    client = env.clients[0]
    client.release.clear()
    task = asyncio.create_task(env.orchestrator._publish_replica_post_warmup_memory(0, 0, client))
    try:
        await asyncio.wait_for(client.started.wait(), timeout=2)
        env.orchestrator._shutdown_stages()
    finally:
        client.release.set()
        await task
    assert env.values(0) == (None, None)
