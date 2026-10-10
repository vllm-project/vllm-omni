# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Finished requests release inter-stage transfer resources in per-replica batches."""

import asyncio
from dataclasses import dataclass, field

import pytest

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _ModelConfig:
    async_chunk: bool = True


@dataclass
class _StageVllmConfig:
    model_config: _ModelConfig = field(default_factory=_ModelConfig)


class _Replica:
    def __init__(self, *, hang: bool = False, fail: bool = False):
        self.calls: list[list[str]] = []
        self.started = asyncio.Event()
        # Released for a healthy replica; a hung replica blocks until the test sets it.
        self.gate = asyncio.Event()
        if not hang:
            self.gate.set()
        self.fail = fail

    async def call_utility_async(self, method, request_ids):
        assert method == "omni_release_request_resources"
        self.calls.append(list(request_ids))
        self.started.set()
        await self.gate.wait()
        if self.fail:
            raise RuntimeError("replica gone")


def _pool(*replicas: _Replica) -> StagePool:
    return StagePool(
        0,
        list(replicas),  # type: ignore[arg-type]
        stage_vllm_config=_StageVllmConfig(),
    )


async def _settle(pool: StagePool) -> None:
    await asyncio.wait_for(asyncio.gather(*pool._release_flushers.values()), timeout=2.0)


@pytest.mark.asyncio
async def test_completions_in_one_loop_pass_share_one_rpc_per_replica() -> None:
    replicas = (_Replica(), _Replica())
    pool = _pool(*replicas)

    pool.schedule_release_request_resources(["a"])
    pool.schedule_release_request_resources(["b", "a"])
    pool.schedule_release_request_resources(["c"])
    await _settle(pool)

    for replica in replicas:
        assert replica.calls == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_completions_during_an_rpc_join_that_replicas_next_batch() -> None:
    replica = _Replica(hang=True)
    pool = _pool(replica)

    pool.schedule_release_request_resources(["a"])
    await asyncio.wait_for(replica.started.wait(), timeout=2.0)
    pool.schedule_release_request_resources(["b"])
    pool.schedule_release_request_resources(["c"])
    await asyncio.sleep(0)
    assert replica.calls == [["a"]]  # never two RPCs in flight for one replica

    replica.gate.set()
    await _settle(pool)
    assert replica.calls == [["a"], ["b", "c"]]


@pytest.mark.asyncio
async def test_hung_replica_does_not_delay_healthy_replica_batches(monkeypatch) -> None:
    hung, healthy = _Replica(hang=True), _Replica()
    pool = _pool(hung, healthy)
    monkeypatch.setattr(pool, "RELEASE_RPC_TIMEOUT_S", 0.2)
    warn = []
    monkeypatch.setattr("vllm_omni.engine.stage_pool.logger.warning", lambda *a: warn.append(a))

    pool.schedule_release_request_resources(["a"])
    await asyncio.sleep(0.01)
    pool.schedule_release_request_resources(["b"])
    await asyncio.sleep(0.01)

    # Well before the hung replica's timeout, the healthy one released both.
    assert healthy.calls == [["a"], ["b"]]
    assert hung.calls == [["a"]]

    await _settle(pool)
    assert hung.calls == [["a"], ["b"]]
    assert len(warn) == 2  # both of the hung replica's RPCs timed out


@pytest.mark.asyncio
async def test_removed_replica_drops_its_pending_releases() -> None:
    replica = _Replica(hang=True)
    pool = _pool(replica)

    pool.schedule_release_request_resources(["a"])
    await asyncio.wait_for(replica.started.wait(), timeout=2.0)
    pool.schedule_release_request_resources(["b"])
    pool.clients[0] = None
    replica.gate.set()
    await _settle(pool)

    assert replica.calls == [["a"]]
    assert not pool._pending_releases


@pytest.mark.asyncio
async def test_failed_release_is_logged_and_the_replica_recovers(monkeypatch) -> None:
    replica = _Replica(fail=True)
    pool = _pool(replica)
    warn = []
    monkeypatch.setattr("vllm_omni.engine.stage_pool.logger.warning", lambda *a: warn.append(a))

    pool.schedule_release_request_resources(["a"])
    await _settle(pool)
    pool.schedule_release_request_resources(["b"])
    await _settle(pool)

    assert replica.calls == [["a"], ["b"]]
    assert len(warn) == 2


def test_orchestrator_queues_releases_on_every_stage_pool(mocker) -> None:
    pools = [mocker.Mock(spec=StagePool) for _ in range(2)]
    orch = Orchestrator.__new__(Orchestrator)
    orch.stage_pools = pools

    orch._release_stage_transfer_resources(["r1", "r2"])
    orch._release_stage_transfer_resources([])

    for pool in pools:
        pool.schedule_release_request_resources.assert_called_once_with(["r1", "r2"])
