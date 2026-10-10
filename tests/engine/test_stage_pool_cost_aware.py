# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from time import time
from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator import (
    CostAwareBalancer,
    ReplicaInfo,
    ReplicaList,
    ReplicaStatus,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _replica(input_addr: str, queue_length: int = 0) -> ReplicaInfo:
    now = time()
    return ReplicaInfo(
        input_addr=input_addr,
        output_addr=input_addr + "-out",
        stage_id=0,
        status=ReplicaStatus.UP,
        queue_length=queue_length,
        last_heartbeat=now,
        registered_at=now,
    )


class _FakeHub:
    def __init__(self, replicas: list[ReplicaInfo]):
        self._replica_list = ReplicaList(replicas=replicas, timestamp=time())

    def get_replicas_for_stage(self, stage_id: int) -> ReplicaList:  # noqa: ARG002
        return self._replica_list


class _FakeStageClient:
    stage_type = "llm"
    final_output = False

    def __init__(self, input_address: str):
        self.client_addresses = {"input_address": input_address}


class _FailingDiffusionClient:
    stage_type = "diffusion"
    final_output = False

    def __init__(self, request_address: str):
        self.request_address = request_address

    async def add_request_async(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("submission failed")


def _make_distributed_pool(
    balancer: CostAwareBalancer,
    addrs: list[str],
    queue_lengths: list[int] | None = None,
) -> StagePool:
    clients = [_FakeStageClient(addr) for addr in addrs]
    pool = StagePool(0, clients)  # type: ignore[arg-type]
    lengths = queue_lengths or [0] * len(addrs)
    pool.attach_hub(_FakeHub([_replica(addr, length) for addr, length in zip(addrs, lengths)]))
    pool.attach_load_balancer(balancer)
    return pool


def _make_local_pool(balancer: CostAwareBalancer, addrs: list[str]) -> StagePool:
    pool = StagePool(0, [_FakeStageClient(addr) for addr in addrs])  # type: ignore[arg-type]
    pool.attach_load_balancer(balancer)
    return pool


@pytest.mark.asyncio
async def test_pick_forwards_sampling_params_to_cost_aware_balancer():
    balancer = CostAwareBalancer()
    pool = _make_distributed_pool(balancer, ["a", "b"])

    params = SimpleNamespace(height=1024, width=1024, num_frames=1)
    await pool.pick("r1", sampling_params=params)

    assert balancer._reservations["r1"][1] == pytest.approx(4.0)


def test_preselect_forwards_sampling_params_to_cost_aware_balancer():
    balancer = CostAwareBalancer()
    pool = _make_distributed_pool(balancer, ["a", "b"])

    params = SimpleNamespace(height=512, width=512, num_frames=1)
    assert pool.preselect_replica_id("r1", sampling_params=params) is not None

    assert balancer._reservations["r1"][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_release_binding_releases_reservation_idempotently():
    balancer = CostAwareBalancer()
    pool = _make_distributed_pool(balancer, ["a", "b"])
    await pool.pick("r1", sampling_params=SimpleNamespace(height=512, width=512))

    pool.release_binding("r1")
    pool.release_binding("r1")

    assert "r1" not in balancer._reservations


@pytest.mark.asyncio
async def test_invalidate_addr_releases_reservation():
    balancer = CostAwareBalancer()
    pool = _make_distributed_pool(balancer, ["a", "b"])
    await pool.pick("r1", sampling_params=SimpleNamespace(height=512, width=512))
    selected_addr = pool._affinity["r1"]

    assert pool.invalidate_addr(selected_addr) == ["r1"]
    assert "r1" not in balancer._reservations


@pytest.mark.asyncio
async def test_sticky_affinity_does_not_double_count():
    balancer = CostAwareBalancer()
    pool = _make_distributed_pool(balancer, ["a", "b"])

    first = await pool.pick("r1", sampling_params=SimpleNamespace(height=512, width=512))
    second = await pool.pick("r1", sampling_params=SimpleNamespace(height=1024, width=1024))

    assert second == first
    assert balancer._reservations["r1"][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_explicit_task_overrides_sampling_params():
    balancer = CostAwareBalancer()
    pool = _make_distributed_pool(balancer, ["a", "b"])

    await pool.pick(
        "r1",
        task={"estimated_cost": 99.0},
        sampling_params=SimpleNamespace(height=512, width=512),
    )

    assert balancer._reservations["r1"][1] == 99.0


@pytest.mark.asyncio
async def test_local_pool_routes_by_projected_cost(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        lambda candidates: candidates[0],
    )
    pool = _make_local_pool(CostAwareBalancer(), ["a", "b"])

    heavy = await pool.pick("heavy", task={"estimated_cost": 100.0})
    light_1 = await pool.pick("light-1", task={"estimated_cost": 1.0})
    light_2 = await pool.pick("light-2", task={"estimated_cost": 1.0})

    assert heavy == 0
    assert light_1 == light_2 == 1


def test_local_pool_without_attached_balancer_keeps_legacy_round_robin():
    pool = StagePool(0, [_FakeStageClient("a"), _FakeStageClient("b")])  # type: ignore[arg-type]

    assert pool.select_replica_id("r1") == 0
    assert pool.select_replica_id("r2") == 1


@pytest.mark.asyncio
async def test_failed_diffusion_submission_releases_reservation():
    balancer = CostAwareBalancer()
    address = "tcp://diffusion:1000"
    pool = StagePool(0, [_FailingDiffusionClient(address)])  # type: ignore[list-item]
    pool.attach_hub(_FakeHub([_replica(address)]))
    pool.attach_load_balancer(balancer)
    params = OmniDiffusionSamplingParams(height=512, width=512)
    request_state = SimpleNamespace(sampling_params_list=[params])

    with pytest.raises(RuntimeError, match="submission failed"):
        await pool.submit_initial("r1", request_state, object())

    assert pool.get_bound_replica_id("r1") is None
    assert "r1" not in balancer._reservations


@pytest.mark.asyncio
async def test_failed_unbound_diffusion_update_releases_reservation():
    balancer = CostAwareBalancer()
    address = "tcp://diffusion:1000"
    pool = StagePool(0, [_FailingDiffusionClient(address)])  # type: ignore[list-item]
    pool.attach_hub(_FakeHub([_replica(address)]))
    pool.attach_load_balancer(balancer)
    params = OmniDiffusionSamplingParams(height=512, width=512)
    request_state = SimpleNamespace(sampling_params_list=[params])

    with pytest.raises(RuntimeError, match="submission failed"):
        await pool.submit_update("r1", request_state, object())

    assert pool.get_bound_replica_id("r1") is None
    assert "r1" not in balancer._reservations
