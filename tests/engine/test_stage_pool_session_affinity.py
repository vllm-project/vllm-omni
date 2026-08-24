# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-to-replica affinity in StagePool (RFC #6226, Phase 0).

A long-lived AR-Diffusion session keeps its persistent state on one worker, so
every tick of that session has to land on the replica owning it. These tests
cover both dispatch paths (legacy ``select_replica_id`` and distributed
``pick``) plus the fail-closed behavior when the owning replica disappears.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator.messages import ReplicaInfo, ReplicaStatus
from vllm_omni.engine.stage_pool import (
    SessionOwnerLostError,
    StagePool,
    StageUnavailableError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _client(addr: str) -> SimpleNamespace:
    # StagePool._client_input_addr() reads `request_address` for diffusion
    # clients; that address is what keys the replica in the hub.
    return SimpleNamespace(request_address=addr, stage_type="diffusion", final_output=False)


def _replica(addr: str, status: ReplicaStatus = ReplicaStatus.UP) -> ReplicaInfo:
    return ReplicaInfo(
        input_addr=addr,
        output_addr=f"{addr}-out",
        stage_id=0,
        status=status,
        queue_length=0,
        last_heartbeat=0.0,
        registered_at=0.0,
    )


class _Hub:
    def __init__(self, replicas: list[ReplicaInfo]) -> None:
        self.replicas = replicas

    def get_replicas_for_stage(self, _stage_id: int) -> SimpleNamespace:
        return SimpleNamespace(replicas=list(self.replicas))


class _RoundRobinLB:
    """Deterministic stand-in for the real load balancer."""

    def __init__(self) -> None:
        self._n = 0

    def select(self, _task, candidates):
        index = self._n % len(candidates)
        self._n += 1
        return index


def _local_pool(num_replicas: int = 3) -> StagePool:
    """A pool with no hub attached: the legacy select_replica_id path."""
    return StagePool(0, [_client(f"tcp://replica-{i}") for i in range(num_replicas)])


def _distributed_pool(num_replicas: int = 3) -> tuple[StagePool, _Hub]:
    clients = [_client(f"tcp://replica-{i}") for i in range(num_replicas)]
    pool = StagePool(0, clients)
    hub = _Hub([_replica(c.request_address) for c in clients])
    pool.attach_hub(hub)
    pool.attach_load_balancer(_RoundRobinLB())
    return pool, hub


# ---------------------------------------------------------------- legacy path


def test_ticks_of_one_session_stay_on_the_owner_replica():
    pool = _local_pool()

    owner = pool.select_replica_id("tick-0", session_id="world-1")
    later = [pool.select_replica_id(f"tick-{i}", session_id="world-1") for i in range(1, 12)]

    assert later == [owner] * 11


def test_distinct_sessions_are_spread_by_the_existing_policy():
    pool = _local_pool()

    owners = {sid: pool.select_replica_id(f"{sid}-0", session_id=sid) for sid in ("a", "b", "c")}

    # Placement of a *new* session still uses the ordinary round-robin policy.
    assert sorted(owners.values()) == [0, 1, 2]
    # And each one then stays put.
    assert all(pool.select_replica_id(f"{sid}-1", session_id=sid) == owner for sid, owner in owners.items())


def test_stateless_routing_is_unchanged():
    """Requests that pass no session_id keep the plain round-robin behavior."""
    pool = _local_pool()

    assert [pool.select_replica_id(f"r{i}") for i in range(6)] == [0, 1, 2, 0, 1, 2]


def test_stateless_requests_still_spread_while_a_session_is_pinned():
    """A pinned session must not starve or capture ordinary request routing."""
    pool = _local_pool()
    owner = pool.select_replica_id("tick-0", session_id="world-1")

    # Interleave ticks of the pinned session with ordinary requests.
    stateless = []
    for i in range(6):
        pool.select_replica_id(f"tick-{i + 1}", session_id="world-1")
        stateless.append(pool.select_replica_id(f"r{i}"))

    assert sorted(set(stateless)) == [0, 1, 2]  # every replica still reachable
    assert pool.select_replica_id("tick-99", session_id="world-1") == owner


def test_release_lets_the_session_be_placed_again():
    pool = _local_pool()
    pool.select_replica_id("tick-0", session_id="world-1")

    pool.release_session("world-1")

    assert not pool.has_session("world-1")
    assert pool.get_session_replica_id("world-1") is None
    # Re-placing the same id afterwards is allowed.
    pool.select_replica_id("tick-1", session_id="world-1")
    assert pool.has_session("world-1")


def test_release_is_idempotent_and_tolerates_unknown_sessions():
    pool = _local_pool()
    pool.select_replica_id("tick-0", session_id="world-1")

    pool.release_session("world-1")
    pool.release_session("world-1")
    pool.release_session("never-existed")

    assert not pool.has_session("world-1")


def test_bind_session_rejects_an_out_of_range_replica():
    pool = _local_pool(2)

    with pytest.raises(ValueError, match="no replica 7"):
        pool.bind_session("world-1", 7)


def test_session_ids_for_replica_reports_ownership():
    pool = _local_pool(2)
    owners = {sid: pool.select_replica_id(f"{sid}-0", session_id=sid) for sid in ("a", "b", "c", "d")}

    expected = sorted(sid for sid, owner in owners.items() if owner == 0)

    assert sorted(pool.session_ids_for_replica(0)) == expected


# ----------------------------------------------------------- failure handling


def test_lost_owner_fails_closed_instead_of_silently_rerouting():
    pool = _local_pool()
    owner = pool.select_replica_id("tick-0", session_id="world-1")

    pool.mark_replica_unavailable(owner)

    # Phase 0 does not migrate or rebuild state, so the tick must fail rather
    # than land on a replica holding none of this session's state.
    with pytest.raises(SessionOwnerLostError, match="world-1"):
        pool.select_replica_id("tick-1", session_id="world-1")


def test_lost_owner_keeps_failing_until_the_session_is_released():
    pool = _local_pool()
    owner = pool.select_replica_id("tick-0", session_id="world-1")
    pool.mark_replica_unavailable(owner)

    with pytest.raises(SessionOwnerLostError):
        pool.select_replica_id("tick-1", session_id="world-1")
    # The tombstone survives the first failure: a retry must not be re-placed
    # onto a healthy-but-empty replica.
    with pytest.raises(SessionOwnerLostError):
        pool.select_replica_id("tick-2", session_id="world-1")

    pool.release_session("world-1")

    assert pool.select_replica_id("tick-3", session_id="world-1") != owner


def test_session_owner_lost_is_a_stage_unavailable_error():
    """Keeps the orchestrator's dispatch guard failing one request, not the server."""
    assert issubclass(SessionOwnerLostError, StageUnavailableError)


def test_replica_death_leaves_other_sessions_alone():
    pool = _local_pool()
    owners = {sid: pool.select_replica_id(f"{sid}-0", session_id=sid) for sid in ("a", "b", "c")}
    victim = owners["a"]

    pool.mark_replica_unavailable(victim)

    for sid, owner in owners.items():
        if owner == victim:
            continue
        assert pool.select_replica_id(f"{sid}-1", session_id=sid) == owner


def test_replica_death_drops_the_route_but_remembers_the_session():
    pool = _local_pool()
    owner = pool.select_replica_id("tick-0", session_id="world-1")

    orphaned = pool.mark_replica_unavailable(owner)

    assert "world-1" not in orphaned  # request ids, not session ids
    assert pool.get_session_replica_id("world-1") is None  # no dangling route
    assert pool.has_session("world-1")  # but still known, so ticks fail closed


# ----------------------------------------------------------- distributed path


@pytest.mark.asyncio
async def test_distributed_ticks_stay_on_the_owner_replica():
    pool, _hub = _distributed_pool()

    owner = await pool.pick("tick-0", session_id="world-1")
    later = [await pool.pick(f"tick-{i}", session_id="world-1") for i in range(1, 10)]

    assert later == [owner] * 9


@pytest.mark.asyncio
async def test_distributed_stateless_requests_still_load_balance():
    pool, _hub = _distributed_pool()

    picks = [await pool.pick(f"r{i}") for i in range(6)]

    assert sorted(set(picks)) == [0, 1, 2]


@pytest.mark.asyncio
async def test_distributed_down_owner_fails_closed():
    pool, hub = _distributed_pool()
    owner = await pool.pick("tick-0", session_id="world-1")
    for replica in hub.replicas:
        if pool.get_replica_id_by_addr(replica.input_addr) == owner:
            replica.status = ReplicaStatus.DOWN

    with pytest.raises(SessionOwnerLostError, match="world-1"):
        await pool.pick("tick-1", session_id="world-1")


@pytest.mark.asyncio
async def test_invalidate_addr_orphans_sessions_on_that_replica():
    pool, _hub = _distributed_pool()
    owner = await pool.pick("tick-0", session_id="world-1")

    pool.invalidate_addr(f"tcp://replica-{owner}")

    assert pool.get_session_replica_id("world-1") is None
    with pytest.raises(SessionOwnerLostError):
        await pool.pick("tick-1", session_id="world-1")


@pytest.mark.asyncio
async def test_concurrent_first_ticks_cannot_create_two_owners():
    """The RFC's key correctness requirement for the initial placement."""
    clients = [_client(f"tcp://replica-{i}") for i in range(3)]
    pool = StagePool(0, clients)
    # An empty replica set parks every pick() in its bounded retry loop, which
    # is the only suspension point where two first ticks could interleave.
    hub = _Hub([])
    pool.attach_hub(hub)
    pool.attach_load_balancer(_RoundRobinLB())
    pool.DISPATCH_RETRY_INTERVAL_S = 0.01
    pool.DISPATCH_WAIT_TIMEOUT_S = 5.0

    tasks = [asyncio.create_task(pool.pick(f"tick-{i}", session_id="world-1")) for i in range(8)]
    await asyncio.sleep(0.05)
    hub.replicas = [_replica(c.request_address) for c in clients]
    owners = await asyncio.gather(*tasks)

    assert len(set(owners)) == 1
