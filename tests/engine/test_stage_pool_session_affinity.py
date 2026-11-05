# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session affinity in the engine's stage pool.

The deployment diagram puts session management above the pool and requires a
session's follow-up requests to reach the replica that holds its state. The pool
routed per request id and round-robin per replica, so a follow-up could land on a
replica with no history for that session.
"""

from __future__ import annotations

import asyncio

import pytest

from vllm_omni.engine.stage_pool import StagePool, session_id_of

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Replica:
    stage_type = "diffusion"
    final_output = False


def _pool(mocker, replicas: int = 2) -> StagePool:
    config = mocker.Mock(model_config=mocker.Mock(async_chunk=False))
    return StagePool(0, [_Replica() for _ in range(replicas)], stage_vllm_config=config)


def test_a_follow_up_lands_on_the_replica_that_holds_the_session(mocker):
    pool = _pool(mocker)
    first = pool.select_replica_id("r1", session_id="s1")

    second = pool.select_replica_id("r2", session_id="s1")

    assert second == first, "the follow-up was routed away from the session's replica"
    assert pool.session_replica_id("s1") == first


def test_different_sessions_can_land_on_different_replicas(mocker):
    pool = _pool(mocker)
    pool.select_replica_id("r1", session_id="s1")

    other = pool.select_replica_id("r2", session_id="s2")

    assert other != pool.session_replica_id("s1")


def test_a_session_whose_replica_died_is_rebound(mocker):
    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 1)

    pool.mark_replica_unavailable(holder)

    assert pool.session_replica_id("s1") is None, "a dead replica still claims the session"
    rebound = pool.select_replica_id("r2", session_id="s1")
    assert rebound != holder
    assert pool.session_replica_id("s1") == rebound


def test_releasing_a_session_frees_its_binding(mocker):
    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 0)

    assert pool.release_session("s1") == holder
    assert pool.session_replica_id("s1") is None
    assert pool.release_session("s1") is None


def test_binding_a_session_to_an_unavailable_replica_is_refused(mocker):
    from vllm_omni.engine.stage_pool import StageUnavailableError

    pool = _pool(mocker)
    pool.mark_replica_unavailable(1)

    with pytest.raises(StageUnavailableError):
        pool.bind_session_replica("s1", 1)


def test_an_active_session_outlives_newer_idle_ones(mocker, monkeypatch):
    """Eviction drops the least recently routed binding, not the oldest bind."""
    from vllm_omni.engine import stage_pool

    monkeypatch.setattr(stage_pool, "MAX_SESSION_BINDINGS", 2)
    pool = _pool(mocker)
    pool.select_replica_id("r1", session_id="active")
    pool.select_replica_id("r2", session_id="idle")
    pool.select_replica_id("r3", session_id="active")

    pool.select_replica_id("r4", session_id="new")

    assert pool.session_replica_id("active") is not None
    assert pool.session_replica_id("idle") is None
    assert pool.session_binding_count() == 2


def test_the_session_identity_is_read_from_the_request_params():
    class _Params:
        extra_args = {"session_id": "s1"}

    class _NoSession:
        extra_args = {}

    class _NoExtra:
        pass

    assert session_id_of(_Params()) == "s1"
    assert session_id_of(_NoSession()) is None
    assert session_id_of(_NoExtra()) is None


# ---- Distributed mode: a session hit must also record the request's route ----


def _distributed_pool(mocker, replicas: int = 2) -> StagePool:
    pool = _pool(mocker, replicas)
    pool.attach_hub(mocker.Mock())
    pool.attach_load_balancer(mocker.Mock())
    for replica_id in range(replicas):
        pool.add_client(f"tcp://replica-{replica_id}", mocker.Mock(), replica_id=replica_id)
    return pool


def test_a_distributed_session_follow_up_records_its_route(mocker):
    """pick() returns the holder; abort and interaction look the route up later."""
    pool = _distributed_pool(mocker)
    holder = pool.bind_session_replica("s1", 1)

    picked = asyncio.run(pool.pick("r2", session_id="s1"))

    assert picked == holder
    assert pool.get_bound_replica_id("r2") == holder
    assert pool._affinity["r2"] == "tcp://replica-1"


def test_a_session_hit_replaces_a_stale_affinity_entry(mocker):
    pool = _distributed_pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    pool._affinity["r2"] = "tcp://replica-0"

    asyncio.run(pool.pick("r2", session_id="s1"))

    assert pool._affinity["r2"] == "tcp://replica-1"
    assert pool.get_bound_replica_id("r2") == holder


def test_preselection_of_a_session_follow_up_records_its_route(mocker):
    pool = _distributed_pool(mocker)
    holder = pool.bind_session_replica("s1", 1)

    assert pool.preselect_replica_id("r3", session_id="s1") == holder
    assert pool.get_bound_replica_id("r3") == holder
    assert pool._affinity["r3"] == "tcp://replica-1"


def test_an_abort_reaches_the_replica_that_holds_the_session(mocker):
    pool = _distributed_pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    asyncio.run(pool.pick("r4", session_id="s1"))
    holder_client = pool.clients[holder]
    holder_client.abort_requests_async = mocker.AsyncMock()
    other_client = pool.clients[1 - holder]
    other_client.abort_requests_async = mocker.AsyncMock()

    asyncio.run(pool.abort_requests(["r4"]))

    holder_client.abort_requests_async.assert_awaited_once_with(["r4"])
    other_client.abort_requests_async.assert_not_awaited()


def test_a_mid_request_interaction_follows_the_session_holder(mocker):
    pool = _distributed_pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    asyncio.run(pool.pick("r5", session_id="s1"))
    holder_client = pool.clients[holder]
    holder_client.submit_interaction_async = mocker.AsyncMock(return_value=None)

    asyncio.run(pool.submit_interaction("r5", mocker.Mock()))

    holder_client.submit_interaction_async.assert_awaited_once()
