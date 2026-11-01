# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session affinity in the engine's stage pool.

The deployment diagram puts session management above the pool and requires a
session's follow-up requests to reach the replica that holds its state. The pool
routed per request id and round-robin per replica, so a follow-up could land on a
replica with no history for that session.
"""

from __future__ import annotations

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