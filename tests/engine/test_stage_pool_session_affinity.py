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
from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator import ReplicaStatus
from vllm_omni.engine.stage_pool import StagePool, session_id_of, session_reset_requested

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


def test_a_dead_holder_is_recorded_as_a_cold_restart(mocker):
    """A session that lost its replica must not look like a healthy binding."""
    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 1)

    pool.mark_replica_unavailable(holder)

    assert pool.session_replica_id("s1") is None, "a dead replica still claims the session"
    assert pool.is_session_stale("s1") is True, "the loss was not recorded"

    rebound = pool.select_replica_id("r2", session_id="s1")
    assert rebound != holder
    assert pool.session_replica_id("s1") == rebound
    assert pool.is_session_stale("s1") is True, "a cold restart erased the history loss"


def test_a_continuation_may_be_refused_when_its_holder_is_gone(mocker):
    """The router owns the decision, so a caller can refuse instead of going cold."""
    from vllm_omni.engine.stage_pool import StageUnavailableError

    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    # Mark the holder dead without releasing its routing state: this is the window
    # in which the pool still knows where the session was.
    pool._unavailable_replicas.add(holder)

    with pytest.raises(StageUnavailableError, match="cannot be served from its history"):
        pool.session_replica_id("s1", allow_cold=False)


def test_a_stale_session_forgets_the_loss_only_when_released(mocker):
    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    pool.mark_replica_unavailable(holder)

    assert pool.is_session_stale("s1") is True
    pool.release_session("s1")
    assert pool.is_session_stale("s1") is False


def test_a_pin_that_contradicts_the_session_holder_is_refused(mocker):
    """A caller pin and a session binding must not be two ledgers."""
    from vllm_omni.engine.stage_pool import StageUnavailableError

    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    pinned = 1 - holder
    pool.reserve_requested_replica("r9", pinned)

    with pytest.raises(StageUnavailableError, match="pinned replica"):
        pool.preselect_replica_id("r9", session_id="s1")


def test_a_pin_on_the_holder_keeps_both_ledgers_in_step(mocker):
    pool = _pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    pool.reserve_requested_replica("r9", holder)

    assert pool.preselect_replica_id("r9", session_id="s1") == holder
    assert pool.session_replica_id("s1") == holder


def test_a_pin_adopts_the_session_when_it_has_no_holder_yet(mocker):
    pool = _pool(mocker)
    pool.reserve_requested_replica("r9", 1)

    assert pool.preselect_replica_id("r9", session_id="fresh") == 1
    assert pool.session_replica_id("fresh") == 1


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


def test_the_serialized_tick_is_the_stage_pool_session_identity():
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

    class _TickOnly:
        extra_args = ARDiffusionTickRequest(
            session_id="tick-session", request_id="req-1", chunk_index=0, reset=True
        ).to_extra_args()

    class _ConflictingLegacy:
        extra_args = {
            "session_id": "legacy-session",
            "reset": False,
            **ARDiffusionTickRequest(
                session_id="tick-session", request_id="req-1", chunk_index=0, reset=True
            ).to_extra_args(),
        }

    assert session_id_of(_TickOnly()) == "tick-session"
    assert session_reset_requested(_TickOnly()) is True
    assert session_id_of(_ConflictingLegacy()) == "tick-session"
    assert session_reset_requested(_ConflictingLegacy()) is True


# ---- Distributed mode: a session hit must also record the request's route ----


def _distributed_pool(mocker, replicas: int = 2) -> StagePool:
    pool = _pool(mocker, replicas)
    pool.attach_hub(mocker.Mock())
    pool.attach_load_balancer(mocker.Mock())
    for replica_id in range(replicas):
        pool.add_client(f"tcp://replica-{replica_id}", mocker.Mock(), replica_id=replica_id)
    pool._hub.get_replicas_for_stage.return_value = SimpleNamespace(
        replicas=[
            SimpleNamespace(input_addr=f"tcp://replica-{replica_id}", status=ReplicaStatus.UP)
            for replica_id in range(replicas)
        ]
    )
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


def test_fresh_preselection_binds_session_before_follow_up(mocker):
    pool = _distributed_pool(mocker)
    pool._lb.select.return_value = 1
    first = pool.preselect_replica_id("first", session_id="new-session")
    pool._lb.select.return_value = 0
    follow_up = pool.preselect_replica_id("follow-up", session_id="new-session")
    assert first == follow_up == 1
    assert pool.session_replica_id("new-session", allow_cold=False) == 1
    assert pool.get_bound_replica_id("first") == 1
    assert pool.get_bound_replica_id("follow-up") == 1
    pool._lb.select.assert_called_once()


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


# ---- Diffusion submit path: the request's session decides the replica ----


def _diffusion_pool(mocker, replicas: int = 2):
    pool = _pool(mocker, replicas)
    for replica_id in range(replicas):
        # Keep the replica object (its ``stage_type`` selects the diffusion path)
        # and give it just the async submit the pool calls.
        pool.clients[replica_id].add_request_async = mocker.AsyncMock()
    return pool


def _submit(pool, mocker, request_id: str, *, extra: dict | None = None):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    params = OmniDiffusionSamplingParams(extra_args=extra or {})
    req_state = mocker.Mock(sampling_params_list=[params], trace=None)
    return asyncio.run(pool.submit_initial(request_id, req_state, request=mocker.Mock(payload_sender_info=None)))


def test_submit_initial_routes_a_diffusion_follow_up_to_its_holder(mocker):
    """The submit path is where a session's request actually chooses a replica."""
    pool = _diffusion_pool(mocker)

    first = _submit(pool, mocker, "r1", extra={"session_id": "s1"})
    second = _submit(pool, mocker, "r2", extra={"session_id": "s1"})

    assert second == first, "the follow-up was submitted to another replica"
    assert pool.session_replica_id("s1") == first


def _submit_tick(pool, mocker, request_id: str, *, session_id: str, reset: bool = False) -> int:
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    tick = ARDiffusionTickRequest(session_id=session_id, request_id=request_id, chunk_index=0, reset=reset)
    params = OmniDiffusionSamplingParams(extra_args=tick.to_extra_args())
    req_state = mocker.Mock(sampling_params_list=[params], trace=None)
    return asyncio.run(pool.submit_initial(request_id, req_state, request=mocker.Mock(payload_sender_info=None)))


def test_submit_initial_reset_stays_on_the_holder_while_worker_releases_its_state(mocker):
    """A namespaced tick reset reaches the worker that owns its KV and grant."""
    pool = _diffusion_pool(mocker)
    first = _submit_tick(pool, mocker, "r1", session_id="s1")
    old_holder = pool.clients[first]
    other = pool.clients[1 - first]
    old_holder.add_request_async.reset_mock()

    second = _submit_tick(pool, mocker, "r2", session_id="s1", reset=True)

    assert second == first, "reset moved before the old worker could release its state"
    old_holder.add_request_async.assert_awaited_once()
    other.add_request_async.assert_not_awaited()
    assert pool.session_replica_id("s1") == first


def test_submit_initial_without_a_session_is_unaffected(mocker):
    pool = _diffusion_pool(mocker)

    _submit(pool, mocker, "r1")
    _submit(pool, mocker, "r2")

    assert pool.session_binding_count() == 0


@pytest.mark.parametrize("pin_kind", ["requested", "pinned"])
def test_submit_initial_rejects_explicit_replica_conflicting_with_session_holder(mocker, pin_kind):
    """The submit path must enforce affinity even when preselection was bypassed."""
    from vllm_omni.engine.stage_pool import StageUnavailableError
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pool = _diffusion_pool(mocker)
    holder = _submit(pool, mocker, "r1", extra={"session_id": "s1"})
    other = 1 - holder
    params = OmniDiffusionSamplingParams(extra_args={"session_id": "s1"})
    req_state = mocker.Mock(sampling_params_list=[params], trace=None)
    kwargs = {"requested_replica_id": other} if pin_kind == "requested" else {"pinned_replica_id": other}

    with pytest.raises(StageUnavailableError, match="held by replica"):
        asyncio.run(pool.submit_initial("r2", req_state, request=mocker.Mock(payload_sender_info=None), **kwargs))

    assert pool.session_replica_id("s1") == holder, "the rejected pin changed the session owner"


def test_submit_initial_refuses_continuation_after_holder_loss(mocker):
    """A continuation cannot silently start cold after its session holder dies."""
    from vllm_omni.engine.stage_pool import StageUnavailableError

    pool = _diffusion_pool(mocker)
    holder = pool.bind_session_replica("s1", 0)
    pool.mark_replica_unavailable(holder)

    with pytest.raises(StageUnavailableError, match="cannot be served from its history"):
        _submit(pool, mocker, "r2", extra={"session_id": "s1"})


def test_submit_initial_reset_can_start_cold_after_holder_loss(mocker):
    """An explicit reset discards stale history instead of refusing recovery."""
    pool = _diffusion_pool(mocker)
    holder = pool.bind_session_replica("s1", 0)
    pool.mark_replica_unavailable(holder)

    replica = _submit(pool, mocker, "r2", extra={"session_id": "s1", "reset": True})

    assert replica != holder
    assert pool.session_replica_id("s1") == replica
    assert pool.is_session_stale("s1") is True


def test_distributed_session_lookup_refuses_an_attached_but_down_holder(mocker):
    """An attached slot alone is not enough when the coordinator marks it DOWN."""
    from vllm_omni.engine.stage_pool import StageUnavailableError

    pool = _distributed_pool(mocker)
    holder = pool.bind_session_replica("s1", 1)
    snapshot = pool._hub.get_replicas_for_stage.return_value
    snapshot.replicas[holder].status = ReplicaStatus.DOWN

    with pytest.raises(StageUnavailableError, match="cannot be served from its history"):
        pool.session_replica_id("s1", allow_cold=False)

    assert pool.session_replica_id("s1", allow_cold=True) is None
    assert pool.is_session_stale("s1") is True
