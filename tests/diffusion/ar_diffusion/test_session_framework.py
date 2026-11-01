# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework seams for session placement and capacity reservation.

The roadmap splits this work across layers (2.1/2.2 replica affinity, 2.3
scheduler-side reservation). These cases pin the decisions each layer owns:
which replica must serve a session, what a follow-up request has to reserve, and
what the runner does when either answer is "no".
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.kv_cache.session import NoisyKVSessionRegistry
from vllm_omni.experimental.ar_diffusion.reservation import (
    CapacityDemand,
    ReservationDeniedError,
    SessionCapacityLedger,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.session_affinity import SessionAffinityError, SessionRouter
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_the_first_request_binds_the_session_to_a_replica():
    router = SessionRouter()

    assert router.route("s1", [0, 1]) == 0
    assert router.owner("s1") == 0


def test_a_follow_up_routes_to_the_replica_that_owns_it():
    router = SessionRouter()
    router.bind("s1", 1)

    assert router.route("s1", [0, 1]) == 1
    assert router.route("s1", [1]) == 1


def test_a_session_whose_replica_died_is_refused():
    router = SessionRouter()
    router.bind("s1", 3)

    with pytest.raises(SessionAffinityError):
        router.route("s1", [0, 1])


def test_allow_cold_rebinds_the_session_and_records_the_loss():
    router = SessionRouter()
    router.bind("s1", 3)

    assert router.route("s1", [0, 1], allow_cold=True) == 0
    assert router.owner("s1") == 0
    assert router.is_stale("s1")


def test_release_replica_drops_the_sessions_it_held():
    router = SessionRouter()
    router.bind("s1", 2)
    router.bind("s2", 2)
    router.bind("s3", 0)

    dropped = router.release_replica(2)

    assert sorted(dropped) == ["s1", "s2"]
    assert router.owner("s1") is None
    assert router.is_stale("s1")
    assert router.owner("s3") == 0


def test_binding_the_same_session_to_another_replica_is_refused():
    router = SessionRouter()
    router.bind("s1", 0)

    with pytest.raises(SessionAffinityError):
        router.bind("s1", 1)


def test_no_live_replica_refuses_every_session():
    router = SessionRouter()

    with pytest.raises(SessionAffinityError):
        router.route("s1", [])


def test_a_follow_up_reserves_only_the_increment():
    ledger = SessionCapacityLedger(budget_bytes=1000)
    demand = CapacityDemand(versions=6, bytes=400)

    first = ledger.reserve("s1", demand)
    second = ledger.reserve("s1", demand)

    assert first.incremental and first.bytes == 400
    assert not second.incremental, "a follow-up reserved the history it already holds"
    assert ledger.held_bytes("s1") == 400
    assert ledger.used_bytes() == 400


def test_a_demand_beyond_the_budget_is_denied():
    ledger = SessionCapacityLedger(budget_bytes=100)

    with pytest.raises(ReservationDeniedError):
        ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))


def test_two_sessions_share_one_budget():
    ledger = SessionCapacityLedger(budget_bytes=600)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    with pytest.raises(ReservationDeniedError):
        ledger.reserve("s2", CapacityDemand(versions=6, bytes=400))

    ledger.release("s1")
    granted = ledger.reserve("s2", CapacityDemand(versions=6, bytes=400))

    assert granted.bytes == 400
    assert ledger.free_bytes() == 200


def test_release_returns_the_holding():
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    assert ledger.release("s1") == 400
    assert ledger.held_bytes("s1") == 0
    assert ledger.grant("s1") is None
    assert ledger.free_bytes() == 1000


def test_a_negative_demand_is_rejected():
    with pytest.raises(ValueError):
        CapacityDemand(versions=-1, bytes=10)


class _Capability:
    def bind_ar_diffusion_cell_context(self, ctx: object):
        del ctx
        return nullcontext()


def _request(**extra: object) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="a world",
        request_id="r1",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, extra_args={"session_id": "s1", **extra}),
    )


def _runner(mocker) -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.noisy_kv_cache = object()
    runner._ar_diffusion_cell_capability = _Capability()
    runner._noisy_sessions = NoisyKVSessionRegistry(capacity=1)
    runner._perf_e2e_times = []
    runner._session_router = None
    runner._replica_id = 0
    runner._capacity_ledger = None
    runner._session_reservation_versions = 0
    runner._session_reservation_bytes = 0
    ctx = SimpleNamespace(inflight=[], pending=[])
    mocker.patch.object(ARDiffusionModelRunner, "_make_chunk_context", return_value=ctx)
    mocker.patch.object(DiffusionModelRunner, "execute_model", return_value=DiffusionOutput(output=None))
    return runner


def test_the_runner_refuses_a_session_bound_to_another_replica(mocker):
    runner = _runner(mocker)
    router = SessionRouter()
    router.bind("s1", 3)
    runner.bind_session_router(router, replica_id=0)

    with pytest.raises(SessionAffinityError):
        runner.execute_model(_request())


def test_the_runner_reserves_the_session_before_running_it(mocker):
    runner = _runner(mocker)
    ledger = SessionCapacityLedger(budget_bytes=1000)
    runner.bind_capacity_ledger(ledger, versions=6, reserved_bytes=400)

    runner.execute_model(_request())

    grant = ledger.grant("s1")
    assert grant is not None and grant.bytes == 400
    assert ledger.held_bytes("s1") == 400


def test_the_runner_returns_a_request_whose_session_does_not_fit(mocker):
    runner = _runner(mocker)
    ledger = SessionCapacityLedger(budget_bytes=100)
    runner.bind_capacity_ledger(ledger, versions=6, reserved_bytes=400)

    with pytest.raises(ReservationDeniedError):
        runner.execute_model(_request())

    assert ledger.held_bytes("s1") == 0


def test_closing_a_session_returns_its_reservation(mocker):
    runner = _runner(mocker)
    ledger = SessionCapacityLedger(budget_bytes=1000)
    runner.bind_capacity_ledger(ledger, versions=6, reserved_bytes=400)
    runner.execute_model(_request())

    runner.execute_model(_request(close_session=True))

    assert ledger.held_bytes("s1") == 0
    assert runner._noisy_sessions.get("s1") is None