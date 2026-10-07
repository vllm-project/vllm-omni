# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from time import time

import pytest

from vllm_omni.distributed.omni_coordinator import (
    CostAwareBalancer,
    LeastQueueLengthBalancer,
    RandomBalancer,
    ReplicaInfo,
    ReplicaStatus,
    RoundRobinBalancer,
    estimate_request_cost,
)

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


def test_load_balancer_select_returns_valid_index():
    """Verify RandomBalancer.select() returns a valid index for replicas."""
    # Task structure mirrors async_omni; RandomBalancer ignores task contents.
    task: dict = {
        "request_id": "test",
        "engine_inputs": None,
        "sampling_params": None,
    }

    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RandomBalancer()

    index = balancer.select(task, replicas)

    assert isinstance(index, int)
    assert 0 <= index < len(replicas)


def test_round_robin_balancer_cycles_replicas():
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RoundRobinBalancer()
    results = [balancer.select({}, replicas) for _ in range(5)]

    # Default start_index=0 => 0,1,2,0,1
    assert results == [0, 1, 2, 0, 1]


def test_round_robin_balancer_empty_replicas_raises():
    with pytest.raises(ValueError, match="replicas must not be empty"):
        RoundRobinBalancer().select({}, [])


def test_round_robin_balancer_after_large_index_and_shorter_list():
    """Large start_index % len(replicas) then counter wraps with shorter list."""
    now = time()
    two = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    balancer = RoundRobinBalancer(start_index=7)
    assert balancer.select({}, two) == 1  # 7 % 2
    assert balancer.select({}, two) == 0  # next index wrapped to 0


def test_least_queue_length_balancer_picks_min_queue():
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=5,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = LeastQueueLengthBalancer()
    index = balancer.select({}, replicas)
    assert index == 1


def test_least_queue_length_balancer_empty_replicas_raises():
    with pytest.raises(ValueError, match="replicas must not be empty"):
        LeastQueueLengthBalancer().select({}, [])


def test_least_queue_length_balancer_equal_queues_uses_choice(mocker):
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    balancer = LeastQueueLengthBalancer()
    mocker.patch(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        return_value=2,
    )
    assert balancer.select({}, replicas) == 2


def test_least_queue_length_balancer_negative_queue_raises():
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=-1,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    with pytest.raises(ValueError, match="queue_length must be non-negative"):
        LeastQueueLengthBalancer().select({}, replicas)


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


class _FakeDiffusionParams:
    """Minimal stand-in for OmniDiffusionSamplingParams scalar fields."""

    def __init__(
        self,
        *,
        height=None,
        width=None,
        resolution=None,
        num_frames=None,
        num_inference_steps=None,
        num_outputs_per_prompt=None,
    ):
        self.height = height
        self.width = width
        self.resolution = resolution
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.num_outputs_per_prompt = num_outputs_per_prompt


def test_estimate_cost_explicit_wins():
    task = {"estimated_cost": 42.0, "sampling_params": _FakeDiffusionParams(height=1024, width=1024)}
    assert estimate_request_cost(task) == 42.0


def test_estimate_cost_explicit_invalid_falls_back_to_sampling_params():
    # Non-positive explicit cost is ignored; diffusion estimate takes over.
    task = {"estimated_cost": 0.0, "sampling_params": _FakeDiffusionParams(height=512, width=512)}
    assert estimate_request_cost(task) == 1.0


def test_estimate_cost_no_signal_returns_unit():
    assert estimate_request_cost({}) == 1.0
    assert estimate_request_cost({"sampling_params": None}) == 1.0


def test_estimate_cost_baseline_512x512_is_unit():
    params = _FakeDiffusionParams(height=512, width=512, num_frames=1, num_inference_steps=1, num_outputs_per_prompt=1)
    assert estimate_request_cost({"sampling_params": params}) == 1.0


def test_estimate_cost_spatial_scales_with_area():
    p1 = _FakeDiffusionParams(height=512, width=512)
    p4 = _FakeDiffusionParams(height=1024, width=1024)
    assert estimate_request_cost({"sampling_params": p4}) == pytest.approx(
        4.0 * estimate_request_cost({"sampling_params": p1})
    )


def test_estimate_cost_resolution_square_fallback():
    by_hw = _FakeDiffusionParams(height=640, width=640)
    by_res = _FakeDiffusionParams(resolution=640)
    assert estimate_request_cost({"sampling_params": by_hw}) == pytest.approx(
        estimate_request_cost({"sampling_params": by_res})
    )


def test_estimate_cost_frames_steps_outputs_multiply():
    base = _FakeDiffusionParams(height=512, width=512)
    frames = _FakeDiffusionParams(height=512, width=512, num_frames=17)
    steps = _FakeDiffusionParams(height=512, width=512, num_inference_steps=49)
    outs = _FakeDiffusionParams(height=512, width=512, num_outputs_per_prompt=4)
    base_cost = estimate_request_cost({"sampling_params": base})
    assert estimate_request_cost({"sampling_params": frames}) == pytest.approx(17 * base_cost)
    assert estimate_request_cost({"sampling_params": steps}) == pytest.approx(49 * base_cost)
    assert estimate_request_cost({"sampling_params": outs}) == pytest.approx(4 * base_cost)


def test_estimate_cost_invalid_components_fall_back():
    # height/width present but one invalid -> resolution fallback.
    params = _FakeDiffusionParams(height=-1, width=512, resolution=512)
    assert estimate_request_cost({"sampling_params": params}) == 1.0

    # num_frames non-positive -> treated as 1.
    params = _FakeDiffusionParams(height=512, width=512, num_frames=0)
    assert estimate_request_cost({"sampling_params": params}) == 1.0

    # Non-finite explicit cost -> falls through to unit.
    assert estimate_request_cost({"estimated_cost": math.inf}) == 1.0
    assert estimate_request_cost({"estimated_cost": float("nan")}) == 1.0
    assert estimate_request_cost({"estimated_cost": True}) == 1.0

    # Untrusted metadata must not overflow while being normalized.
    huge = 10**10000
    params = _FakeDiffusionParams(height=huge, width=huge, num_frames=huge)
    assert estimate_request_cost({"sampling_params": params}) == 1.0


# ---------------------------------------------------------------------------
# CostAwareBalancer
# ---------------------------------------------------------------------------


def test_cost_aware_empty_replicas_raises():
    with pytest.raises(ValueError, match="replicas must not be empty"):
        CostAwareBalancer().select({"request_id": "r"}, [])


def test_cost_aware_negative_queue_raises():
    with pytest.raises(ValueError, match="queue_length must be non-negative"):
        CostAwareBalancer().select({"request_id": "r"}, [_replica("a", queue_length=-1)])


def test_cost_aware_picks_min_effective_load():
    balancer = CostAwareBalancer()
    replicas = [_replica("a", queue_length=0), _replica("b", queue_length=0)]
    # First request goes to one replica; second (same cost) should go to the
    # other because the first now has tracked work.
    first = balancer.select({"request_id": "r1", "estimated_cost": 10.0}, replicas)
    second = balancer.select({"request_id": "r2", "estimated_cost": 10.0}, replicas)
    assert first != second


def test_cost_aware_routes_heterogeneous_costs_to_balance_work(monkeypatch):
    """17/49/81-frame requests spread so projected work is evened out."""
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        lambda candidates: candidates[0],
    )
    balancer = CostAwareBalancer()
    replicas = [_replica("a"), _replica("b"), _replica("c")]
    costs = [17.0, 49.0, 81.0, 17.0, 49.0, 81.0]
    loads = {"a": 0.0, "b": 0.0, "c": 0.0}
    for i, c in enumerate(costs):
        idx = balancer.select({"request_id": f"r{i}", "estimated_cost": c}, replicas)
        loads[replicas[idx].input_addr] += c
    # First-candidate round-robin would produce loads [34, 98, 162]. Online
    # cost-aware list scheduling reduces the peak to 130 for this order.
    assert max(loads.values()) == 130.0


def test_cost_aware_release_removes_reservation():
    balancer = CostAwareBalancer()
    replicas = [_replica("a"), _replica("b")]
    balancer.select({"request_id": "r1", "estimated_cost": 100.0}, replicas)
    balancer.release("r1")
    assert "r1" not in balancer._reservations


def test_cost_aware_reselection_is_idempotent():
    balancer = CostAwareBalancer()
    replicas = [_replica("a"), _replica("b")]
    balancer.select({"request_id": "r1", "estimated_cost": 100.0}, replicas)
    # Re-select r1 with a small cost; the old large reservation must be dropped
    # first so it does not double-count.
    balancer.select({"request_id": "r1", "estimated_cost": 1.0}, replicas)
    assert len(balancer._reservations) == 1
    assert balancer._reservations["r1"][1] == 1.0


def test_cost_aware_does_not_reserve_without_request_id():
    balancer = CostAwareBalancer()
    replicas = [_replica("a"), _replica("b")]
    balancer.select({"estimated_cost": 10.0}, replicas)
    assert balancer._reservations == {}


def test_cost_aware_prunes_disappeared_replicas(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        lambda candidates: candidates[0],
    )
    balancer = CostAwareBalancer()
    two = [_replica("a"), _replica("b")]
    balancer.select({"request_id": "r1", "estimated_cost": 10.0}, two)
    # Replica "b" disappears; r1 was on "a" so it stays. Add r2 on the only
    # remaining replica set.
    one = [_replica("a")]
    idx = balancer.select({"request_id": "r2", "estimated_cost": 5.0}, one)
    assert idx == 0
    # Now "a" disappears too; r1's stale reservation must be pruned.
    other = [_replica("c")]
    idx = balancer.select({"request_id": "r3", "estimated_cost": 1.0}, other)
    assert idx == 0


def test_cost_aware_stale_queue_length_does_not_double_count():
    """When tracked_count exceeds reported queue_length, unknown_count clamps to 0."""
    balancer = CostAwareBalancer()
    # Route twice while only "a" is a candidate, giving it two exact local
    # reservations even though the next coordinator snapshot reports one.
    only_a = [_replica("a", queue_length=0)]
    balancer.select({"request_id": "r1", "estimated_cost": 5.0}, only_a)
    balancer.select({"request_id": "r2", "estimated_cost": 5.0}, only_a)
    # Coordinator reports queue_length=1 for "a" (stale/lagging). tracked_count
    # is 2, so unknown_count = max(1 - 2, 0) = 0. Effective load for "a" is
    # 10.0 (tracked work only), not 10 + 1*rep_cost.
    stale = [_replica("a", queue_length=1), _replica("b", queue_length=0)]
    idx = balancer.select({"request_id": "r3", "estimated_cost": 1.0}, stale)
    # "b" has 0 tracked work and 0 queue -> effective 0. "a" has 10 tracked.
    assert idx == 1


def test_cost_aware_unknown_remote_work_uses_representative_cost():
    balancer = CostAwareBalancer()
    # Replica "a" has 3 remote requests we don't know about (queue_length=3).
    # We have one local reservation of cost 10.0 on "b". representative_cost
    # is the mean local cost = 10.0. So "a"'s effective load = 0 + 3 * 10 = 30,
    # "b"'s = 10 + max(0 - 1, 0) * 10 = 10. New unit-cost request picks "b".
    replicas = [_replica("a", queue_length=3), _replica("b", queue_length=0)]
    balancer.select({"request_id": "r1", "estimated_cost": 10.0}, replicas)
    idx = balancer.select({"request_id": "r2", "estimated_cost": 1.0}, replicas)
    assert replicas[idx].input_addr == "b"


def test_cost_aware_falls_back_to_least_queue_when_all_unit_cost():
    """With unit costs and empty local state, behaves like least-queue-length."""
    balancer = CostAwareBalancer()
    replicas = [_replica("a", queue_length=5), _replica("b", queue_length=1), _replica("c", queue_length=3)]
    idx = balancer.select({"request_id": "r1"}, replicas)
    assert idx == 1


def test_cost_aware_tie_breaks_by_queue_length_then_random(monkeypatch):
    balancer = CostAwareBalancer()
    replicas = [_replica("a", queue_length=2), _replica("b", queue_length=1), _replica("c", queue_length=1)]
    # All have zero tracked work, so effective load = queue_length * rep_cost.
    # rep_cost = incoming cost = 1.0. "a" has load 2, "b" and "c" have load 1.
    # Tie between b and c broken by queue_length (both 1) then random.
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        lambda candidates: 2,
    )
    assert balancer.select({"request_id": "r1", "estimated_cost": 1.0}, replicas) == 2


# ---------------------------------------------------------------------------
# Deterministic simulation: cost-aware vs least-queue makespan
# ---------------------------------------------------------------------------


def test_cost_aware_lowers_peak_makespan_vs_least_queue(monkeypatch):
    """Mixed workload: cost-aware balances projected work, least-queue does not.

    A stream of requests with heterogeneous costs is routed one at a time to
    the replica with the lower current projected load. Least-queue treats
    every request as weight 1, so it round-robins and can pile heavy requests
    on one replica. Cost-aware weights by estimated cost and keeps the peak
    assigned work strictly lower.
    """
    costs = [81, 17, 49, 81, 17, 49, 81, 17, 49]
    n_replicas = 3
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        lambda candidates: candidates[0],
    )

    def simulate(balancer_factory):
        balancer = balancer_factory()
        replicas = [_replica(f"r{i}") for i in range(n_replicas)]
        loads = [0.0] * n_replicas
        for i, c in enumerate(costs):
            idx = balancer.select({"request_id": f"req-{i}", "estimated_cost": float(c)}, replicas)
            loads[idx] += c
            # Model the next heartbeat snapshot. CostAwareBalancer subtracts
            # its locally tracked count, while LeastQueueLengthBalancer uses
            # this count directly.
            replicas[idx].queue_length += 1
        return max(loads)

    least_peak = simulate(LeastQueueLengthBalancer)
    cost_peak = simulate(CostAwareBalancer)
    assert least_peak == 243.0
    assert cost_peak == 164.0
