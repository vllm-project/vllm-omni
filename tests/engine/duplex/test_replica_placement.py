# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure session placement and capacity accounting; no engine or GPU execution."""

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.engine.duplex.replica_placement import DuplexReplicaPlacement

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_aligned_groups_balance_sessions_and_keep_routes_stable() -> None:
    placement = DuplexReplicaPlacement(stage_replica_counts=[3, 3], max_sessions_per_replica=2)
    reservations = [placement.reserve(f"s{index}") for index in range(6)]
    assert [reservation.group_id for reservation in reservations] == [0, 1, 2, 0, 1, 2]
    assert placement.snapshot()["groups"] == [{0: 0, 1: 0}, {0: 1, 1: 1}, {0: 2, 1: 2}]
    for index, reservation in enumerate(reservations):
        assert placement.reserve(f"s{index}") is reservation
        assert placement.route(f"s{index}", 0) == placement.route(f"s{index}", 1) == index % 3
    assert all(count == 2 for count in placement.snapshot()["reserved"].values())

    with pytest.raises(DuplexSessionError, match="replica_capacity_exhausted") as exc:
        placement.reserve("overflow")
    assert exc.value.code == "resource_exhausted"
    assert exc.value.retryable is True
    assert exc.value.session_id == "overflow"


def test_round_robin_ties_advance_even_when_sessions_release_immediately() -> None:
    placement = DuplexReplicaPlacement(stage_replica_counts=[2], max_sessions_per_replica=1)
    for index in range(5):
        assert placement.reserve("session").group_id == index % 2
        placement.release("session")
        placement.release("session")
    assert placement.snapshot()["reserved"] == {(0, 0): 0, (0, 1): 0}


def test_group_score_uses_peak_occupancy_then_sum_before_round_robin() -> None:
    placement = DuplexReplicaPlacement(
        stage_replica_counts=[2, 2],
        max_sessions_per_replica=4,
        replica_groups=[{0: 0, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 0}],
    )
    # The fourth choice skips group 0 even though the cursor points at it:
    # all groups have peak occupancy 2, but group 0 has the higher total.
    assert [placement.reserve(f"s{index}").group_id for index in range(5)] == [0, 1, 2, 1, 2]


def test_shared_replica_capacity_is_not_duplicated_by_groups() -> None:
    placement = DuplexReplicaPlacement(
        stage_replica_counts=[1, 2],
        max_sessions_per_replica=2,
        replica_groups=[{0: 0, 1: 0}, {0: 0, 1: 1}],
    )
    assert placement.reserve("a").group_id == 0
    assert placement.reserve("b").group_id == 1
    assert placement.snapshot()["reserved"] == {(0, 0): 2, (1, 0): 1, (1, 1): 1}
    with pytest.raises(DuplexSessionError, match="replica_capacity_exhausted"):
        placement.reserve("c")
    placement.release("a")
    assert placement.reserve("c").group_id == 0
    assert placement.snapshot()["reserved"] == {(0, 0): 2, (1, 0): 1, (1, 1): 1}


def test_failure_finds_all_owners_and_rejects_routes_through_other_stages() -> None:
    placement = DuplexReplicaPlacement(stage_replica_counts=[2, 2], max_sessions_per_replica=2)
    for session_id in ("a", "b", "c"):
        placement.reserve(session_id)
    assert placement.mark_unavailable(1, 0) == {"a", "c"}
    # This also rejects an open that was awaiting plugin preparation when a
    # downstream replica failed, before its Stage0 request was registered.
    with pytest.raises(DuplexSessionError) as exc:
        placement.route("a", 0)
    assert exc.value.code == "replica_unavailable"
    assert placement.route("b", 0) == 1
    assert placement.reserve("d").group_id == 1
    placement.release("a")
    placement.release("c")
    assert placement.mark_unavailable(1, 0) == set()
    with pytest.raises(DuplexSessionError, match="replica_capacity_exhausted"):
        placement.reserve("e")


def test_reservations_and_snapshots_do_not_expose_mutable_placement_state() -> None:
    placement = DuplexReplicaPlacement(stage_replica_counts=[1], max_sessions_per_replica=1)
    reservation = placement.reserve("a")
    with pytest.raises(TypeError):
        reservation.replicas[0] = 1
    with pytest.raises(FrozenInstanceError):
        reservation.group_id = 1
    snapshot = placement.snapshot()
    snapshot["groups"][0][0] = 99
    snapshot["reserved"][(0, 0)] = 99
    snapshot["sessions"].clear()
    assert placement.route("a", 0) == 0
    assert placement.snapshot()["reserved"] == {(0, 0): 1}
    placement.release("a")
    with pytest.raises(DuplexSessionError) as exc:
        placement.route("a", 0)
    assert exc.value.code == "unknown_session"
    with pytest.raises(ValueError, match="Unknown stage replica"):
        placement.mark_unavailable(0, 9)


@pytest.mark.parametrize(
    ("counts", "capacity", "groups", "message"),
    [
        ([], 1, None, "positive replica count"),
        ([0], 1, None, "positive replica count"),
        ([True], 1, None, "positive replica count"),
        ([1], 0, None, "positive integer"),
        ([1], True, None, "positive integer"),
        ([1, 2], 1, None, "unequal stage replica counts"),
        ([1], 1, [], "at least one group"),
        ([1, 1], 1, [{0: 0}], "cover every stage"),
        ([1], 1, [{0: 0, 1: 0}], "cover every stage"),
        ([1], 1, [{"0": 0}], "integer stage IDs"),
        ([1], 1, [{False: 0}], "integer stage IDs"),
        ([1], 1, [{0: 1}], "replica_id for stage 0"),
        ([1], 1, [{0: -1}], "replica_id for stage 0"),
        ([1], 1, [{0: True}], "replica_id for stage 0"),
        ([1], 1, [{0: 0}, {0: 0}], "duplicates"),
        ([2], 1, [{0: 0}], "cover every replica"),
    ],
)
def test_invalid_topologies_are_rejected(counts, capacity, groups, message) -> None:
    with pytest.raises(ValueError, match=message):
        DuplexReplicaPlacement(
            stage_replica_counts=counts,
            max_sessions_per_replica=capacity,
            replica_groups=groups,
        )


@pytest.mark.parametrize(("replicas", "frontends"), [(8, 1), (8, 2), (8, 4), (8, 8), (7, 3), (5, 2)])
def test_frontend_ownership_partitions_all_replicas(replicas: int, frontends: int) -> None:
    seen: set[int] = set()
    for frontend in range(frontends):
        placement = DuplexReplicaPlacement(
            stage_replica_counts=[replicas, replicas],
            max_sessions_per_replica=2,
            frontend_count=frontends,
            frontend_index=frontend,
        )
        owned = list(range(frontend, replicas, frontends))
        assert placement.owned_group_count == len(owned)
        assert set(placement.snapshot()["reserved"]) == {(stage, replica) for stage in range(2) for replica in owned}
        assert not seen.intersection(owned)
        seen.update(owned)
        # Releasing immediately must still visit every owned group. IDs stay
        # global, so taking the cursor modulo the local group count is wrong.
        for index in range(2 * len(owned)):
            reservation = placement.reserve("warmup")
            assert reservation.group_id == owned[index % len(owned)]
            placement.release("warmup")
        for index in range(2 * len(owned)):
            reservation = placement.reserve(f"s{index}")
            assert reservation.group_id == owned[index % len(owned)]
            assert dict(reservation.replicas) == {0: reservation.group_id, 1: reservation.group_id}
        with pytest.raises(DuplexSessionError, match="replica_capacity_exhausted"):
            placement.reserve("full")
    assert seen == set(range(replicas))


@pytest.mark.parametrize(
    ("replicas", "frontends", "cap", "global_limit", "expected"),
    [
        (8, 4, 8, 64, [16, 16, 16, 16]),
        (8, 4, 8, 63, [16, 16, 16, 15]),
        (5, 2, 2, 99, [6, 4]),
        (8, 4, 8, 2, [1, 1, 0, 0]),
        (7, 3, 2, 11, [4, 4, 3]),
    ],
)
def test_frontend_admission_budget_splits_global_limit(replicas, frontends, cap, global_limit, expected) -> None:
    limits = [
        DuplexReplicaPlacement(
            stage_replica_counts=[replicas],
            max_sessions_per_replica=cap,
            frontend_count=frontends,
            frontend_index=index,
        ).frontend_session_limit(global_limit)
        for index in range(frontends)
    ]
    assert limits == expected
    assert sum(limits) <= global_limit


@pytest.mark.parametrize(
    ("counts", "groups", "frontends"),
    [
        ([1, 2], [{0: 0, 1: 0}, {0: 0, 1: 1}], 1),
        ([2, 3], [{0: 0, 1: 0}, {0: 1, 1: 1}, {0: 0, 1: 2}], 2),
    ],
)
def test_frontend_admission_budget_matches_shared_replica_capacity(counts, groups, frontends) -> None:
    placement = DuplexReplicaPlacement(
        stage_replica_counts=counts,
        max_sessions_per_replica=4,
        replica_groups=groups,
        frontend_count=frontends,
        frontend_index=0,
    )
    assert placement.frontend_session_limit(100) == 4
    for index in range(4):
        placement.reserve(f"s{index}")
    with pytest.raises(DuplexSessionError, match="replica_capacity_exhausted") as exc:
        placement.reserve("overflow")
    assert exc.value.retryable is True


@pytest.mark.parametrize(
    ("frontends", "index", "message"),
    [
        (0, 0, "positive integer"),
        (True, 0, "positive integer"),
        (3, 0, "must not exceed"),
        (2, -1, "frontend_index"),
        (2, 2, "frontend_index"),
        (2, True, "frontend_index"),
    ],
)
def test_invalid_frontend_topology_is_rejected(frontends, index, message) -> None:
    with pytest.raises(ValueError, match=message):
        DuplexReplicaPlacement(
            stage_replica_counts=[2],
            max_sessions_per_replica=1,
            frontend_count=frontends,
            frontend_index=index,
        )


def test_shared_replica_cannot_belong_to_different_frontends() -> None:
    for index in range(2):
        with pytest.raises(ValueError, match="sharing stage 0 replica 0 must belong to the same frontend"):
            DuplexReplicaPlacement(
                stage_replica_counts=[1, 2],
                max_sessions_per_replica=2,
                replica_groups=[{0: 0, 1: 0}, {0: 0, 1: 1}],
                frontend_count=2,
                frontend_index=index,
            )


def test_shared_replica_groups_with_one_frontend_owner_preserve_capacity() -> None:
    placement = DuplexReplicaPlacement(
        stage_replica_counts=[2, 3],
        max_sessions_per_replica=2,
        replica_groups=[{0: 0, 1: 0}, {0: 1, 1: 1}, {0: 0, 1: 2}],
        frontend_count=2,
        frontend_index=0,
    )
    assert [placement.reserve(session).group_id for session in ("a", "b")] == [0, 2]
    assert placement.snapshot()["reserved"] == {(0, 0): 2, (1, 0): 1, (1, 2): 1}
    with pytest.raises(DuplexSessionError, match="replica_capacity_exhausted"):
        placement.reserve("overflow")
    assert placement.mark_unavailable(0, 1) == set()
    assert placement.route("a", 0) == 0
    assert placement.mark_unavailable(0, 0) == {"a", "b"}
