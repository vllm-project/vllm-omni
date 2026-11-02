# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the replica-wide group confirmation (RFC 9536 01-4)."""

from __future__ import annotations

import pickle
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.weight_stack.group as group_module
from vllm_omni.diffusion.weight_stack.errors import (
    ConfigError,
    ConfirmationTimeoutError,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
    WriteError,
)
from vllm_omni.diffusion.weight_stack.group import (
    DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    GroupConfirmation,
    Point0Payload,
    Point1Payload,
    RoundMessage,
    assemble_topology,
    check_collective_parts,
    check_unmatched_policy_keys,
    create_allgather_groups,
    destroy_allgather_groups,
    group_table_keys,
    raise_if_any_failed,
    read_rank_coordinates,
)
from vllm_omni.diffusion.weight_stack.interfaces import (
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

WORLD_ONE = ParallelCoordinate(size=1, rank=0)


@pytest.fixture(autouse=True)
def clean_group_table():
    group_module.destroy_allgather_groups()
    yield
    group_module.destroy_allgather_groups()


def _coordinates(rank: int = 0, world_size: int = 1) -> RankCoordinates:
    """All parallel dimensions are 1 except TP, which covers the replica."""
    return RankCoordinates(
        tp=ParallelCoordinate(size=world_size, rank=rank),
        sp=WORLD_ONE,
        ulysses=WORLD_ONE,
        ring=WORLD_ONE,
        cfg=WORLD_ONE,
        dp=WORLD_ONE,
        pp=WORLD_ONE,
        ep=None,
        fs=None,
    )


def _hsdp_coordinates(rank: int, world_size: int) -> RankCoordinates:
    """Pure HSDP: singleton tp/sp/pp/cfg/dp groups and one FS group over the replica."""
    return RankCoordinates(
        tp=WORLD_ONE,
        sp=WORLD_ONE,
        ulysses=WORLD_ONE,
        ring=WORLD_ONE,
        cfg=WORLD_ONE,
        dp=WORLD_ONE,
        pp=WORLD_ONE,
        ep=None,
        fs=ParallelCoordinate(size=world_size, rank=rank),
    )


def _fingerprints(
    rank: int,
    *,
    content_identities: dict[str, str] | None = None,
    collective: dict[str, str] | None = None,
) -> RankFingerprints:
    return RankFingerprints(
        rank=rank,
        content_identities=content_identities or {},
        plan_fingerprint=f"plan-{rank}",
        collective=collective or {},
    )


def _point1(
    fingerprints: RankFingerprints,
    *,
    group_keys: frozenset[tuple[tuple[int, ...], str]] = frozenset(),
    unmatched_policy_keys: frozenset[str] = frozenset(),
) -> Point1Payload:
    return Point1Payload(
        fingerprints=fingerprints,
        group_keys=group_keys,
        unmatched_policy_keys=unmatched_policy_keys,
    )


def _ctx(world_size: int = 1, stage_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(world_size=world_size, od_config=SimpleNamespace(stage_id=stage_id))


def _patch_fake_dist(
    monkeypatch,
    *,
    rank: int = 0,
    initialized: bool = True,
    new_group=None,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    if new_group is not None:
        monkeypatch.setattr(torch.distributed, "new_group", new_group)


def _recording_new_group(created: list[tuple[tuple[int, ...], str]]):
    def new_group(ranks, backend=None):
        created.append((tuple(ranks), backend))
        return SimpleNamespace(ranks=tuple(ranks))

    return new_group


# ---------------------------------------------------------------------------
# Round messages and the exchange protocol
# ---------------------------------------------------------------------------


def test_round_message_of_success_carries_the_payload():
    message = RoundMessage.of("point1_elections", 2, None, {"record": 1})
    assert (message.point, message.rank, message.ok) == ("point1_elections", 2, True)
    assert message.error is None
    assert message.payload == {"record": 1}
    assert pickle.loads(pickle.dumps(message)) == message


def test_round_message_of_pending_error_reports_summary():
    pending = WriteError("tensor write failed")
    pending.rank = 1
    pending.stage_id = 0
    message = RoundMessage.of("write", 1, pending, {"ignored": True})

    assert message.ok is False
    assert message.payload is None
    assert message.error is not None
    assert message.error.error_type == "WriteError"
    assert message.error.phase == "write"
    assert message.error.rank == 1
    assert "tensor write failed" in message.error.message


class _ForbiddenDistributed:
    """Any attribute access means the confirmation path touched torch.distributed."""

    def __getattr__(self, name):
        raise AssertionError(f"world_size 1 must not touch torch.distributed.{name}")


@pytest.mark.parametrize(
    "point",
    ["point0_topology", "point1_fingerprints", "point1_elections", "point2_ready", "point3_commit"],
)
def test_exchange_world_size_one_never_touches_torch_distributed(monkeypatch, point):
    monkeypatch.setattr(group_module, "dist", _ForbiddenDistributed())
    channel = GroupConfirmation(world_size=1, timeout_seconds=5)
    message = RoundMessage.of(point, 0, None, {"payload": 1})
    assert channel.exchange(point, message) == (message,)


def test_exchange_rejects_a_message_for_another_point_or_rank():
    channel = GroupConfirmation(world_size=2, cpu_group=object(), timeout_seconds=5, rank=1)
    with pytest.raises(ValueError, match="does not match round point"):
        channel.exchange("point0_topology", RoundMessage.of("point1_fingerprints", 1, None, None))
    with pytest.raises(ValueError, match="does not match this rank"):
        channel.exchange("point0_topology", RoundMessage.of("point0_topology", 0, None, None))


def test_exchange_detects_ranks_at_different_confirmation_points(monkeypatch):
    channel = GroupConfirmation(world_size=2, cpu_group=object(), timeout_seconds=5, rank=0)
    monkeypatch.setattr(torch.distributed, "monitored_barrier", lambda **kwargs: None)

    def fake_all_gather_object(gathered, message, group=None):
        gathered[0] = message
        gathered[1] = RoundMessage.of("point3_commit", 1, None, None)

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(GroupConfirmationError, match="not at the same confirmation point") as exc_info:
        channel.exchange("point2_ready", RoundMessage.of("point2_ready", 0, None, None))
    assert exc_info.value.phase == "point2_ready"
    assert "rank 1: 'point3_commit'" in str(exc_info.value)


def test_exchange_timeout_on_rank_zero_lists_the_missing_ranks(monkeypatch):
    channel = GroupConfirmation(world_size=3, cpu_group=object(), timeout_seconds=5, rank=0)
    barrier_error = RuntimeError("[Rank 0]: Ranks 1, 2 failed to pass monitoredBarrier in 5000 ms")

    def failing_barrier(**kwargs):
        raise barrier_error

    monkeypatch.setattr(torch.distributed, "monitored_barrier", failing_barrier)

    with pytest.raises(ConfirmationTimeoutError) as exc_info:
        channel.exchange("point0_topology", RoundMessage.of("point0_topology", 0, None, None))

    error = exc_info.value
    assert error.__cause__ is barrier_error
    assert error.phase == "point0_topology"
    assert error.rank == 0
    assert error.world_size == 3
    assert "rank(s) [1, 2] did not reach confirmation point" in str(error)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("[Rank 0]: Ranks 1, 2 failed to pass monitoredBarrier in 5000 ms", (1, 2)),
        ("[Rank 0]: Rank 3 failed to pass monitoredBarrier in 100 ms", (3,)),
        ("[Rank 1]: Ranks 0 failed to pass monitoredBarrier in 100 ms", (0,)),
        ("[Rank 0]: Ranks 1 failed to pass monitoredBarrier in 1800000 ms", (1,)),
        ("Connection closed by peer [127.0.0.1]:1234", None),
        ("Rank 1 successfully reached monitoredBarrier, but received errors", None),
        ("", None),
    ],
)
def test_missing_ranks_parser_handles_gloo_timeout_messages(message, expected):
    assert group_module._missing_ranks_from_barrier_error(RuntimeError(message)) == expected


def test_exchange_early_transport_error_propagates_raw(monkeypatch):
    channel = GroupConfirmation(world_size=2, cpu_group=object(), timeout_seconds=30, rank=1)

    def failing_barrier(**kwargs):
        raise RuntimeError("Connection closed by peer [127.0.0.1]:1234")

    monkeypatch.setattr(torch.distributed, "monitored_barrier", failing_barrier)

    with pytest.raises(RuntimeError, match="Connection closed by peer"):
        channel.exchange("point0_topology", RoundMessage.of("point0_topology", 1, None, None))


def test_exchange_non_rank_zero_timeout_is_reported_as_confirmation_timeout(monkeypatch):
    channel = GroupConfirmation(world_size=2, cpu_group=object(), timeout_seconds=3, rank=1)
    clock = iter([0.0, 3.2])

    def failing_barrier(**kwargs):
        raise RuntimeError("received errors while waiting for send/recv from rank 0")

    monkeypatch.setattr(group_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(torch.distributed, "monitored_barrier", failing_barrier)

    with pytest.raises(ConfirmationTimeoutError, match="timed out after 3s"):
        channel.exchange("point1_fingerprints", RoundMessage.of("point1_fingerprints", 1, None, None))


def test_exchange_local_watchdog_bounds_a_blocked_barrier(monkeypatch):
    """Non-zero ranks must return within the timeout even when rank 0 never arrives."""
    channel = GroupConfirmation(world_size=2, cpu_group=object(), timeout_seconds=0.01, rank=1)
    release = threading.Event()

    def blocking_barrier(**kwargs):
        release.wait(2.0)

    monkeypatch.setattr(group_module, "_CONFIRMATION_WATCHDOG_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(torch.distributed, "monitored_barrier", blocking_barrier)

    started = time.monotonic()
    try:
        with pytest.raises(ConfirmationTimeoutError, match="timed out after 0.01s"):
            channel.exchange("point0_topology", RoundMessage.of("point0_topology", 1, None, None))
    finally:
        release.set()
    # The local watchdog, not the (blocked) barrier, bounded the call.
    assert time.monotonic() - started < 0.5


def test_exchange_propagates_all_gather_transport_errors_raw(monkeypatch):
    channel = GroupConfirmation(world_size=2, cpu_group=object(), timeout_seconds=5, rank=0)
    monkeypatch.setattr(torch.distributed, "monitored_barrier", lambda **kwargs: None)

    def failing_all_gather_object(gathered, message, group=None):
        raise RuntimeError("gloo all_gather_object failed")

    monkeypatch.setattr(torch.distributed, "all_gather_object", failing_all_gather_object)

    with pytest.raises(RuntimeError, match="gloo all_gather_object failed"):
        channel.exchange("point1_elections", RoundMessage.of("point1_elections", 0, None, None))


def test_timeout_property_is_a_timedelta():
    channel = GroupConfirmation(world_size=1, timeout_seconds=12)
    assert channel.timeout == timedelta(seconds=12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"world_size": 0}, "world_size must be positive"),
        ({"world_size": 2, "rank": 2}, "outside the replica"),
        ({"world_size": 2, "rank": -1}, "outside the replica"),
        ({"world_size": 2}, "cpu_group is required"),
        ({"world_size": 1, "timeout_seconds": 0}, "timeout_seconds must be positive"),
    ],
)
def test_confirmation_channel_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GroupConfirmation(**kwargs)


# ---------------------------------------------------------------------------
# Failure aggregation
# ---------------------------------------------------------------------------


def test_raise_if_any_failed_raises_this_ranks_own_error_first():
    pending = WriteError("local failure")
    gathered = [RoundMessage.of("write", 0, None, None)]
    with pytest.raises(WriteError) as exc_info:
        raise_if_any_failed(gathered, pending)
    assert exc_info.value is pending


def test_raise_if_any_failed_lists_every_peer_failure():
    failing = WriteError("peer write failed")
    failing.rank = 1
    gathered = [
        RoundMessage.of("write", 0, None, None),
        RoundMessage.of("write", 1, failing, None),
        RoundMessage.of("write", 2, None, None),
    ]
    with pytest.raises(PeerFailureError) as exc_info:
        raise_if_any_failed(gathered, None)

    message = str(exc_info.value)
    assert exc_info.value.phase == "write"
    assert "failed on 1 rank(s)" in message
    assert "rank 1: " in message
    assert "peer write failed" in message


def test_raise_if_any_failed_lists_multiple_peer_failures():
    first = WriteError("first peer failure")
    first.rank = 1
    second = WriteError("second peer failure")
    second.rank = 2
    gathered = [
        RoundMessage.of("write", 0, None, None),
        RoundMessage.of("write", 1, first, None),
        RoundMessage.of("write", 2, second, None),
    ]
    with pytest.raises(PeerFailureError) as exc_info:
        raise_if_any_failed(gathered, None)

    message = str(exc_info.value)
    assert "failed on 2 rank(s)" in message
    assert "rank 1: " in message and "first peer failure" in message
    assert "rank 2: " in message and "second peer failure" in message


def test_raise_if_any_failed_returns_when_every_rank_succeeded():
    raise_if_any_failed([RoundMessage.of("point2_ready", 0, None, None)], None)


# ---------------------------------------------------------------------------
# Point 0: topology assembly
# ---------------------------------------------------------------------------


def test_assemble_topology_world_size_one():
    payload = Point0Payload(
        record={"pid": 1},
        coordinates=_coordinates(),
        deployment=("a" * 32, False),
    )
    topology = assemble_topology([payload], _ctx(world_size=1, stage_id=7))

    assert topology.deployment_id == "a" * 32
    assert topology.deployment_scoped is False
    assert topology.stage_id == 7
    assert topology.world_size == 1
    assert topology.records == ({"pid": 1},)
    assert topology.coordinates == (_coordinates(),)


def test_assemble_topology_adopts_rank_zero_deployment_pair():
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("d" * 32, True)),
        Point0Payload(record=object(), coordinates=_coordinates(1, 2), deployment=("other", False)),
    ]
    topology = assemble_topology(payloads, _ctx(world_size=2))
    assert topology.deployment_id == "d" * 32
    assert topology.deployment_scoped is True
    assert topology.world_size == 2
    assert topology.coordinates == (_coordinates(0, 2), _coordinates(1, 2))


def test_assemble_topology_rejects_scoped_deployment_mismatch():
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, True)),
        Point0Payload(record=object(), coordinates=_coordinates(1, 2), deployment=("b" * 32, True)),
    ]
    with pytest.raises(GroupConfirmationError, match="scoped to its own stage") as exc_info:
        assemble_topology(payloads, _ctx(world_size=2))
    assert exc_info.value.phase == "point0_topology"
    assert "rank 1" in str(exc_info.value)


def test_assemble_topology_rejects_scoped_rank_when_rank_zero_is_unscoped():
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, False)),
        Point0Payload(record=object(), coordinates=_coordinates(1, 2), deployment=("b" * 32, True)),
    ]
    with pytest.raises(GroupConfirmationError, match="scoped to its own stage"):
        assemble_topology(payloads, _ctx(world_size=2))


def test_assemble_topology_ignores_unscoped_rank_deployment_id():
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, True)),
        Point0Payload(record=object(), coordinates=_coordinates(1, 2), deployment=("b" * 32, False)),
    ]
    topology = assemble_topology(payloads, _ctx(world_size=2))
    assert topology.deployment_id == "a" * 32
    assert topology.deployment_scoped is True


def test_assemble_topology_rejects_invalid_rank_zero_deployment():
    payload = Point0Payload(record=object(), coordinates=_coordinates(), deployment=("", False))
    with pytest.raises(GroupConfirmationError, match="invalid deployment id"):
        assemble_topology([payload], _ctx(world_size=1))


def test_assemble_topology_rejects_wrong_payload_count():
    payload = Point0Payload(record=object(), coordinates=_coordinates(), deployment=("a" * 32, False))
    with pytest.raises(GroupConfirmationError, match="expected 2 point-0 payloads, collected 1"):
        assemble_topology([payload], _ctx(world_size=2))


def test_assemble_topology_rejects_out_of_bounds_coordinate():
    bad = RankCoordinates(
        tp=ParallelCoordinate(size=2, rank=2),
        sp=WORLD_ONE,
        ulysses=WORLD_ONE,
        ring=WORLD_ONE,
        cfg=WORLD_ONE,
        dp=WORLD_ONE,
        pp=WORLD_ONE,
        ep=None,
        fs=None,
    )
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, False)),
        Point0Payload(record=object(), coordinates=bad, deployment=("a" * 32, False)),
    ]
    with pytest.raises(GroupConfirmationError, match="outside"):
        assemble_topology(payloads, _ctx(world_size=2))


def test_assemble_topology_rejects_non_uniform_coordinates():
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, False)),
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, False)),
    ]
    with pytest.raises(GroupConfirmationError, match="not a uniform covering"):
        assemble_topology(payloads, _ctx(world_size=2))


def test_assemble_topology_rejects_missing_optional_coordinate_on_some_ranks():
    from dataclasses import replace

    with_ep = replace(_coordinates(1, 2), ep=ParallelCoordinate(size=2, rank=1))
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 2), deployment=("a" * 32, False)),
        Point0Payload(record=object(), coordinates=with_ep, deployment=("a" * 32, False)),
    ]
    with pytest.raises(GroupConfirmationError, match="'ep' coordinate"):
        assemble_topology(payloads, _ctx(world_size=2))


def test_assemble_topology_rejects_parallel_size_product_mismatch():
    payloads = [
        Point0Payload(record=object(), coordinates=_coordinates(0, 1), deployment=("a" * 32, False)),
        Point0Payload(record=object(), coordinates=_coordinates(0, 1), deployment=("a" * 32, False)),
    ]
    with pytest.raises(GroupConfirmationError, match="does not equal the replica world size"):
        assemble_topology(payloads, _ctx(world_size=2))


def test_assemble_topology_accepts_pure_hsdp_coordinates():
    """Pure HSDP uses singleton orthogonal groups, so the size product is not the world size."""
    payloads = [
        Point0Payload(record=object(), coordinates=_hsdp_coordinates(rank, 4), deployment=("a" * 32, False))
        for rank in range(4)
    ]
    topology = assemble_topology(payloads, _ctx(world_size=4))
    assert [coordinates.fs for coordinates in topology.coordinates] == [
        ParallelCoordinate(size=4, rank=rank) for rank in range(4)
    ]


def test_assemble_topology_accepts_ulysses_as_a_sub_dimension_of_sp():
    """``sp`` already contains ulysses*ring*allgather, so it must not be multiplied twice."""
    payloads = [
        Point0Payload(
            record=object(),
            coordinates=RankCoordinates(
                tp=WORLD_ONE,
                sp=ParallelCoordinate(size=2, rank=rank % 2),
                ulysses=ParallelCoordinate(size=2, rank=rank % 2),
                ring=WORLD_ONE,
                cfg=WORLD_ONE,
                dp=ParallelCoordinate(size=2, rank=rank // 2),
                pp=WORLD_ONE,
                ep=None,
                fs=None,
            ),
            deployment=("a" * 32, False),
        )
        for rank in range(4)
    ]
    topology = assemble_topology(payloads, _ctx(world_size=4))
    assert [coordinates.sp for coordinates in topology.coordinates] == [
        ParallelCoordinate(size=2, rank=rank % 2) for rank in range(4)
    ]


def test_assemble_topology_accepts_ring_as_a_sub_dimension_of_sp():
    payloads = [
        Point0Payload(
            record=object(),
            coordinates=RankCoordinates(
                tp=WORLD_ONE,
                sp=ParallelCoordinate(size=2, rank=rank),
                ulysses=WORLD_ONE,
                ring=ParallelCoordinate(size=2, rank=rank),
                cfg=WORLD_ONE,
                dp=WORLD_ONE,
                pp=WORLD_ONE,
                ep=None,
                fs=None,
            ),
            deployment=("a" * 32, False),
        )
        for rank in range(2)
    ]
    topology = assemble_topology(payloads, _ctx(world_size=2))
    assert [coordinates.ring for coordinates in topology.coordinates] == [
        ParallelCoordinate(size=2, rank=rank) for rank in range(2)
    ]


def test_assemble_topology_defaults_missing_stage_id_to_zero():
    payload = Point0Payload(record=object(), coordinates=_coordinates(), deployment=("a" * 32, False))
    ctx = SimpleNamespace(world_size=1, od_config=SimpleNamespace(stage_id=None))
    assert assemble_topology([payload], ctx).stage_id == 0


# ---------------------------------------------------------------------------
# Rank coordinates
# ---------------------------------------------------------------------------


def test_read_rank_coordinates_reads_parallel_state(monkeypatch):
    from vllm.distributed import parallel_state as vllm_parallel_state

    from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

    def fake_ep_group():
        return SimpleNamespace(world_size=4, rank_in_group=3)

    def fake_fs_group():
        return SimpleNamespace(world_size=2, rank_in_group=1)

    for name, value in (
        ("get_tensor_model_parallel_world_size", 2),
        ("get_tensor_model_parallel_rank", 1),
    ):
        monkeypatch.setattr(vllm_parallel_state, name, lambda value=value: value)
    monkeypatch.setattr(vllm_parallel_state, "get_ep_group", fake_ep_group)
    for name, value in (
        ("get_sequence_parallel_world_size", 8),
        ("get_sequence_parallel_rank", 5),
        ("get_ulysses_parallel_world_size", 4),
        ("get_ulysses_parallel_rank", 3),
        ("get_ring_parallel_world_size", 2),
        ("get_ring_parallel_rank", 1),
        ("get_classifier_free_guidance_world_size", 2),
        ("get_classifier_free_guidance_rank", 0),
        ("get_data_parallel_world_size", 2),
        ("get_data_parallel_rank", 1),
        ("get_pipeline_parallel_world_size", 1),
        ("get_pipeline_parallel_rank", 0),
    ):
        monkeypatch.setattr(omni_parallel_state, name, lambda value=value: value)
    monkeypatch.setattr(omni_parallel_state, "get_fs_group", fake_fs_group)

    coordinates = read_rank_coordinates()

    assert coordinates.tp == ParallelCoordinate(size=2, rank=1)
    assert coordinates.sp == ParallelCoordinate(size=8, rank=5)
    assert coordinates.ulysses == ParallelCoordinate(size=4, rank=3)
    assert coordinates.ring == ParallelCoordinate(size=2, rank=1)
    assert coordinates.cfg == ParallelCoordinate(size=2, rank=0)
    assert coordinates.dp == ParallelCoordinate(size=2, rank=1)
    assert coordinates.pp == ParallelCoordinate(size=1, rank=0)
    assert coordinates.ep == ParallelCoordinate(size=4, rank=3)
    assert coordinates.fs == ParallelCoordinate(size=2, rank=1)


def test_read_rank_coordinates_optional_groups_absent(monkeypatch):
    from vllm.distributed import parallel_state as vllm_parallel_state

    from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

    for module in (vllm_parallel_state, omni_parallel_state):
        for name in dir(module):
            if name.startswith("get_") and ("world_size" in name or name.endswith("_rank")):
                monkeypatch.setattr(module, name, lambda: 1)

    def not_initialized(*args, **kwargs):
        raise AssertionError("group is not initialized")

    monkeypatch.setattr(vllm_parallel_state, "get_ep_group", not_initialized)
    monkeypatch.setattr(omni_parallel_state, "get_fs_group", not_initialized)

    coordinates = read_rank_coordinates()
    assert coordinates.ep is None
    assert coordinates.fs is None


# ---------------------------------------------------------------------------
# Point 1: collective-part checks and policy keys
# ---------------------------------------------------------------------------


def test_check_collective_parts_accepts_consistent_fingerprints():
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    check_collective_parts(fingerprints)


def test_check_collective_parts_rejects_missing_collective_on_declaring_rank():
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={}),
    ]
    with pytest.raises(GroupConfirmationError, match="rank\\(s\\) \\[1\\] declare it without") as exc_info:
        check_collective_parts(fingerprints)
    assert exc_info.value.phase == "point1_fingerprints"


def test_check_collective_parts_rejects_collective_without_declaration():
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig", "vae": "vae-sig"}),
    ]
    with pytest.raises(GroupConfirmationError, match="do not declare it") as exc_info:
        check_collective_parts(fingerprints)
    assert "rank(s) [1]" in str(exc_info.value)
    assert exc_info.value.phase == "point1_fingerprints"


def test_check_collective_parts_rejects_fingerprint_mismatch_inside_identity_group():
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig-a"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig-b"}),
    ]
    with pytest.raises(GroupConfirmationError, match="different allgather fingerprints") as exc_info:
        check_collective_parts(fingerprints)
    assert "rank 0: 'sig-a'" in str(exc_info.value)
    assert "rank 1: 'sig-b'" in str(exc_info.value)


def test_check_collective_parts_ignores_rank_local_and_partial_components():
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0", "vae": "vae"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    check_collective_parts(fingerprints)


def test_check_collective_parts_allows_different_identity_groups():
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "tp-0"}, collective={"dit": "sig-tp0"}),
        _fingerprints(1, content_identities={"dit": "tp-1"}, collective={"dit": "sig-tp1"}),
    ]
    check_collective_parts(fingerprints)


def test_check_unmatched_policy_keys_ignores_keys_matched_on_some_rank():
    gathered = [
        _point1(_fingerprints(0, content_identities={"dit": "id"}), unmatched_policy_keys=frozenset({"vae"})),
        _point1(_fingerprints(1, content_identities={"dit": "id"}), unmatched_policy_keys=frozenset()),
    ]
    check_unmatched_policy_keys(gathered)


def test_check_unmatched_policy_keys_raises_l01_with_the_available_context():
    """The driver passes no pipeline/roles (they are not in the payload), so the fallback text is used."""
    gathered = [
        _point1(
            _fingerprints(0, content_identities={"dit": "id", "vae": "id"}),
            unmatched_policy_keys=frozenset({"typo", "shared"}),
        ),
        _point1(
            _fingerprints(1, content_identities={"dit": "id"}),
            unmatched_policy_keys=frozenset({"shared"}),
        ),
    ]
    with pytest.raises(PlanningError) as exc_info:
        check_unmatched_policy_keys(gathered)

    error = exc_info.value
    assert error.phase == "plan"
    assert error.what == (
        "offload policy key(s) 'shared' match no component on any rank; declared on this replica: dit, vae"
    )
    assert error.fix == "remove the key(s), or use one of the declared component names or roles"
    assert "'typo'" not in str(error)


def test_check_unmatched_policy_keys_uses_optional_pipeline_and_roles_context():
    gathered = [
        _point1(
            _fingerprints(0, content_identities={"dit": "id", "vae": "id"}),
            unmatched_policy_keys=frozenset({"shared"}),
        ),
        _point1(_fingerprints(1, content_identities={"dit": "id"}), unmatched_policy_keys=frozenset({"shared"})),
    ]
    with pytest.raises(PlanningError) as exc_info:
        check_unmatched_policy_keys(gathered, pipeline="WanPipeline", roles={"dit": "dit", "vae": "vae"})

    message = str(exc_info.value)
    assert "offload policy key(s) 'shared' match no component of WanPipeline on any rank" in message
    assert "declared on this replica: dit (dit), vae (vae)" in message


def test_check_unmatched_policy_keys_message_is_identical_on_every_rank():
    payloads = [
        _point1(_fingerprints(0, content_identities={"dit": "id"}), unmatched_policy_keys=frozenset({"x"})),
        _point1(_fingerprints(1, content_identities={"dit": "id"}), unmatched_policy_keys=frozenset({"x"})),
    ]
    messages = []
    for gathered in (payloads, list(reversed(payloads))):
        with pytest.raises(PlanningError) as exc_info:
            check_unmatched_policy_keys(gathered)
        messages.append(str(exc_info.value))
    assert messages[0] == messages[1]


# ---------------------------------------------------------------------------
# Point 1: AllGather group calculation, creation, reuse and destruction
# ---------------------------------------------------------------------------


def test_create_allgather_groups_rejects_single_rank_group_before_new_group(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, new_group=_recording_new_group(created))
    fingerprints = [_fingerprints(0, content_identities={"dit": "id"}, collective={"dit": "sig"})]
    gathered = [_point1(fingerprints[0])]

    with pytest.raises(GroupConfirmationError, match="single rank") as exc_info:
        create_allgather_groups(fingerprints, gathered, backend="gloo")

    assert exc_info.value.phase == "point1_fingerprints"
    assert created == []


def test_create_allgather_groups_calls_new_group_on_every_rank(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    gathered = [_point1(item) for item in fingerprints]

    groups = create_allgather_groups(fingerprints, gathered, backend="gloo")

    assert created == [((0, 1), "gloo")]
    assert list(groups) == ["dit"]
    assert len(groups["dit"]) == 1
    assert groups["dit"][0].component == "dit"
    assert groups["dit"][0].ranks == (0, 1)
    assert groups["dit"][0].group is not None
    assert group_table_keys() == frozenset({((0, 1), "gloo")})


def test_create_allgather_groups_world_size_one_never_touches_torch_distributed(monkeypatch):
    monkeypatch.setattr(group_module, "dist", _ForbiddenDistributed())

    # No allgather component: no process group is needed at all.
    fingerprints = [_fingerprints(0, content_identities={"dit": "id"}, collective={})]
    assert create_allgather_groups(fingerprints, [_point1(fingerprints[0])]) == {}

    # One allgather component in a single-rank replica is a confirmation error,
    # raised before any new_group call.
    fingerprints = [_fingerprints(0, content_identities={"dit": "id"}, collective={"dit": "sig"})]
    with pytest.raises(GroupConfirmationError, match="single rank"):
        create_allgather_groups(fingerprints, [_point1(fingerprints[0])])
    assert group_table_keys() == frozenset()


def test_create_allgather_groups_tp_two_creates_two_groups_in_rank_order(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=1, new_group=_recording_new_group(created))
    identities = ["tp-0", "tp-1", "tp-0", "tp-1"]
    fingerprints = [
        _fingerprints(rank, content_identities={"dit": identity}, collective={"dit": f"sig-{identity}"})
        for rank, identity in enumerate(identities)
    ]
    gathered = [_point1(item) for item in fingerprints]

    groups = create_allgather_groups(fingerprints, gathered, backend="gloo")

    # Same order on every rank: (component, smallest rank).
    assert created == [((0, 2), "gloo"), ((1, 3), "gloo")]
    assert [group.ranks for group in groups["dit"]] == [(0, 2), (1, 3)]
    # This rank (1) is only a member of the second group.
    assert groups["dit"][0].group is None
    assert groups["dit"][1].group is not None


def _patch_platform_backend(monkeypatch, dist_backend):
    """Patch the platform the production code reads lazily for the group backend."""
    from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

    monkeypatch.setattr(
        omni_parallel_state,
        "current_omni_platform",
        SimpleNamespace(dist_backend=dist_backend),
    )


def test_create_allgather_groups_uses_the_platform_distributed_backend(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    _patch_platform_backend(monkeypatch, "gloo")
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]

    groups = create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints])

    assert created == [((0, 1), "gloo")]
    assert groups["dit"][0].group is not None


def test_create_allgather_groups_rejects_an_invalid_platform_backend(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    _patch_platform_backend(monkeypatch, None)
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]

    with pytest.raises(GroupConfirmationError, match="dist_backend is not a backend name"):
        create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints])
    assert created == []


def test_create_allgather_groups_reuses_groups_when_every_rank_reports_the_key(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    first = create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints], backend="gloo")
    table_keys = group_table_keys()

    second_gathered = [_point1(item, group_keys=table_keys) for item in fingerprints]
    second = create_allgather_groups(fingerprints, second_gathered, backend="gloo")

    assert len(created) == 1  # no second new_group call
    assert second["dit"][0].group is first["dit"][0].group


def test_create_allgather_groups_recreates_when_one_rank_lacks_the_key(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    first = create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints], backend="gloo")
    table_keys = group_table_keys()

    # Rank 0 knows the group, rank 1 does not: every rank must create it again.
    gathered = [
        _point1(fingerprints[0], group_keys=table_keys),
        _point1(fingerprints[1], group_keys=frozenset()),
    ]
    second = create_allgather_groups(fingerprints, gathered, backend="gloo")

    assert len(created) == 2
    first_group = first["dit"][0].group
    second_group = second["dit"][0].group
    assert second_group is not first_group
    assert group_table_keys() == table_keys  # replaced entry keeps the same key

    destroyed: list[object] = []
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda group=None: destroyed.append(group))
    destroy_allgather_groups()
    # Both the replaced group and its replacement are released exactly once each.
    assert {id(group) for group in destroyed} == {id(first_group), id(second_group)}


def test_create_allgather_groups_rejects_missing_local_table_entry(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints], backend="gloo")
    table_keys = group_table_keys()

    # Simulate a process whose table lost the entry although every rank reports it.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    destroy_allgather_groups()
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))

    gathered = [_point1(item, group_keys=table_keys) for item in fingerprints]
    with pytest.raises(GroupConfirmationError, match="no such group in its process-level table"):
        create_allgather_groups(fingerprints, gathered, backend="gloo")


def test_create_allgather_groups_partial_failure_keeps_created_groups_registered(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    calls = {"count": 0}
    fake_new_group = _recording_new_group(created)

    def failing_new_group(ranks, backend=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("new_group failed")
        return fake_new_group(ranks, backend=backend)

    _patch_fake_dist(monkeypatch, rank=0, new_group=failing_new_group)
    fingerprints = [
        _fingerprints(
            rank,
            content_identities={"dit": "id-0", "vae": "vae-0"},
            collective={"dit": "sig-dit", "vae": "sig-vae"},
        )
        for rank in range(2)
    ]
    gathered = [_point1(item) for item in fingerprints]

    with pytest.raises(RuntimeError, match="new_group failed"):
        create_allgather_groups(fingerprints, gathered, backend="gloo")

    # The first group stays registered so the worker shutdown path can destroy it.
    assert created == [((0, 1), "gloo")]
    assert group_table_keys() == frozenset({((0, 1), "gloo")})


def test_destroy_allgather_groups_is_idempotent_and_only_destroys_own_groups(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    destroyed: list[object] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda group=None: destroyed.append(group))
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints], backend="gloo")

    destroy_allgather_groups()
    destroy_allgather_groups()

    assert len(destroyed) == 1
    assert group_table_keys() == frozenset()


def test_destroy_allgather_groups_clears_records_without_distributed(monkeypatch):
    created: list[tuple[tuple[int, ...], str]] = []
    _patch_fake_dist(monkeypatch, rank=0, new_group=_recording_new_group(created))
    fingerprints = [
        _fingerprints(0, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
        _fingerprints(1, content_identities={"dit": "id-0"}, collective={"dit": "sig"}),
    ]
    create_allgather_groups(fingerprints, [_point1(item) for item in fingerprints], backend="gloo")
    assert group_table_keys()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    destroy_allgather_groups()
    assert group_table_keys() == frozenset()


# ---------------------------------------------------------------------------
# Channel construction from config
# ---------------------------------------------------------------------------


def test_from_config_uses_default_timeout_without_distributed(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    channel = GroupConfirmation.from_config(SimpleNamespace(dist_timeout=None))
    assert channel.world_size == 1
    assert channel.rank == 0
    assert channel.cpu_group is None
    assert channel.timeout_seconds == DEFAULT_CONFIRMATION_TIMEOUT_SECONDS


def test_from_config_reads_dist_timeout(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    channel = GroupConfirmation.from_config(SimpleNamespace(dist_timeout=42))
    assert channel.timeout_seconds == 42


@pytest.mark.parametrize("timeout", ["5", 0, -1, True])
def test_from_config_rejects_invalid_dist_timeout(monkeypatch, timeout):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(ConfigError, match="dist_timeout"):
        GroupConfirmation.from_config(SimpleNamespace(dist_timeout=timeout))


def test_from_config_uses_world_group_for_multi_rank(monkeypatch):
    from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    cpu_group = object()
    monkeypatch.setattr(
        omni_parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(cpu_group=cpu_group),
    )

    channel = GroupConfirmation.from_config(SimpleNamespace(dist_timeout=5))
    assert channel.world_size == 2
    assert channel.rank == 1
    assert channel.cpu_group is cpu_group
