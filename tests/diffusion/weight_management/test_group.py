# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU unit tests for RFC 01-4 replica confirmation and AllGather groups."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.weight_management import group
from vllm_omni.diffusion.weight_management.errors import (
    ConfigError,
    ConfirmationTimeoutError,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
)
from vllm_omni.diffusion.weight_management.interfaces import (
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    StartupContext,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _coords(tp_size: int = 1, tp_rank: int = 0) -> RankCoordinates:
    one = ParallelCoordinate(1, 0)
    return RankCoordinates(
        tp=ParallelCoordinate(tp_size, tp_rank),
        sp=one,
        ulysses=one,
        ring=one,
        cfg=one,
        dp=one,
        pp=one,
        ep=None,
        fs=None,
    )


def _ctx(world_size: int = 1, stage_id: int = 0) -> StartupContext:
    return StartupContext(
        od_config=SimpleNamespace(stage_id=stage_id),  # type: ignore[arg-type]
        vllm_config=object(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=world_size,
        load_format="default",
        custom_pipeline_name=None,
        memory_pools=object(),  # type: ignore[arg-type]
        policy=None,
        compile_enabled=False,
        faults=None,
    )


def test_world_size_one_skips_distributed_and_rejects_allgather(monkeypatch):
    def _forbidden(*args, **kwargs):
        pytest.fail("world_size=1 must not call torch.distributed")

    monkeypatch.setattr(dist, "monitored_barrier", _forbidden)
    monkeypatch.setattr(dist, "all_gather_object", _forbidden)

    ch = group.GroupConfirmation(world_size=1)
    msg = group.RoundMessage.of("point2_ready", 0, None, "ok")
    assert ch.exchange("point2_ready", msg) == (msg,)

    pending = ConfigError("bad config")
    failed = ch.exchange("point2_ready", group.RoundMessage.of("point2_ready", 0, pending, None))
    with pytest.raises(ConfigError) as exc_info:
        group.raise_if_any_failed(failed, pending)
    assert exc_info.value is pending

    fp = RankFingerprints(0, {"dit": "id0"}, "plan0", {"dit": "sig0"})
    payload = group.Point1Payload(fp, frozenset(), frozenset())
    with pytest.raises(GroupConfirmationError, match="at least 2 ranks"):
        group.create_allgather_groups((fp,), (payload,), backend="gloo")


def test_exchange_point_mismatch_peer_failure_and_transport_passthrough(monkeypatch):
    monkeypatch.setattr(dist, "monitored_barrier", lambda **kw: None)
    peer_err = ConfigError("peer broke")
    messages = [
        group.RoundMessage.of("point1_elections", 0, None, None),
        group.RoundMessage.of("point1_elections", 1, peer_err, None),
    ]

    def _gather(out, value, **kw):
        out[:] = messages

    monkeypatch.setattr(dist, "all_gather_object", _gather)
    ch = group.GroupConfirmation(world_size=2, cpu_group=object())

    gathered = ch.exchange("point1_elections", messages[0])
    with pytest.raises(PeerFailureError, match="rank 1.*peer broke"):
        group.raise_if_any_failed(gathered, None)

    with pytest.raises(GroupConfirmationError, match="inconsistent coordination point"):
        ch.exchange("point2_ready", messages[0])

    # Non-timeout Gloo transport error propagates directly.
    transport_err = RuntimeError("Gloo connection closed by peer")
    monkeypatch.setattr(dist, "monitored_barrier", lambda **kw: (_ for _ in ()).throw(transport_err))
    with pytest.raises(RuntimeError) as exc_info:
        ch.exchange("point2_ready", group.RoundMessage.of("point2_ready", 0, None, None))
    assert exc_info.value is transport_err

    # Monitored barrier timeout wraps into ConfirmationTimeoutError.
    barrier_timeout = RuntimeError("Rank 1 failed to pass monitoredBarrier in 1000 ms")
    monkeypatch.setattr(dist, "monitored_barrier", lambda **kw: (_ for _ in ()).throw(barrier_timeout))
    with pytest.raises(ConfirmationTimeoutError, match="Rank 1 failed to pass monitoredBarrier"):
        ch.exchange("point2_ready", group.RoundMessage.of("point2_ready", 0, None, None))


def test_assemble_topology_adopts_rank0_and_checks_propagated_id():
    p0 = group.Point0Payload("rec0", _coords(), ("a" * 32, False))
    p1 = group.Point0Payload("rec1", _coords(), ("b" * 32, False))
    topo = group.assemble_topology((p0, p1), _ctx(world_size=2, stage_id=3))
    assert topo.deployment_id == "a" * 32
    assert topo.deployment_scoped is False
    assert topo.stage_id == 3
    assert topo.records == ("rec0", "rec1")

    # Matching propagated IDs pass; mismatched propagated ID fails.
    p0_scoped = group.Point0Payload("rec0", _coords(), ("a" * 32, True))
    p1_same = group.Point0Payload("rec1", _coords(), ("a" * 32, True))
    assert group.assemble_topology((p0_scoped, p1_same), _ctx(world_size=2)).deployment_scoped is True

    p1_diff = group.Point0Payload("rec1", _coords(), ("c" * 32, True))
    with pytest.raises(GroupConfirmationError, match="deployment_id"):
        group.assemble_topology((p0, p1_diff), _ctx(world_size=2))


def test_unmatched_policy_keys_uses_intersection():
    fp0 = RankFingerprints(0, {"dit": "a"}, "p0", {})
    fp1 = RankFingerprints(1, {"dit": "a", "vae": "b"}, "p1", {})
    # "vae" unmatched only on rank 0 -> intersection empty -> passes.
    group.check_unmatched_policy_keys(
        (
            group.Point1Payload(fp0, frozenset(), frozenset({"vae"})),
            group.Point1Payload(fp1, frozenset(), frozenset()),
        )
    )
    # "typo" unmatched on both -> PlanningError naming key and declared components.
    with pytest.raises(PlanningError, match="typo") as exc_info:
        group.check_unmatched_policy_keys(
            (
                group.Point1Payload(fp0, frozenset(), frozenset({"vae", "typo"})),
                group.Point1Payload(fp1, frozenset(), frozenset({"typo"})),
            )
        )
    assert "dit" in str(exc_info.value) and "vae" in str(exc_info.value)


def test_collective_consistency_checks():
    # Rank 1 declares dit but omits it from collective.
    fps_missing = (
        RankFingerprints(0, {"dit": "id"}, "p0", {"dit": "sig"}),
        RankFingerprints(1, {"dit": "id"}, "p1", {}),
    )
    with pytest.raises(GroupConfirmationError, match="dit"):
        group.check_collective_parts(fps_missing)

    # Same content identity, different collective signature.
    fps_mismatch = (
        RankFingerprints(0, {"dit": "id"}, "p0", {"dit": "sig-a"}),
        RankFingerprints(1, {"dit": "id"}, "p1", {"dit": "sig-b"}),
    )
    with pytest.raises(GroupConfirmationError, match="dit"):
        group.check_collective_parts(fps_mismatch)


def test_allgather_groups_order_reuse_replacement_and_destroy(monkeypatch):
    created: list[tuple[tuple[int, ...], str, object]] = []
    destroyed: list[object] = []

    def _new_group(ranks, *, backend):
        handle = object()
        created.append((tuple(ranks), backend, handle))
        return handle

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "new_group", _new_group)
    monkeypatch.setattr(dist, "destroy_process_group", destroyed.append)

    try:
        # 4 ranks, TP=2: dit has identity groups (0, 2) and (1, 3); vae only on ranks (2, 3).
        fps = (
            RankFingerprints(0, {"dit": "tp0"}, "p0", {"dit": "s0"}),
            RankFingerprints(1, {"dit": "tp1"}, "p1", {"dit": "s1"}),
            RankFingerprints(2, {"dit": "tp0", "vae": "v"}, "p2", {"dit": "s0", "vae": "sv"}),
            RankFingerprints(3, {"dit": "tp1", "vae": "v"}, "p3", {"dit": "s1", "vae": "sv"}),
        )
        payloads = tuple(group.Point1Payload(fp, frozenset(), frozenset()) for fp in fps)
        groups = group.create_allgather_groups(fps, payloads, backend="gloo")

        assert [r for r, _, _ in created] == [(0, 2), (1, 3), (2, 3)]
        assert [g.ranks for g in groups["dit"]] == [(0, 2), (1, 3)]
        assert groups["dit"][0].group is created[0][2]  # rank 0 is member
        assert groups["dit"][1].group is None  # rank 0 is nonmember
        assert groups["vae"][0].group is None  # rank 0 does not own vae

        # All ranks have cached keys -> reuse with zero new_group calls.
        keys = group.group_table_keys()
        reused = group.create_allgather_groups(
            fps, tuple(group.Point1Payload(fp, keys, frozenset()) for fp in fps), backend="gloo"
        )
        assert len(created) == 3
        assert reused["dit"][0].group is created[0][2]

        # One rank missing key -> recreate and retain previous owned handle for shutdown.
        partial = (
            group.Point1Payload(fps[0], keys, frozenset()),
            group.Point1Payload(fps[1], frozenset(), frozenset()),
            group.Point1Payload(fps[2], keys, frozenset()),
            group.Point1Payload(fps[3], keys, frozenset()),
        )
        replaced = group.create_allgather_groups(fps, partial, backend="gloo")
        assert len(created) == 6
        assert replaced["dit"][0].group is created[3][2]

        # Destroy destroys both old and new owned handles (only groups rank 0 belongs to) and is idempotent.
        group.destroy_allgather_groups()
        assert destroyed == [created[3][2], created[0][2]]
        assert group.group_table_keys() == frozenset()
        group.destroy_allgather_groups()
        assert len(destroyed) == 2
    finally:
        group.destroy_allgather_groups()


def test_read_rank_coordinates_handles_optional_ep_and_fs(monkeypatch):
    from vllm.distributed import parallel_state as vllm_ps

    from vllm_omni.diffusion.distributed import parallel_state as diff_ps

    def _pg(size: int, rank: int):
        return SimpleNamespace(world_size=size, rank_in_group=rank)

    sp = SimpleNamespace(
        world_size=4, rank_in_group=1, ulysses_world_size=2, ulysses_rank=1, ring_world_size=2, ring_rank=0
    )
    monkeypatch.setattr(vllm_ps, "get_tp_group", lambda: _pg(2, 1))
    monkeypatch.setattr(diff_ps, "get_sp_group", lambda: sp)
    monkeypatch.setattr(diff_ps, "get_cfg_group", lambda: _pg(1, 0))
    monkeypatch.setattr(diff_ps, "get_dp_group", lambda: _pg(2, 0))
    monkeypatch.setattr(diff_ps, "get_pp_group", lambda: _pg(1, 0))
    monkeypatch.setattr(vllm_ps, "get_ep_group", lambda: (_ for _ in ()).throw(AssertionError("no ep")))
    monkeypatch.setattr(diff_ps, "get_fs_group", lambda: (_ for _ in ()).throw(AssertionError("no fs")))

    coords = group.read_rank_coordinates()
    assert coords.tp == ParallelCoordinate(2, 1)
    assert coords.sp == ParallelCoordinate(4, 1)
    assert coords.ulysses == ParallelCoordinate(2, 1)
    assert coords.ring == ParallelCoordinate(2, 0)
    assert coords.dp == ParallelCoordinate(2, 0)
    assert coords.ep is None
    assert coords.fs is None
