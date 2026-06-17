# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure planning tests for mixed-parallel KV receive decisions."""

import pytest

from vllm_omni.distributed.omni_connectors.utils.parallel_plan import (
    KVParallelRankCoord,
    ParallelAxis,
    auto_connector_port_axes,
    build_kv_receive_distribution_plan,
    flatten_parallel_coord,
    select_connector_port_from_list,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _coord(
    *,
    cfg_size: int = 1,
    cfg_rank: int = 0,
    sp_size: int = 1,
    sp_rank: int = 0,
    ring_size: int = 1,
    ring_rank: int = 0,
    ulysses_size: int = 1,
    ulysses_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> KVParallelRankCoord:
    return KVParallelRankCoord(
        axes=(
            ParallelAxis("tp", tp_size, tp_rank, "tensor_shard"),
            ParallelAxis("pp", pp_size, pp_rank, "tensor_shard"),
            ParallelAxis("ring", ring_size, ring_rank, "tensor_shard"),
            ParallelAxis("ulysses", ulysses_size, ulysses_rank, "tensor_shard"),
            ParallelAxis("cfg", cfg_size, cfg_rank, "branch"),
            ParallelAxis("sp", sp_size, sp_rank, "container"),
            ParallelAxis("dp", 1, 0, "replica"),
            ParallelAxis("ep", ep_size, ep_rank, "replica"),
        )
    )


def _plan(coord: KVParallelRankCoord, *, source_tp: int = 1, target_tp: int = 1):
    return build_kv_receive_distribution_plan(
        coord=coord,
        world_size=8,
        world_rank=0,
        source_tp_size=source_tp,
        target_tp_size=target_tp,
    )


def test_legacy_sp_without_ring_uses_sp_rank_zero_owner():
    plan = _plan(_coord(sp_size=2, sp_rank=1))

    assert plan.mode == "cfg_sp_local_distribution"
    assert plan.sp_broadcast_enabled
    assert not plan.owner_receives


def test_ring_inside_sp_uses_sp_rank_zero_transport_owner():
    plan = _plan(_coord(sp_size=4, sp_rank=1, ring_size=2, ring_rank=1))

    assert plan.mode == "cfg_sp_local_distribution"
    assert plan.sequence_sharded
    assert plan.sp_broadcast_enabled
    assert not plan.owner_receives


def test_cfg_with_ring_has_one_transport_owner_per_sp_group():
    owner = _plan(_coord(cfg_size=2, cfg_rank=0, sp_size=4, sp_rank=0, ring_size=2, ring_rank=0))
    cfg_sp_root = _plan(_coord(cfg_size=2, cfg_rank=1, sp_size=4, sp_rank=0, ring_size=2, ring_rank=0))
    sp_follower = _plan(_coord(cfg_size=2, cfg_rank=1, sp_size=4, sp_rank=3, ring_size=2, ring_rank=1))

    assert owner.mode == "cfg_sp_local_distribution"
    assert owner.sp_broadcast_enabled
    assert owner.owner_receives
    assert not cfg_sp_root.owner_receives
    assert cfg_sp_root.cfg_follower_receives_from_cfg_leader
    assert not sp_follower.owner_receives
    assert not sp_follower.cfg_follower_receives_from_cfg_leader


def test_pp_sharded_rank_receives_independently_when_no_local_distribution():
    plan = _plan(_coord(pp_size=2, pp_rank=1))

    assert plan.mode == "independent_remote"
    assert plan.owner_receives


def test_tp_sharded_rank_preserves_existing_independent_path():
    plan = _plan(_coord(), source_tp=2, target_tp=4)

    assert plan.mode == "independent_remote"
    assert plan.tp_active
    assert plan.owner_receives


def test_ep_replica_axis_does_not_change_world_broadcast_mode():
    plan = _plan(_coord(ep_size=2, ep_rank=1))

    assert plan.mode == "world_broadcast"
    assert not plan.uses_replica_fanout
    assert not plan.coord.axis("ep").shards_kv


def test_ep_enables_fanout_without_changing_independent_remote_mode():
    plan = _plan(_coord(ep_size=2, ep_rank=1), source_tp=2, target_tp=2)

    assert plan.mode == "independent_remote"
    assert plan.uses_replica_fanout
    assert plan.replica_fanout_group == "ep"
    assert ("ep", 1, 2) not in plan.replica_fanout_identity


def test_ep_enables_fanout_without_changing_cfg_local_distribution_mode():
    plan = _plan(_coord(cfg_size=2, cfg_rank=0, ep_size=2))

    assert plan.mode == "cfg_sp_local_distribution"
    assert plan.uses_replica_fanout
    assert plan.owner_receives


def test_replica_fanout_identity_tracks_kv_sharding_axes():
    base = _plan(_coord(tp_size=2, tp_rank=0, cfg_size=2, cfg_rank=0, ep_size=2))
    different_ep = _plan(_coord(tp_size=2, tp_rank=0, cfg_size=2, cfg_rank=0, ep_size=2, ep_rank=1))
    different_tp = _plan(_coord(tp_size=2, tp_rank=1, cfg_size=2, cfg_rank=0, ep_size=2))
    different_cfg = _plan(_coord(tp_size=2, tp_rank=0, cfg_size=2, cfg_rank=1, ep_size=2))

    assert base.replica_fanout_identity == different_ep.replica_fanout_identity
    assert base.replica_fanout_identity != different_tp.replica_fanout_identity
    assert base.replica_fanout_identity != different_cfg.replica_fanout_identity


def test_connector_port_axes_include_active_non_dp_dimensions():
    coord = _coord(
        tp_size=2,
        tp_rank=1,
        cfg_size=2,
        cfg_rank=1,
        sp_size=4,
        sp_rank=3,
        ulysses_size=2,
        ulysses_rank=1,
        ring_size=2,
        ring_rank=0,
        ep_size=2,
        ep_rank=1,
    )

    axes = auto_connector_port_axes(coord)
    assert tuple(axis.name for axis in axes) == ("tp", "cfg", "ulysses", "ring", "ep")

    rank, world_size, axis_names = flatten_parallel_coord(coord, axes)
    assert axis_names == ("tp", "cfg", "ulysses", "ring", "ep")
    assert world_size == 32
    assert rank == 29

    port, port_rank, port_axes = select_connector_port_from_list(list(range(10000, 10032)), coord)
    assert port_rank == rank
    assert port_axes == axis_names
    assert port == 10029


def test_connector_port_axes_use_sp_only_when_ulysses_and_ring_are_absent():
    coord = _coord(sp_size=2, sp_rank=1)

    rank, world_size, axis_names = flatten_parallel_coord(coord)

    assert axis_names == ("sp",)
    assert world_size == 2
    assert rank == 1
