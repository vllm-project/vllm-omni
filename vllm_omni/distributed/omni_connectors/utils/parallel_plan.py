# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel-coordinate planning for Omni KV cache transfer.

This module intentionally contains pure planning primitives.  The transfer
manager still owns connector I/O and request mutation; the plan only answers
which ranks should receive remotely and which local collectives are safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm_omni.config.omni_config import OmniStageParallelConfig

KVAxisRole = Literal["tensor_shard", "branch", "replica", "container"]
KVReplicaFanoutGroup = Literal["ep"]
# Identity axes used by EP fanout. EP is intentionally omitted because EP ranks
# are attention-KV replicas; ranks that only differ by EP rank can share one
# remotely received payload inside the EP group.
DEFAULT_REPLICA_FANOUT_IDENTITY_AXES = ("tp", "pp", "ring", "ulysses", "cfg")


@dataclass(frozen=True)
class KVReceiveRankInfo:
    """Runtime rank coordinates needed to project a ParallelConfig into KV semantics.

    Size fields come from the unified ``OmniStageParallelConfig`` hierarchy.
    This object carries only runtime ranks and the EP group size, because EP
    size is supplied by vLLM's live expert-parallel group.
    """

    tp_rank: int = 0
    pp_rank: int = 0
    ring_rank: int = 0
    ulysses_rank: int = 0
    cfg_rank: int = 0
    sp_rank: int = 0
    dp_rank: int = 0
    dp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1


@dataclass(frozen=True)
class ParallelAxis:
    """One coordinate in the local parallel mesh.

    Example:
        ParallelAxis("tp", size=2, rank=0, role="tensor_shard") means the
        current rank is tensor-parallel rank 0 in a TP-2 group. Since TP shards
        attention KV, this axis participates in KV identity.
    """

    name: str
    size: int = 1
    rank: int = 0
    role: KVAxisRole = "container"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Parallel axis {self.name!r} has invalid size={self.size}")
        if self.rank < 0 or self.rank >= self.size:
            raise ValueError(f"Parallel axis {self.name!r} has rank={self.rank} outside [0, {self.size})")

    @property
    def active(self) -> bool:
        return self.size > 1

    @property
    def shards_kv(self) -> bool:
        return self.active and self.role == "tensor_shard"


@dataclass(frozen=True)
class KVParallelRankCoord:
    """Current rank's coordinate in all KV-relevant parallel dimensions.

    Example:
        A DiT rank with TP2+EP2 may have axes
        (tp rank 0/2, ep rank 1/2). For KV identity, TP is included because it
        shards KV, while EP is a replica axis used for local fanout.
    """

    axes: tuple[ParallelAxis, ...]

    def axis(self, name: str) -> ParallelAxis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        return ParallelAxis(name=name)

    def size(self, name: str) -> int:
        return self.axis(name).size

    def rank(self, name: str) -> int:
        return self.axis(name).rank

    @property
    def tensor_shard_axes(self) -> tuple[ParallelAxis, ...]:
        return tuple(axis for axis in self.axes if axis.shards_kv)

    @property
    def sequence_sharded(self) -> bool:
        """Whether SP internals are expected to need distinct KV sequence shards."""
        return self.size("ring") > 1 or self.size("ulysses") > 1

    @property
    def pp_sharded(self) -> bool:
        return self.size("pp") > 1


def build_kv_parallel_rank_coord(
    *,
    parallel_config: OmniStageParallelConfig,
    ranks: KVReceiveRankInfo,
    target_tp_size: int,
) -> KVParallelRankCoord:
    """Build a KV rank coordinate from the unified per-stage parallel config."""
    return KVParallelRankCoord(
        axes=(
            ParallelAxis("tp", target_tp_size, ranks.tp_rank, "tensor_shard"),
            ParallelAxis(
                "pp",
                int(parallel_config.pipeline_parallel_size),
                ranks.pp_rank,
                "tensor_shard",
            ),
            ParallelAxis(
                "ring",
                int(getattr(parallel_config, "ring_degree", 1)),
                ranks.ring_rank,
                "tensor_shard",
            ),
            ParallelAxis(
                "ulysses",
                int(getattr(parallel_config, "ulysses_degree", 1)),
                ranks.ulysses_rank,
                "tensor_shard",
            ),
            ParallelAxis(
                "cfg",
                int(getattr(parallel_config, "cfg_parallel_size", 1)),
                ranks.cfg_rank,
                "branch",
            ),
            ParallelAxis(
                "sp",
                int(getattr(parallel_config, "sequence_parallel_size", 1)),
                ranks.sp_rank,
                "container",
            ),
            ParallelAxis(
                "dp",
                int(getattr(parallel_config, "data_parallel_size", ranks.dp_size)),
                ranks.dp_rank,
                "replica",
            ),
            ParallelAxis("ep", ranks.ep_size, ranks.ep_rank, "replica"),
        )
    )


@dataclass(frozen=True)
class KVReceiveDistributionPlan:
    """How a receive-side rank participates in remote receive/local fanout."""

    coord: KVParallelRankCoord
    world_size: int
    world_rank: int
    mode: Literal[
        "single",
        "independent_remote",
        "cfg_sp_local_distribution",
        "world_broadcast",
    ]
    tp_active: bool
    cfg_active: bool
    sp_broadcast_enabled: bool
    sequence_sharded: bool
    owner_receives: bool
    replica_fanout_enabled: bool = False
    replica_fanout_group: KVReplicaFanoutGroup | None = None
    replica_fanout_identity_axes: tuple[str, ...] = DEFAULT_REPLICA_FANOUT_IDENTITY_AXES

    @property
    def cfg_size(self) -> int:
        return self.coord.size("cfg")

    @property
    def cfg_rank(self) -> int:
        return self.coord.rank("cfg")

    @property
    def sp_size(self) -> int:
        return self.coord.size("sp")

    @property
    def sp_rank(self) -> int:
        return self.coord.rank("sp")

    @property
    def uses_local_distribution(self) -> bool:
        return self.mode == "cfg_sp_local_distribution"

    @property
    def independently_receives_remote(self) -> bool:
        return self.mode in ("single", "independent_remote")

    @property
    def uses_world_broadcast(self) -> bool:
        return self.mode == "world_broadcast"

    @property
    def uses_replica_fanout(self) -> bool:
        return self.replica_fanout_enabled and self.replica_fanout_group is not None

    @property
    def replica_fanout_identity(self) -> tuple[tuple[str, int, int], ...]:
        """KV identity used to decide whether replica ranks can share payload."""
        return tuple(
            (axis_name, self.coord.rank(axis_name), self.coord.size(axis_name))
            for axis_name in self.replica_fanout_identity_axes
        )

    @property
    def cfg_follower_receives_from_cfg_leader(self) -> bool:
        if not self.uses_local_distribution or not self.cfg_active:
            return False
        if self.cfg_rank == 0:
            return False
        if self.sp_broadcast_enabled:
            return self.sp_rank == 0
        return True


def build_kv_receive_distribution_plan(
    *,
    coord: KVParallelRankCoord,
    world_size: int,
    world_rank: int,
    source_tp_size: int,
    target_tp_size: int,
) -> KVReceiveDistributionPlan:
    """Build the receive-side distribution plan for this rank.

    The rule is conservative:
    * TP/PP/Ring/Ulysses are treated as KV-sharding dimensions, so ranks along
      those axes should not receive an arbitrary peer's final payload.
    * CFG keeps the existing branch-local distribution behavior.
    * SP/Ring/Ulysses ranks share the current TP-rank-aware transport key, so
      only sp-rank 0 receives from the remote connector and then broadcasts
      within the SP group.  This avoids multiple sequence ranks consuming the
      same one-shot SHM segment.
    * If no KV-sharding or local-distribution axis is active, retain the
      historical world-rank-0 receive + broadcast fallback.
    """
    tp_active = source_tp_size > 1 or target_tp_size > 1
    cfg_active = coord.size("cfg") > 1
    sp_active = coord.size("sp") > 1
    sequence_sharded = coord.sequence_sharded
    sp_broadcast_enabled = sp_active
    pp_sharded = coord.pp_sharded
    local_distribution = cfg_active or sp_broadcast_enabled

    def build_plan(
        *,
        mode: Literal[
            "single",
            "independent_remote",
            "cfg_sp_local_distribution",
            "world_broadcast",
        ],
        owner_receives: bool,
    ) -> KVReceiveDistributionPlan:
        # Current vLLM-Omni EP is not an independent parallel axis in the
        # unified config. ``get_ep_group()`` is derived from the existing
        # TP/SP/CFG/DP mesh for MoE expert dispatch, so ranks in that group are
        # not guaranteed to be KV-equivalent replicas. Keep KV replica fanout
        # disabled until an explicit ep_size axis exists.
        replica_fanout_enabled = False
        return KVReceiveDistributionPlan(
            coord=coord,
            world_size=world_size,
            world_rank=world_rank,
            mode=mode,
            tp_active=tp_active,
            cfg_active=cfg_active,
            sp_broadcast_enabled=sp_broadcast_enabled,
            sequence_sharded=sequence_sharded,
            owner_receives=owner_receives,
            replica_fanout_enabled=replica_fanout_enabled,
            replica_fanout_group="ep" if replica_fanout_enabled else None,
        )

    if world_size <= 1:
        return build_plan(
            mode="single",
            owner_receives=True,
        )

    if local_distribution:
        owner_receives = coord.rank("cfg") == 0
        if sp_broadcast_enabled:
            owner_receives = owner_receives and coord.rank("sp") == 0
        return build_plan(
            mode="cfg_sp_local_distribution",
            owner_receives=owner_receives,
        )

    if tp_active or sequence_sharded or pp_sharded:
        return build_plan(
            mode="independent_remote",
            owner_receives=True,
        )

    return build_plan(
        mode="world_broadcast",
        owner_receives=world_rank == 0,
    )
