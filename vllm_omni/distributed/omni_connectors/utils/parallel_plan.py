# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel-coordinate planning for Omni KV cache transfer.

This module intentionally contains pure planning primitives.  The transfer
manager still owns connector I/O and request mutation; the plan only answers
which ranks should receive remotely and which local collectives are safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KVAxisRole = Literal["tensor_shard", "branch", "replica", "container"]
KVReplicaFanoutGroup = Literal["ep"]
DEFAULT_REPLICA_FANOUT_IDENTITY_AXES = ("tp", "pp", "ring", "ulysses", "cfg")
DEFAULT_CONNECTOR_PORT_AXIS_ORDER = (
    "tp",
    "pp",
    "cfg",
    "ulysses",
    "ring",
    "sp",
    "ep",
    "fs",
    "vae_patch",
)
CONNECTOR_PORT_EXCLUDED_AXES = frozenset({"dp"})


@dataclass(frozen=True)
class ParallelAxis:
    """One coordinate in the local parallel mesh."""

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
    """Current rank's coordinate in all KV-relevant parallel dimensions."""

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


def auto_connector_port_axes(coord: KVParallelRankCoord) -> tuple[ParallelAxis, ...]:
    """Return active non-DP axes that should index connector endpoint ports.

    DP is intentionally excluded because the orchestrator owns replica-level
    placement and should provide a separate port list per replica.  SP is a
    container for Ulysses and Ring; include SP only as a compatibility fallback
    when neither Ulysses nor Ring is exposed as an active axis.
    """
    ulysses_or_ring_active = coord.size("ulysses") > 1 or coord.size("ring") > 1
    axes: list[ParallelAxis] = []
    for axis_name in DEFAULT_CONNECTOR_PORT_AXIS_ORDER:
        if axis_name in CONNECTOR_PORT_EXCLUDED_AXES:
            continue
        if axis_name == "sp" and ulysses_or_ring_active:
            continue
        axis = coord.axis(axis_name)
        if axis.active:
            axes.append(axis)
    return tuple(axes)


def flatten_parallel_coord(
    coord: KVParallelRankCoord,
    axes: tuple[ParallelAxis, ...] | None = None,
) -> tuple[int, int, tuple[str, ...]]:
    """Flatten selected coordinate axes into a stable row-major rank index."""
    selected_axes = auto_connector_port_axes(coord) if axes is None else axes
    rank = 0
    world_size = 1
    names: list[str] = []
    for axis in selected_axes:
        rank = rank * axis.size + axis.rank
        world_size *= axis.size
        names.append(axis.name)
    return rank, world_size, tuple(names)


def select_connector_port_from_list(
    port_list: list[int] | tuple[int, ...],
    coord: KVParallelRankCoord,
) -> tuple[int, int, tuple[str, ...]]:
    """Select a connector port from a list using the auto-selected axes."""
    index, world_size, axes = flatten_parallel_coord(coord)
    if len(port_list) < world_size:
        raise ValueError(
            f"connector port list has {len(port_list)} entries, "
            f"but selected axes {axes} require at least {world_size}"
        )
    return int(port_list[index]), index, axes


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
    ep_active = coord.size("ep") > 1

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
        replica_fanout_enabled = ep_active and mode in (
            "independent_remote",
            "cfg_sp_local_distribution",
        )
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
