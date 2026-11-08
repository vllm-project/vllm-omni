# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-lifetime replica reservations, independent of request routing.

All mutations run synchronously on the orchestrator loop. A reservation counts
once against each physical stage replica, even when several groups share it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict

from vllm_omni.engine.duplex.messages import DuplexSessionError


@dataclass(frozen=True)
class ReplicaReservation:
    group_id: int
    replicas: Mapping[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "replicas", MappingProxyType(dict(self.replicas)))


class ReplicaPlacementSnapshot(TypedDict):
    groups: list[dict[int, int]]
    reserved: dict[tuple[int, int], int]
    sessions: dict[str, int]
    unavailable: frozenset[tuple[int, int]]


class DuplexReplicaPlacement:
    """Choose a complete group by occupancy and reserve it atomically."""

    def __init__(
        self,
        *,
        stage_replica_counts: Sequence[int],
        max_sessions_per_replica: int,
        replica_groups: Sequence[Mapping[int, int]] | None = None,
        frontend_count: int = 1,
        frontend_index: int = 0,
    ) -> None:
        counts = tuple(stage_replica_counts)
        if not counts or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("session replica placement requires a positive replica count for every stage")
        if type(max_sessions_per_replica) is not int or max_sessions_per_replica < 1:
            raise ValueError("max_sessions_per_replica must be a positive integer")
        if type(frontend_count) is not int or frontend_count < 1:
            raise ValueError("frontend_count must be a positive integer")
        if type(frontend_index) is not int or not 0 <= frontend_index < frontend_count:
            raise ValueError("frontend_index must be in [0, frontend_count)")
        if replica_groups is None:
            if len(set(counts)) != 1:
                raise ValueError("unequal stage replica counts require explicit duplex_session.replica_groups")
            replica_groups = [
                {stage_id: replica_id for stage_id in range(len(counts))} for replica_id in range(counts[0])
            ]
        if not replica_groups:
            raise ValueError("replica_groups must contain at least one group")

        expected_stages = set(range(len(counts)))
        all_replicas = {(stage_id, replica_id) for stage_id, count in enumerate(counts) for replica_id in range(count)}
        covered: set[tuple[int, int]] = set()
        seen: set[tuple[int, ...]] = set()
        groups: list[ReplicaReservation] = []
        for group_id, replicas in enumerate(replica_groups):
            if not isinstance(replicas, Mapping) or any(type(stage_id) is not int for stage_id in replicas):
                raise ValueError(f"replica_groups[{group_id}] must map integer stage IDs to replica IDs")
            if set(replicas) != expected_stages:
                raise ValueError(
                    f"replica_groups[{group_id}] must cover every stage exactly: {sorted(expected_stages)}"
                )
            for stage_id, replica_id in replicas.items():
                if type(replica_id) is not int or not 0 <= replica_id < counts[stage_id]:
                    raise ValueError(
                        f"replica_groups[{group_id}] replica_id for stage {stage_id} must be in [0, {counts[stage_id]})"
                    )
            key = tuple(replicas[stage_id] for stage_id in range(len(counts)))
            if key in seen:
                raise ValueError(f"replica_groups[{group_id}] duplicates an earlier group")
            seen.add(key)
            covered.update(replicas.items())
            groups.append(ReplicaReservation(group_id, replicas))
        if covered != all_replicas:
            raise ValueError(f"replica_groups must cover every replica; missing {sorted(all_replicas - covered)}")

        if frontend_count > len(groups):
            raise ValueError(
                f"api_server_count ({frontend_count}) must not exceed the number of replica groups ({len(groups)})"
            )
        replica_frontends: dict[tuple[int, int], int] = {}
        for group in groups:
            owner = group.group_id % frontend_count
            for replica in group.replicas.items():
                previous = replica_frontends.setdefault(replica, owner)
                if previous != owner:
                    raise ValueError(
                        f"replica_groups sharing stage {replica[0]} replica {replica[1]} must belong to the same "
                        f"frontend; group {group.group_id} belongs to frontend {owner}, already owned by {previous}"
                    )

        self._frontend_count = frontend_count
        self._frontend_index = frontend_index
        self._group_count = len(groups)
        self._groups = tuple(group for group in groups if group.group_id % frontend_count == frontend_index)
        owned_replicas = {replica for replica, owner in replica_frontends.items() if owner == frontend_index}
        # Every frontend still attaches to every replica and observes its death.
        # Only owned replicas participate in reservations and metrics.
        self._all_replicas = all_replicas
        self._capacity = max_sessions_per_replica
        self._reserved = dict.fromkeys(sorted(owned_replicas), 0)
        self._sessions: dict[str, ReplicaReservation] = {}
        self._owners: dict[tuple[int, int], set[str]] = {replica: set() for replica in owned_replicas}
        self._unavailable: set[tuple[int, int]] = set()
        self._cursor = 0

    @property
    def owned_group_count(self) -> int:
        return len(self._groups)

    def frontend_session_limit(self, global_max_sessions: int) -> int:
        """Bound this frontend by its equal share and its owned replica capacity.

        Each session occupies one replica on every stage, so the stage with
        the fewest owned replicas determines the capacity bound. If the group
        topology cannot fill that bound, ``reserve()`` remains authoritative
        and rejects excess sessions with a retryable error.
        """
        share, remainder = divmod(global_max_sessions, self._frontend_count)
        owned_replicas_per_stage = Counter(stage_id for stage_id, _ in self._reserved)
        owned_capacity = min(owned_replicas_per_stage.values()) * self._capacity
        return min(share + (self._frontend_index < remainder), owned_capacity)

    def reserve(self, session_id: str) -> ReplicaReservation:
        """Return an existing reservation or atomically choose an eligible group."""
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        eligible = [
            group
            for group in self._groups
            if all(
                replica not in self._unavailable and self._reserved[replica] < self._capacity
                for replica in group.replicas.items()
            )
        ]
        if not eligible:
            raise DuplexSessionError(
                "replica_capacity_exhausted: no replica group has capacity on every stage",
                code="resource_exhausted",
                retryable=True,
                session_id=session_id,
            )

        def score(group: ReplicaReservation) -> tuple[float, int, int]:
            occupancy = [self._reserved[replica] for replica in group.replicas.items()]
            return (
                max(occupancy) / self._capacity,
                sum(occupancy),
                (group.group_id - self._cursor) % self._group_count,
            )

        reservation = min(eligible, key=score)
        for replica in reservation.replicas.items():
            self._reserved[replica] += 1
            self._owners[replica].add(session_id)
        self._sessions[session_id] = reservation
        self._cursor = (reservation.group_id + 1) % self._group_count
        return reservation

    def release(self, session_id: str) -> None:
        """Release the whole session, independent of its request/turn lifetime."""
        reservation = self._sessions.pop(session_id, None)
        if reservation is None:
            return
        for replica in reservation.replicas.items():
            self._reserved[replica] -= 1
            self._owners[replica].remove(session_id)

    def route(self, session_id: str, stage_id: int) -> int:
        reservation = self._sessions.get(session_id)
        if reservation is None:
            raise DuplexSessionError(
                f"Unknown replica reservation for duplex session {session_id}",
                code="unknown_session",
                session_id=session_id,
            )
        unavailable = self._unavailable.intersection(reservation.replicas.items())
        if unavailable:
            raise DuplexSessionError(
                f"Replica group for duplex session {session_id} is unavailable: {sorted(unavailable)}",
                code="replica_unavailable",
                session_id=session_id,
            )
        return reservation.replicas[stage_id]

    def mark_unavailable(self, stage_id: int, replica_id: int) -> set[str]:
        replica = (stage_id, replica_id)
        if replica not in self._all_replicas:
            raise ValueError(f"Unknown stage replica {replica}")
        if replica not in self._owners:
            return set()
        self._unavailable.add(replica)
        return set(self._owners[replica])

    def snapshot(self) -> ReplicaPlacementSnapshot:
        """Detached state for structured logs, metrics and diagnostics."""
        return {
            "groups": [dict(group.replicas) for group in self._groups],
            "reserved": dict(self._reserved),
            "sessions": {session_id: group.group_id for session_id, group in self._sessions.items()},
            "unavailable": frozenset(self._unavailable),
        }
