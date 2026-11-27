# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-lifetime replica reservations, independent of request routing.

All mutations run synchronously on the orchestrator loop. A reservation counts
once against each physical stage replica, even when several groups share it.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
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
    # None represents the implicit product described by stage_replica_counts.
    groups: list[dict[int, int]] | None
    stage_replica_counts: tuple[int, ...]
    group_count: int
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
        all_replicas = {(stage_id, replica_id) for stage_id, count in enumerate(counts) for replica_id in range(count)}
        groups: list[ReplicaReservation] | None = None
        if replica_groups is None:
            group_count = prod(counts)
            if frontend_count > group_count:
                raise ValueError(
                    f"api_server_count ({frontend_count}) must not exceed the number of replica groups ({group_count})"
                )
            if frontend_count > 1 and len(counts) > 1:
                # Consecutive product IDs differ in the last nontrivial stage
                # and share every other stage's replica, violating ownership.
                varying_stage = max(stage for stage, count in enumerate(counts) if count > 1)
                shared_stage = next(stage for stage in range(len(counts)) if stage != varying_stage)
                raise ValueError(
                    f"replica_groups sharing stage {shared_stage} replica 0 must belong to the same frontend; "
                    "use explicit disjoint replica_groups for multiple frontends"
                )
            replica_frontends = {replica: replica[1] % frontend_count for replica in all_replicas}
        else:
            if not replica_groups:
                raise ValueError("replica_groups must contain at least one group")
            expected_stages = set(range(len(counts)))
            covered: set[tuple[int, int]] = set()
            seen: set[tuple[int, ...]] = set()
            groups = []
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
                            f"replica_groups[{group_id}] replica_id for stage {stage_id} "
                            f"must be in [0, {counts[stage_id]})"
                        )
                key = tuple(replicas[stage_id] for stage_id in range(len(counts)))
                if key in seen:
                    raise ValueError(f"replica_groups[{group_id}] duplicates an earlier group")
                seen.add(key)
                covered.update(replicas.items())
                groups.append(ReplicaReservation(group_id, replicas))
            if covered != all_replicas:
                raise ValueError(f"replica_groups must cover every replica; missing {sorted(all_replicas - covered)}")

            group_count = len(groups)
            if frontend_count > group_count:
                raise ValueError(
                    f"api_server_count ({frontend_count}) must not exceed the number of replica groups ({group_count})"
                )
            replica_frontends = {}
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
        self._stage_replica_counts = counts
        self._group_count = group_count
        self._groups = (
            tuple(group for group in groups if group.group_id % frontend_count == frontend_index)
            if groups is not None
            else None
        )
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
        if self._groups is None:
            return self._group_count if self._frontend_count == 1 else len(self._reserved)
        return len(self._groups)

    @property
    def live_group_count(self) -> int:
        """Complete groups owned by this frontend, including groups currently at capacity."""
        if not self._unavailable:
            return self.owned_group_count
        if self._groups is not None:
            return sum(self._unavailable.isdisjoint(group.replicas.items()) for group in self._groups)
        # The implicit product needs one live, owned replica on every stage.
        live_per_stage = Counter(stage for stage, _ in self._reserved.keys() - self._unavailable)
        return prod(live_per_stage[stage] for stage in range(len(self._stage_replica_counts)))

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

    def check_session_availability(self, session_id: str) -> None:
        """Raise if this frontend has no complete group, regardless of occupancy."""
        if self.live_group_count == 0:
            raise DuplexSessionError(
                "No complete duplex replica group is available on this frontend",
                code="replica_unavailable",
                retryable=self._frontend_count > 1,
                session_id=session_id,
            )

    def reserve(self, session_id: str) -> ReplicaReservation:
        """Return an existing reservation or atomically choose an eligible group."""
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        reservation = self._choose_implicit_group() if self._groups is None else self._choose_explicit_group()
        if reservation is None:
            self.check_session_availability(session_id)
            raise DuplexSessionError(
                "replica_capacity_exhausted: no replica group has capacity on every stage",
                code="resource_exhausted",
                retryable=True,
                session_id=session_id,
            )
        for replica in reservation.replicas.items():
            self._reserved[replica] += 1
            self._owners[replica].add(session_id)
        self._sessions[session_id] = reservation
        self._cursor = (reservation.group_id + 1) % self._group_count
        return reservation

    def _choose_implicit_group(self) -> ReplicaReservation | None:
        # The minimum peak and sum occupancy over an unrestricted product are
        # attained by choosing a minimum-occupancy replica independently per stage.
        choices: list[list[int]] = []
        for stage, count in enumerate(self._stage_replica_counts):
            minimum = self._capacity
            candidates: list[int] = []
            for replica_id in range(count):
                replica = (stage, replica_id)
                if replica in self._unavailable:
                    continue
                occupancy = self._reserved.get(replica, self._capacity)
                if occupancy < minimum:
                    minimum = occupancy
                    candidates = [replica_id]
                elif occupancy == minimum and occupancy < self._capacity:
                    candidates.append(replica_id)
            if not candidates:
                return None
            choices.append(candidates)

        # Find the first eligible mixed-radix ID at/after the cursor, wrapping
        # once. This preserves explicit-product round-robin ties without visiting
        # combinations. Backtracking only visits the already selected prefix.
        cursor = self._cursor
        digits = [0] * len(choices)
        for stage in range(len(digits) - 1, -1, -1):
            cursor, digits[stage] = divmod(cursor, self._stage_replica_counts[stage])
        selected: list[int] = []
        for stage, candidates in enumerate(choices):
            index = bisect_left(candidates, digits[stage])
            if index < len(candidates):
                selected.append(candidates[index])
                if candidates[index] > digits[stage]:
                    selected.extend(values[0] for values in choices[stage + 1 :])
                    break
            else:
                for previous in range(stage - 1, -1, -1):
                    index = bisect_right(choices[previous], selected[previous])
                    if index < len(choices[previous]):
                        selected = selected[:previous] + [choices[previous][index]]
                        selected.extend(values[0] for values in choices[previous + 1 :])
                        break
                else:
                    selected = [values[0] for values in choices]
                break
        group_id = 0
        for replica_id, count in zip(selected, self._stage_replica_counts):
            group_id = group_id * count + replica_id
        return ReplicaReservation(group_id, dict(enumerate(selected)))

    def _choose_explicit_group(self) -> ReplicaReservation | None:
        assert self._groups is not None

        def score(group: ReplicaReservation) -> tuple[float, int, int]:
            occupancy = [self._reserved[replica] for replica in group.replicas.items()]
            return (
                max(occupancy) / self._capacity,
                sum(occupancy),
                (group.group_id - self._cursor) % self._group_count,
            )

        eligible = [
            group
            for group in self._groups
            if all(
                replica not in self._unavailable and self._reserved[replica] < self._capacity
                for replica in group.replicas.items()
            )
        ]
        return min(eligible, key=score) if eligible else None

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
            # A dynamically attached replica can die at an id the startup topology
            # never declared. It owns no reservation, so there is nothing to fail
            # here; the caller still evicts it and fails its ordinary requests.
            return set()
        if replica not in self._owners:
            return set()
        self._unavailable.add(replica)
        return set(self._owners[replica])

    def snapshot(self) -> ReplicaPlacementSnapshot:
        """Detached state for structured logs, metrics and diagnostics."""
        return {
            "groups": [dict(group.replicas) for group in self._groups] if self._groups is not None else None,
            "stage_replica_counts": self._stage_replica_counts,
            "group_count": self._group_count,
            "reserved": dict(self._reserved),
            "sessions": {session_id: group.group_id for session_id, group in self._sessions.items()},
            "unavailable": frozenset(self._unavailable),
        }
