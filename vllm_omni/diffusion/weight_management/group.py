# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replica-wide coordination and process-owned AllGather groups (RFC 01-4)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import torch.distributed as dist

from vllm_omni.diffusion.weight_management.errors import (
    ConfigError,
    ConfirmationTimeoutError,
    ErrorSummary,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
    WeightStackError,
)
from vllm_omni.diffusion.weight_management.interfaces import (
    AllGatherGroup,
    AllGatherGroups,
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    StartupContext,
    Topology,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm_omni.diffusion.weight_management.host import LocalRecord

logger = logging.getLogger(__name__)

DEFAULT_CONFIRMATION_TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True)
class RoundMessage:
    """One rank's message exchanged at a confirmation point."""

    point: str
    rank: int
    ok: bool
    error: ErrorSummary | None
    payload: object | None

    @classmethod
    def of(
        cls,
        point: str,
        rank: int,
        pending: WeightStackError | None,
        payload: object | None,
    ) -> RoundMessage:
        if pending is not None:
            return cls(point=point, rank=rank, ok=False, error=pending.summary(), payload=None)
        return cls(point=point, rank=rank, ok=True, error=None, payload=payload)


@dataclass(frozen=True)
class Point0Payload:
    """Point-0 payload: node-local record, rank coordinates, and deployment ID claim."""

    record: LocalRecord
    coordinates: RankCoordinates
    deployment: tuple[str, bool]


@dataclass(frozen=True)
class Point1Payload:
    """Point-1 payload: fingerprints, cached group keys, and locally unmatched strict policy keys."""

    fingerprints: RankFingerprints
    group_keys: frozenset[tuple[tuple[int, ...], str]]
    unmatched_policy_keys: frozenset[str]


class GroupConfirmation:
    """Replica-wide confirmation channel over the world Gloo group."""

    def __init__(
        self,
        *,
        world_size: int = 1,
        rank: int = 0,
        cpu_group: ProcessGroup | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        seconds = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
        if seconds <= 0:
            raise ConfigError(f"dist_timeout must be positive, got {timeout_seconds!r}")
        if world_size > 1 and cpu_group is None:
            raise ConfigError("multi-rank confirmation requires the replica world Gloo group")
        self.world_size = world_size
        self.rank = rank
        self.cpu_group = cpu_group
        self.timeout = timedelta(seconds=seconds)

    @classmethod
    def from_config(cls, config: object) -> GroupConfirmation:
        timeout = getattr(config, "dist_timeout", None)
        if not dist.is_initialized():
            return cls(world_size=1, timeout_seconds=timeout)
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        world = get_world_group()
        return cls(
            world_size=world.world_size,
            rank=world.rank_in_group,
            cpu_group=world.cpu_group,
            timeout_seconds=timeout,
        )

    def exchange(self, point: str, message: RoundMessage) -> tuple[RoundMessage, ...]:
        """Synchronize all ranks on ``point`` and return rank-ordered messages."""
        if self.world_size == 1:
            gathered: list[object] = [message]
        else:
            self._barrier(point)
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, message, group=self.cpu_group)

        for rank, item in enumerate(gathered):
            if not isinstance(item, RoundMessage) or item.point != point:
                raise GroupConfirmationError(
                    f"inconsistent coordination point {point!r}: rank {rank} sent {item!r}",
                    phase=point,
                )
        return cast(tuple[RoundMessage, ...], tuple(gathered))

    def _barrier(self, point: str) -> None:
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                dist.monitored_barrier(group=self.cpu_group, timeout=self.timeout, wait_all_ranks=True)
            except BaseException as exc:
                errors.append(exc)

        # monitored_barrier enforces timeout on rank 0 only; bound other ranks with a watchdog.
        started = time.monotonic()
        seconds = self.timeout.total_seconds()
        thread = threading.Thread(target=_run, name="weight-stack-barrier", daemon=True)
        thread.start()
        thread.join(seconds + 0.5)
        if not thread.is_alive() and not errors:
            return

        cause = errors[0] if errors else None
        if cause is not None:
            msg = str(cause).lower()
            elapsed = time.monotonic() - started
            is_timeout = isinstance(cause, RuntimeError) and (
                "monitoredbarrier" in msg or ("timed out" in msg and elapsed >= seconds - 0.1)
            )
            if not is_timeout:
                raise cause

        detail = f": {cause}" if cause is not None else " (local watchdog expired)"
        err = ConfirmationTimeoutError(
            f"confirmation point {point!r} timed out after {seconds:g}s{detail}",
            phase=point,
        )
        err.rank = self.rank
        err.world_size = self.world_size
        raise err from cause


def raise_if_any_failed(gathered: Sequence[RoundMessage], pending: WeightStackError | None) -> None:
    """Re-raise local pending error or aggregate peer failures."""
    if pending is not None:
        raise pending
    failed = [m for m in gathered if not m.ok]
    if failed:
        details = "; ".join(
            f"rank {m.rank}: {m.error.message if m.error is not None else 'missing error summary'}" for m in failed
        )
        raise PeerFailureError(details, phase=failed[0].point)


def read_rank_coordinates() -> RankCoordinates:
    """Read local parallel coordinates; disabled EP/HSDP coordinates are None."""
    from vllm.distributed import parallel_state as vllm_ps

    from vllm_omni.diffusion.distributed import parallel_state as diff_ps

    def _coord(group: object) -> ParallelCoordinate:
        return ParallelCoordinate(size=group.world_size, rank=group.rank_in_group)  # type: ignore[attr-defined]

    def _opt(getter: Callable[[], object]) -> ParallelCoordinate | None:
        try:
            return _coord(getter())
        except AssertionError:
            return None

    sp = diff_ps.get_sp_group()
    return RankCoordinates(
        tp=_coord(vllm_ps.get_tp_group()),
        sp=_coord(sp),
        ulysses=ParallelCoordinate(size=sp.ulysses_world_size, rank=sp.ulysses_rank),
        ring=ParallelCoordinate(size=sp.ring_world_size, rank=sp.ring_rank),
        cfg=_coord(diff_ps.get_cfg_group()),
        dp=_coord(diff_ps.get_dp_group()),
        pp=_coord(diff_ps.get_pp_group()),
        ep=_opt(vllm_ps.get_ep_group),
        fs=_opt(diff_ps.get_fs_group),
    )


def assemble_topology(gathered: Sequence[Point0Payload], ctx: StartupContext) -> Topology:
    """Adopt rank 0's deployment identity, verify propagated IDs, and build Topology."""
    if len(gathered) != ctx.world_size:
        raise GroupConfirmationError(
            f"expected {ctx.world_size} point-0 payloads, got {len(gathered)}",
            phase="point0_topology",
        )
    adopted_id, scoped = gathered[0].deployment
    for rank, payload in enumerate(gathered):
        dep_id, propagated = payload.deployment
        if propagated and dep_id != adopted_id:
            raise GroupConfirmationError(
                f"rank {rank} reported deployment_id {dep_id!r}, expected {adopted_id!r}",
                phase="point0_topology",
            )
    return Topology(
        deployment_id=adopted_id,
        deployment_scoped=scoped,
        stage_id=ctx.od_config.stage_id,
        world_size=ctx.world_size,
        records=tuple(p.record for p in gathered),
        coordinates=tuple(p.coordinates for p in gathered),
    )


def check_unmatched_policy_keys(gathered: Sequence[Point1Payload]) -> None:
    """Raise L01 PlanningError if any strict policy key matched no component on any rank."""
    if not gathered:
        return
    unmatched = set(gathered[0].unmatched_policy_keys)
    for payload in gathered[1:]:
        unmatched.intersection_update(payload.unmatched_policy_keys)
    if unmatched:
        keys = ", ".join(repr(k) for k in sorted(unmatched))
        declared = sorted({name for p in gathered for name in p.fingerprints.content_identities})
        raise PlanningError(
            f"offload policy key(s) {keys} match no component on any rank; declared: {', '.join(declared) or '<none>'}",
            fix="remove the key(s) or use a declared component name or role",
        )


def _plan_groups(fingerprints: Sequence[RankFingerprints]) -> list[tuple[str, tuple[int, ...]]]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    components = sorted({name for fp in fingerprints for name in fp.collective})
    for comp in components:
        by_identity: dict[str, list[int]] = {}
        for rank, fp in enumerate(fingerprints):
            in_decl = comp in fp.content_identities
            in_coll = comp in fp.collective
            if in_decl != in_coll:
                raise GroupConfirmationError(
                    f"component {comp!r}: rank {rank} disagrees on collective participation",
                    phase="point1_fingerprints",
                )
            if in_decl:
                by_identity.setdefault(fp.content_identities[comp], []).append(rank)
        for ranks in by_identity.values():
            if len({fingerprints[r].collective[comp] for r in ranks}) != 1:
                raise GroupConfirmationError(
                    f"component {comp!r}: ranks {ranks} disagree on collective layout",
                    phase="point1_fingerprints",
                )
            groups.append((comp, tuple(ranks)))
    return sorted(groups, key=lambda item: (item[0], item[1][0]))


def check_collective_parts(fingerprints: Sequence[RankFingerprints]) -> None:
    """Validate collective participation and digest agreement across ranks."""
    _plan_groups(fingerprints)


# Process-level table of AllGather groups; _owned_groups retains replaced handles until shutdown.
_allgather_groups: dict[tuple[tuple[int, ...], str], ProcessGroup | None] = {}
_owned_groups: list[ProcessGroup] = []


def group_table_keys() -> frozenset[tuple[tuple[int, ...], str]]:
    """Return cached ``(ranks, backend)`` keys held by this process."""
    return frozenset(_allgather_groups)


def create_allgather_groups(
    fingerprints: Sequence[RankFingerprints],
    gathered: Sequence[Point1Payload],
    *,
    backend: str | None = None,
) -> AllGatherGroups:
    """Create or reuse AllGather process groups in deterministic order on every rank."""
    planned = _plan_groups(fingerprints)
    for comp, ranks in planned:
        if len(ranks) < 2:
            raise GroupConfirmationError(
                f"component {comp!r}: AllGather group {ranks} requires at least 2 ranks",
                phase="point1_fingerprints",
            )
    if not planned:
        return {}
    if backend is None:
        from vllm_omni.diffusion.distributed.parallel_state import current_omni_platform

        backend = current_omni_platform.dist_backend

    my_rank = dist.get_rank()
    created_this_round: set[tuple[tuple[int, ...], str]] = set()
    result: dict[str, list[AllGatherGroup]] = {}
    for comp, ranks in planned:
        key = (ranks, backend)
        if key not in created_this_round and not all(key in p.group_keys for p in gathered):
            pg = dist.new_group(list(ranks), backend=backend)
            _allgather_groups[key] = pg if my_rank in ranks else None
            if my_rank in ranks:
                _owned_groups.append(pg)
            created_this_round.add(key)
        result.setdefault(comp, []).append(AllGatherGroup(component=comp, ranks=ranks, group=_allgather_groups[key]))
    return {comp: tuple(gs) for comp, gs in result.items()}


def destroy_allgather_groups() -> None:
    """Destroy all process-owned AllGather groups before distributed teardown. Idempotent."""
    _allgather_groups.clear()
    owned = list(reversed(_owned_groups))
    _owned_groups.clear()
    if not owned or not dist.is_initialized():
        return
    for pg in owned:
        try:
            dist.destroy_process_group(pg)
        except Exception:
            logger.exception("Failed to destroy an owned AllGather group")
