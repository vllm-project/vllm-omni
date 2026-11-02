# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replica-wide group confirmation for the unified weight stack.

RFC #9536, ``rfcs/01-startup-transaction.md`` section 7 (sub-RFC 01-4).

The confirmation protocol runs on the replica's world Gloo group, i.e. the
``cpu_group`` of ``vllm_omni.diffusion.distributed.parallel_state``'s world
coordinator.  Each round first synchronizes with ``monitored_barrier`` so that a
rank which never arrives turns into a bounded timeout instead of an indefinite
wait, and then exchanges exactly one picklable :class:`RoundMessage` per rank
with ``all_gather_object``.  A round is complete only when every rank reports
the same confirmation point; a rank that already failed sends ``ok=False`` plus
an :class:`~vllm_omni.diffusion.weight_stack.errors.ErrorSummary``, and every
rank then raises.

This module also owns the replica-wide topology assembly (point 0), the
deterministic AllGather process-group calculation/creation/reuse (point 1) and
the process-level AllGather group table.  The table is created lazily on the
first startup and destroyed only by :func:`destroy_allgather_groups`; closing a
startup transaction or an ``OffloadRuntime`` never touches process groups.  The
startup driver (sub-RFC 01-5) owns the state machine, resource release and the
``Election`` lifecycle; this module never destroys resources other than the
AllGather groups it created itself.

``torch.distributed.monitored_barrier`` only bounds rank 0, so a daemon-thread
watchdog bounds every other rank locally as well.  The process-level table is
single-threaded state of the startup path: it is not inherited safely across
``fork()`` and is not protected by a lock, matching the RFC's requirement that
only one unified startup runs per process at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm_omni.diffusion.weight_stack.errors import (
    ConfigError,
    ConfirmationTimeoutError,
    ErrorSummary,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
    WeightStackError,
)
from vllm_omni.diffusion.weight_stack.interfaces import (
    AllGatherGroup,
    AllGatherGroups,
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    Topology,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm_omni.diffusion.weight_stack.host import LocalRecord
    from vllm_omni.diffusion.weight_stack.interfaces import StartupContext

logger = logging.getLogger(__name__)

DEFAULT_CONFIRMATION_TIMEOUT_SECONDS = 1800.0
"""Timeout of one confirmation round when ``od_config.dist_timeout`` is unset."""

_CONFIRMATION_TIMEOUT_TOLERANCE_SECONDS = 0.5
"""Slack used to tell "barrier ran into its timeout" from "a peer died early"."""

_CONFIRMATION_WATCHDOG_GRACE_SECONDS = 1.0
"""Extra time a rank gives the monitored barrier beyond the round timeout."""


class _BarrierWatchdogTimeout:
    """Sentinel returned when the local watchdog fires before the barrier returns."""


_BARRIER_WATCHDOG_TIMEOUT = _BarrierWatchdogTimeout()

_MONITORED_BARRIER_FAILURE_MARKER = "failed to pass monitoredbarrier"

_REQUIRED_COORDINATES = ("tp", "sp", "ulysses", "ring", "cfg", "dp", "pp")
_OPTIONAL_COORDINATES = ("ep", "fs")
_ORTHOGONAL_COORDINATES = ("tp", "sp", "pp", "cfg", "dp")
# ``sp`` already contains ulysses*ring*allgather, so only the orthogonal
# dimensions multiply to the replica world size.


@dataclass(frozen=True)
class RoundMessage:
    """One rank's contribution to one confirmation round."""

    point: str
    rank: int
    ok: bool
    error: ErrorSummary | None  # set when ok is False
    payload: object | None  # set when ok is True; must be picklable

    @classmethod
    def of(
        cls,
        point: str,
        rank: int,
        pending: WeightStackError | None,
        payload: object | None,
    ) -> RoundMessage:
        """Build the message for this round; a pending error makes the round fail."""
        if pending is not None:
            return cls(point=point, rank=rank, ok=False, error=pending.summary(), payload=None)
        return cls(point=point, rank=rank, ok=True, error=None, payload=payload)


@dataclass(frozen=True)
class Point0Payload:
    """What each rank contributes to confirmation point 0 (topology collection)."""

    record: LocalRecord  # HostWeightManager.local_record()
    coordinates: RankCoordinates  # this rank's size and rank in every parallel dimension
    deployment: tuple[str, bool]  # HostWeightManager.deployment_id()


@dataclass(frozen=True)
class Point1Payload:
    """What each rank contributes to the first round of confirmation point 1."""

    fingerprints: RankFingerprints
    group_keys: frozenset[tuple[tuple[int, ...], str]]  # (members, backend) already in the process-level table
    unmatched_policy_keys: frozenset[str]  # policy keys with no matching component on this rank


class GroupConfirmation:
    """The replica-wide confirmation channel of one startup.

    ``world_size == 1`` completes every round in-process and never touches
    ``torch.distributed``.  For larger replicas ``cpu_group`` must be the Gloo
    process group over all replica ranks.
    """

    def __init__(
        self,
        *,
        world_size: int,
        cpu_group: ProcessGroup | None = None,
        timeout_seconds: float | None = None,
        rank: int = 0,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside the replica of world_size {world_size}")
        if world_size > 1 and cpu_group is None:
            raise ValueError("cpu_group is required when world_size is greater than 1")
        resolved_timeout = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
        if resolved_timeout <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {resolved_timeout}")
        self.world_size = world_size
        self.rank = rank
        self.cpu_group = cpu_group
        self.timeout_seconds = resolved_timeout

    @property
    def timeout(self) -> timedelta:
        """Timeout of one round, as expected by ``monitored_barrier``."""
        return timedelta(seconds=self.timeout_seconds)

    @classmethod
    def from_config(cls, config: object, *, rank: int | None = None) -> GroupConfirmation:
        """Build the channel from ``od_config`` and the current distributed state.

        The timeout is ``od_config.dist_timeout`` seconds when set, otherwise
        1800 seconds.  When the distributed environment is not initialized the
        replica is treated as a single rank.
        """
        timeout_seconds = _timeout_seconds_from_config(config)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            resolved_rank = dist.get_rank() if rank is None else rank
        else:
            world_size = 1
            resolved_rank = 0 if rank is None else rank
        cpu_group: ProcessGroup | None = None
        if world_size > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            cpu_group = get_world_group().cpu_group
        return cls(
            world_size=world_size,
            cpu_group=cpu_group,
            timeout_seconds=timeout_seconds,
            rank=resolved_rank,
        )

    def exchange(self, point: str, message: RoundMessage) -> tuple[RoundMessage, ...]:
        """Run one confirmation round and return every rank's message in rank order."""
        if message.point != point:
            raise ValueError(f"round message point {message.point!r} does not match round point {point!r}")
        if message.rank != self.rank:
            raise ValueError(f"round message rank {message.rank} does not match this rank {self.rank}")
        if self.world_size == 1:
            return (message,)
        if self.cpu_group is None:  # pragma: no cover - guarded by the constructor
            raise ValueError("cpu_group is required when world_size is greater than 1")

        started = time.monotonic()
        barrier_result = _monitored_barrier_with_watchdog(self.cpu_group, self.timeout)
        if barrier_result is _BARRIER_WATCHDOG_TIMEOUT:
            # The local watchdog fired: this rank did not see the barrier return
            # within the timeout (for example because rank 0 itself never
            # arrived).  ``torch.distributed`` alone only bounds rank 0.
            raise _confirmation_timeout_error(point, self, None)
        if isinstance(barrier_result, BaseException):
            missing_ranks = _missing_ranks_from_barrier_error(barrier_result)
            elapsed = time.monotonic() - started
            if missing_ranks is None and elapsed + _CONFIRMATION_TIMEOUT_TOLERANCE_SECONDS < self.timeout_seconds:
                # A peer process died before the barrier timed out: the transport
                # error is not a confirmation timeout, so let it propagate raw.
                raise barrier_result
            raise _confirmation_timeout_error(point, self, missing_ranks) from barrier_result

        gathered: list[RoundMessage] = [None] * self.world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, message, group=self.cpu_group)
        result = tuple(gathered)

        points = {item.point for item in result}
        if len(points) != 1 or point not in points:
            details = ", ".join(f"rank {index}: {item.point!r}" for index, item in enumerate(result))
            raise GroupConfirmationError(
                f"ranks are not at the same confirmation point, expected {point!r} ({details})",
                phase=point,
            )
        return result


def raise_if_any_failed(gathered: Sequence[RoundMessage], pending: WeightStackError | None) -> None:
    """Raise this rank's own pending error, or a peer-failure error listing every failed rank."""
    if pending is not None:
        raise pending
    failed = [message for message in gathered if not message.ok]
    if not failed:
        return
    point = gathered[0].point if gathered else "unknown"
    details = []
    for message in failed:
        message_text = message.error.message if message.error is not None else "no error summary"
        details.append(f"rank {message.rank}: {message_text}")
    raise PeerFailureError(
        f"confirmation point {point!r} failed on {len(failed)} rank(s): " + "; ".join(details),
        phase=point,
    )


def read_rank_coordinates() -> RankCoordinates:
    """Read this rank's coordinates from the Omni and vLLM parallel state.

    ``ep`` is ``None`` when no expert-parallel group exists (vLLM only creates
    one for MoE models) and ``fs`` is ``None`` when HSDP is not enabled.
    """
    from vllm.distributed import parallel_state as vllm_parallel_state

    from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

    def coordinate(world_size_getter: Callable[[], int], rank_getter: Callable[[], int]) -> ParallelCoordinate:
        return ParallelCoordinate(size=int(world_size_getter()), rank=int(rank_getter()))

    return RankCoordinates(
        tp=coordinate(
            vllm_parallel_state.get_tensor_model_parallel_world_size,
            vllm_parallel_state.get_tensor_model_parallel_rank,
        ),
        sp=coordinate(
            omni_parallel_state.get_sequence_parallel_world_size,
            omni_parallel_state.get_sequence_parallel_rank,
        ),
        ulysses=coordinate(
            omni_parallel_state.get_ulysses_parallel_world_size,
            omni_parallel_state.get_ulysses_parallel_rank,
        ),
        ring=coordinate(
            omni_parallel_state.get_ring_parallel_world_size,
            omni_parallel_state.get_ring_parallel_rank,
        ),
        cfg=coordinate(
            omni_parallel_state.get_classifier_free_guidance_world_size,
            omni_parallel_state.get_classifier_free_guidance_rank,
        ),
        dp=coordinate(
            omni_parallel_state.get_data_parallel_world_size,
            omni_parallel_state.get_data_parallel_rank,
        ),
        pp=coordinate(
            omni_parallel_state.get_pipeline_parallel_world_size,
            omni_parallel_state.get_pipeline_parallel_rank,
        ),
        ep=_optional_group_coordinate(vllm_parallel_state.get_ep_group),
        fs=_optional_group_coordinate(omni_parallel_state.get_fs_group),
    )


def assemble_topology(gathered: Sequence[Point0Payload], ctx: StartupContext) -> Topology:
    """Assemble the replica-wide topology from the collected point-0 payloads.

    Every rank adopts rank 0's ``(deployment id, deployment scoped)`` pair.  A
    rank that reports a deployment id scoped to its own stage while its id
    differs from the adopted one fails the whole replica.  Coordinates must be
    complete, in bounds and a uniform covering of the replica so that the same
    ``Topology`` is produced on every rank.
    """
    world_size = int(ctx.world_size)
    if len(gathered) != world_size:
        raise GroupConfirmationError(
            f"expected {world_size} point-0 payloads, collected {len(gathered)}",
            phase="point0_topology",
        )
    deployment_id, deployment_scoped = _adopt_deployment_id(gathered)
    coordinates = tuple(payload.coordinates for payload in gathered)
    _validate_coordinates(coordinates, world_size)
    return Topology(
        deployment_id=deployment_id,
        deployment_scoped=deployment_scoped,
        stage_id=_stage_id_from(ctx),
        world_size=world_size,
        records=tuple(payload.record for payload in gathered),
        coordinates=coordinates,
    )


def group_table_keys() -> frozenset[tuple[tuple[int, ...], str]]:
    """Return the (members, backend) keys currently present in the process-level AllGather group table."""
    return frozenset(_ALLGATHER_GROUPS)


def check_collective_parts(fingerprints: Sequence[RankFingerprints]) -> None:
    """Check the parts of the replica that take part in collectives (RFC 01 section 7.2).

    For every component that has AllGather Units, every rank that declares the
    component must also report a collective fingerprint for it, a rank must not
    report a collective fingerprint for a component it does not declare, and all
    ranks sharing one content identity must report the same fingerprint.
    """
    world_size = len(fingerprints)
    components = sorted({component for item in fingerprints for component in item.collective})
    for component in components:
        declaring = [rank for rank in range(world_size) if component in fingerprints[rank].content_identities]
        missing = [rank for rank in declaring if component not in fingerprints[rank].collective]
        if missing:
            raise GroupConfirmationError(
                f"component {component!r} has allgather Units but rank(s) {missing} declare it without a "
                f"collective fingerprint; declaring ranks: {declaring}",
                phase="point1_fingerprints",
            )
        undeclared = [
            rank
            for rank in range(world_size)
            if component in fingerprints[rank].collective and component not in fingerprints[rank].content_identities
        ]
        if undeclared:
            raise GroupConfirmationError(
                f"component {component!r} has a collective fingerprint on rank(s) {undeclared} that do not "
                f"declare it in their content identities",
                phase="point1_fingerprints",
            )
        identity_groups: dict[str, list[int]] = {}
        for rank in declaring:
            identity_groups.setdefault(fingerprints[rank].content_identities[component], []).append(rank)
        for ranks in identity_groups.values():
            signatures = {fingerprints[rank].collective[component] for rank in ranks}
            if len(signatures) > 1:
                details = ", ".join(f"rank {rank}: {fingerprints[rank].collective[component]!r}" for rank in ranks)
                raise GroupConfirmationError(
                    f"component {component!r} has different allgather fingerprints inside one "
                    f"content-identity group ({details})",
                    phase="point1_fingerprints",
                )


def check_unmatched_policy_keys(
    gathered: Sequence[Point1Payload],
    *,
    pipeline: str | None = None,
    roles: Mapping[str, str] | None = None,
) -> None:
    """Raise ``PlanningError`` (rule L01) when a policy key matches no rank's components.

    The policy key only fails when it is unmatched on every rank, so the check
    uses the intersection of the per-rank unmatched sets.  The reported
    component list is the union over the replica, which keeps the message
    identical on every rank (RFC 01 section 7.2).

    ``pipeline`` and ``roles`` are optional context for the L01 message; the
    confirmation payload does not carry them, so the message falls back to the
    replica-wide component names when they are not supplied.
    """
    if not gathered:
        return
    common = set(gathered[0].unmatched_policy_keys)
    for payload in gathered[1:]:
        common &= set(payload.unmatched_policy_keys)
    if not common:
        return
    keys = ", ".join(repr(key) for key in sorted(common))
    declared = sorted({component for payload in gathered for component in payload.fingerprints.content_identities})
    if roles is not None:
        declared_text = ", ".join(f"{name} ({roles[name]})" if name in roles else name for name in declared)
    else:
        declared_text = ", ".join(declared) if declared else "<none>"
    if pipeline is None:
        what = f"offload policy key(s) {keys} match no component on any rank; declared on this replica: {declared_text}"
    else:
        what = (
            f"offload policy key(s) {keys} match no component of {pipeline} on any rank; "
            f"declared on this replica: {declared_text}"
        )
    raise PlanningError(
        what,
        fix="remove the key(s), or use one of the declared component names or roles",
    )


def create_allgather_groups(
    fingerprints: Sequence[RankFingerprints],
    gathered: Sequence[Point1Payload],
    *,
    backend: str | None = None,
) -> AllGatherGroups:
    """Create or reuse the AllGather groups determined by the collected fingerprints.

    Groups are formed per component from the content identity of every rank that
    declares the component, ordered by (component name, smallest rank).  A group
    whose (members, backend) key is present in *every* rank's ``group_keys`` is
    reused; otherwise every rank of the replica, including ranks outside the
    group and ranks that do not declare the component, calls
    ``torch.distributed.new_group`` in the same order.  A replaced group of the
    same membership stays in the process-level table's retired list and is
    destroyed by :func:`destroy_allgather_groups`.

    A single-rank group is a confirmation error; it is detected before any
    ``new_group`` call.  A ``new_group`` failure propagates as-is and leaves the
    groups created so far in the process-level table, so the worker shutdown
    path still destroys them.
    """
    if len(fingerprints) != len(gathered):
        raise GroupConfirmationError(
            f"expected one point-1 payload per rank, got {len(gathered)} payloads for {len(fingerprints)} fingerprints",
            phase="point1_fingerprints",
        )
    planned = _plan_allgather_groups(fingerprints)
    if not planned:
        return {}
    if not (dist.is_available() and dist.is_initialized()):
        raise GroupConfirmationError(
            "AllGather groups require an initialized torch.distributed process group",
            phase="point1_fingerprints",
        )
    resolved_backend = _dist_backend() if backend is None else backend
    rank = dist.get_rank()
    per_rank_keys = [set(payload.group_keys) for payload in gathered]
    groups: dict[str, list[AllGatherGroup]] = {}
    for component, ranks in planned:
        key = (ranks, resolved_backend)
        reusable = all(key in keys for keys in per_rank_keys)
        record = _ALLGATHER_GROUPS.get(key) if reusable else None
        if reusable and record is None:
            raise GroupConfirmationError(
                f"every rank reports AllGather group {list(ranks)} with backend {resolved_backend!r}, "
                f"but this process has no such group in its process-level table",
                phase="point1_fingerprints",
            )
        if record is None:
            record = _create_group(ranks, resolved_backend, rank)
            _ALLGATHER_GROUPS[key] = record
        groups.setdefault(component, []).append(
            AllGatherGroup(
                component=component,
                ranks=ranks,
                group=record.process_group if rank in ranks else None,
            )
        )
    return {component: tuple(component_groups) for component, component_groups in groups.items()}


def destroy_allgather_groups() -> None:
    """Destroy every AllGather process group created by this process.

    Idempotent, and safe to call when no group was created or when the
    distributed environment has already been torn down.  Only groups recorded
    in this module's own table are destroyed; groups owned by other components
    are never touched.
    """
    records = list(_ALLGATHER_GROUPS.values()) + _RETIRED_GROUPS
    _ALLGATHER_GROUPS.clear()
    _RETIRED_GROUPS.clear()
    if not records:
        return
    if not (dist.is_available() and dist.is_initialized()):
        logger.warning(
            "Dropping %d AllGather group record(s) without destroying them: torch.distributed is not initialized",
            len(records),
        )
        return
    destroyed = 0
    for record in records:
        if record.process_group is None:
            continue
        try:
            dist.destroy_process_group(record.process_group)
            destroyed += 1
        except Exception:
            logger.exception("Failed to destroy AllGather process group %s", list(record.ranks))
    logger.debug("Destroyed %d AllGather process group(s)", destroyed)


@dataclass
class _GroupRecord:
    """One entry of the process-level AllGather group table."""

    ranks: tuple[int, ...]
    backend: str
    process_group: ProcessGroup | None  # None on ranks that are not members of the group


_ALLGATHER_GROUPS: dict[tuple[tuple[int, ...], str], _GroupRecord] = {}
_RETIRED_GROUPS: list[_GroupRecord] = []


def _timeout_seconds_from_config(config: object) -> float:
    timeout = getattr(config, "dist_timeout", None)
    if timeout is None:
        return DEFAULT_CONFIRMATION_TIMEOUT_SECONDS
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise ConfigError(
            f"dist_timeout must be a number of seconds, got {type(timeout).__name__}",
            where="od_config.dist_timeout",
            fix="set dist_timeout to a positive number of seconds or leave it unset",
        )
    timeout_seconds = float(timeout)
    if timeout_seconds <= 0:
        raise ConfigError(
            f"dist_timeout must be positive, got {timeout_seconds}",
            where="od_config.dist_timeout",
            fix="set dist_timeout to a positive number of seconds or leave it unset",
        )
    return timeout_seconds


def _confirmation_timeout_error(
    point: str,
    channel: GroupConfirmation,
    missing_ranks: tuple[int, ...] | None,
) -> ConfirmationTimeoutError:
    if missing_ranks:
        what = (
            f"rank(s) {list(missing_ranks)} did not reach confirmation point {point!r} "
            f"within {channel.timeout_seconds:g}s"
        )
    else:
        what = f"confirmation point {point!r} timed out after {channel.timeout_seconds:g}s"
    error = ConfirmationTimeoutError(
        what,
        where="torch.distributed.monitored_barrier on the replica world gloo group",
        fix="check the logs of the ranks that did not arrive",
        phase=point,
    )
    error.rank = channel.rank
    error.world_size = channel.world_size
    return error


def _monitored_barrier_with_watchdog(
    cpu_group: ProcessGroup,
    timeout: timedelta,
) -> BaseException | _BarrierWatchdogTimeout | None:
    """Run the monitored barrier with a local watchdog on every rank.

    ``torch.distributed.monitored_barrier`` bounds rank 0, the monitor; a rank
    that waits for a hung rank 0 has no local timeout of its own.  The watchdog
    therefore gives every rank a bounded return: it returns
    :data:`_BARRIER_WATCHDOG_TIMEOUT` when the barrier did not finish within
    ``timeout + grace``, the barrier exception when one was raised, and ``None``
    on success.  A barrier that outlives the watchdog is left in its daemon
    thread; the startup has already failed at that point and the worker process
    terminates the replica.

    ``wait_all_ranks=True`` makes rank 0 wait one full timeout per missing rank,
    so when two or more ranks never arrive rank 0 can need up to ``k * timeout``
    to raise.  The watchdog deliberately bounds rank 0 at ``timeout + grace``:
    startup failure stays bounded (the RFC's primary requirement) at the cost of
    a generic timeout message on rank 0 instead of the full missing-rank list in
    that multi-missing-rank case.  The common single-hung-rank case still lists
    the missing rank.
    """
    outcome: dict[str, BaseException] = {}

    def target() -> None:
        try:
            dist.monitored_barrier(group=cpu_group, timeout=timeout, wait_all_ranks=True)
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            outcome["error"] = exc

    thread = threading.Thread(
        target=target,
        name="weight-stack-confirmation-barrier",
        daemon=True,
    )
    thread.start()
    thread.join(timeout.total_seconds() + _CONFIRMATION_WATCHDOG_GRACE_SECONDS)
    if thread.is_alive():
        return _BARRIER_WATCHDOG_TIMEOUT
    return outcome.get("error")


def _missing_ranks_from_barrier_error(exc: BaseException) -> tuple[int, ...] | None:
    """Extract the ranks that failed to reach the barrier from rank 0's timeout error.

    Gloo's rank 0 reports ``[Rank 0]: Ranks 1, 2 failed to pass monitoredBarrier
    in <timeout> ms``.  The message is parsed without a regular expression so the
    module keeps the RFC's standard-library-only import surface.
    """
    text = str(exc)
    marker_index = text.lower().find(_MONITORED_BARRIER_FAILURE_MARKER)
    if marker_index < 0:
        return None
    prefix = text[:marker_index].rstrip()
    digits: list[str] = []
    while prefix and (prefix[-1].isdigit() or prefix[-1] in ", "):
        digits.append(prefix[-1])
        prefix = prefix[:-1]
    if not digits:
        return None
    tokens = "".join(reversed(digits)).replace(",", " ").split()
    ranks = sorted({int(token) for token in tokens})
    return tuple(ranks) if ranks else None


def _optional_group_coordinate(get_group: Callable[[], object]) -> ParallelCoordinate | None:
    try:
        group = get_group()
    except AssertionError:
        return None
    return ParallelCoordinate(size=int(group.world_size), rank=int(group.rank_in_group))  # type: ignore[attr-defined]


def _adopt_deployment_id(gathered: Sequence[Point0Payload]) -> tuple[str, bool]:
    deployment = gathered[0].deployment
    if not isinstance(deployment, tuple) or len(deployment) != 2:
        raise GroupConfirmationError(
            f"rank 0 reported an invalid deployment identity: {deployment!r}",
            phase="point0_topology",
        )
    deployment_id, deployment_scoped = deployment
    if not isinstance(deployment_id, str) or not deployment_id:
        raise GroupConfirmationError(
            f"rank 0 reported an invalid deployment id: {deployment_id!r}",
            phase="point0_topology",
        )
    if not isinstance(deployment_scoped, bool):
        raise GroupConfirmationError(
            f"rank 0 reported an invalid deployment scoped flag: {deployment_scoped!r}",
            phase="point0_topology",
        )
    for rank, payload in enumerate(gathered[1:], start=1):
        rank_deployment = payload.deployment
        if not isinstance(rank_deployment, tuple) or len(rank_deployment) != 2:
            raise GroupConfirmationError(
                f"rank {rank} reported an invalid deployment identity: {rank_deployment!r}",
                phase="point0_topology",
            )
        rank_id, rank_scoped = rank_deployment
        if not isinstance(rank_id, str) or not isinstance(rank_scoped, bool):
            raise GroupConfirmationError(
                f"rank {rank} reported an invalid deployment identity: ({rank_id!r}, {rank_scoped!r})",
                phase="point0_topology",
            )
        if rank_scoped and rank_id != deployment_id:
            raise GroupConfirmationError(
                f"rank {rank} reported deployment id {rank_id!r} scoped to its own stage, but rank 0 "
                f"adopted {deployment_id!r}; all ranks must use the deployment id issued by the orchestrator",
                phase="point0_topology",
            )
    return deployment_id, deployment_scoped


def _validate_coordinates(coordinates: Sequence[RankCoordinates], world_size: int) -> None:
    for rank, rank_coordinates in enumerate(coordinates):
        for field_name in _REQUIRED_COORDINATES:
            _validate_coordinate(getattr(rank_coordinates, field_name), field_name, rank, world_size)
        for field_name in _OPTIONAL_COORDINATES:
            coordinate = getattr(rank_coordinates, field_name)
            if coordinate is not None:
                _validate_coordinate(coordinate, field_name, rank, world_size)

    for field_name in _REQUIRED_COORDINATES + _OPTIONAL_COORDINATES:
        values = [getattr(rank_coordinates, field_name) for rank_coordinates in coordinates]
        present = [value for value in values if value is not None]
        if not present:
            continue
        if len(present) != len(values):
            missing = [rank for rank, value in enumerate(values) if value is None]
            raise GroupConfirmationError(
                f"rank(s) {missing} do not report the {field_name!r} coordinate while other ranks do",
                phase="point0_topology",
            )
        sizes = {value.size for value in present}
        if len(sizes) != 1:
            raise GroupConfirmationError(
                f"rank coordinates disagree on the size of {field_name!r}: {sorted(sizes)}",
                phase="point0_topology",
            )
        size = sizes.pop()
        if world_size % size != 0:
            raise GroupConfirmationError(
                f"{field_name!r} size {size} does not divide the replica world size {world_size}",
                phase="point0_topology",
            )
        expected_count = world_size // size
        counts = Counter(value.rank for value in present)
        if set(counts) != set(range(size)) or any(count != expected_count for count in counts.values()):
            raise GroupConfirmationError(
                f"rank coordinates of {field_name!r} are not a uniform covering of the replica: expected "
                f"each of ranks 0..{size - 1} exactly {expected_count} time(s), got {dict(sorted(counts.items()))}",
                phase="point0_topology",
            )

    # Without HSDP the orthogonal dimensions are a decomposition of the replica,
    # so their sizes multiply to the world size.  ``ulysses``/``ring`` are
    # already contained in ``sp`` and must not be multiplied again.  With HSDP
    # the FS dimension overlaps the orthogonal ones (pure HSDP uses singleton
    # tp/sp/pp/cfg/dp groups over the whole world), so the product check does not
    # apply; the per-dimension uniform-covering checks above still do.
    if coordinates[0].fs is None:
        product = 1
        for field_name in _ORTHOGONAL_COORDINATES:
            product *= getattr(coordinates[0], field_name).size
        if product != world_size:
            raise GroupConfirmationError(
                f"the product of the parallel sizes (tp*sp*pp*cfg*dp = {product}) does not equal the replica "
                f"world size {world_size}",
                phase="point0_topology",
            )


def _validate_coordinate(
    coordinate: ParallelCoordinate,
    field_name: str,
    rank: int,
    world_size: int,
) -> None:
    if not isinstance(coordinate, ParallelCoordinate):
        raise GroupConfirmationError(
            f"rank {rank} reported an invalid {field_name!r} coordinate: {coordinate!r}",
            phase="point0_topology",
        )
    if coordinate.size < 1:
        raise GroupConfirmationError(
            f"rank {rank} reported a non-positive {field_name!r} size: {coordinate.size}",
            phase="point0_topology",
        )
    if not 0 <= coordinate.rank < coordinate.size:
        raise GroupConfirmationError(
            f"rank {rank} reported {field_name!r} rank {coordinate.rank} outside [0, {coordinate.size})",
            phase="point0_topology",
        )
    if coordinate.size > world_size:
        raise GroupConfirmationError(
            f"rank {rank} reported {field_name!r} size {coordinate.size} larger than the replica world size "
            f"{world_size}",
            phase="point0_topology",
        )


def _stage_id_from(ctx: StartupContext) -> int:
    od_config = getattr(ctx, "od_config", None)
    stage_id = getattr(od_config, "stage_id", 0)
    if stage_id is None:
        return 0
    if isinstance(stage_id, bool) or not isinstance(stage_id, int):
        raise GroupConfirmationError(
            f"od_config.stage_id must be an int, got {type(stage_id).__name__}",
            phase="point0_topology",
        )
    return stage_id


def _plan_allgather_groups(fingerprints: Sequence[RankFingerprints]) -> list[tuple[str, tuple[int, ...]]]:
    world_size = len(fingerprints)
    components = sorted({component for item in fingerprints for component in item.collective})
    planned: list[tuple[str, tuple[int, ...]]] = []
    for component in components:
        declaring = [rank for rank in range(world_size) if component in fingerprints[rank].content_identities]
        identity_groups: dict[str, list[int]] = {}
        for rank in declaring:
            identity_groups.setdefault(fingerprints[rank].content_identities[component], []).append(rank)
        for ranks in identity_groups.values():
            members = tuple(sorted(ranks))
            if len(members) < 2:
                raise GroupConfirmationError(
                    f"component {component!r} uses allgather but its AllGather group has a single rank "
                    f"({members[0]}); an allgather Unit needs at least 2 ranks",
                    phase="point1_fingerprints",
                )
            planned.append((component, members))
    planned.sort(key=lambda item: (item[0], item[1][0]))
    return planned


def _create_group(ranks: tuple[int, ...], backend: str, rank: int) -> _GroupRecord:
    key = (ranks, backend)
    process_group = torch.distributed.new_group(list(ranks), backend=backend)
    old_record = _ALLGATHER_GROUPS.get(key)
    if old_record is not None:
        # The replaced group may still be referenced by the previous startup's
        # plan/layout, so keep it alive until destroy_allgather_groups().
        _RETIRED_GROUPS.append(old_record)
    return _GroupRecord(
        ranks=ranks,
        backend=backend,
        process_group=process_group if rank in ranks else None,
    )


def _dist_backend() -> str:
    """Backend of the AllGather process groups (``current_omni_platform.dist_backend``).

    The platform object is read lazily from the distributed module so a test
    platform installed before startup (for example the CPU Gloo test shim) is
    honored instead of a platform bound at import time.
    """
    from vllm_omni.diffusion.distributed import parallel_state

    backend = getattr(parallel_state.current_omni_platform, "dist_backend", None)
    if not isinstance(backend, str) or not backend:
        raise GroupConfirmationError(
            f"current_omni_platform.dist_backend is not a backend name: {backend!r}",
            phase="point1_fingerprints",
        )
    return backend
