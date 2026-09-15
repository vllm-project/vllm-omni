# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Topology-wide session lifecycle ordering for a disaggregated AR pipeline.

A split session keeps state on several stages (encode owns VAE history, denoise
owns paged KV), and each stage used to release its own independently, so an
eviction in denoise left encode conditioning on history whose KV was gone.

This coordinator owns the ordering: one end-to-end request in flight, one
generation per begin/reset registered on every participant, reset/close fanned
out once at the topology boundary, and worker-initiated releases drained back
and replayed onto the peers. It takes an async ``rpc`` callable rather than
importing the orchestrator, so it runs directly on CPU.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.release_events import ARDiffusionReleaseEvent

logger = init_logger(__name__)

# Roles holding per-session state; a trailing postprocess stage owns nothing to
# retire but still takes part in lifecycle RPC.
_STATE_OWNING_ROLES = frozenset({"full", "encode", "denoise", "denoise_decode"})

DEFAULT_MAX_LIVE_SESSIONS = 64

# Async RPC into one stage: (method, stage_id, args) -> per-replica results.
StageRPC = Callable[[str, int, tuple[Any, ...]], Awaitable[Any]]


class SessionLifecycleError(RuntimeError):
    """A coordinated lifecycle operation could not be completed safely."""


class SessionNotLiveError(SessionLifecycleError):
    """A continuation refers to a session this topology no longer tracks.

    VAE history cannot be rebuilt from a later chunk, so recovery is an explicit
    new rollout. Raised before any stage mutates counters, history or KV.
    """


@dataclass(frozen=True)
class SessionControls:
    """The lifecycle intent one request carries, normalized once."""

    session_id: str
    reset: bool = False
    close_session: bool = False


def read_session_controls(sampling_params_list: Sequence[Any]) -> SessionControls | None:
    """Read session controls from a request's per-stage sampling params.

    ``None`` means the request carries none, which is how a non-session workload
    on the same engine opts out. A typed tick wins over the flat controls,
    matching the runner: a typed ``False`` is not overridden by a stale ``True``.
    """
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

    for params in sampling_params_list or ():
        extra_args = getattr(params, "extra_args", None)
        if not isinstance(extra_args, Mapping):
            continue
        tick = ARDiffusionTickRequest.from_extra_args(extra_args)
        if tick is not None:
            return SessionControls(
                session_id=str(tick.session_id),
                reset=bool(tick.reset),
                close_session=bool(tick.close_session),
            )
        session_id = extra_args.get("session_id")
        if session_id is None:
            continue
        return SessionControls(
            session_id=str(session_id),
            reset=bool(extra_args.get("reset", False)),
            close_session=bool(extra_args.get("close_session", False)),
        )
    return None


@dataclass(frozen=True)
class DiffusionStageLifecycleTopology:
    """Which stages take part, and which of them own session state."""

    stage_ids: tuple[int, ...]
    state_owning_stage_ids: tuple[int, ...]

    @classmethod
    def from_stage_configs(cls, stage_configs: Sequence[Any]) -> DiffusionStageLifecycleTopology | None:
        """Build the topology from declared stage configs, or None if not opted in."""
        from vllm_omni.config.stage_config import resolve_diffusion_stage_role

        stage_ids: list[int] = []
        state_owning: list[int] = []
        for index, stage_config in enumerate(stage_configs or ()):
            if not bool(getattr(stage_config, "coordinated_session_lifecycle", False)):
                continue
            stage_id = int(getattr(stage_config, "stage_id", index))
            stage_ids.append(stage_id)
            role = resolve_diffusion_stage_role(
                getattr(stage_config, "stage_role", None),
                getattr(stage_config, "model_stage", None),
            )
            if str(role.value) in _STATE_OWNING_ROLES:
                state_owning.append(stage_id)
        if not stage_ids:
            return None
        return cls(stage_ids=tuple(stage_ids), state_owning_stage_ids=tuple(state_owning))


@dataclass
class _PendingRetirement:
    """One retirement whose remote outcome is not yet confirmed."""

    session_id: str
    generation: int
    reason: str
    # Whether worker cleanup was dispatched: before that, aborting the wait
    # changed nothing remotely; after it, the outcome is uncertain.
    dispatched: bool = False
    confirmed: bool = False
    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def describe(self) -> str:
        return f"{self.session_id!r} generation {self.generation} ({self.reason})"


@dataclass
class _InFlight:
    request_id: str
    session_id: str
    generation: int
    close_session: bool
    # Retired by this request's own admission, so their release events are
    # expected and must not be fanned out again.
    coordinated_sessions: set[str] = field(default_factory=set)


class DiffusionStageLifecycleCoordinator:
    """Serialize and order session lifecycle across a declared topology."""

    def __init__(
        self,
        topology: DiffusionStageLifecycleTopology,
        rpc: StageRPC,
        *,
        max_live_sessions: int = DEFAULT_MAX_LIVE_SESSIONS,
        replica_count: Callable[[int], int] | None = None,
    ) -> None:
        if max_live_sessions <= 0:
            raise ValueError(f"max_live_sessions must be positive, got {max_live_sessions}")
        self.topology = topology
        self._rpc = rpc
        self._replica_count = replica_count
        self._max_live_sessions = int(max_live_sessions)
        # Bounded registry; the generation counter is global and monotonic, so a
        # reused id never inherits an earlier generation and needs no tombstone.
        self._live: OrderedDict[str, int] = OrderedDict()
        self._next_generation = 0
        self._gate = asyncio.Lock()
        self._inflight: _InFlight | None = None
        self._blocked_reason: str | None = None
        # Ids whose close is queued or settling, counted so one operation
        # finishing cannot clear the fence another still needs.
        self._close_pending: Counter[str] = Counter()
        # Retirements whose remote outcome is unknown, keyed by operation id.
        self._unresolved: dict[str, _PendingRetirement] = {}

    @property
    def live_sessions(self) -> dict[str, int]:
        return dict(self._live)

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    def is_active(self, session_id: str) -> bool:
        return str(session_id) in self._live

    def generation_of(self, session_id: str) -> int | None:
        return self._live.get(str(session_id))

    @property
    def unresolved_retirements(self) -> tuple[str, ...]:
        """Human-readable descriptions of retirements awaiting reconciliation."""
        return tuple(record.describe() for record in self._unresolved.values())

    def clear_block(self, *, force: bool = False) -> None:
        """Allow admission again once recovery is confirmed.

        Refused while a retirement is still unresolved: an elapsed timer or an
        empty registry is not evidence that a worker finished its cleanup. Use
        ``force`` only after the workers were restarted or reconciled by hand.
        """
        if self._unresolved and not force:
            raise SessionLifecycleError(
                "Cannot clear the coordinated lifecycle block while "
                f"{len(self._unresolved)} retirement(s) are unconfirmed: " + "; ".join(self.unresolved_retirements)
            )
        self._unresolved.clear()
        self._blocked_reason = None

    async def _call(self, method: str, stage_ids: Iterable[int], *args: Any) -> dict[int, Any]:
        """Invoke one method per stage, converting transport errors into results.

        A raised transport error must not abandon the remaining participants: the
        caller still has cleanup to attempt on them.
        """
        results: dict[int, Any] = {}
        for stage_id in stage_ids:
            try:
                results[stage_id] = await self._rpc(method, stage_id, tuple(args))
            except Exception as exc:  # noqa: BLE001 - reported per stage, not raised
                results[stage_id] = {"supported": False, "error": f"{type(exc).__name__}: {exc}"}
        return results

    @staticmethod
    def _collect_support(value: Any, *, errors: list[str], supported: list[bool]) -> None:
        """Flatten per-replica/per-rank results into support and error lists.

        Only ``True`` counts as an acknowledgement. ``False``, ``None``, an empty
        reply and an unrecognized shape are all failures: a lifecycle operation
        that cannot be confirmed has not happened.
        """
        if isinstance(value, bool):
            supported.append(value)
            if not value:
                errors.append("worker declined the operation (returned False)")
            return
        if isinstance(value, (list, tuple)):
            if not value:
                errors.append("empty reply, so no worker acknowledged")
                return
            for item in value:
                DiffusionStageLifecycleCoordinator._collect_support(item, errors=errors, supported=supported)
            return
        if isinstance(value, Mapping):
            if value.get("supported") is False:
                errors.append(str(value.get("error") or "stage does not support this lifecycle RPC"))
                return
            if "result" in value:
                DiffusionStageLifecycleCoordinator._collect_support(value["result"], errors=errors, supported=supported)
                return
            errors.append(f"unrecognized reply {sorted(value)!r}")
            return
        if value is None:
            errors.append("worker returned nothing, so it did not acknowledge")
            return
        errors.append(f"unrecognized reply of type {type(value).__name__}")

    def _validate_acknowledgements(self, method: str, results: Mapping[int, Any]) -> list[str]:
        """Return one failure string per stage that did not fully acknowledge."""
        failures: list[str] = []
        for stage_id, value in results.items():
            errors: list[str] = []
            supported: list[bool] = []
            self._collect_support(value, errors=errors, supported=supported)
            if errors or not supported or not all(supported):
                detail = "; ".join(errors) or "no acknowledgement"
                failures.append(f"stage {stage_id}: {detail}")
        return failures

    async def _fan_out(self, method: str, session_id: str, stage_ids: Sequence[int]) -> None:
        """Run one lifecycle method on every listed stage, failing loudly."""
        results = await self._call(method, stage_ids, session_id)
        failures = self._validate_acknowledgements(method, results)
        if failures:
            raise SessionLifecycleError(
                f"{method} for session {session_id!r} did not complete on every participant: " + "; ".join(failures)
            )

    def _require_single_replica_layout(self) -> None:
        """Refuse a multi-replica layout until session-affine routing exists."""
        if self._replica_count is None:
            return
        for stage_id in self.topology.stage_ids:
            count = int(self._replica_count(stage_id))
            if count > 1:
                raise SessionLifecycleError(
                    f"Coordinated session lifecycle supports one replica per stage until "
                    f"session-affine routing exists, but stage {stage_id} has {count}. "
                    "TP ranks inside one replica are supported; multiple replicas are not."
                )
            if count == 0:
                raise SessionLifecycleError(
                    f"Coordinated session lifecycle stage {stage_id} has no live replica; "
                    "the session's state cannot be reached."
                )

    def _retire_locally(self, session_id: str) -> None:
        self._live.pop(str(session_id), None)

    async def _evict_oldest_if_needed(self) -> None:
        """Keep the registry bounded, retiring the oldest through the fan-out.

        A victim's history is released irreversibly, so it is never resurrected in
        the registry. If its cleanup cannot be confirmed, block reuse instead of
        leaving a stage holding state nothing tracks.
        """
        while len(self._live) > self._max_live_sessions:
            victim, generation = next(iter(self._live.items()))
            logger.warning(
                "Coordinated session registry is full (%d); retiring least-recently-used session %s",
                self._max_live_sessions,
                victim,
            )
            del generation
            await self._retire_session(victim, reason="registry_eviction")

    async def admit(self, request_id: str, controls: SessionControls) -> int:
        """Order one request against the topology and return its generation.

        Blocks until the previous end-to-end request finished and acknowledged
        its cleanup. Raises before any stage is mutated when the request cannot
        be served; the caller must then fail it instead of submitting it.
        """
        await self._gate.acquire()
        session_id = str(controls.session_id)
        # Nothing is published until every registration acknowledges, so a failed
        # admission cannot leave a half-registered session continuable.
        candidate: int | None = None
        committed = False
        try:
            if self._blocked_reason is not None:
                raise SessionLifecycleError(
                    f"Coordinated session lifecycle is blocked pending recovery: {self._blocked_reason}"
                )
            self._require_single_replica_layout()
            if session_id in self._close_pending:
                raise SessionLifecycleError(
                    f"Session {session_id!r} has a close in progress; a new rollout under the same "
                    "id cannot start until it settles, or the close would retire the new one."
                )
            coordinated: set[str] = set()

            if controls.reset:
                # Phase one: retire the old generation. The gate already drained
                # older work, and this fans out once so no downstream stage
                # repeats it. A failure or cancellation here fences the id through
                # the shared retirement path -- there is no candidate yet to roll
                # back, and the old session is never restored, because a peer that
                # did succeed has already discarded its history.
                if session_id in self._live:
                    await self._retire_session(session_id, reason="coordinated_reset")
                    coordinated.add(session_id)
                # Phase two: allocate a candidate. Monotonic, so recovery may skip
                # numbers but never reuses one.
                self._next_generation += 1
                candidate = self._next_generation
            else:
                candidate = self._live.get(session_id, 0)
                if not candidate:
                    raise SessionNotLiveError(
                        f"DreamZero session {session_id!r} has no live state on this topology; "
                        "its history was released (explicit close, eviction, or a failed request) "
                        "and a continuation cannot rebuild it. Start a new rollout with an "
                        "explicit reset."
                    )

            await self._register_generation(session_id, candidate)

            # Registration is confirmed on every participant: publish.
            self._live[session_id] = candidate
            self._live.move_to_end(session_id)
            await self._evict_oldest_if_needed()
            self._inflight = _InFlight(
                request_id=str(request_id),
                session_id=session_id,
                generation=candidate,
                close_session=bool(controls.close_session),
                coordinated_sessions=coordinated,
            )
            committed = True
            return candidate
        except BaseException:
            self._inflight = None
            if candidate and not committed:
                # A participant may already have taken the candidate; clean it
                # everywhere so it cannot be continued, and block reuse when the
                # rollback itself cannot be confirmed.
                await self._rollback_candidate(session_id, candidate)
            raise
        finally:
            if not committed:
                self._gate.release()

    async def _register_generation(self, session_id: str, generation: int) -> None:
        """Bind the generation on every participant before payload execution.

        Identity only, so the stateless postprocess stage allocates nothing. A
        participant that declines cannot fence a stale payload, so anything short
        of an acknowledgement from every one of them fails the admission.
        """
        method = "register_ar_diffusion_generation"
        results = await self._call(method, self.topology.stage_ids, session_id, generation)
        failures = self._validate_acknowledgements(method, results)
        if failures:
            raise SessionLifecycleError(
                f"Registering generation {generation} for session {session_id!r} was not "
                "acknowledged by every participant: " + "; ".join(failures)
            )

    async def _rollback_candidate(self, session_id: str, generation: int) -> None:
        """Undo a failed admission, or block reuse when it cannot be confirmed.

        The generation counter is never rewound: a consumed id stays consumed, so
        a retried begin cannot collide with state a participant already took.
        """
        try:
            await self._retire_session(session_id, reason="admission_rollback")
        except (SessionLifecycleError, asyncio.CancelledError):
            # Already fenced and recorded by the retirement path.
            return
        logger.info(
            "Rolled back failed admission of session %s generation %d on every participant",
            session_id,
            generation,
        )

    async def _retire_session(self, session_id: str, *, reason: str) -> None:
        """Clear a session on every participant, fencing the id unless confirmed.

        The single retirement path for explicit close, reset, request failure,
        rollback and registry eviction, so no caller can leave an untracked
        partial cleanup behind. A cancelled await does not prove the remote
        cleanup stopped, so cancellation is recorded as unresolved before it is
        allowed to propagate.
        """
        key = str(session_id)
        record = _PendingRetirement(
            session_id=key,
            # Read before the registry is mutated, so diagnostics keep the
            # generation whose state is in question.
            generation=int(self._live.get(key, 0)),
            reason=reason,
        )
        self._unresolved[record.operation_id] = record
        self._retire_locally(key)
        try:
            record.dispatched = True
            results = await self._call(
                "close_ar_diffusion_session",
                self.topology.state_owning_stage_ids,
                key,
            )
        except BaseException as exc:
            # Includes cancellation: a timed-out close may still be running on a
            # worker, so the id stays fenced until something confirms otherwise.
            self._block(
                f"retirement of session {key!r} generation {record.generation} ({reason}) was "
                f"interrupted by {type(exc).__name__} after cleanup was dispatched; its remote "
                "state is unconfirmed"
            )
            raise
        failures = self._validate_acknowledgements("close_ar_diffusion_session", results)
        if failures:
            self._block(
                f"retirement of session {key!r} generation {record.generation} ({reason}) was not "
                "confirmed: " + "; ".join(failures)
            )
            raise SessionLifecycleError(self._blocked_reason or "retirement failed")
        record.confirmed = True
        self._unresolved.pop(record.operation_id, None)
        logger.info("Coordinated session %s retired across the topology (%s)", key, reason)

    def _block(self, reason: str) -> None:
        """Fence the topology; the first reason is kept as the root cause."""
        if self._blocked_reason is None:
            self._blocked_reason = reason
        else:
            self._blocked_reason = f"{self._blocked_reason} | {reason}"
        logger.error("Coordinated session lifecycle blocked: %s", reason)

    def is_inflight(self, request_id: str) -> bool:
        """Whether this request currently holds the topology's admission slot."""
        return self._inflight is not None and self._inflight.request_id == str(request_id)

    async def settle(self, request_id: str, *, success: bool) -> None:
        """Finish one request's cross-stage lifecycle, keeping the gate held.

        Runs for a returned error and a raised exception alike; either path can
        leave release events that must drain before the state is reused. Raises
        when the topology could not be synchronized, so the caller can withhold a
        terminal success it has not earned. The gate stays held until
        ``release_admission``, so no other generation starts while the outcome of
        this one is still being decided.
        """
        inflight = self._inflight
        if inflight is None or inflight.request_id != str(request_id):
            # Already settled, or never admitted.
            return
        failures: list[str] = []
        try:
            await self._drain_release_events(inflight)
        except SessionLifecycleError as exc:
            failures.append(str(exc))

        if not success:
            # The generation is invalid everywhere; never retried as a
            # continuation.
            try:
                await self._retire_session(inflight.session_id, reason="request_failure")
            except SessionLifecycleError as exc:
                failures.append(str(exc))
        elif inflight.close_session and inflight.session_id in self._live:
            try:
                await self._retire_session(inflight.session_id, reason="explicit_close")
            except SessionLifecycleError as exc:
                failures.append(str(exc))

        if failures:
            # Block reuse rather than report a synchronization we did not achieve.
            self._blocked_reason = "; ".join(failures)
            raise SessionLifecycleError(self._blocked_reason)

    async def release_admission(self, request_id: str) -> None:
        """Let the next topology request in, once this one's outcome is published."""
        if self._inflight is None or self._inflight.request_id != str(request_id):
            return
        self._inflight = None
        self._gate.release()

    async def complete(self, request_id: str, *, success: bool) -> None:
        """Settle one request and immediately reopen admission.

        For callers with no terminal output to publish between the two steps.
        """
        try:
            await self.settle(request_id, success=success)
        finally:
            await self.release_admission(request_id)

    def request_close(self, session_id: str) -> None:
        """Fence a session against a new begin while its close is settling.

        Recorded synchronously, before the close waits on the gate: otherwise a
        begin for the same id could take the gate first and the close behind it
        would retire that newer rollout instead. Counted, so two concurrent
        closes for one id each hold their own share of the fence.
        """
        self._close_pending[str(session_id)] += 1

    def _release_close(self, session_id: str) -> None:
        """Return this operation's share of the close fence."""
        key = str(session_id)
        remaining = self._close_pending.get(key, 0) - 1
        if remaining > 0:
            self._close_pending[key] = remaining
        else:
            self._close_pending.pop(key, None)

    async def close(self, session_id: str) -> None:
        """Explicit close with no inference: clear participants and mark inactive.

        Absent state is an idempotent success only while nothing about this
        topology is unresolved; a prior cleanup failure must not be reported as a
        clean close just because the registry entry is already gone.
        """
        key = str(session_id)
        self.request_close(key)
        gate_held = False
        try:
            # Cancellation while queued behind the gate has dispatched nothing, so
            # it leaves no unresolved record; only the fence share is returned.
            await self._gate.acquire()
            gate_held = True
            if self._blocked_reason is not None:
                raise SessionLifecycleError(
                    f"Cannot confirm close of session {key!r} while the topology is blocked "
                    f"pending recovery: {self._blocked_reason}"
                )
            # The retirement path records an unconfirmed or cancelled outcome and
            # fences the id, so there is nothing extra to do on failure here.
            await self._retire_session(key, reason="explicit_close")
        finally:
            self._release_close(key)
            if gate_held:
                self._gate.release()

    async def invalidate_all(self, *, reason: str) -> None:
        """Drop every live session and clean whatever peer state is reachable.

        For a dead participant: its KV did not survive, so nothing that spanned
        it may continue. Deliberately skips the admission gate, since it runs
        while the request that lost its worker is still being torn down.
        """
        sessions = list(self._live)
        self._live.clear()
        if not sessions:
            return
        logger.error(
            "Invalidating %d coordinated session(s) after %s; a continuation now requires an explicit reset",
            len(sessions),
            reason,
        )
        failures: list[str] = []
        for session_id in sessions:
            try:
                await self._fan_out(
                    "close_ar_diffusion_session",
                    session_id,
                    self.topology.state_owning_stage_ids,
                )
            except SessionLifecycleError as exc:
                failures.append(str(exc))
        if failures:
            self._blocked_reason = f"{reason}: " + "; ".join(failures)

    async def _drain_release_events(self, inflight: _InFlight) -> None:
        """Replay worker-initiated releases onto the peers, then acknowledge them."""
        stage_ids = self.topology.state_owning_stage_ids
        if not stage_ids:
            return
        raw = await self._call("get_ar_diffusion_release_events", stage_ids)

        conflicts: list[str] = []
        # session -> generation seen, and stage -> event ids to acknowledge
        victims: dict[str, int] = {}
        acks: dict[int, list[str]] = {}
        for stage_id, value in raw.items():
            events, decode_errors = _decode_release_events(value)
            conflicts.extend(f"stage {stage_id}: {error}" for error in decode_errors)
            for event in events:
                acks.setdefault(stage_id, []).append(event.event_id)
                if event.cleanup_failed:
                    conflicts.append(
                        f"stage {stage_id} failed to clean up session {event.session_id!r} "
                        f"({event.reason}); peer state cannot be assumed released"
                    )
                registered = self._live.get(event.session_id)
                if event.generation and registered is not None and event.generation != registered:
                    conflicts.append(
                        f"stage {stage_id} released session {event.session_id!r} at generation "
                        f"{event.generation} but generation {registered} is registered"
                    )
                    continue
                if event.generation and registered is None:
                    # Already retired here: a cleanup this coordinator drove, or
                    # a late duplicate.
                    continue
                previous = victims.get(event.session_id)
                if previous is not None and event.generation and previous and previous != event.generation:
                    conflicts.append(
                        f"session {event.session_id!r} was released at conflicting generations "
                        f"{previous} and {event.generation}"
                    )
                victims[event.session_id] = event.generation or (previous or 0)

        retire_errors: list[str] = []
        for session_id in list(victims):
            if session_id in inflight.coordinated_sessions:
                continue
            if session_id == inflight.session_id:
                logger.warning(
                    "Coordinated session %s was released by a worker while its own request was in flight",
                    session_id,
                )
            try:
                await self._retire_session(session_id, reason="worker_release_event")
            except SessionLifecycleError as exc:
                retire_errors.append(str(exc))

        if conflicts or retire_errors:
            # Leave them unacknowledged so a retry still sees them.
            raise SessionLifecycleError("; ".join(conflicts + retire_errors))

        ack_failures: list[str] = []
        for stage_id, event_ids in acks.items():
            if not event_ids:
                continue
            results = await self._call("ack_ar_diffusion_release_events", [stage_id], event_ids)
            # An unconfirmed acknowledgement leaves the events pending on the
            # worker; assuming it consumed them would replay or lose them.
            ack_failures.extend(self._validate_ack_counts(results, expected=len(event_ids)))
        if ack_failures:
            raise SessionLifecycleError("; ".join(ack_failures))

    def _validate_ack_counts(self, results: Mapping[int, Any], *, expected: int) -> list[str]:
        """Check that every stage answered the acknowledgement.

        A count below ``expected`` is not a failure: acknowledgement has to be
        idempotent, because a retry after a partially failed ack re-sends ids some
        ranks have already dropped. What must not pass is an error, an unsupported
        participant or a reply nobody answered -- those leave the events pending on
        the worker, and assuming otherwise would lose or replay them.
        """
        failures: list[str] = []
        for stage_id, value in results.items():
            counts: list[int] = []
            errors: list[str] = []
            _collect_ack_counts(value, counts=counts, errors=errors)
            if errors:
                failures.append(f"stage {stage_id} release-event ack: {'; '.join(errors)}")
                continue
            if not counts:
                failures.append(f"stage {stage_id} release-event ack was not acknowledged")
                continue
            if any(count != expected for count in counts):
                # Ranks that disagree about what they held are worth seeing; a
                # genuine divergence is already caught when the events are decoded.
                logger.debug(
                    "Stage %d acknowledged %s of %d release event(s)",
                    stage_id,
                    counts,
                    expected,
                )
        return failures


def _collect_ack_counts(value: Any, *, counts: list[int], errors: list[str]) -> None:
    """Flatten an ``ack_ar_diffusion_release_events`` reply into removal counts."""
    if isinstance(value, bool):
        errors.append("worker returned a flag instead of a removal count")
        return
    if isinstance(value, int):
        counts.append(value)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            errors.append("empty reply, so no worker acknowledged")
            return
        for item in value:
            _collect_ack_counts(item, counts=counts, errors=errors)
        return
    if isinstance(value, Mapping):
        if value.get("supported") is False:
            errors.append(str(value.get("error") or "stage does not support release-event acknowledgement"))
            return
        if "result" in value:
            _collect_ack_counts(value["result"], counts=counts, errors=errors)
            return
        errors.append(f"unrecognized reply {sorted(value)!r}")
        return
    errors.append(f"unrecognized reply of type {type(value).__name__}")


def _decode_release_events(value: Any) -> tuple[list[ARDiffusionReleaseEvent], list[str]]:
    """Flatten one stage's RPC result into release events, deduped by event id."""
    events: dict[str, ARDiffusionReleaseEvent] = {}
    errors: list[str] = []

    def add(event: ARDiffusionReleaseEvent) -> None:
        # TP ranks report the same releases under the same ids: agreeing records
        # collapse, disagreeing ones are a synchronization bug.
        existing = events.get(event.event_id)
        if existing is not None and existing != event:
            errors.append(
                f"release event {event.event_id} disagrees across ranks: {existing.to_dict()} vs {event.to_dict()}"
            )
            return
        events[event.event_id] = event

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, ARDiffusionReleaseEvent):
            add(node)
            return
        if isinstance(node, Mapping):
            if node.get("supported") is False:
                errors.append(str(node.get("error") or "stage does not support release-event reporting"))
                return
            if "event_id" in node:
                try:
                    event = ARDiffusionReleaseEvent.from_dict(node)
                except (TypeError, ValueError) as exc:
                    errors.append(f"malformed release event: {exc}")
                    return
                add(event)
                return
            if "result" in node:
                walk(node["result"])
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        errors.append(f"unexpected release-event payload of type {type(node).__name__}")

    walk(value)
    return list(events.values()), errors
