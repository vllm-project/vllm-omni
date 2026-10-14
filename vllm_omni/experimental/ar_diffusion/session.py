# SPDX-License-Identifier: Apache-2.0
"""Internal realtime session control plane for AR-Diffusion.

This module owns transport-level identity, event ordering, backpressure, and
chunk-boundary snapshots. Model-defined control payloads remain opaque here.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)


class ARDiffusionSessionError(RuntimeError):
    """Base error for realtime AR-Diffusion session operations."""


class ARDiffusionSessionStateError(ARDiffusionSessionError):
    """Raised when an operation is invalid for the current session state."""


class ARDiffusionEventQueueFullError(ARDiffusionSessionError):
    """Raised when accepting an event would exceed the bounded queue."""


class ARDiffusionStaleEventError(ARDiffusionSessionError):
    """Raised when an event is older than the current monotonic boundary."""


class ARDiffusionChunkMetadataError(ARDiffusionSessionError):
    """Raised when a tick consumer returns metadata for a different chunk."""


class ARDiffusionEventAcceptanceStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


class ARDiffusionSessionStatus(str, Enum):
    ACTIVE = "active"
    RESETTING = "resetting"
    FAILED = "failed"
    CLOSING = "closing"
    CLEANUP_FAILED = "cleanup_failed"
    CLOSED = "closed"


def _copy_control(control: ARDiffusionControlInput) -> ARDiffusionControlInput:
    return ARDiffusionControlInput(
        track=control.track,
        schema=control.schema,
        data=control.data,
    )


@dataclass(frozen=True)
class ARDiffusionSessionEvent:
    """One atomically accepted prompt/control update.

    ``prompt=None`` means that this event does not update the prompt. Controls
    are keyed by ``track`` when the chunk snapshot is assembled; their schema
    and data are otherwise opaque to this layer.
    """

    event_id: int
    prompt: str | None = None
    controls: tuple[ARDiffusionControlInput, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.event_id, bool) or not isinstance(self.event_id, int) or self.event_id < 0:
            raise ValueError("event_id must be a non-negative integer.")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string or None.")
        controls = tuple(self.controls)
        if any(not isinstance(control, ARDiffusionControlInput) for control in controls):
            raise ValueError("controls must contain ARDiffusionControlInput values.")
        tracks = [control.track for control in controls]
        if len(tracks) != len(set(tracks)):
            raise ValueError("A composite event may update each control track at most once.")
        if self.prompt is None and not controls:
            raise ValueError("An event must update the prompt, one or more controls, or both.")
        object.__setattr__(self, "controls", controls)


@dataclass(frozen=True)
class ARDiffusionEventAcceptance:
    """Immediate queue acceptance result; this does not mean applied."""

    session_id: str
    event_id: int
    status: ARDiffusionEventAcceptanceStatus
    pending_event_count: int


TickOutputT = TypeVar("TickOutputT")


class ARDiffusionTickConsumer(Protocol[TickOutputT]):
    """Consumes one immutable chunk-boundary tick."""

    async def execute_tick(self, tick: ARDiffusionTickRequest) -> TickOutputT: ...

    def chunk_metadata(self, output: TickOutputT) -> ARDiffusionChunkMetadata: ...


@dataclass(frozen=True)
class ARDiffusionPreparedControls:
    """One reducer-owned, transactional control snapshot.

    ``private_state`` is opaque to the session layer and becomes durable only
    after the corresponding model tick succeeds.
    """

    controls: tuple[ARDiffusionControlInput, ...]
    private_state: Any = None


class ARDiffusionControlReducer(Protocol):
    """Optional model adapter for stateful controls sampled per AR block."""

    def prepare(
        self,
        *,
        current_controls: Mapping[str, ARDiffusionControlInput],
        events: Sequence[ARDiffusionSessionEvent],
        chunk_index: int,
    ) -> ARDiffusionPreparedControls: ...

    def commit(self, prepared: ARDiffusionPreparedControls) -> None: ...

    def reset(self) -> None: ...


class ARDiffusionSessionLifecycle(Protocol):
    """Worker-facing session lifecycle operations."""

    async def reset_session(self, session_id: str) -> None: ...

    async def close_session(self, session_id: str) -> None: ...


class ARDiffusionCollectiveRPCClient(Protocol):
    """Subset of the engine control RPC surface used by session lifecycle."""

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]: ...


def _collect_rpc_support_results(value: Any, *, errors: list[str], supported: list[bool]) -> None:
    if isinstance(value, bool):
        supported.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_rpc_support_results(item, errors=errors, supported=supported)
        return
    if isinstance(value, Mapping):
        if value.get("supported") is False:
            errors.append(str(value.get("error") or "target stage does not support this lifecycle RPC"))
            return
        if "result" in value:
            _collect_rpc_support_results(value["result"], errors=errors, supported=supported)


class ARDiffusionWorkerLifecycle:
    """Routes reset/close through the existing orchestrator collective RPC."""

    def __init__(
        self,
        rpc_client: ARDiffusionCollectiveRPCClient,
        *,
        stage_ids: Sequence[int],
        timeout: float | None = None,
    ) -> None:
        normalized_stage_ids = tuple(stage_ids)
        if not normalized_stage_ids:
            raise ValueError("stage_ids must select at least one AR-Diffusion stage.")
        if any(isinstance(stage_id, bool) or not isinstance(stage_id, int) or stage_id < 0 for stage_id in stage_ids):
            raise ValueError("stage_ids must contain non-negative integers.")
        self._rpc_client = rpc_client
        self._stage_ids = normalized_stage_ids
        self._timeout = timeout

    async def _invoke(self, method: str, session_id: str) -> None:
        results = await self._rpc_client.collective_rpc(
            method=method,
            timeout=self._timeout,
            args=(session_id,),
            stage_ids=list(self._stage_ids),
        )
        errors: list[str] = []
        supported: list[bool] = []
        _collect_rpc_support_results(results, errors=errors, supported=supported)
        if errors:
            raise ARDiffusionSessionError(f"{method} failed: {'; '.join(errors)}")
        if not supported or not all(supported):
            raise ARDiffusionSessionError(f"{method} is not supported by every selected worker.")

    async def reset_session(self, session_id: str) -> None:
        await self._invoke("reset_ar_diffusion_session", session_id)

    async def close_session(self, session_id: str) -> None:
        await self._invoke("close_ar_diffusion_session", session_id)


def _default_request_id(session_id: str, chunk_index: int) -> str:
    return f"ar-{session_id}-{chunk_index}-{uuid.uuid4().hex}"


class ARDiffusionSession(Generic[TickOutputT]):
    """Transactional event queue and chunk snapshot for one realtime world."""

    def __init__(
        self,
        session_id: str,
        *,
        tick_consumer: ARDiffusionTickConsumer[TickOutputT],
        lifecycle: ARDiffusionSessionLifecycle,
        max_pending_events: int,
        request_id_factory: Callable[[str, int], str] | None = None,
        control_reducer: ARDiffusionControlReducer | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if isinstance(max_pending_events, bool) or not isinstance(max_pending_events, int) or max_pending_events <= 0:
            raise ValueError("max_pending_events must be a positive integer.")
        self.session_id = session_id
        self._tick_consumer = tick_consumer
        self._lifecycle = lifecycle
        self._max_pending_events = max_pending_events
        self._request_id_factory = request_id_factory or _default_request_id
        self._control_reducer = control_reducer

        self._status = ARDiffusionSessionStatus.ACTIVE
        self._pending_events: dict[int, ARDiffusionSessionEvent] = {}
        self._recent_event_ids: OrderedDict[int, None] = OrderedDict()
        self._event_floor = -1
        self._inflight_event_floor: int | None = None
        self._chunk_index = 0
        self._prompt: str | None = None
        self._controls: dict[str, ARDiffusionControlInput] = {}

        self._state_lock = asyncio.Lock()
        self._tick_lock = asyncio.Lock()

    @property
    def status(self) -> ARDiffusionSessionStatus:
        return self._status

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    @property
    def pending_event_count(self) -> int:
        return len(self._pending_events)

    def _require_active(self) -> None:
        if self._status is not ARDiffusionSessionStatus.ACTIVE:
            raise ARDiffusionSessionStateError(f"AR-Diffusion session {self.session_id!r} is {self._status.value}.")

    def _require_resettable(self) -> None:
        if self._status not in {
            ARDiffusionSessionStatus.ACTIVE,
            ARDiffusionSessionStatus.FAILED,
        }:
            raise ARDiffusionSessionStateError(f"AR-Diffusion session {self.session_id!r} is {self._status.value}.")

    async def accept_event(self, event: ARDiffusionSessionEvent) -> ARDiffusionEventAcceptance:
        """Atomically enqueue an event or return an idempotent duplicate result."""
        if not isinstance(event, ARDiffusionSessionEvent):
            raise TypeError("event must be an ARDiffusionSessionEvent.")
        async with self._state_lock:
            self._require_active()
            if event.event_id in self._pending_events:
                return ARDiffusionEventAcceptance(
                    session_id=self.session_id,
                    event_id=event.event_id,
                    status=ARDiffusionEventAcceptanceStatus.DUPLICATE,
                    pending_event_count=len(self._pending_events),
                )
            if event.event_id in self._recent_event_ids:
                return ARDiffusionEventAcceptance(
                    session_id=self.session_id,
                    event_id=event.event_id,
                    status=ARDiffusionEventAcceptanceStatus.DUPLICATE,
                    pending_event_count=len(self._pending_events),
                )

            event_floor = self._event_floor
            if self._inflight_event_floor is not None:
                event_floor = max(event_floor, self._inflight_event_floor)
            if event.event_id <= event_floor:
                raise ARDiffusionStaleEventError(
                    f"event_id {event.event_id} is stale; the session event floor is {event_floor}."
                )
            if len(self._pending_events) >= self._max_pending_events:
                raise ARDiffusionEventQueueFullError(
                    f"AR-Diffusion session {self.session_id!r} event queue is full "
                    f"({self._max_pending_events} pending)."
                )

            stored_event = ARDiffusionSessionEvent(
                event_id=event.event_id,
                prompt=event.prompt,
                controls=tuple(_copy_control(control) for control in event.controls),
            )
            self._pending_events[event.event_id] = stored_event
            self._recent_event_ids[event.event_id] = None
            while len(self._recent_event_ids) > self._max_pending_events:
                self._recent_event_ids.popitem(last=False)
            return ARDiffusionEventAcceptance(
                session_id=self.session_id,
                event_id=event.event_id,
                status=ARDiffusionEventAcceptanceStatus.ACCEPTED,
                pending_event_count=len(self._pending_events),
            )

    def _build_tick_locked(
        self,
    ) -> tuple[
        ARDiffusionTickRequest,
        str | None,
        dict[str, ARDiffusionControlInput],
        ARDiffusionPreparedControls | None,
    ]:
        event_ids = tuple(sorted(self._pending_events))
        events = tuple(self._pending_events[event_id] for event_id in event_ids)
        prompt = self._prompt
        for event in events:
            if event.prompt is not None:
                prompt = event.prompt

        prepared_controls: ARDiffusionPreparedControls | None = None
        if self._control_reducer is None:
            controls = {track: _copy_control(control) for track, control in self._controls.items()}
            for event in events:
                for control in event.controls:
                    controls[control.track] = _copy_control(control)
        else:
            prepared_controls = self._control_reducer.prepare(
                current_controls={track: _copy_control(control) for track, control in self._controls.items()},
                events=events,
                chunk_index=self._chunk_index,
            )
            if not isinstance(prepared_controls, ARDiffusionPreparedControls):
                raise TypeError("control reducer prepare() must return ARDiffusionPreparedControls.")
            controls = {}
            for control in prepared_controls.controls:
                if not isinstance(control, ARDiffusionControlInput):
                    raise TypeError("prepared controls must contain ARDiffusionControlInput values.")
                if control.track in controls:
                    raise ValueError("prepared controls must contain at most one value per track.")
                controls[control.track] = _copy_control(control)

        chunk_index = self._chunk_index
        request_id = self._request_id_factory(self.session_id, chunk_index)
        tick = ARDiffusionTickRequest(
            session_id=self.session_id,
            request_id=request_id,
            chunk_index=chunk_index,
            applied_event_ids=event_ids,
            prompt=prompt,
            controls=tuple(_copy_control(controls[track]) for track in sorted(controls)),
        )
        self._inflight_event_floor = event_ids[-1] if event_ids else None
        return tick, prompt, controls, prepared_controls

    async def _commit_tick(
        self,
        tick: ARDiffusionTickRequest,
        prompt: str | None,
        controls: dict[str, ARDiffusionControlInput],
        prepared_controls: ARDiffusionPreparedControls | None,
    ) -> None:
        async with self._state_lock:
            # Reducer state and the session snapshot form one logical commit.
            # Run the only fallible operation before mutating session fields so
            # a reducer failure cannot acknowledge events or advance the chunk.
            if prepared_controls is not None:
                assert self._control_reducer is not None
                self._control_reducer.commit(prepared_controls)
            for event_id in tick.applied_event_ids:
                self._pending_events.pop(event_id, None)
            if tick.applied_event_ids:
                self._event_floor = tick.applied_event_ids[-1]
            self._prompt = prompt
            self._controls = controls
            self._chunk_index += 1
            self._inflight_event_floor = None

    async def _fail_tick(self, tick_error: BaseException) -> None:
        async with self._state_lock:
            self._inflight_event_floor = None
            self._status = ARDiffusionSessionStatus.FAILED
        try:
            # A model forward may already have released its state. This
            # idempotent close also covers successful forwards whose metadata
            # or local reducer commit could not be accepted.
            await self._lifecycle.close_session(self.session_id)
        except Exception as cleanup_error:
            raise ARDiffusionSessionError(
                "AR-Diffusion tick failed "
                f"({type(tick_error).__name__}: {tick_error}) and worker "
                f"session cleanup also failed: {cleanup_error}"
            ) from tick_error

    async def next_chunk(self) -> TickOutputT:
        """Execute and atomically commit one chunk-boundary snapshot."""
        async with self._tick_lock:
            async with self._state_lock:
                self._require_active()
                tick, prompt, controls, prepared_controls = self._build_tick_locked()

            expected_metadata = ARDiffusionChunkMetadata.from_tick(tick)
            try:
                output = await self._tick_consumer.execute_tick(tick)
                metadata = self._tick_consumer.chunk_metadata(output)
                if not isinstance(metadata, ARDiffusionChunkMetadata) or metadata != expected_metadata:
                    raise ARDiffusionChunkMetadataError(
                        "Tick consumer metadata does not match the submitted AR-Diffusion tick."
                    )
            except BaseException as tick_error:
                await self._fail_tick(tick_error)
                raise

            commit_task = asyncio.create_task(
                self._commit_tick(
                    tick,
                    prompt,
                    controls,
                    prepared_controls,
                )
            )
            try:
                await asyncio.shield(commit_task)
            except asyncio.CancelledError:
                # Once a consumer reports success, the local chunk/event state
                # must commit even if the transport task is being cancelled.
                try:
                    await commit_task
                except BaseException as commit_error:
                    await self._fail_tick(commit_error)
                    raise
                raise
            except BaseException as commit_error:
                await self._fail_tick(commit_error)
                raise
            return output

    async def reset(self) -> None:
        """Reset worker state and discard queued/current transport state."""
        async with self._tick_lock:
            async with self._state_lock:
                self._require_resettable()
                self._status = ARDiffusionSessionStatus.RESETTING
            try:
                await self._lifecycle.reset_session(self.session_id)
            except BaseException:
                async with self._state_lock:
                    self._status = ARDiffusionSessionStatus.FAILED
                raise
            async with self._state_lock:
                if self._pending_events:
                    self._event_floor = max(self._event_floor, max(self._pending_events))
                self._pending_events.clear()
                self._prompt = None
                self._controls.clear()
                if self._control_reducer is not None:
                    self._control_reducer.reset()
                self._inflight_event_floor = None
                self._chunk_index = 0
                self._status = ARDiffusionSessionStatus.ACTIVE

    async def close(self) -> None:
        """Close worker state, retaining a tombstone until cleanup succeeds."""
        async with self._tick_lock:
            async with self._state_lock:
                if self._status is ARDiffusionSessionStatus.CLOSED:
                    return
                self._status = ARDiffusionSessionStatus.CLOSING
            try:
                await self._lifecycle.close_session(self.session_id)
            except BaseException:
                async with self._state_lock:
                    self._status = ARDiffusionSessionStatus.CLEANUP_FAILED
                raise
            else:
                async with self._state_lock:
                    self._pending_events.clear()
                    self._prompt = None
                    self._controls.clear()
                    self._inflight_event_floor = None
                    self._status = ARDiffusionSessionStatus.CLOSED

    async def disconnect(self) -> None:
        """Disconnect uses the same owning-worker cleanup as explicit close."""
        await self.close()


class ARDiffusionSessionManager(Generic[TickOutputT]):
    """Creates and owns transport sessions sharing one tick/lifecycle backend."""

    def __init__(
        self,
        *,
        tick_consumer: ARDiffusionTickConsumer[TickOutputT],
        lifecycle: ARDiffusionSessionLifecycle,
        max_pending_events: int,
        request_id_factory: Callable[[str, int], str] | None = None,
        session_id_factory: Callable[[], str] | None = None,
        control_reducer_factory: Callable[[], ARDiffusionControlReducer] | None = None,
    ) -> None:
        self._tick_consumer = tick_consumer
        self._lifecycle = lifecycle
        self._max_pending_events = max_pending_events
        self._request_id_factory = request_id_factory
        self._session_id_factory = session_id_factory or (lambda: uuid.uuid4().hex)
        self._control_reducer_factory = control_reducer_factory
        self._sessions: dict[str, ARDiffusionSession[TickOutputT]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str | None = None) -> ARDiffusionSession[TickOutputT]:
        resolved_session_id = session_id or self._session_id_factory()
        async with self._lock:
            if resolved_session_id in self._sessions:
                raise ARDiffusionSessionStateError(f"AR-Diffusion session {resolved_session_id!r} already exists.")
            session = ARDiffusionSession(
                resolved_session_id,
                tick_consumer=self._tick_consumer,
                lifecycle=self._lifecycle,
                max_pending_events=self._max_pending_events,
                request_id_factory=self._request_id_factory,
                control_reducer=(
                    self._control_reducer_factory() if self._control_reducer_factory is not None else None
                ),
            )
            self._sessions[resolved_session_id] = session
            return session

    async def get_session(self, session_id: str) -> ARDiffusionSession[TickOutputT]:
        async with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise ARDiffusionSessionStateError(f"AR-Diffusion session {session_id!r} does not exist.") from exc

    async def reset_session(self, session_id: str) -> None:
        session = await self.get_session(session_id)
        await session.reset()

    async def close_session(self, session_id: str) -> None:
        session = await self.get_session(session_id)
        await session.close()
        async with self._lock:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id)

    async def disconnect(self, session_id: str) -> None:
        """Release the owning worker route when its transport disconnects."""
        await self.close_session(session_id)
