# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-resident session runner: one ordered mailbox and one session state per duplex session.

``DuplexSessionRunner`` owns the whole lifecycle of one session on the
orchestrator loop (the session is never touched from another thread):

* every mutation of ``DuplexEngineSession`` happens on this loop, through the
  mailbox worker (commands, stage outputs, internal items) or through tracked
  tasks that re-validate ``(epoch, turn_id)`` after each ``await``;
* appends are planned with the model plugin and submitted to the stage port
  in wire order on the per-session append tail (no RPC hop);
* stage outputs are pushed in by ``DuplexOrchestrator._intercept_stage_output``
  instead of being polled through request queues;
* everything the session says leaves as typed events through
  ``DuplexSessionManager.emit`` after terminal-acceptance / stale-epoch
  filtering, so a cancelled epoch can never speak again.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import uuid
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from vllm.logger import init_logger

from vllm_omni.engine.duplex.audio import convert_input_audio_with_rate
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    AppendText,
    BargeIn,
    CancelInput,
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
    DeleteItem,
    DuplexCommand,
    Heartbeat,
    SignalTurn,
    TruncateItem,
    UpdateSession,
)
from vllm_omni.engine.duplex.commit_policy import CommitAction, CommitSnapshot, decide_commit_action
from vllm_omni.engine.duplex.config import (
    DuplexCommittedInput,
    DuplexConfigError,
    DuplexOverlapPolicy,
    DuplexPlaybackCommitPolicy,
    DuplexSessionState,
    DuplexTurnEventType,
    ResponseCreateOptions,
    realtime_item_to_history_message,
)
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputContext,
    DuplexOutputDecision,
    DuplexStagePort,
    DuplexStageSubmission,
    duplex_data_plane_request_info,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.events import (
    DOMAIN_TERMINAL_EVENTS,
    MODEL_OUTPUT_EVENTS,
    DuplexEvent,
    ErrorEvent,
    InputCleared,
    OverlapDecision,
    PlaybackAcknowledged,
    SessionExpired,
    SessionHeartbeatAck,
    error_event,
)
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    PcmAppendReservation,
    coerce_int,
    payload_turn_id,
)
from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    discard_pending_input_audio,
    note_input_append,
    project_internal_event,
    resolve_cancel_response,
    resolve_clear_output_audio,
    resolve_commit,
    resolve_create_item,
    resolve_delete_item,
    resolve_truncate_item,
    retrieve_item_events,
)
from vllm_omni.engine.duplex.session import DuplexEngineSession, DuplexFenceMismatchError
from vllm_omni.engine.duplex.turn_detection import (
    PendingTurnDetectionUpdate,
    ServerTurnDetector,
    ServerVADUnavailableError,
    TurnDetectionConfig,
    TurnDetectionResult,
    apply_turn_detection_result,
)
from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.duplex import attach_duplex_output_decision

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.outputs import RequestOutput

    from vllm_omni.engine.duplex.session_manager import DuplexSessionManager

logger = init_logger(__name__)

_OffloadT = TypeVar("_OffloadT")


# --------------------------------------------------------------------------- #
# Task bookkeeping                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DuplexAppendTaskMeta:
    epoch: int
    final: bool
    response_bound: bool


@dataclass
class DuplexSessionTasks:
    """Tracked task handles of one session (append tail, active response, pending silence)."""

    append_tasks: dict[asyncio.Task[bool], DuplexAppendTaskMeta] = field(default_factory=dict)
    append_tail: asyncio.Task[bool] | None = None
    active_response_task: asyncio.Task[None] | None = None

    def track_append_task(
        self,
        task: asyncio.Task[bool],
        *,
        epoch: int,
        final: bool,
        response_bound: bool,
    ) -> None:
        self.append_tasks[task] = DuplexAppendTaskMeta(epoch, final, response_bound)
        task.add_done_callback(self.append_tasks.pop)

    def has_response_bound_append_tasks(self) -> bool:
        return any(meta.response_bound for meta in self.append_tasks.values())

    async def cancel_append_tasks(self, timeout_s: float = 0.25, *, response_bound_only: bool = False) -> bool:
        tasks = [task for task, meta in self.append_tasks.items() if not response_bound_only or meta.response_bound]
        if not tasks:
            return False
        cancelled_tail = self.append_tail if self.append_tail in tasks else None
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
        except TimeoutError:
            pass
        if cancelled_tail is not None and self.append_tail is cancelled_tail:
            self.append_tail = None
        return True


# --------------------------------------------------------------------------- #
# Mailbox items                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _StageOutput:
    stage_id: int
    output: RequestOutput
    metrics: StageRequestStats | None
    request_id: str
    context: DuplexOutputContext
    decision: DuplexOutputDecision | None


@dataclass(frozen=True, slots=True)
class _Internal:
    """Runner-internal mailbox item (its payload runs through the dict-based handlers)."""

    kind: str
    payload: dict[str, object] = field(default_factory=dict)


_CANCEL_EVENTS = frozenset({"input.cancel", "response.cancel", "barge_in", "output_audio_buffer.clear"})


class DuplexSessionRunner:
    """Owns one ``DuplexEngineSession`` on the orchestrator loop (see module docstring)."""

    # One MiniCPM model unit (1 s at 16 kHz) is the compatibility default.
    _NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO = base64.b64encode(bytes(16000 * 4)).decode("ascii")
    _NATIVE_RESPONSE_MAX_CONTINUATION_UNITS = 8
    _NATIVE_AUTO_RESPONSE_MAX_CONTINUATION_UNITS = 64

    def __init__(
        self,
        *,
        session: DuplexEngineSession,
        plugin: DuplexModelPlugin,
        stage_port: DuplexStagePort,
        manager: DuplexSessionManager,
        model_config: ModelConfig | None,
    ) -> None:
        self.session = session
        self.plugin = plugin
        self.stage_port = stage_port
        self.manager = manager
        self.model_config = model_config
        if session.model_state is None:
            session.model_state = plugin.create_session_state()
        self.model_state: DuplexModelSessionState = session.model_state
        self.tasks = DuplexSessionTasks()
        self._mailbox: asyncio.Queue[DuplexCommand | _StageOutput | _Internal] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._worker_stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        self._close_reason: str | None = None
        self._closed_emitted = False
        self._closed_deferred = False
        self._runtime_closed = False
        #: Request id of the resumable data-plane stream currently bound to this session.
        self._stream_request_id: str | None = None
        self._turn_detection_config: TurnDetectionConfig | None = None
        self._turn_detector: ServerTurnDetector | None = None
        self._pending_turn_detection: PendingTurnDetectionUpdate | None = None
        self._projector: RealtimeProjectionState | None = session.projector

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        session = self.session
        if self._projector is None:
            default_payload = session.config.extra_body.get("realtime_session_payload")
            self._projector = RealtimeProjectionState(
                session_id=session.session_id,
                model=session.config.model,
                default_payload=default_payload if isinstance(default_payload, Mapping) else None,
                initial_session_update=True,
            )
            session.projector = self._projector
        # Every client speaks the Realtime protocol now; the old runner forced
        # the ACK-only playback ledger for that path.
        session.config.playback_commit_policy = DuplexPlaybackCommitPolicy.ACK_ONLY.value
        self._init_turn_detection()
        self._worker = self._loop.create_task(self._run(), name=f"duplex-session-{session.session_id}")
        self.emit({"type": "session.created", "session": session.as_public_dict()})

    def submit(self, command: DuplexCommand) -> None:
        self._mailbox.put_nowait(command)

    def on_stage_output(
        self,
        stage_id: int,
        output: RequestOutput,
        metrics: StageRequestStats | None,
        *,
        request_id: str,
        context: DuplexOutputContext,
    ) -> bool:
        """Accept one stage output (orchestrator loop); return True when it must not be forwarded."""
        decision: DuplexOutputDecision | None = None
        if stage_id < context.final_stage_id:
            decision = self._decide_output(stage_id, output, context)
        consume = decision is not None or stage_id >= context.final_stage_id
        if not consume:
            # Stage0 text without a direct decision feeds the TTS stage as before.
            # Its metrics still have to reach the client: before sessions moved
            # into the engine the orchestrator published them as a standalone
            # ``StageMetricsMessage``, a path session-owned requests no longer
            # take. Hand them to the session instead of dropping them, on the
            # mailbox so they stay ordered with this session's other work.
            snapshot = self._stage_metrics_snapshot(stage_id, metrics, output)
            if snapshot is not None and not self._closing and self.session.state != DuplexSessionState.CLOSED:
                self._mailbox.put_nowait(_Internal("stage_metrics", {"stage_metrics": snapshot}))
            return False
        if self._closing or self.session.state == DuplexSessionState.CLOSED:
            return True
        self._mailbox.put_nowait(
            _StageOutput(
                stage_id=stage_id,
                output=output,
                metrics=metrics,
                request_id=request_id,
                context=context,
                decision=decision,
            )
        )
        return True

    def on_stage_failure(self, stage_id: int, exc: BaseException) -> None:
        """A stage rejected this session's request: fail the active response now.

        Runs synchronously on the loop (no mailbox hop): the orchestrator
        expires the session right after this call, so a queued item could be
        cancelled with the worker and the client would only see
        ``session.expired``. Emitting here keeps the order
        ``error`` -> failed ``response.done`` -> ``session.expired``.
        """
        session = self.session
        if session.state == DuplexSessionState.CLOSED:
            return
        self._emit_error(
            "runtime_data_plane_stream_failed",
            f"Stage-{stage_id} input processor failed: {type(exc).__name__}: {exc}",
        )
        response_id = session.active_response_id
        if response_id is not None:
            session.end_response(commit_text=False)
            self.emit(
                {
                    "type": "response.done",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": session.epoch,
                    "committed": False,
                    "status": "failed",
                    "status_details": {"type": "failed", "reason": "runtime_data_plane_stream_failed"},
                    "playback": session.playback.as_dict(),
                }
            )

    @property
    def closed_emitted(self) -> bool:
        """Whether ``session.closed`` / ``session.expired`` already left this runner."""
        return self._closed_emitted

    @property
    def closing(self) -> bool:
        """Whether an irreversible close has begun (commands and control ops are refused)."""
        return self._closing or self.session.state != DuplexSessionState.OPEN

    async def close(self, reason: str, *, emit_closed: bool = True) -> None:
        """Graceful close: cancel work, release the data plane, emit ``session.closed``.

        With ``emit_closed=False`` the manager emits ``session.closed`` itself
        once the stage resources are released, so the event also means "the
        admission slot is free again".
        """
        session = self.session
        if session.state == DuplexSessionState.CLOSED and (self._closed_emitted or self._closed_deferred):
            await self._stop_worker()
            return
        self._begin_close(reason)
        self.model_state.audio_buffer.clear()
        session.release_all_input_bytes()
        self.model_state.input_since_commit = False
        self.model_state.speech_since_commit = False
        self.model_state.clear_committed_audio()
        await self.tasks.cancel_append_tasks()
        self._cancel_data_plane_stream()
        await self._cancel_active_response(self.tasks.active_response_task, reason=reason, notify=False)
        self.tasks.active_response_task = None
        self._cleanup_duplex_session_state()
        if not self._closed_emitted and not self._closed_deferred:
            self._close_reason = self._close_reason or reason
            if emit_closed:
                self._closed_emitted = True
                self.emit({"type": "session.closed", "session_id": session.session_id, "reason": reason})
            else:
                self._closed_deferred = True
        session.close()
        await self._stop_worker()

    async def expire(self, reason: str, *, emit_expired: bool = True) -> None:
        """Lease expiry / runtime cleanup: emit ``session.expired`` and tear down.

        With ``emit_expired=False`` the manager emits the event after the stage
        cleanup (see ``close``).
        """
        session = self.session
        self._begin_close(reason)
        if not self._closed_emitted:
            if emit_expired:
                # Also when the event was deferred: the manager only emits a
                # deferred terminal for the teardown it drives itself, and an
                # expiry that arrives first (a stage failure racing a wire
                # ``session.close``) takes the runner away from it. Emitting
                # here keeps "every session ends with one terminal event".
                self._closed_emitted = True
                self._emit_events([SessionExpired(reason=reason)])
            else:
                self._closed_deferred = True
        await self.tasks.cancel_append_tasks()
        self._cancel_data_plane_stream()
        active_response_task = self.tasks.active_response_task
        if active_response_task is not None and not active_response_task.done():
            active_response_task.cancel()
            await asyncio.gather(active_response_task, return_exceptions=True)
        self.tasks.active_response_task = None
        if session.active_response_id is not None:
            session.end_response(commit_text=False)
        self._cleanup_duplex_session_state()
        session.close()
        await self._stop_worker()

    async def shutdown(self) -> None:
        self._closing = True
        self._closed_emitted = True
        await self.tasks.cancel_append_tasks()
        self._cancel_data_plane_stream()
        active_response_task = self.tasks.active_response_task
        if active_response_task is not None and not active_response_task.done():
            active_response_task.cancel()
            await asyncio.gather(active_response_task, return_exceptions=True)
        self.session.close()
        await self._stop_worker()

    # ------------------------------------------------------------------ #
    # Worker                                                             #
    # ------------------------------------------------------------------ #

    async def _run(self) -> None:
        while not self._worker_stopped:
            item = await self._mailbox.get()
            try:
                await self._handle_item(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Duplex session %s failed handling %r: %s", self.session.session_id, item, exc)
                self._emit_error("internal_error", str(exc))

    async def _stop_worker(self) -> None:
        worker = self._worker
        self._worker = None
        # Set even when the worker itself is stopping (a wire ``session.close``
        # runs inside it): its loop exits after the current item instead of
        # parking on the mailbox forever.
        self._worker_stopped = True
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        if worker is not None and worker is not asyncio.current_task() and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    def _spawn(self, coro: Awaitable[None], *, name: str) -> asyncio.Task[None]:
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _offload(self, fn: Callable[..., _OffloadT], *args: object, **kwargs: object) -> _OffloadT:
        loop = self._loop or asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(self.manager.executor, lambda: fn(*args, **kwargs))
        return await loop.run_in_executor(self.manager.executor, fn, *args)

    async def _handle_item(self, item: DuplexCommand | _StageOutput | _Internal) -> None:
        session = self.session
        if isinstance(item, _StageOutput):
            await self._on_stage_output_item(item)
            return
        if isinstance(item, _Internal):
            await self._on_internal(item)
            return
        if self._closing or session.state == DuplexSessionState.CLOSED:
            if isinstance(item, Commit):
                session.release_pending_turn()
            elif isinstance(item, AppendAudio):
                session.release_input_bytes(len(item.audio))
            return
        await self._on_command(item)

    async def _on_internal(self, item: _Internal) -> None:
        if item.kind == "stage_metrics":
            stage_metrics = item.payload.get("stage_metrics")
            if isinstance(stage_metrics, Mapping):
                self.session.stash_stage_metrics(stage_metrics)
            return
        if item.kind == "promote_deferred_overlap":
            if self._closing or self.session.state != DuplexSessionState.OPEN:
                return
            payload = item.payload.get("payload")
            if not isinstance(payload, dict):
                return
            await self._start_append(
                payload,
                final=True,
                precreate_response=bool(item.payload.get("precreate_response", False)),
                operation_id=self.model_state.committed_audio_operation_id,
                retained_committed_payload=(payload if self.model_state.committed_audio_payload is payload else None),
            )
            return
        if item.kind == "commit":
            await self._on_commit(dict(item.payload))
            return
        if item.kind == "run_payload":
            await self._run_internal_payload(dict(item.payload))
            return
        logger.warning("Unknown duplex runner internal item: %s", item.kind)

    async def _run_internal_payload(self, payload: dict[str, object]) -> None:
        """Run one internal event dictionary through the matching handler."""
        event_type = payload.get("type")
        if event_type == "input_audio_buffer.append":
            await self._on_append_audio(payload)
        elif event_type in {"input_audio_buffer.commit", "input.commit", "response.create"}:
            await self._on_commit(payload)
        elif event_type == "playback.ack":
            self._handle_playback_ack(payload)
        elif event_type in _CANCEL_EVENTS:
            await self._on_cancel(payload)
        elif event_type == "turn.signal":
            await self._on_turn_signal(payload)
        else:
            self._emit_error("unknown_event", f"Unknown duplex event: {event_type}")

    # ------------------------------------------------------------------ #
    # Commands                                                           #
    # ------------------------------------------------------------------ #

    async def _on_command(self, command: DuplexCommand) -> None:
        session = self.session
        projector = self._require_projector()
        if isinstance(command, AppendAudio):
            # The manager reserved the wire size at admission; the handler
            # re-reserves the decoded size around its PCM reservation.
            session.release_input_bytes(len(command.audio))
            await self._on_append_audio(command.payload())
        elif isinstance(command, AppendText):
            session.mark_user_input_activity()
            self._emit_error(
                "native_text_append_unsupported",
                "The selected native duplex runtime accepts audio append only",
                event_id=command.event_id,
            )
        elif isinstance(command, Commit):
            session.release_pending_turn()
            resolved = resolve_commit(projector, command)
            self._emit_events(resolved.events)
            if resolved.reset_vad and self._turn_detector is not None:
                self._turn_detector.reset()
            if resolved.payload is not None:
                await self._on_commit(resolved.payload)
        elif isinstance(command, CreateResponse):
            await self._on_commit(command.payload())
        elif isinstance(command, ClearInput):
            self._on_clear_input()
        elif isinstance(command, CancelResponse):
            resolved = resolve_cancel_response(projector, command)
            self._emit_events(resolved.events)
            for payload in resolved.payloads:
                await self._on_cancel(payload)
        elif isinstance(command, ClearOutputAudio):
            resolved = resolve_clear_output_audio(projector, command)
            self._emit_events(resolved.events)
            for payload in resolved.payloads:
                await self._on_cancel(payload)
        elif isinstance(command, CancelInput | BargeIn):
            await self._on_cancel(command.payload())
        elif isinstance(command, SignalTurn):
            if command.event == "conversation.item.retrieve":
                self._emit_events(
                    retrieve_item_events(
                        projector,
                        {
                            **dict(command.signal_payload),
                            **({"event_id": command.event_id} if command.event_id else {}),
                        },
                    )
                )
                return
            payload = command.payload()
            if command.event in _CANCEL_EVENTS:
                normalized = dict(payload.get("payload") or {})
                normalized.update(payload)
                normalized["type"] = command.event
                await self._on_cancel(normalized)
            else:
                await self._on_turn_signal(payload)
        elif isinstance(command, UpdateSession):
            await self._on_session_update(dict(command.patch), realtime_event_id=command.event_id)
        elif isinstance(command, AckPlayback):
            self._handle_playback_ack(command.payload())
        elif isinstance(command, Heartbeat):
            self._on_heartbeat(command)
        elif isinstance(command, CreateItem):
            resolved = resolve_create_item(projector, command)
            self._emit_events(resolved.events)
            for payload in resolved.payloads:
                await self._run_internal_payload(payload)
        elif isinstance(command, DeleteItem):
            resolved = resolve_delete_item(projector, command)
            self._emit_events(resolved.events)
            for payload in resolved.payloads:
                await self._on_turn_signal(payload)
        elif isinstance(command, TruncateItem):
            resolved = resolve_truncate_item(projector, command)
            self._emit_events(resolved.events)
            for payload in resolved.payloads:
                await self._run_internal_payload(payload)
        elif isinstance(command, CloseSession):
            await self._on_close_command(command.reason)
        else:
            self._emit_error("unknown_event", f"Unknown duplex command: {type(command).__name__}")

    async def _on_close_command(self, reason: str) -> None:
        # A close requested through the ordered command stream (the client's
        # ``session.close``) tears the runner down here; the manager then
        # aborts the stage requests, frees the admission slot and emits
        # ``session.closed`` exactly like a close through the control RPC.
        # Mark the session closing before the first await so a concurrent
        # resume / touch / command is refused instead of racing the teardown.
        self._begin_close(reason)
        await self.close(reason, emit_closed=False)
        self.manager.close_from_runner(self, reason)

    def _on_heartbeat(self, command: Heartbeat) -> None:
        try:
            self.session.touch_lease(DuplexLeaseActivity.HEARTBEAT)
        except Exception as exc:
            self._emit_error("runtime_touch_failed", str(exc), event_id=command.event_id)
            return
        self._emit_events([SessionHeartbeatAck()])

    def _on_clear_input(self) -> None:
        session = self.session
        model_state = self.model_state
        model_state.audio_buffer.clear()
        session.release_all_input_bytes()
        model_state.input_since_commit = False
        model_state.speech_since_commit = False
        model_state.clear_committed_audio()
        session.cancel_pending_input()
        projector = self._projector
        if projector is not None:
            from vllm_omni.engine.duplex.realtime_events import clear_input_buffer

            clear_input_buffer(projector)
        self._emit_events([InputCleared()])

    # ------------------------------------------------------------------ #
    # Emission (was emit_event + _apply_outbound_session_event + writer) #
    # ------------------------------------------------------------------ #

    def _require_projector(self) -> RealtimeProjectionState:
        if self._projector is None:
            self._projector = RealtimeProjectionState(
                session_id=self.session.session_id,
                model=self.session.config.model,
                initial_session_update=True,
            )
            self.session.projector = self._projector
        return self._projector

    def _emit_events(self, events: list[DuplexEvent]) -> None:
        for event in events:
            self.manager.emit(self.session, event)

    def _emit_error(
        self,
        code: str,
        message: str,
        *,
        event_id: object | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Send one typed ``error`` event (``event_id`` is the client event it answers)."""
        extra = {} if retryable is None else {"retryable": retryable}
        self._emit_events([error_event(code, message, event_id=event_id, extra=extra)])

    def emit(self, payload: dict[str, object]) -> None:
        """Apply the domain effects of an internal event, then project it to typed events and send them.

        Only events with domain effects (response / cancel / close terminals) or
        Realtime projection state (response items, content parts) still travel as
        internal dictionaries; stateless events are constructed typed at the
        emit site.
        """
        accepted, deferred_overlap_payload = self._apply_outbound_session_event(payload)
        if not accepted:
            return
        self._emit_events(project_internal_event(self._require_projector(), payload))
        if deferred_overlap_payload is not None and not self._closing:
            precreate_response = self.model_state.deferred_precreate_response
            self.model_state.deferred_precreate_response = False
            self._mailbox.put_nowait(
                _Internal(
                    "promote_deferred_overlap",
                    {"payload": deferred_overlap_payload, "precreate_response": precreate_response},
                )
            )

    def _is_stale_model_output(self, payload: dict[str, object]) -> bool:
        event_type = payload.get("type")
        if event_type in DOMAIN_TERMINAL_EVENTS:
            return False
        if event_type not in MODEL_OUTPUT_EVENTS:
            return False
        if self._closing and event_type != "response.listen":
            return True
        if self.session.state == DuplexSessionState.CLOSED and event_type != "response.listen":
            return True
        epoch = payload.get("epoch")
        return isinstance(epoch, int) and epoch != self.session.epoch

    def _apply_outbound_session_event(self, payload: dict[str, object]) -> tuple[bool, dict[str, object] | None]:
        """Apply domain transitions before an event is projected (moved from serving)."""
        session = self.session
        model_state = self.model_state
        payload_type = payload.get("type")
        is_terminal = payload_type in DOMAIN_TERMINAL_EVENTS
        if is_terminal:
            payload_epoch = payload.get("epoch")
            if isinstance(payload_epoch, int) and payload_epoch != session.epoch:
                return False, None
            if payload_type in {"response.done", "response.listen"} and (
                self._closing or session.state == DuplexSessionState.CLOSED
            ):
                return False, None
        elif self._is_stale_model_output(payload):
            return False, None

        if payload_type == "session.closed":
            self._close_reason = self._close_reason or str(payload.get("reason") or "closed")
            session.mark_closing()

        if not is_terminal:
            return True, None

        terminal_status = payload.get("status")
        terminal_status_details = payload.get("status_details")
        if terminal_status is None and isinstance(terminal_status_details, dict):
            terminal_status = terminal_status_details.get("type")
        response_terminal = payload_type == "response.done" or (
            payload_type == "response.listen" and session.active_response_id is not None
        )
        can_promote_overlap = response_terminal and terminal_status not in {"cancelled", "failed"}
        deferred_overlap_payload: dict[str, object] | None = None
        continuous_input_crosses_terminal = (
            can_promote_overlap
            and self._session_auto_responds()
            and model_state.input_since_commit
            and not model_state.deferred_response_create
        )
        if continuous_input_crosses_terminal:
            session.reset_overlap_speech()
            return True, None
        realtime_input_still_open = (
            can_promote_overlap and model_state.input_since_commit and not model_state.deferred_response_create
        )
        if realtime_input_still_open:
            session.reset_overlap_speech()
            return True, None
        if can_promote_overlap and session.overlap_speech_ms > 0:
            has_deferred_overlap = (
                model_state.audio_buffer.has_pending() or model_state.committed_audio_payload is not None
            )
            should_promote_overlap = (
                session.state == DuplexSessionState.OPEN
                and has_deferred_overlap
                and session.overlap_speech_ms > session.config.overlap_short_ack_ms
            )
            if should_promote_overlap:
                flushed_reserved_bytes = model_state.audio_buffer.pending_byte_count
                deferred_overlap_payload = model_state.audio_buffer.flush(
                    chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                )
                if model_state.committed_audio_payload is not None:
                    if deferred_overlap_payload is not None:
                        deferred_overlap_payload = self._merge_audio_payloads(
                            model_state.committed_audio_payload,
                            deferred_overlap_payload,
                        )
                    else:
                        deferred_overlap_payload = model_state.committed_audio_payload
                if self._session_auto_responds() and deferred_overlap_payload is not None:
                    deferred_overlap_payload = dict(deferred_overlap_payload)
                    deferred_overlap_payload["force_listen"] = False
                if deferred_overlap_payload is not None:
                    model_state.retain_committed_audio(
                        deferred_overlap_payload,
                        operation_id=model_state.committed_audio_operation_id,
                        reserved_bytes=flushed_reserved_bytes,
                    )
                model_state.input_since_commit = deferred_overlap_payload is not None
                # Realtime path: defer the promoted response to the next terminal.
                model_state.deferred_response_create = True
                model_state.deferred_precreate_response = False
                deferred_overlap_payload = None
            else:
                had_pending_overlap_audio = model_state.audio_buffer.has_pending()
                model_state.audio_buffer.clear()
                model_state.input_since_commit = False
                model_state.speech_since_commit = False
                if had_pending_overlap_audio and self._projector is not None:
                    self._emit_events(discard_pending_input_audio(self._projector, session.overlap_speech_ms))
                if payload_type in {"audio.cancelled", "input.cancelled", "session.closed"}:
                    session.release_input_bytes(model_state.clear_committed_audio())

        session.reset_overlap_speech()
        if (
            can_promote_overlap
            and model_state.deferred_response_create
            and model_state.committed_audio_payload is not None
        ):
            deferred_overlap_payload = model_state.committed_audio_payload
            model_state.deferred_response_create = False
            model_state.input_since_commit = False
            model_state.speech_since_commit = False
        return True, deferred_overlap_payload

    # ------------------------------------------------------------------ #
    # Session helpers (moved from OmniDuplexSessionHandler)              #
    # ------------------------------------------------------------------ #

    def _begin_close(self, reason: str) -> None:
        self._closing = True
        self._close_reason = self._close_reason or reason
        self.session.mark_closing()

    def _session_auto_responds(self) -> bool:
        extra = getattr(self.session.config, "extra_body", None)
        if not isinstance(extra, dict):
            return False
        return extra.get("auto_response") is True or extra.get("full_duplex") is True

    def _stage0_request_id(self, epoch: int) -> str:
        return duplex_resource_request_id(DuplexFence(self.session.session_id, epoch=epoch), "stage0")

    def _response_in_progress(self) -> bool:
        session = self.session
        if session.active_response_id is not None:
            return True
        if (
            session.config.playback_commit_policy == DuplexPlaybackCommitPolicy.ACK_ONLY.value
            and session.playback.sent_ms > session.playback.committed_ms
        ):
            return True
        active_task = self.tasks.active_response_task
        if active_task is not None and not active_task.done():
            return True
        if self.tasks.has_response_bound_append_tasks():
            return True
        return False

    def _assistant_playback_active(self) -> bool:
        session = self.session
        return (
            session.config.playback_commit_policy == DuplexPlaybackCommitPolicy.ACK_ONLY.value
            and session.playback.sent_ms > session.playback.committed_ms
        )

    @staticmethod
    def _advance_barge_in_epoch(session: DuplexEngineSession) -> tuple[int, dict[str, int]]:
        old_playback = session.playback.as_dict()
        new_epoch = session.barge_in()
        session.clear_playback_cursor()
        return new_epoch, old_playback

    @staticmethod
    def _commit_played_response_history(
        session: DuplexEngineSession,
        response_id: str | None,
        committed_ms: int,
    ) -> None:
        if not response_id or committed_ms < 0:
            return
        session.truncate_history_item(
            f"item_{response_id}",
            audio_end_ms=committed_ms,
            playback=session.playback_for_response(response_id),
        )

    @staticmethod
    def _should_commit_response_to_history(session: DuplexEngineSession, response_id: str | None) -> bool:
        if response_id is not None and response_id != session.active_response_id:
            return True
        mode = session.response_config.extra_body.get("realtime_response_conversation")
        return not isinstance(mode, str) or mode.strip().lower() != "none"

    def _response_created_payload(
        self,
        response_id: str,
        *,
        epoch: int,
        request_id: str | None = None,
    ) -> dict[str, object]:
        session = self.session
        response_config = session.response_config
        payload: dict[str, object] = {
            "type": "response.created",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": epoch,
            "modalities": list(response_config.modalities),
        }
        if request_id is not None:
            payload["request_id"] = request_id
        metadata = response_config.extra_body.get("realtime_response_metadata")
        if not isinstance(metadata, dict):
            metadata = response_config.extra_body.get("realtime_metadata")
        if isinstance(metadata, dict):
            payload["metadata"] = dict(metadata)
        conversation = response_config.extra_body.get("realtime_response_conversation")
        if isinstance(conversation, str):
            payload["conversation"] = conversation
        prompt = response_config.extra_body.get("realtime_response_prompt")
        if isinstance(prompt, dict):
            payload["prompt"] = dict(prompt)
        return payload

    @staticmethod
    def _barge_in_unsupported_error() -> ErrorEvent:
        return error_event("barge_in_unsupported", "Barge-in is not supported by this duplex model")

    @staticmethod
    def _audio_payload_size_bytes(payload: Mapping[str, object]) -> int:
        audio = payload.get("audio") or payload.get("data")
        if not isinstance(audio, str):
            return 0
        try:
            return len(base64.b64decode(audio, validate=True))
        except (ValueError, binascii.Error):
            return 0

    @staticmethod
    def _input_committed_payload(
        session: DuplexEngineSession,
        committed: DuplexCommittedInput,
        *,
        realtime_item_id: object | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "input.committed",
            "session_id": session.session_id,
            "turn_id": committed.turn_id,
            "epoch": committed.epoch,
            "history_len": len(session.history),
            "message": committed.message,
        }
        if isinstance(realtime_item_id, str) and realtime_item_id:
            payload["realtime_item_id"] = realtime_item_id
        return payload

    @staticmethod
    def _commit_audio_input(
        session: DuplexEngineSession,
        *,
        realtime_item_id: object | None = None,
        transcript: object | None = None,
        turn_id: int | None = None,
    ) -> DuplexCommittedInput:
        clean_transcript = transcript.strip() if isinstance(transcript, str) else None
        committed = session.commit_audio_input(
            transcript=clean_transcript or None,
            turn_id=turn_id,
        )
        if isinstance(realtime_item_id, str) and realtime_item_id:
            session.register_history_item(realtime_item_id, committed.message)
        return committed

    @staticmethod
    def _audio_committed_payload(
        session: DuplexEngineSession,
        *,
        committed: DuplexCommittedInput | None = None,
        realtime_item_id: object | None = None,
        transcript: object | None = None,
    ) -> dict[str, object]:
        message = committed.message if committed is not None else None
        if not isinstance(message, dict):
            input_audio_part: dict[str, object] = {
                "type": "audio_url",
                "audio_url": {"url": "native-duplex:input-audio"},
            }
            if isinstance(transcript, str) and transcript:
                input_audio_part["transcript"] = transcript
            message = {"role": "user", "content": [input_audio_part]}
        payload: dict[str, object] = {
            "type": "input.committed",
            "session_id": session.session_id,
            "turn_id": committed.turn_id if committed is not None else session.turn_id,
            "epoch": committed.epoch if committed is not None else session.epoch,
            "history_len": len(session.history),
            "native_audio": True,
            "message": message,
        }
        if isinstance(transcript, str) and transcript:
            payload["transcript"] = transcript
        if isinstance(realtime_item_id, str) and realtime_item_id:
            payload["realtime_item_id"] = realtime_item_id
        return payload

    def _cleanup_duplex_session_state(self) -> None:
        session = self.session
        self.plugin.data_plane.close_session(session.session_id, active_request_id=session.active_request_id)
        self._stream_request_id = None

    # ------------------------------------------------------------------ #
    # Overlap policy (moved from OmniDuplexSessionHandler)               #
    # ------------------------------------------------------------------ #

    def _overlap_decision(self, event: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
        session = self.session
        duration_ms = self._input_audio_duration_ms(event, payload)
        is_speech = self._input_looks_like_speech(event, payload)
        if not session.capabilities.supports_barge_in and self._event_requests_barge_in(event):
            return self._defer_unsupported_barge_in(session, duration_ms=duration_ms, is_speech=is_speech)
        explicit = event.get("overlap_action") or event.get("overlap")
        if isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {"barge_in", "interrupt", "cancel"}:
                return {
                    "action": "barge_in",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": True,
                }
            if normalized in {"listen", "continue", "continue_output", "ack"}:
                session.reset_overlap_speech()
                return {
                    "action": "listen",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": (
                        normalized == "listen" and is_speech and duration_ms > session.config.overlap_short_ack_ms
                    ),
                    "defer_runtime_append": True,
                }
            if normalized in {"drop", "ignore", "silence"}:
                session.reset_overlap_speech()
                return {
                    "action": "drop",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": False,
                }

        if bool(event.get("force_barge_in", False)):
            return {
                "action": "barge_in",
                "reason": "client_force_barge_in",
                "duration_ms": duration_ms,
                "buffer_audio": True,
            }
        if self._session_auto_responds():
            if is_speech:
                session.accumulate_overlap_speech(duration_ms)
            vad_speech_started = self._vad_speech_started(event, payload)
            if (
                session.capabilities.supports_barge_in
                and is_speech
                and session.config.overlap_policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
                and vad_speech_started is not False
            ):
                return {
                    "action": "barge_in",
                    "reason": ("server_vad_speech_started" if vad_speech_started is True else "barge_in_on_speech"),
                    "cancel_reason": "turn_detected" if vad_speech_started is True else "barge_in",
                    "duration_ms": duration_ms,
                    "overlap_speech_ms": session.overlap_speech_ms,
                    "buffer_audio": True,
                }
            return {
                "action": "listen",
                "reason": "auto_response_continuous",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": False,
                "force_listen": event.get("force_listen") is True or payload.get("force_listen") is True,
                "preserve_realtime_input": True,
            }
        if bool(event.get("force_listen", False)):
            session.reset_overlap_speech()
            return {
                "action": "listen",
                "reason": "client_force_listen",
                "duration_ms": duration_ms,
                "buffer_audio": is_speech,
                "defer_runtime_append": True,
            }

        policy = session.config.overlap_policy
        if not is_speech:
            if session.overlap_speech_ms <= 0:
                session.reset_overlap_speech()
            return {
                "action": "drop",
                "reason": "silence_or_noise",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": False,
            }

        if self._is_short_ack_transcript_hint(event, payload):
            session.reset_overlap_speech()
            return {
                "action": "listen",
                "reason": "short_ack_transcript",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": False,
                "defer_runtime_append": True,
            }

        if policy == DuplexOverlapPolicy.LISTEN_ONLY.value:
            session.accumulate_overlap_speech(duration_ms)
            return {
                "action": "listen",
                "reason": "policy_listen_only",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }

        if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value and not session.capabilities.supports_barge_in:
            return self._defer_unsupported_barge_in(session, duration_ms=duration_ms, is_speech=True)

        session.accumulate_overlap_speech(duration_ms)
        if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value:
            vad_speech_started = self._vad_speech_started(event, payload)
            if vad_speech_started is False:
                return {
                    "action": "listen",
                    "reason": "server_vad_utterance_active",
                    "duration_ms": duration_ms,
                    "overlap_speech_ms": session.overlap_speech_ms,
                    "buffer_audio": True,
                    "defer_runtime_append": False,
                    "force_listen": True,
                    "preserve_realtime_input": True,
                }
            return {
                "action": "barge_in",
                "reason": ("server_vad_speech_started" if vad_speech_started is True else "policy_barge_in_on_speech"),
                "cancel_reason": "turn_detected" if vad_speech_started is True else "barge_in",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
            }

        if (
            duration_ms <= session.config.overlap_short_ack_ms
            and session.overlap_speech_ms <= session.config.overlap_short_ack_ms
        ):
            return {
                "action": "listen",
                "reason": "short_ack",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }
        if session.overlap_speech_ms >= session.config.overlap_barge_in_ms:
            if not session.capabilities.supports_barge_in:
                return {
                    "action": "listen",
                    "reason": "barge_in_unsupported",
                    "duration_ms": duration_ms,
                    "overlap_speech_ms": session.overlap_speech_ms,
                    "buffer_audio": True,
                    "defer_runtime_append": True,
                }
            return {
                "action": "barge_in",
                "reason": "long_overlap_speech",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
            }
        return {
            "action": "listen",
            "reason": "accumulating_overlap_speech",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
            "defer_runtime_append": True,
        }

    @staticmethod
    def _vad_speech_started(event: Mapping[str, object], payload: Mapping[str, object]) -> bool | None:
        for source in (event, payload):
            vad = source.get("vad")
            if isinstance(vad, Mapping) and isinstance(vad.get("speech_started"), bool):
                return bool(vad["speech_started"])
        return None

    @staticmethod
    def _event_requests_barge_in(event: Mapping[str, object]) -> bool:
        if event.get("force_barge_in") is True:
            return True
        explicit = event.get("overlap_action") or event.get("overlap")
        return isinstance(explicit, str) and explicit.strip().lower() in {"barge_in", "interrupt", "cancel"}

    @staticmethod
    def _defer_unsupported_barge_in(
        session: DuplexEngineSession,
        *,
        duration_ms: int,
        is_speech: bool,
    ) -> dict[str, object]:
        if is_speech:
            session.accumulate_overlap_speech(duration_ms)
        return {
            "action": "listen",
            "reason": "barge_in_unsupported",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": is_speech,
            "defer_runtime_append": True,
        }

    @staticmethod
    def _is_short_ack_transcript_hint(event: dict[str, object], payload: dict[str, object]) -> bool:
        raw_text = event.get("transcript") or event.get("text") or payload.get("transcript") or payload.get("text")
        if not isinstance(raw_text, str):
            return False
        normalized = raw_text.strip().lower()
        if not normalized:
            return False
        compact = "".join(ch for ch in normalized if ch.isalnum() or "一" <= ch <= "鿿")
        if compact in {
            "嗯",
            "嗯嗯",
            "对",
            "对的",
            "好",
            "好的",
            "继续",
            "继续说",
            "可以",
            "是的",
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "continue",
            "goon",
            "right",
        }:
            return True
        return normalized in {"go on", "keep going", "please continue"}

    @staticmethod
    def _input_audio_duration_ms(event: dict[str, object], payload: dict[str, object]) -> int:
        for key in ("duration_ms", "audio_duration_ms"):
            value = event.get(key)
            if isinstance(value, int | float):
                return max(0, int(value))
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt == "pcm_f32le" and isinstance(sample_rate_hz, int) and sample_rate_hz > 0 and isinstance(audio, str):
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError):
                return 0
            return int((len(raw) // 4) * 1000 / sample_rate_hz)
        return 0

    @staticmethod
    def _merge_audio_payloads(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
        if first.get("format") != "pcm_f32le" or second.get("format") != "pcm_f32le":
            return second
        first_rate = first.get("sample_rate_hz")
        second_rate = second.get("sample_rate_hz")
        if not isinstance(first_rate, int) or not isinstance(second_rate, int) or first_rate != second_rate:
            return second
        first_audio = first.get("audio")
        second_audio = second.get("audio")
        if not isinstance(first_audio, str) or not isinstance(second_audio, str):
            return second
        try:
            first_raw = base64.b64decode(first_audio, validate=True)
            second_raw = base64.b64decode(second_audio, validate=True)
        except (binascii.Error, ValueError):
            return second
        merged = dict(second)
        merged["audio"] = base64.b64encode(first_raw + second_raw).decode("ascii")
        merged["sample_rate_hz"] = first_rate
        merged_frames = [
            frame
            for source in (first.get("video_frames"), second.get("video_frames"))
            if isinstance(source, list)
            for frame in source
            if isinstance(frame, str) and frame
        ]
        if merged_frames:
            merged["video_frames"] = merged_frames
        else:
            merged.pop("video_frames", None)
        merged["force_listen"] = bool(first.get("force_listen", False)) or bool(second.get("force_listen", False))
        merged.pop("force_speak", None)
        merged["is_speech"] = bool(first.get("is_speech", False)) or bool(second.get("is_speech", False))
        return merged

    def _should_force_listen_for_short_commit(self, event: dict[str, object], payload: dict[str, object]) -> bool:
        if event.get("force_listen") is True or payload.get("force_listen") is True:
            return True
        if event.get("force_barge_in") is True:
            return False
        if event.get("response_create") is not True:
            return False
        duration_ms = self._input_audio_duration_ms(event, payload)
        return 0 < duration_ms <= self.session.config.overlap_short_ack_ms

    def _should_force_listen_for_auto_response_overlap(
        self, event: dict[str, object], payload: dict[str, object]
    ) -> bool:
        if not self._session_auto_responds():
            return False
        if event.get("force_barge_in") is True:
            return False
        return event.get("force_listen") is True or payload.get("force_listen") is True

    def _input_looks_like_speech(self, event: dict[str, object], payload: dict[str, object]) -> bool:
        for key in ("is_speech", "speech"):
            value = event.get(key)
            if isinstance(value, bool):
                return value
        vad = event.get("vad")
        if isinstance(vad, dict):
            value = vad.get("is_speech")
            if isinstance(value, bool):
                return value
            probability = vad.get("speech_probability", vad.get("probability"))
            if isinstance(probability, int | float):
                return float(probability) >= 0.5
        probability = event.get("speech_probability")
        if isinstance(probability, int | float):
            return float(probability) >= 0.5

        fmt = payload.get("format")
        audio = payload.get("audio")
        if fmt in {"pcm_f32le", "pcm16"} and isinstance(audio, str):
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError):
                return True
            if fmt == "pcm_f32le":
                if len(raw) < 4 or len(raw) % 4 != 0:
                    return True
                samples = np.frombuffer(raw, dtype=np.float32)
            else:
                if len(raw) < 2 or len(raw) % 2 != 0:
                    return True
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                return False
            rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            return rms >= self.session.config.overlap_silence_rms
        return True

    def _emit_overlap_decision(self, decision: dict[str, object]) -> None:
        session = self.session
        details: dict[str, object] = {
            "type": "overlap.decision",
            "session_id": session.session_id,
            "epoch": session.epoch,
            "policy": session.config.overlap_policy,
            **decision,
        }
        action = decision.get("action")
        reason = decision.get("reason")
        self._emit_events(
            [
                OverlapDecision(
                    policy=session.config.overlap_policy,
                    action=action if isinstance(action, str) else None,
                    reason=reason if isinstance(reason, str) else None,
                    details=details,
                )
            ]
        )

    # ------------------------------------------------------------------ #
    # Turn detection (server VAD)                                        #
    # ------------------------------------------------------------------ #

    def _init_turn_detection(self) -> None:
        turn_detection = self.session.config.extra_body.get("realtime_turn_detection")
        if not isinstance(turn_detection, dict) or turn_detection.get("type") != "server_vad":
            self._turn_detection_config = None
            self._turn_detector = None
            return
        try:
            config = TurnDetectionConfig.from_realtime(turn_detection)
            self._turn_detection_config = config
            self._turn_detector = config.build_detector()
        except Exception as exc:
            logger.warning("Duplex session %s: turn detection disabled: %s", self.session.session_id, exc)
            self._turn_detection_config = None
            self._turn_detector = None

    async def _run_turn_detection(self, event: dict[str, object]) -> TurnDetectionResult | None:
        detector = self._turn_detector
        if detector is None:
            return None
        audio = event.get("audio")
        if not isinstance(audio, str) or not audio:
            return None
        fmt = event.get("format") if isinstance(event.get("format"), str) else "pcm_f32le"
        sample_rate_hz = event.get("sample_rate_hz")
        try:
            result = await self._offload(
                detector.process,
                audio,
                fmt=fmt,
                sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int) else None,
                audio_end_ms=event.get("audio_end_ms") if isinstance(event.get("audio_end_ms"), int) else None,
            )
        except ServerVADUnavailableError as exc:
            self._emit_error("server_vad_unavailable", str(exc))
            self._turn_detector = None
            return None
        except ValueError as exc:
            self._emit_error("bad_audio", str(exc))
            return None
        apply_turn_detection_result(event, result)
        return result

    # ------------------------------------------------------------------ #
    # Append path (was the audio-append branch + start_native_append)    #
    # ------------------------------------------------------------------ #

    async def _on_append_audio(self, event: dict[str, object]) -> None:
        session = self.session
        model_state = self.model_state
        session.mark_user_input_activity()
        audio = event.get("audio") or event.get("data")
        if not isinstance(audio, str):
            self._emit_error("bad_event", "input_audio_buffer.append requires audio")
            return
        if not session.capabilities.supports_barge_in and self._event_requests_barge_in(event):
            self._emit_events([self._barge_in_unsupported_error()])
            event = dict(event)
            event.pop("force_barge_in", None)
            for key in ("overlap_action", "overlap"):
                value = event.get(key)
                if isinstance(value, str) and value.strip().lower() in {"barge_in", "interrupt", "cancel"}:
                    event.pop(key, None)
        fmt = event.get("format") if isinstance(event.get("format"), str) else "pcm16"
        sr_raw = event.get("sample_rate_hz") or event.get("sample_rate")
        sample_rate_hz = sr_raw if isinstance(sr_raw, int | float) else 16000
        try:
            audio, fmt, sample_rate_hz = await self._offload(
                convert_input_audio_with_rate,
                audio,
                fmt,
                sample_rate_hz=sample_rate_hz,
            )
        except ValueError as exc:
            self._emit_error("bad_event", str(exc))
            return
        if isinstance(fmt, str) and fmt.lower() in {"pcm16", "pcm_s16le", "s16le"}:
            self._emit_error("bad_audio", "input_audio_buffer.append pcm16 audio could not be decoded")
            return
        event["audio"] = audio
        event["format"] = fmt
        event["sample_rate_hz"] = sample_rate_hz
        vad_result = await self._run_turn_detection(event)
        projector = self._require_projector()
        self._emit_events(note_input_append(projector, event, vad_result=vad_result))
        if self._closing or session.state != DuplexSessionState.OPEN:
            return

        force_listen = bool(event.get("force_listen", False))
        payload: dict[str, object] = {
            "type": "audio",
            "audio": audio,
            "format": fmt,
            "sample_rate_hz": sample_rate_hz,
            "force_listen": force_listen,
        }
        video_frames = event.get("video_frames")
        if isinstance(video_frames, list):
            frames = [frame for frame in video_frames if isinstance(frame, str) and frame]
            if frames:
                payload["video_frames"] = frames
        payload["is_speech"] = self._input_looks_like_speech(event, payload)
        defer_append = False
        buffer_overlap_audio = True
        self._mark_pending_silence_superseded()
        overlap_active = self._response_in_progress() and (
            not self._session_auto_responds()
            or (
                session.capabilities.supports_barge_in
                and (
                    session.config.overlap_policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
                    or self._event_requests_barge_in(event)
                )
            )
        )
        if overlap_active:
            decision = self._overlap_decision(event, payload)
            self._emit_overlap_decision(decision)
            action = decision.get("action")
            if action == "drop":
                self._emit_events(discard_pending_input_audio(projector, self._input_audio_duration_ms(event, payload)))
                self._maybe_schedule_vad_commit(vad_result)
                return
            if action == "listen":
                buffer_overlap_audio = bool(decision.get("buffer_audio", True))
                defer_append = bool(decision.get("defer_runtime_append", True))
                if not buffer_overlap_audio and decision.get("preserve_realtime_input") is not True:
                    self._emit_events(
                        discard_pending_input_audio(projector, self._input_audio_duration_ms(event, payload))
                    )
                if decision.get("force_listen", True) is True:
                    payload["force_listen"] = True
            else:
                event["force_barge_in"] = True
                cancelled_fence = session.fence
                playback_was_active = self._assistant_playback_active()
                buffer_overlap_audio = True
                defer_append = False
                model_state.audio_buffer.clear_force_listen()
                session.reset_overlap_speech()
                model_state.input_since_commit = False
                model_state.speech_since_commit = False
                await self.tasks.cancel_append_tasks()
                had_stream = self._stream_request_id is not None
                cancel_reason = str(decision.get("cancel_reason") or "barge_in")
                cancelled = await self._cancel_active_response(
                    self.tasks.active_response_task,
                    reason=cancel_reason,
                )
                had_stream = self._cancel_data_plane_stream() or had_stream
                if not cancelled and had_stream:
                    old_epoch = session.epoch
                    old_response_id = session.active_response_id
                    committed_ms = session.playback.committed_ms
                    self._commit_played_response_history(session, old_response_id, committed_ms)
                    new_epoch, old_playback = self._advance_barge_in_epoch(session)
                    self.emit(
                        {
                            "type": "audio.cancelled",
                            "session_id": session.session_id,
                            "response_id": old_response_id,
                            "reason": cancel_reason,
                            "cancelled_epoch": old_epoch,
                            "epoch": new_epoch,
                            "committed_ms": committed_ms,
                            "playback": old_playback,
                        }
                    )
                    cancelled = True
                if not cancelled and playback_was_active:
                    old_epoch = session.epoch
                    committed_ms = session.playback.committed_ms
                    self._commit_played_response_history(session, session.last_response_id, committed_ms)
                    new_epoch, old_playback = self._advance_barge_in_epoch(session)
                    self.emit(
                        {
                            "type": "audio.cancelled",
                            "session_id": session.session_id,
                            "response_id": session.last_response_id,
                            "reason": cancel_reason,
                            "cancelled_epoch": old_epoch,
                            "epoch": new_epoch,
                            "committed_ms": committed_ms,
                            "playback": old_playback,
                        }
                    )
                    cancelled = True
                if session.epoch > cancelled_fence.epoch:
                    if not await self._signal_cancel_fence(cancelled_fence):
                        return
                self.tasks.active_response_task = None
        elif not self._session_auto_responds() and not self._input_looks_like_speech(event, payload):
            # Turn-mode only: skip silent chunks so they don't open a response.
            self.emit(
                {
                    "type": "response.listen",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "reason": "silence_or_noise",
                }
            )
            self._maybe_schedule_vad_commit(vad_result)
            return
        if self._should_force_listen_for_auto_response_overlap(event, payload):
            payload["force_listen"] = True
        if not buffer_overlap_audio:
            self._maybe_schedule_vad_commit(vad_result)
            return
        session.mark_user_input_activity()
        model_state.input_since_commit = True
        model_state.speech_since_commit = model_state.speech_since_commit or self._input_looks_like_speech(
            event, payload
        )
        raw_audio_bytes = self._audio_payload_size_bytes(payload)
        try:
            if not session.reserve_input_bytes(
                raw_audio_bytes,
                limit=int(self.manager.runtime_config.max_pending_input_bytes_per_session),
            ):
                self._emit_error("input_backpressure", "Duplex session pending input exceeds server limit")
                return
            # Full-duplex: emit each ~chunk_period of audio so the model runs
            # per-chunk generation without an explicit response.create.
            allow_emit = not defer_append and self._session_auto_responds()
            pcm_reservation = model_state.audio_buffer.prepare_append(
                payload,
                operation_id=uuid.uuid4().hex,
                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                allow_emit=allow_emit,
            )
        except ValueError as exc:
            session.release_input_bytes(raw_audio_bytes)
            self._emit_error("bad_event", str(exc))
            return
        if pcm_reservation is None:
            self._maybe_schedule_vad_commit(vad_result)
            return
        if pcm_reservation.byte_count == 0:
            session.release_input_bytes(raw_audio_bytes)
        payload = pcm_reservation.payload
        await self._start_append(payload, final=False, pcm_reservation=pcm_reservation)
        self._maybe_schedule_vad_commit(vad_result)

    def _maybe_schedule_vad_commit(self, vad_result: TurnDetectionResult | None) -> None:
        """Server VAD ended the user turn: run the same commit the old translator synthesized."""
        if vad_result is None or not vad_result.should_commit:
            return
        command = Commit(final=True, create_response=vad_result.create_response)
        resolved = resolve_commit(self._require_projector(), command)
        self._emit_events(resolved.events)
        if resolved.reset_vad and self._turn_detector is not None:
            self._turn_detector.reset()
        if resolved.payload is not None:
            self._mailbox.put_nowait(_Internal("commit", resolved.payload))

    def _clear_completed_pending_silence(self) -> None:
        task = self.model_state.pending_silence_task
        if task is not None and task.done():
            self.model_state.pending_silence_task = None
            self.model_state.pending_silence_owner_id = None

    def _mark_pending_silence_superseded(self) -> None:
        task = self.model_state.pending_silence_task
        if task is None:
            return
        if task.done():
            self.model_state.pending_silence_task = None
        # Do not cancel the task here: a silence append may already have
        # reached the stage; before_append skips silence that has not started.
        self.model_state.pending_silence_owner_id = None

    def _real_input_waiting(self) -> bool:
        self._clear_completed_pending_silence()
        return self.model_state.audio_buffer.has_pending() or self.model_state.audio_buffer.has_reserved()

    async def _start_append(
        self,
        payload: dict[str, object],
        *,
        final: bool,
        precreate_response: bool = False,
        pcm_reservation: PcmAppendReservation | None = None,
        operation_id: str | None = None,
        retained_committed_payload: dict[str, object] | None = None,
        silence_continuation: bool = False,
        before_append: Callable[[], bool] | None = None,
    ) -> asyncio.Task[bool]:
        session = self.session
        model_state = self.model_state
        if not silence_continuation:
            self._mark_pending_silence_superseded()
        append_epoch = session.epoch
        append_turn_id = payload_turn_id(payload)
        if append_turn_id is None:
            append_turn_id = session.turn_id
        request_id = self._stage0_request_id(append_epoch)
        if final or precreate_response:
            session.bind_request(request_id)
        if precreate_response:
            session.bind_response_turn(append_turn_id)
        if precreate_response and session.active_response_id is None:
            response_id = session.begin_response(turn_id=append_turn_id)
            self.emit(self._response_created_payload(response_id, epoch=append_epoch))
        precreated_response_id = session.active_response_id if precreate_response else None

        def _discard_retained_committed_audio() -> None:
            if (
                retained_committed_payload is not None
                and model_state.committed_audio_payload is retained_committed_payload
            ):
                session.release_input_bytes(model_state.clear_committed_audio())

        async def _run() -> bool:
            try:
                append_ok, emitted_response = await self._append_runtime_input(
                    payload,
                    operation_id=(pcm_reservation.operation_id if pcm_reservation is not None else operation_id),
                    final=final,
                    expected_epoch=append_epoch,
                )
                if append_ok:
                    model_state.context_locked = True
                    if pcm_reservation is not None:
                        pcm_reservation.commit()
                        session.release_input_bytes(pcm_reservation.byte_count)
                    if (
                        retained_committed_payload is not None
                        and model_state.committed_audio_payload is retained_committed_payload
                    ):
                        session.release_input_bytes(model_state.clear_committed_audio())
                else:
                    if pcm_reservation is not None:
                        pcm_reservation.rollback()
                    _discard_retained_committed_audio()
                if (
                    not append_ok
                    and precreated_response_id is not None
                    and session.active_response_id == precreated_response_id
                ):
                    session.end_response(commit_text=False)
                    self.emit(
                        {
                            "type": "response.done",
                            "session_id": session.session_id,
                            "response_id": precreated_response_id,
                            "epoch": session.epoch,
                            "committed": False,
                            "status": "failed",
                            "status_details": {"type": "failed", "reason": "runtime_append_failed"},
                            "playback": session.playback.as_dict(),
                        }
                    )
                if not append_ok and session.state == DuplexSessionState.CLOSED:
                    self._runtime_closed = True
                    return False
                if not emitted_response and session.epoch == append_epoch:
                    if session.active_request_id == self._stage0_request_id(append_epoch):
                        session.clear_request(request_id)
                    if final:
                        self._emit_events([session.signal_turn(DuplexTurnEventType.USER_STARTED.value)])
                return append_ok
            except asyncio.CancelledError:
                if pcm_reservation is not None:
                    pcm_reservation.rollback()
                raise
            except Exception as exc:
                if pcm_reservation is not None:
                    pcm_reservation.rollback()
                _discard_retained_committed_audio()
                logger.exception("Native duplex append task failed: %s", exc)
                self._send_runtime_error("runtime_append_task_failed", exc)
                if session.state != DuplexSessionState.CLOSED:
                    self._begin_close("runtime_append_task_failed")
                    self._spawn(self._close_from_runtime("runtime_append_task_failed"), name="duplex-runtime-close")
                return False

        async def _run_in_wire_order(predecessor: asyncio.Task[bool] | None) -> bool:
            if predecessor is not None:
                try:
                    predecessor_ok = await predecessor
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    predecessor_ok = False
                except Exception:
                    predecessor_ok = False
                if not predecessor_ok:
                    if pcm_reservation is not None:
                        pcm_reservation.rollback()
                    _discard_retained_committed_audio()
                    return False
            if self._closing or self._runtime_closed or session.state != DuplexSessionState.OPEN:
                if pcm_reservation is not None:
                    pcm_reservation.rollback()
                _discard_retained_committed_audio()
                return False
            if before_append is not None and not before_append():
                if pcm_reservation is not None:
                    pcm_reservation.rollback()
                _discard_retained_committed_audio()
                return True
            if pcm_reservation is not None and not pcm_reservation.active:
                _discard_retained_committed_audio()
                return False
            return await _run()

        def _release_cancelled_retained_audio(done: asyncio.Task[bool]) -> None:
            if done.cancelled():
                _discard_retained_committed_audio()
                return
            try:
                append_ok = done.result()
            except Exception:
                append_ok = False
            if not append_ok:
                _discard_retained_committed_audio()

        predecessor = self.tasks.append_tail
        if predecessor is not None and predecessor.done():
            try:
                predecessor_ok = predecessor.result()
            except (asyncio.CancelledError, Exception):
                predecessor_ok = False
            if not predecessor_ok:
                # Appends queued behind a failed predecessor stop; a later
                # command is an explicit retry and starts a new chain.
                predecessor = None
        task = asyncio.create_task(_run_in_wire_order(predecessor))
        task.add_done_callback(_release_cancelled_retained_audio)
        self.tasks.append_tail = task
        self.tasks.track_append_task(
            task,
            epoch=append_epoch,
            final=final,
            response_bound=final or precreate_response,
        )
        if silence_continuation:
            model_state.pending_silence_task = task

            def _clear_done_pending_silence(done: asyncio.Task[bool]) -> None:
                if model_state.pending_silence_task is done:
                    model_state.pending_silence_task = None
                    model_state.pending_silence_owner_id = None

            task.add_done_callback(_clear_done_pending_silence)
        # Let this wire-order effect start before the next mailbox item can cancel it.
        await asyncio.sleep(0)
        return task

    async def _wait_for_append_tail(self) -> bool:
        predecessor = self.tasks.append_tail
        if predecessor is None:
            return True
        try:
            return await predecessor
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return False
        except Exception:
            return False

    async def _schedule_silence_continuation(
        self,
        payload: object,
        *,
        request_id: str,
        owner_id: str,
        response_id: str | None,
        response_owned: bool,
        expected_epoch: int | None,
        expected_model_turn_id: int | None,
    ) -> bool:
        session = self.session
        model_state = self.model_state
        self._clear_completed_pending_silence()
        pending_silence = model_state.pending_silence_task
        if pending_silence is not None and not pending_silence.done():
            if pending_silence is asyncio.current_task():
                return False
            try:
                if not await pending_silence:
                    return False
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                return False
            except Exception:
                return False
            self._clear_completed_pending_silence()
            pending_silence = model_state.pending_silence_task
            if pending_silence is not None and not pending_silence.done():
                return False
        append_tail = self.tasks.append_tail
        if (append_tail is None or append_tail.done()) and self._real_input_waiting():
            return False
        continuation_delay_s = max(0.0, float(session.capabilities.chunk_period_ms or 1000) / 1000.0)
        if continuation_delay_s > 0:
            await asyncio.sleep(continuation_delay_s)
            if (
                self.tasks.append_tail is not append_tail
                or ((append_tail is None or append_tail.done()) and self._real_input_waiting())
                or self._silence_continuation_is_stale(
                    request_id=request_id,
                    response_id=response_id,
                    response_owned=response_owned,
                    expected_epoch=expected_epoch,
                    expected_model_turn_id=expected_model_turn_id,
                )
            ):
                return False

        def _still_valid() -> bool:
            return not self._real_input_waiting() and not self._silence_continuation_is_stale(
                request_id=request_id,
                response_id=response_id,
                response_owned=response_owned,
                expected_epoch=expected_epoch,
                expected_model_turn_id=expected_model_turn_id,
            )

        model_state.pending_silence_owner_id = owner_id
        task = await self._start_append(
            dict(payload) if isinstance(payload, dict) else {},
            final=False,
            silence_continuation=True,
            before_append=_still_valid,
        )
        return task is not None

    # ------------------------------------------------------------------ #
    # Engine bridge: plan an append with the plugin and submit it        #
    # ------------------------------------------------------------------ #

    async def _append_runtime_input(
        self,
        payload: object,
        *,
        operation_id: str | None = None,
        final: bool,
        expected_epoch: int | None = None,
    ) -> tuple[bool, bool]:
        session = self.session
        if not session.capabilities.supports_input_append:
            return True, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        try:
            result = await self._append_via_data_plane(
                payload,
                final=final,
                operation_id=operation_id,
                expected_epoch=expected_epoch,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to append duplex runtime input: %s", exc)
            self._send_runtime_error("runtime_append_failed", exc)
            return False, False
        if result is None:
            return True, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        request_id, _ = duplex_data_plane_request_info(result)
        if request_id is not None:
            self.plugin.data_plane.begin_request(request_id)
        close_reason, emitted_response = await self._send_model_output_events(result, expected_epoch=expected_epoch)
        emitted_response = emitted_response or request_id is not None
        if close_reason is None and self._start_data_plane_stream(result):
            emitted_response = True
        if close_reason is not None:
            await self._close_from_runtime(close_reason)
            return False, emitted_response
        return True, emitted_response

    async def _append_via_data_plane(
        self,
        payload: object,
        *,
        final: bool,
        operation_id: str | None,
        expected_epoch: int | None,
    ) -> dict[str, object] | None:
        """Submit one planned append to the stage port and bind the resulting stage request."""
        session = self.session
        if self.stage_port.stage_count == 0:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        payload_turn = payload_turn_id(payload)
        fence = DuplexFence(
            session.session_id,
            epoch=session.epoch,
            turn_id=(
                payload_turn
                if payload_turn is not None
                else (
                    session.active_response_turn_id if session.active_response_turn_id is not None else session.turn_id
                )
            ),
        )
        lease_operation_id = f"append:{operation_id or uuid.uuid4().hex}"
        operation_started = False
        stage_id = 0
        request_id = self.manager.stage_request_id(fence, stage_id=stage_id)
        try:
            session.begin_lease_operation(fence, lease_operation_id)
            operation_started = True
            reservation = session.prepare_append(fence)
            already_submitted = session.stage_request_submitted(stage_id, request_id)
            request_context = self.manager.ensure_stage_request(session, stage_id=stage_id, fence=fence)
            if request_context is None:
                raise RuntimeError("duplex_data_plane_has_no_stage")
            append_plan = self.plugin.plan_append(
                request_id=request_id,
                fence=fence,
                session_config=dict(request_context.session_config),
                runtime_config=dict(request_context.runtime_config),
                seq=reservation.update.seq,
                turn_seq=reservation.update.turn_seq,
                payload=payload,
                final=final,
                sampling_params=request_context.stage_sampling_params,
            )
            if not isinstance(append_plan, DuplexAppendPlan):
                raise TypeError("duplex plugin plan_append() must return DuplexAppendPlan")
            submission = DuplexStageSubmission(
                context=request_context,
                prompt=append_plan.prompt,
                already_submitted=already_submitted,
            )
            submission_result = await self.stage_port.submit(submission)
            try:
                if submission_result.request_id != request_id or submission_result.stage_id != stage_id:
                    raise RuntimeError("duplex stage adapter returned a mismatched submission result")
                if expected_epoch is not None and session.epoch != expected_epoch:
                    raise DuplexFenceMismatchError(session.fence, fence)
                update = session.commit_append(reservation)
                session.bind_stage_request(stage_id, request_id, fence=fence)
            except BaseException:
                try:
                    await self.stage_port.cleanup([request_id])
                except Exception as cleanup_exc:
                    logger.warning(
                        "duplex append compensation remains pending for session %s: %s",
                        session.session_id,
                        cleanup_exc,
                    )
                raise
            session.touch_lease(DuplexLeaseActivity.APPEND)
            return {
                "ok": True,
                "operation": "append",
                "session_id": session.session_id,
                "stage_results": [
                    {
                        "stage_id": stage_id,
                        "replica_id": submission_result.replica_id,
                        "result": {
                            "supported": True,
                            "data_plane_append": True,
                            "request_id": request_id,
                            "response_stage_id": request_context.final_stage_id,
                            "seq": update.seq,
                            "turn_id": update.turn_id,
                            "turn_seq": update.turn_seq,
                            "resumable": True,
                        },
                    }
                ],
            }
        finally:
            if operation_started and lease_operation_id in session.lease.active_operations:
                try:
                    session.end_lease_operation(lease_operation_id)
                except Exception:
                    session.lease.active_operations.discard(lease_operation_id)

    def _start_data_plane_stream(self, result: object) -> bool:
        """Bind the resumable stage request as the session's active data-plane stream."""
        request_id, _ = duplex_data_plane_request_info(result) if isinstance(result, dict) else (None, None)
        if request_id is None or self._data_plane_outputs_finished(result):
            return False
        session = self.session
        session.bind_request(request_id)
        if self._stream_request_id == request_id:
            return False
        self._stream_request_id = request_id
        return True

    def _cancel_data_plane_stream(self) -> bool:
        had_stream = self._stream_request_id is not None
        self._stream_request_id = None
        return had_stream

    async def _signal_cancel_fence(self, cancelled_fence: DuplexFence) -> bool:
        """In-process equivalent of the old ``barge_in`` control signal."""
        session = self.session
        try:
            next_fence = session.sync_fence()
            if next_fence.epoch > cancelled_fence.epoch:
                stale_request_ids = session.cancel_fence(cancelled_fence, next_fence)
            else:
                stale_request_ids = []
            if stale_request_ids:
                await self.stage_port.cleanup(list(stale_request_ids), abort=True)
            session.touch_lease(DuplexLeaseActivity.SIGNAL)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to signal duplex runtime session: %s", exc)
            self._send_runtime_error("runtime_signal_failed", exc)
            return False
        return True

    async def _close_from_runtime(self, reason: str) -> None:
        """A runtime failure closed the session from inside the engine."""
        session = self.session
        if session.state == DuplexSessionState.CLOSED:
            return
        self._runtime_closed = True
        self._begin_close(reason)
        self._cleanup_duplex_session_state()
        if not self._closed_emitted:
            self._closed_emitted = True
            self.emit({"type": "session.closed", "session_id": session.session_id, "reason": reason})
        session.close()
        # The manager aborts the stage requests (if any) and frees the
        # admission slot; a session without stage requests must not stay
        # registered until its idle TTL.
        self.manager.close_from_runner(self, reason)

    def _send_runtime_error(self, code: str, exc: BaseException) -> None:
        if isinstance(exc, DuplexRuntimeConfigError):
            code = exc.code
        self._emit_error(code, str(exc), retryable=False)

    @staticmethod
    def _data_plane_outputs_finished(result: object) -> bool:
        if not isinstance(result, dict):
            return False
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        return bool(getattr(outputs[-1], "finished", False))

    def _runtime_data_plane_context(self) -> object:
        session = self.session
        response_config = session.response_config
        return self.plugin.data_plane_context(
            epoch=session.epoch,
            turn_id=session.turn_id,
            active_response_turn_id=session.active_response_turn_id,
            active_response_id=session.active_response_id,
            auto_responds=self._session_auto_responds(),
            response_format=response_config.response_format,
            speed=response_config.speed,
            modalities=tuple(response_config.modalities),
        )

    # ------------------------------------------------------------------ #
    # Stage outputs (was collect_outputs + drain + _send_one_native...)  #
    # ------------------------------------------------------------------ #

    def _decide_output(
        self, stage_id: int, output: RequestOutput, context: DuplexOutputContext
    ) -> DuplexOutputDecision | None:
        """Pure plugin decision (no session mutation: this runs inline on the orchestrator loop)."""
        decision = self.plugin.decide_output(
            stage_id=stage_id,
            final_stage_id=context.final_stage_id,
            segment_finished=context.segment_finished,
            segment_token_ids=context.segment_token_ids,
            segment_output_metadata=dict(context.segment_output_metadata),
            output=output,
        )
        if decision is not None and not isinstance(decision, DuplexOutputDecision):
            raise TypeError("duplex plugin decide_output() must return DuplexOutputDecision or None")
        return decision

    @staticmethod
    def _stage_metrics_snapshot(stage_id: int, metrics: object, output: object) -> dict[str, dict[str, object]] | None:
        if not isinstance(metrics, StageRequestStats):
            return None
        event = metrics
        if event.stage_id is None:
            event = replace(event, stage_id=stage_id)
        if event.final_output_type is None:
            final_output_type = getattr(output, "final_output_type", None)
            if isinstance(final_output_type, str):
                event = replace(event, final_output_type=final_output_type)
        try:
            merged = OrchestratorAggregator._merge_stage_metric_event(None, event)
        except Exception:
            return None
        return {str(stage_id): merged}

    def _build_stage_output(self, item: _StageOutput) -> OmniRequestOutput:
        output = item.output
        finished = bool(getattr(output, "finished", False)) or item.context.segment_finished
        if item.decision is not None:
            engine_output = OmniRequestOutput.from_stage_output(
                output,
                request_id=item.request_id,
                finished=True,
                stage_id=item.stage_id,
                final_output_type=item.decision.final_output_type,
            )
            engine_output = attach_duplex_output_decision(engine_output, item.decision)
        elif isinstance(output, OmniRequestOutput):
            engine_output = output
            if finished:
                engine_output.finished = True
        else:
            final_output_type = getattr(output, "final_output_type", None)
            engine_output = OmniRequestOutput.from_stage_output(
                output,
                request_id=item.request_id,
                finished=finished,
                stage_id=item.stage_id,
                final_output_type=final_output_type if isinstance(final_output_type, str) else "audio",
            )
        snapshot = self._stage_metrics_snapshot(item.stage_id, item.metrics, output)
        if snapshot is not None:
            existing = engine_output.metrics if isinstance(engine_output.metrics, dict) else {}
            existing_stage_metrics = existing.get("stage_metrics")
            merged_stage_metrics = (
                {**existing_stage_metrics, **snapshot} if isinstance(existing_stage_metrics, dict) else snapshot
            )
            engine_output.metrics = {**existing, "stage_metrics": merged_stage_metrics}
        return engine_output

    async def _on_stage_output_item(self, item: _StageOutput) -> None:
        session = self.session
        if session.state == DuplexSessionState.CLOSED or self._closing:
            return
        expected_epoch = item.context.identity.fence.epoch
        if session.epoch != expected_epoch:
            return
        if session.lease.terminal_reason is not None:
            # Lease already closed/expired: the output belongs to a dead session.
            return
        session.touch_lease(DuplexLeaseActivity.MODEL_OUTPUT)
        if self.plugin.data_plane.is_terminal(item.request_id):
            return
        if self._session_auto_responds():
            active_request_id = session.active_request_id
            if active_request_id is not None and active_request_id != item.request_id:
                return
        engine_output = self._build_stage_output(item)
        drain_result = {"data_plane_outputs": [engine_output]}
        close_reason, emitted_response = await self._send_model_output_events(
            drain_result, expected_epoch=expected_epoch
        )
        if close_reason is not None:
            await self._close_from_runtime(close_reason)
            return
        finished = self._data_plane_outputs_finished(drain_result)
        if finished and emitted_response and not self._session_auto_responds():
            # A finished, emitted response releases the per-request projector
            # cursor on its way out and offers the model another
            # silence unit.
            self.plugin.data_plane.close_stream(item.request_id)
            if self._stream_request_id == item.request_id:
                self._stream_request_id = None
            self._spawn(
                self._maybe_continue_response(expected_epoch=expected_epoch),
                name="duplex-continue",
            )

    async def _send_model_output_events(
        self,
        result: object,
        *,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        session = self.session
        if expected_epoch is not None and session.epoch != expected_epoch:
            return None, False
        close_reason: str | None = None
        emitted_response = False
        request_id, _ = duplex_data_plane_request_info(result) if isinstance(result, dict) else (None, None)
        if self.plugin.data_plane.is_terminal(request_id):
            return None, False
        if request_id is not None and session.active_request_id is None:
            session.bind_request(request_id)
        context = self._runtime_data_plane_context()
        for model_result in self.plugin.data_plane.project(result, context=context):
            close_reason_for_result, did_emit = await self._send_one_model_output_event(
                model_result,
                expected_epoch=expected_epoch,
            )
            emitted_response = emitted_response or did_emit
            close_reason = close_reason or close_reason_for_result
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None, emitted_response
        return close_reason, emitted_response

    async def _send_one_model_output_event(
        self,
        model_result: dict[str, object],
        *,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        session = self.session
        model_state = self.model_state
        data_plane = self.plugin.data_plane
        close_reason: str | None = None
        emitted_response = False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return close_reason, emitted_response
        data_plane_request_id = model_result.get("data_plane_request_id")
        if isinstance(data_plane_request_id, str) and data_plane.is_terminal(data_plane_request_id):
            return close_reason, emitted_response
        auto_response = self._session_auto_responds()
        active_request_matches = session.active_request_id == data_plane_request_id or (
            auto_response and session.active_request_id is None
        )
        if isinstance(data_plane_request_id, str) and not active_request_matches:
            return close_reason, emitted_response
        if isinstance(model_result.get("error_code"), str):
            response_id = session.active_response_id
            self._emit_error(
                str(model_result.get("error_code")),
                str(model_result.get("error") or "Duplex native data-plane error"),
            )
            if response_id is not None:
                session.end_response(commit_text=False)
                self.emit(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": False,
                        "status": "failed",
                        "status_details": {"type": "failed", "reason": model_result.get("error_code")},
                        "playback": session.playback.as_dict(),
                    }
                )
            return close_reason, True
        if model_result.get("function_call") is True:
            self.emit(
                {
                    "type": "function_call.done",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "call_id": model_result.get("call_id"),
                    "name": model_result.get("name"),
                    "arguments": model_result.get("arguments", ""),
                }
            )
            return close_reason, True
        if model_result.get("requires_stage_handoff") is True or model_result.get("requires_tts_stage") is True:
            # Reserve the protocol response only when Stage1 emits text/audio.
            return close_reason, False
        is_listen = model_result.get("is_listen")
        model_turn_id = coerce_int(model_result.get("model_turn_id"))
        if model_result.get("is_buffering") is True or model_result.get("prefill_success") is False:
            if model_result.get("data_plane_request_id") == session.active_request_id:
                session.clear_request()
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "reason": model_result.get("reason") or "buffering",
                "model_listen": False,
                "buffering": True,
            }
            self._attach_runtime_metadata(payload, model_result)
            self.emit(payload)
            return close_reason, emitted_response
        if is_listen is True:
            self._end_active_response_before_future_model_turn(model_turn_id=model_turn_id)
            if (
                session.active_response_id is not None
                and model_turn_id is not None
                and not session.active_response_accepts_model_turn(model_turn_id)
            ):
                return close_reason, emitted_response
            active_response_id = session.active_response_id
            auto_continuations_remaining = active_response_id is None or self._response_continuations_remaining(
                active_response_id
            )
            non_terminal_auto_listen = (
                auto_response
                and active_response_id is not None
                and session.active_request_id is not None
                and model_result.get("end_of_turn") is not True
                and auto_continuations_remaining
            )
            if non_terminal_auto_listen:
                self._spawn(self._maybe_continue_response(expected_epoch=expected_epoch), name="duplex-continue")
                return close_reason, emitted_response
            if not auto_response and data_plane_request_id == session.active_request_id:
                session.clear_request()
            model_listen = model_result.get("model_listen")
            if not isinstance(model_listen, bool):
                model_listen = model_result.get("reason") in {None, "", "model_listen"}
            response_id = session.active_response_id
            if isinstance(data_plane_request_id, str) and not auto_response:
                data_plane.mark_terminal(data_plane_request_id)
            emitted_response = True
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "reason": model_result.get("reason") or "model_listen",
                "model_listen": model_listen,
            }
            if response_id is not None:
                payload["response_id"] = response_id
            self._attach_runtime_metadata(payload, model_result)
            self.emit(payload)
            if model_result.get("abort_data_plane_request") is True and isinstance(data_plane_request_id, str):
                await self._abort_request_background(data_plane_request_id, notify=False)
            if response_id is not None:
                if not auto_response and self._response_continuations_remaining(response_id):
                    self._spawn(self._maybe_continue_response(expected_epoch=expected_epoch), name="duplex-continue")
                    return close_reason, emitted_response
                if auto_response:
                    model_state.clear_continuation()
                    if not auto_continuations_remaining:
                        completed_turn_id = model_turn_id
                        if completed_turn_id is None:
                            completed_turn_id = session.active_response_turn_id
                        if completed_turn_id is not None:
                            session.complete_model_turn(completed_turn_id)
                session.end_response(commit_text=False, preserve_request=auto_response)
                self.emit(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": False,
                        "playback": session.playback.as_dict(),
                    }
                )
            return close_reason, emitted_response

        text = model_result.get("text")
        audio = model_result.get("audio_data", model_result.get("audio"))
        end_of_turn = bool(model_result.get("end_of_turn", False))
        has_text = isinstance(text, str) and bool(text)
        has_audio = isinstance(audio, str) and bool(audio)
        if not has_text and not has_audio and not end_of_turn:
            tts_segment_ended = (
                model_result.get("stage_role") == "tts" and model_result.get("abort_data_plane_request") is True
            )
            if (
                tts_segment_ended
                and auto_response
                and (model_turn_id is not None or session.active_response_id is not None)
            ):
                self._spawn(
                    self._maybe_continue_response(expected_epoch=expected_epoch, expected_model_turn_id=model_turn_id),
                    name="duplex-continue",
                )
            return close_reason, emitted_response
        if end_of_turn and not has_text and not has_audio and session.active_response_id is None:
            if isinstance(data_plane_request_id, str):
                if not auto_response and data_plane_request_id == session.active_request_id:
                    session.clear_request()
                if not auto_response:
                    data_plane.mark_terminal(data_plane_request_id)
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            if auto_response:
                model_state.clear_continuation()
                emitted_response = True
                payload = {
                    "type": "response.listen",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "reason": "model_turn_completed_without_output",
                    "model_listen": True,
                }
                self._attach_runtime_metadata(payload, model_result)
                self.emit(payload)
            return close_reason, emitted_response
        if session.active_response_id is None and model_turn_id is not None and model_turn_id < session.turn_id:
            # Late audio of a completed model turn must not reserve a second response.
            return close_reason, emitted_response
        self._end_active_response_before_future_model_turn(model_turn_id=model_turn_id)
        if (
            session.active_response_id is not None
            and model_turn_id is not None
            and not session.active_response_accepts_model_turn(model_turn_id)
        ):
            return close_reason, emitted_response
        emitted_response = True
        response_created = False
        response_id = session.active_response_id
        if response_id is None:
            response_id = session.begin_response(turn_id=model_turn_id)
            response_created = True
            self.emit(self._response_created_payload(response_id, epoch=session.epoch))
        response_stage_metrics = session.accumulate_response_stage_metrics(
            model_result.get("stage_metrics") if isinstance(model_result.get("stage_metrics"), Mapping) else None
        )
        if response_created:
            speak_payload = {
                "type": "response.speak",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "text": text if isinstance(text, str) else "",
                "end_of_turn": end_of_turn,
                "model_speak": True,
            }
            self._attach_runtime_metadata(speak_payload, model_result, stage_metrics=response_stage_metrics)
            self.emit(speak_payload)
        previous_sent_ms = session.playback.sent_ms
        text_chars_before_append = len("".join(session.assistant_text_buffer))
        if isinstance(text, str):
            session.append_assistant_text(text)
        duration_ms = model_result.get("audio_duration_ms")
        text_chars = len("".join(session.assistant_text_buffer))
        mark_duration_ms = None
        mark_text_chars: int | None = text_chars
        if model_result.get("audio_text_mark") is False:
            mark_text_chars = None
        if isinstance(duration_ms, int | float):
            mark_duration_ms = int(duration_ms)
            if model_result.get("audio_duration_is_cumulative") is not True:
                mark_duration_ms += session.playback.sent_ms
        audio_text_marks = model_result.get("audio_text_marks")
        audio_text_marks = self._normalize_audio_text_marks(
            audio_text_marks if isinstance(audio_text_marks, list) else None,
            audio_offset_ms=(
                0
                if model_result.get("audio_text_marks_are_cumulative") is True
                or model_result.get("audio_duration_is_cumulative") is True
                else previous_sent_ms
            ),
            text_offset_chars=(
                0 if model_result.get("audio_text_marks_are_cumulative") is True else text_chars_before_append
            ),
        )
        session.mark_audio_sent(
            mark_duration_ms,
            text_chars=mark_text_chars if mark_duration_ms is not None else None,
            audio_text_marks=audio_text_marks,
        )
        payload = {
            "type": "response.output_audio.delta",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": session.epoch,
            "text": text if isinstance(text, str) else "",
            "audio": audio if isinstance(audio, str) else "",
            "format": (
                model_result.get("audio_format")
                if isinstance(model_result.get("audio_format"), str)
                else session.response_config.response_format
            ),
            "end_of_turn": end_of_turn,
            "model_speak": True,
        }
        if mark_duration_ms is not None:
            payload["audio_duration_ms"] = mark_duration_ms
        if audio_text_marks:
            payload["audio_text_marks"] = audio_text_marks
        elif mark_duration_ms is not None and mark_text_chars is not None:
            payload["audio_text_marks"] = [
                {"text_chars": max(0, int(mark_text_chars)), "audio_end_ms": max(0, int(mark_duration_ms))}
            ]
        payload["playback"] = session.playback.as_dict()
        sample_rate_hz = model_result.get("sample_rate_hz") or model_result.get("audio_sample_rate_hz")
        if isinstance(sample_rate_hz, int | float) and int(sample_rate_hz) > 0:
            payload["sample_rate_hz"] = int(sample_rate_hz)
        self._attach_runtime_metadata(payload, model_result, stage_metrics=response_stage_metrics)
        self.emit(payload)
        if (
            not end_of_turn
            and model_result.get("stage_role") == "tts"
            and model_result.get("abort_data_plane_request") is True
            and auto_response
        ):
            self._spawn(self._maybe_continue_response(expected_epoch=expected_epoch), name="duplex-continue")
        if end_of_turn:
            data_plane_request_id = model_result.get("data_plane_request_id")
            if isinstance(data_plane_request_id, str) and not auto_response:
                data_plane.close_stream(data_plane_request_id)
            if isinstance(data_plane_request_id, str):
                if not auto_response and data_plane_request_id == session.active_request_id:
                    session.clear_request()
                if not auto_response:
                    data_plane.mark_terminal(data_plane_request_id)
            should_commit = self._should_commit_response_to_history(session, response_id)
            committed_message = session.end_response(commit_text=should_commit, preserve_request=auto_response)
            model_turn_id = coerce_int(model_result.get("model_turn_id"))
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            if should_commit:
                session.register_history_item(f"item_{response_id}", committed_message)
            self.emit(
                {
                    "type": "response.done",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": session.epoch,
                    "committed": committed_message is not None,
                    "playback": session.playback.as_dict(),
                }
            )
        return close_reason, emitted_response

    def _end_active_response_before_future_model_turn(self, *, model_turn_id: int | None) -> None:
        session = self.session
        if not self._session_auto_responds():
            return
        response_id = session.active_response_id
        active_turn_id = session.active_response_turn_id
        if response_id is None or model_turn_id is None or active_turn_id is None:
            return
        if int(model_turn_id) <= int(active_turn_id):
            return
        session.complete_model_turn(int(model_turn_id) - 1)
        should_commit = self._should_commit_response_to_history(session, response_id)
        committed_message = session.end_response(commit_text=should_commit, preserve_request=True)
        if should_commit:
            session.register_history_item(f"item_{response_id}", committed_message)
        self.emit(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "committed": committed_message is not None,
                "playback": session.playback.as_dict(),
            }
        )

    @staticmethod
    def _normalize_audio_text_marks(
        audio_text_marks: list[object] | None,
        *,
        audio_offset_ms: int,
        text_offset_chars: int,
    ) -> list[dict[str, int]] | None:
        if not audio_text_marks:
            return None
        normalized: list[dict[str, int]] = []
        for raw_mark in audio_text_marks:
            if not isinstance(raw_mark, dict):
                continue
            raw_text_chars = raw_mark.get("text_chars")
            raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
            if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                continue
            normalized.append(
                {
                    "text_chars": max(0, int(raw_text_chars) + int(text_offset_chars)),
                    "audio_end_ms": max(0, int(raw_audio_end_ms) + int(audio_offset_ms)),
                }
            )
        return normalized or None

    @staticmethod
    def _attach_runtime_metadata(
        payload: dict[str, object],
        model_result: dict[str, object],
        *,
        stage_metrics: Mapping[str, object] | None = None,
    ) -> None:
        metadata: dict[str, object] = {}
        runtime_impl = model_result.get("runtime_impl")
        if isinstance(runtime_impl, str) and runtime_impl:
            metadata["runtime_impl"] = runtime_impl
        owned_runtime = model_result.get("owned_runtime")
        if isinstance(owned_runtime, bool):
            metadata["owned_runtime"] = owned_runtime
        model_turn_id = coerce_int(model_result.get("model_turn_id"))
        if model_turn_id is not None:
            metadata["model_turn_id"] = model_turn_id
        for name in ("uses_model_runner_scheduler", "runner_kv_backed"):
            value = model_result.get(name)
            if isinstance(value, bool):
                metadata[name] = value
        effective_stage_metrics = stage_metrics if stage_metrics is not None else model_result.get("stage_metrics")
        if isinstance(effective_stage_metrics, Mapping):
            metadata["stage_metrics"] = {
                str(stage_id): dict(values)
                for stage_id, values in effective_stage_metrics.items()
                if isinstance(values, Mapping)
            }
        if metadata:
            payload["vllm_omni"] = metadata

    # ------------------------------------------------------------------ #
    # Silence continuation                                               #
    # ------------------------------------------------------------------ #

    def _silence_unit_payload(self) -> dict[str, object]:
        samples = int(self.plugin.silence_continuation_samples)
        audio = (
            self._NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO
            if samples == 16000
            else base64.b64encode(bytes(samples * 4)).decode("ascii")
        )
        return {"type": "audio", "audio": audio, "format": "pcm_f32le", "sample_rate_hz": 16000}

    def _response_continuations_remaining(self, response_id: str) -> bool:
        model_state = self.model_state
        owner_id = f"response:{response_id}"
        count = model_state.continuation_units if model_state.continuation_owner_id == owner_id else 0
        limit = (
            self._NATIVE_AUTO_RESPONSE_MAX_CONTINUATION_UNITS
            if self._session_auto_responds()
            else self._NATIVE_RESPONSE_MAX_CONTINUATION_UNITS
        )
        return count < limit

    async def _finish_bounded_auto_response(
        self,
        *,
        expected_epoch: int | None,
        model_turn_id: int | None = None,
    ) -> None:
        session = self.session
        model_state = self.model_state
        response_id = session.active_response_id
        response_epoch = session.epoch
        response_turn_id = model_turn_id if model_turn_id is not None else session.active_response_turn_id
        if expected_epoch is not None and response_epoch != expected_epoch:
            return
        model_state.clear_continuation()
        payload: dict[str, object] = {
            "type": "response.listen",
            "session_id": session.session_id,
            "epoch": response_epoch,
            "reason": "continuation_limit",
            "model_listen": False,
        }
        if response_id is not None:
            payload["response_id"] = response_id
        self.emit(payload)
        if response_id is None:
            if session.epoch == response_epoch and response_turn_id is not None:
                session.complete_model_turn(response_turn_id)
            return
        if session.epoch != response_epoch or session.active_response_id != response_id:
            return
        if response_turn_id is not None:
            session.complete_model_turn(response_turn_id)
        session.end_response(commit_text=False, preserve_request=True)
        self.emit(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": response_epoch,
                "committed": False,
                "playback": session.playback.as_dict(),
            }
        )

    def _silence_continuation_is_stale(
        self,
        *,
        request_id: str,
        response_id: str | None,
        response_owned: bool,
        expected_epoch: int | None,
        expected_model_turn_id: int | None,
    ) -> bool:
        session = self.session
        stale_common_owner = (
            session.state == DuplexSessionState.CLOSED
            or session.active_request_id != request_id
            or (expected_epoch is not None and session.epoch != expected_epoch)
        )
        stale_response_owner = response_owned and session.active_response_id != response_id
        stale_model_turn_owner = not response_owned and (
            session.active_response_id is not None or session.turn_id != expected_model_turn_id
        )
        return stale_common_owner or stale_response_owner or stale_model_turn_owner

    async def _maybe_continue_response(
        self,
        *,
        expected_epoch: int | None,
        expected_model_turn_id: int | None = None,
    ) -> None:
        session = self.session
        model_state = self.model_state
        response_id = session.active_response_id
        if session.state == DuplexSessionState.CLOSED or self._closing:
            model_state.clear_continuation()
            return
        request_id = session.active_request_id
        if request_id is None:
            model_state.clear_continuation()
            return
        if expected_epoch is not None and session.epoch != expected_epoch:
            return
        auto_response = self._session_auto_responds()
        response_owned = response_id is not None
        if response_owned:
            owner_id = f"response:{response_id}"
            payload_turn_id_value = (
                session.active_response_turn_id if session.active_response_turn_id is not None else session.turn_id
            )
        else:
            if not auto_response or expected_model_turn_id is None or session.turn_id != expected_model_turn_id:
                model_state.clear_continuation()
                return
            owner_id = f"model-turn:{expected_model_turn_id}"
            payload_turn_id_value = expected_model_turn_id
        count = model_state.continuation_units if model_state.continuation_owner_id == owner_id else 0
        continuation_limit = (
            self._NATIVE_AUTO_RESPONSE_MAX_CONTINUATION_UNITS
            if auto_response
            else self._NATIVE_RESPONSE_MAX_CONTINUATION_UNITS
        )
        if count >= continuation_limit:
            if auto_response:
                await self._finish_bounded_auto_response(
                    expected_epoch=expected_epoch,
                    model_turn_id=payload_turn_id_value,
                )
            return
        payload = self._silence_unit_payload()
        if auto_response and count + 1 == continuation_limit:
            payload["force_listen"] = True
        payload["duplex_turn_id"] = payload_turn_id_value
        try:
            scheduled = await self._schedule_silence_continuation(
                payload,
                request_id=request_id,
                owner_id=owner_id,
                response_id=response_id,
                response_owned=response_owned,
                expected_epoch=expected_epoch,
                expected_model_turn_id=expected_model_turn_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to schedule duplex native response continuation: %s", exc)
            scheduled = False
        if scheduled:
            model_state.continuation_owner_id = owner_id
            model_state.continuation_units = count + 1

    # ------------------------------------------------------------------ #
    # Cancel / barge-in                                                  #
    # ------------------------------------------------------------------ #

    async def _on_cancel(self, event: dict[str, object]) -> None:
        session = self.session
        model_state = self.model_state
        event_type = str(event.get("type"))
        if event_type == "barge_in" and not session.capabilities.supports_barge_in:
            self._emit_events([self._barge_in_unsupported_error()])
            return
        cancel_reason = (
            "output_audio_buffer_clear"
            if event_type == "output_audio_buffer.clear"
            else "client_cancelled"
            if event_type == "response.cancel"
            else "barge_in"
        )
        cancelled_fence = session.fence
        if event_type == "response.cancel":
            requested_response_id = event.get("response_id")
            has_active_response_work = self._response_in_progress()
            if (
                isinstance(requested_response_id, str)
                and session.active_response_id is not None
                and requested_response_id != session.active_response_id
            ):
                self._emit_error(
                    "response_not_active",
                    f"Response is not active: {requested_response_id}",
                    event_id=event.get("realtime_event_id"),
                )
                return
            if not has_active_response_work:
                if isinstance(requested_response_id, str):
                    return
                self._emit_error(
                    "response_not_active",
                    "response.cancel requires an active response",
                    event_id=event.get("realtime_event_id"),
                )
                return
        had_unbuffered_append = model_state.input_since_commit and not model_state.audio_buffer.has_pending()
        playback_was_active = self._assistant_playback_active()
        if event_type in {"input.cancel", "barge_in"}:
            model_state.audio_buffer.clear()
            session.release_all_input_bytes()
            model_state.input_since_commit = False
            model_state.speech_since_commit = False
            model_state.clear_committed_audio()
        had_append = await self.tasks.cancel_append_tasks(
            response_bound_only=event_type in {"response.cancel", "output_audio_buffer.clear"},
        )
        if event_type == "response.cancel":
            session.release_input_bytes(model_state.clear_committed_audio())
        had_stream = self._stream_request_id is not None
        cancelled = await self._cancel_active_response(self.tasks.active_response_task, reason=cancel_reason)
        had_stream = self._cancel_data_plane_stream() or had_stream
        if not cancelled and (had_stream or had_append or had_unbuffered_append):
            old_epoch = session.epoch
            old_response_id = session.active_response_id
            committed_ms = session.playback.committed_ms
            self._commit_played_response_history(session, old_response_id, committed_ms)
            new_epoch, old_playback = self._advance_barge_in_epoch(session)
            self.emit(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": old_response_id,
                    "reason": cancel_reason,
                    "cancelled_epoch": old_epoch,
                    "epoch": new_epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
            cancelled = True
        if not cancelled and playback_was_active:
            old_epoch = session.epoch
            committed_ms = session.playback.committed_ms
            self._commit_played_response_history(session, session.last_response_id, committed_ms)
            new_epoch, old_playback = self._advance_barge_in_epoch(session)
            self.emit(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": session.last_response_id,
                    "reason": cancel_reason,
                    "cancelled_epoch": old_epoch,
                    "epoch": new_epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
            cancelled = True
        if not cancelled and event_type == "response.cancel":
            old_epoch = session.epoch
            old_response_id = session.active_response_id
            committed_ms = session.playback.committed_ms
            self._commit_played_response_history(session, old_response_id, committed_ms)
            new_epoch, old_playback = self._advance_barge_in_epoch(session)
            self.emit(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": old_response_id,
                    "reason": cancel_reason,
                    "cancelled_epoch": old_epoch,
                    "epoch": new_epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
            cancelled = True
        if not cancelled and event_type == "output_audio_buffer.clear":
            old_playback = session.playback.as_dict()
            committed_ms = session.playback.committed_ms
            session.clear_playback_cursor()
            self.emit(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": session.active_response_id,
                    "reason": cancel_reason,
                    "cancelled_epoch": session.epoch,
                    "epoch": session.epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
            self.tasks.active_response_task = None
            return
        if not cancelled:
            self._cancel_pending_input(reason="barge_in")
        if not await self._signal_cancel_fence(cancelled_fence):
            return
        self.tasks.active_response_task = None

    async def _cancel_active_response(
        self,
        active_task: asyncio.Task[None] | None,
        *,
        reason: str,
        notify: bool = True,
    ) -> bool:
        session = self.session
        has_running_task = active_task is not None and not active_task.done()
        if not has_running_task and session.active_request_id is None and session.active_response_id is None:
            return False

        old_epoch = session.epoch
        old_request_id = session.active_request_id
        old_response_id = session.active_response_id
        committed_ms = session.playback.committed_ms
        committed_message = session.end_response(
            commit_text=self._should_commit_response_to_history(session, old_response_id),
            playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value,
        )
        if old_response_id is not None:
            item_id = f"item_{old_response_id}"
            if committed_message is not None:
                session.register_history_item(item_id, committed_message)
            elif committed_ms > 0 and not session.playback_ack_is_too_late(old_response_id, item_id):
                session.truncate_history_item(item_id, audio_end_ms=committed_ms)
        # The epoch bump is the atomic part: from here on every model output
        # and append of the old epoch is dropped by the stale-epoch filter in
        # ``emit`` / the append tail, whatever the awaits below interleave with.
        new_epoch, old_playback = self._advance_barge_in_epoch(session)
        if old_request_id is not None:
            # Release projector/parser cursors so cancelled epochs do not
            # accumulate until the whole session closes.
            self.plugin.data_plane.close_stream(old_request_id)
            await self._abort_request_background(old_request_id, notify=notify)
        if has_running_task and active_task is not None:
            active_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(active_task, return_exceptions=True), timeout=0.25)
            except TimeoutError:
                pass
        if notify:
            self.emit(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": old_response_id,
                    "reason": reason,
                    "cancelled_epoch": old_epoch,
                    "epoch": new_epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
        return True

    async def _abort_request_background(self, request_id: str, *, notify: bool) -> None:
        try:
            await self.stage_port.abort_requests([request_id])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to abort duplex request %s: %s", request_id, exc)
            if notify and self.session.state != DuplexSessionState.CLOSED:
                self._send_runtime_error("runtime_abort_failed", exc)

    def _cancel_pending_input(self, *, reason: str) -> None:
        session = self.session
        cancelled = session.cancel_pending_input()
        self._advance_barge_in_epoch(session)
        self.emit(
            {
                "type": "input.cancelled",
                "session_id": session.session_id,
                "reason": reason,
                "epoch": session.epoch,
                "cancelled": cancelled,
            }
        )

    # ------------------------------------------------------------------ #
    # turn.signal (session.update / conversation items / local turns)    #
    # ------------------------------------------------------------------ #

    async def _on_turn_signal(self, event: dict[str, object]) -> None:
        session = self.session
        turn_event = event.get("event")
        realtime_event_id = event.get("realtime_event_id")
        if not isinstance(turn_event, str):
            self._emit_error("bad_event", "turn.signal requires event")
            return
        if turn_event == "barge_in" and not session.capabilities.supports_barge_in:
            self._emit_events([self._barge_in_unsupported_error()])
            return
        if turn_event == "session.update":
            payload = event.get("payload")
            if not isinstance(payload, dict):
                self._emit_error("bad_event", "session.update requires a session payload", event_id=realtime_event_id)
                return
            await self._on_session_update(
                payload,
                realtime_event_id=realtime_event_id if isinstance(realtime_event_id, str) else None,
            )
            return
        if turn_event == "conversation.item.create":
            await self._on_conversation_item_create(event)
            return
        if turn_event == "conversation.item.delete":
            payload = event.get("payload")
            item_id = payload.get("item_id") if isinstance(payload, dict) else None
            deleted = session.delete_history_item(item_id) if isinstance(item_id, str) else False
            self.emit(
                {
                    "type": "conversation.item.deleted",
                    "session_id": session.session_id,
                    "item_id": item_id,
                    "deleted": deleted,
                }
            )
            return
        if turn_event == "conversation.item.truncate":
            payload = event.get("payload")
            item_id = payload.get("item_id") if isinstance(payload, dict) else None
            audio_end_ms = payload.get("audio_end_ms") if isinstance(payload, dict) else None
            truncated = (
                session.truncate_history_item(
                    item_id,
                    audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                    hard=True,
                )
                if isinstance(item_id, str)
                else False
            )
            self.emit(
                {
                    "type": "conversation.item.truncated",
                    "session_id": session.session_id,
                    "item_id": item_id,
                    "content_index": (payload.get("content_index", 0) if isinstance(payload, dict) else 0),
                    "audio_end_ms": audio_end_ms,
                    "truncated": truncated,
                }
            )
            return
        self._emit_events([session.signal_turn(turn_event, event)])

    async def _on_session_update(self, payload: dict[str, object], *, realtime_event_id: str | None) -> None:
        session = self.session
        model_state = self.model_state
        pending_turn_detection: PendingTurnDetectionUpdate | None = None

        def reject_update() -> None:
            if pending_turn_detection is not None:
                pending_turn_detection.reject()

        try:
            pending_turn_detection = PendingTurnDetectionUpdate.prepare(payload)
        except Exception as exc:
            self._emit_error("unsupported_turn_detection", str(exc), event_id=realtime_event_id)
            return
        if not await self._wait_for_append_tail():
            self._emit_error(
                "session_update_aborted",
                "session.update was not applied because the preceding append failed",
                event_id=realtime_event_id,
            )
            reject_update()
            return
        try:
            self.plugin.validate_client_extra_body(payload.get("extra_body"))
        except DuplexRuntimeConfigError as exc:
            self._emit_error(exc.code, str(exc), event_id=realtime_event_id)
            reject_update()
            return
        candidate_config = deepcopy(session.config)
        audio_started = session.playback.generated_ms > 0 or session.playback.sent_ms > 0
        try:
            candidate_config.apply_realtime_update(
                payload,
                session_id=session.session_id,
                audio_started=audio_started,
            )
        except DuplexConfigError as exc:
            self._emit_error(exc.code, str(exc) or "session.update was rejected", event_id=realtime_event_id)
            reject_update()
            return
        if candidate_config.instructions != session.config.instructions and model_state.context_locked:
            self._emit_error(
                "instructions_update_unsupported",
                "session.update cannot change instructions after the native duplex context is initialized",
                event_id=realtime_event_id,
            )
            reject_update()
            return
        requests_audio = any(str(modality).lower() == "audio" for modality in candidate_config.modalities)
        if (
            requests_audio
            and "ref_audio_data" not in session.runtime_config
            and getattr(self.plugin, "requires_ref_audio", False)
        ):
            self._emit_error(
                "ref_audio_required", "Native duplex audio output requires ref_audio", event_id=realtime_event_id
            )
            reject_update()
            return
        try:
            candidate_runtime_config = self.plugin.runtime_config_for_update(
                candidate_config,
                dict(session.runtime_config),
            )
            # Validate the sampling policy for the candidate before adopting it.
            self.plugin.configure_sampling_params(
                runtime_config=dict(candidate_runtime_config),
                defaults=tuple(self.stage_port.sampling_defaults()),
            )
        except (DuplexRuntimeConfigError, DuplexConfigError) as exc:
            self._emit_error(exc.code, str(exc), event_id=realtime_event_id)
            reject_update()
            return
        except Exception as exc:
            self._emit_error("runtime_signal_failed", str(exc), event_id=realtime_event_id)
            reject_update()
            return
        session.replace_config(candidate_config)
        session.replace_runtime_config(candidate_runtime_config)
        try:
            session.touch_lease(DuplexLeaseActivity.SIGNAL)
        except Exception:
            pass
        if pending_turn_detection is not None:
            self._turn_detection_config, self._turn_detector = pending_turn_detection.commit(self._turn_detector)
        projector = self._require_projector()
        projector.apply_session_defaults(payload)
        self.emit({"type": "session.updated", "session": session.as_public_dict()})

    async def _on_conversation_item_create(self, event: dict[str, object]) -> None:
        session = self.session
        payload = event.get("payload")
        item = payload.get("item") if isinstance(payload, dict) else None
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "function_call_output" and isinstance(item, dict):
            if not await self._wait_for_append_tail():
                return
            try:
                candidate_runtime_config = self.plugin.runtime_config_for_function_output(
                    session.config,
                    dict(session.runtime_config),
                    item,
                )
            except DuplexRuntimeConfigError as exc:
                self._emit_error(exc.code, str(exc))
                return
            if candidate_runtime_config is not None:
                session.replace_runtime_config(candidate_runtime_config)
            self.emit(
                {
                    "type": "conversation.item.created",
                    "session_id": session.session_id,
                    "item": item,
                    "created": True,
                }
            )
            self._spawn(
                self._maybe_continue_response(
                    expected_epoch=session.epoch,
                    expected_model_turn_id=session.turn_id,
                ),
                name="duplex-continue",
            )
            return
        message = realtime_item_to_history_message(item)
        item_id = item.get("id") if isinstance(item, dict) else None
        if message is not None:
            session.append_history_message(message)
            session.register_history_item(item_id if isinstance(item_id, str) else None, message)
        self.emit(
            {
                "type": "conversation.item.created",
                "session_id": session.session_id,
                "item": item,
                "created": message is not None,
            }
        )

    # ------------------------------------------------------------------ #
    # Commit / response.create                                           #
    # ------------------------------------------------------------------ #

    def _apply_response_create_options(self, response_payload: dict[str, object]) -> str | None:
        session = self.session
        try:
            options = ResponseCreateOptions.from_realtime(
                response_payload,
                private_runtime_config_keys=self.plugin.private_runtime_config_keys,
            )
        except DuplexConfigError as exc:
            return exc.code
        try:
            session.reserve_response_options(options)
        except Exception:
            return "response_already_active"
        return None

    async def _on_commit(self, event: dict[str, object]) -> None:
        session = self.session
        model_state = self.model_state
        event_type = str(event.get("type"))
        realtime_item_id = event.get("realtime_item_id")
        realtime_validated_audio_commit = (
            event_type == "input_audio_buffer.commit" and isinstance(realtime_item_id, str) and bool(realtime_item_id)
        )
        if event_type in {"input.commit", "input_audio_buffer.commit"} and not await self._wait_for_append_tail():
            return
        if event_type == "input_audio_buffer.commit" and event.get("is_speech") is False:
            model_state.input_since_commit = False
            model_state.speech_since_commit = False
            model_state.audio_buffer.clear()
            session.release_all_input_bytes()
            model_state.clear_committed_audio()
            self.emit(
                {
                    "type": "input.committed",
                    "session_id": session.session_id,
                    "turn_id": session.turn_id,
                    "epoch": session.epoch,
                    "empty": True,
                    "is_speech": False,
                    "no_response": True,
                }
            )
            self.emit(
                {
                    "type": "response.listen",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "reason": "silence_or_noise",
                }
            )
            return
        should_create_response = (
            event_type == "response.create"
            or bool(event.get("response_create", event_type == "input.commit"))
            or (event_type == "input_audio_buffer.commit" and self._session_auto_responds())
        )
        precreate_response_requested = event_type == "response.create" or bool(
            event.get("response_create", event_type == "input.commit")
        )
        if event_type == "response.create":
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                response_options_error = self._apply_response_create_options(response_payload)
                if response_options_error is not None:
                    error_message = (
                        "The selected native duplex runtime does not support generation "
                        "overrides for instructions, voice, temperature, max tokens, tools, or "
                        "tool_choice."
                        if response_options_error == "unsupported_native_response_options"
                        else "response.create cannot reserve options while another response is active."
                    )
                    self._emit_error(response_options_error, error_message, event_id=event.get("realtime_event_id"))
                    return
        if event_type == "input_audio_buffer.commit":
            has_pending_audio = (
                model_state.input_since_commit
                or model_state.audio_buffer.has_pending()
                or model_state.committed_audio_payload is not None
                or realtime_validated_audio_commit
            )
            if not has_pending_audio and not self.tasks.append_tasks and self._stream_request_id is None:
                self._emit_error(
                    "input_audio_buffer_empty", "input_audio_buffer.commit requires a non-empty input audio buffer."
                )
                return
            commit_action = decide_commit_action(
                CommitSnapshot(
                    auto_responds=self._session_auto_responds(),
                    speech_since_commit=model_state.speech_since_commit,
                    active_response_id=session.active_response_id,
                    overlap_speech_ms=session.overlap_speech_ms,
                    response_in_progress=self._response_in_progress(),
                    playback_active=self._assistant_playback_active(),
                )
            )
            if commit_action is CommitAction.DEFER_ACTIVE_RESPONSE:
                if session.overlap_speech_ms <= session.config.overlap_short_ack_ms:
                    model_state.audio_buffer.clear()
                    session.release_all_input_bytes()
                    model_state.input_since_commit = False
                    model_state.speech_since_commit = False
                    model_state.clear_committed_audio()
                    self._emit_events(discard_pending_input_audio(self._require_projector(), session.overlap_speech_ms))
                    self.emit(
                        {
                            "type": "input.committed",
                            "session_id": session.session_id,
                            "turn_id": session.turn_id,
                            "epoch": session.epoch,
                            "empty": True,
                            "is_speech": False,
                            "overlap_ack": True,
                            "no_response": True,
                        }
                    )
                    session.reset_overlap_speech()
                    session.discard_response_options()
                    return

                commit_reservation = model_state.audio_buffer.prepare_commit(
                    operation_id=uuid.uuid4().hex,
                    chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                )
                deferred_payload = commit_reservation.payload
                if deferred_payload is None:
                    commit_reservation.commit()
                else:
                    if model_state.committed_audio_payload is not None:
                        deferred_payload = self._merge_audio_payloads(
                            model_state.committed_audio_payload,
                            deferred_payload,
                        )
                    model_state.retain_committed_audio(
                        deferred_payload,
                        operation_id=commit_reservation.operation_id,
                        reserved_bytes=commit_reservation.byte_count,
                    )
                    commit_reservation.commit()
                    model_state.deferred_response_create = should_create_response
                    model_state.deferred_precreate_response = precreate_response_requested
                    model_state.input_since_commit = False
                    model_state.speech_since_commit = False
                    committed = self._commit_audio_input(
                        session,
                        realtime_item_id=realtime_item_id,
                        transcript=event.get("transcript"),
                    )
                    committed_payload = self._audio_committed_payload(
                        session,
                        committed=committed,
                        realtime_item_id=realtime_item_id,
                        transcript=event.get("transcript"),
                    )
                    committed_payload["overlap_deferred"] = True
                    committed_payload["response_create_deferred"] = should_create_response
                    self.emit(committed_payload)
                    return
            if commit_action is CommitAction.START_AUTO_RESPONSE:
                commit_reservation = model_state.audio_buffer.prepare_commit(
                    operation_id=uuid.uuid4().hex,
                    chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                )
                committed_input = commit_reservation.payload
                final_payload = committed_input
                if model_state.committed_audio_payload is not None:
                    if final_payload is not None:
                        final_payload = self._merge_audio_payloads(model_state.committed_audio_payload, final_payload)
                    else:
                        final_payload = model_state.committed_audio_payload
                commit_reservation.commit()
                if final_payload is not None:
                    model_state.retain_committed_audio(
                        final_payload,
                        operation_id=commit_reservation.operation_id,
                        reserved_bytes=commit_reservation.byte_count,
                    )
                model_state.deferred_response_create = False
                model_state.input_since_commit = False
                model_state.speech_since_commit = False
                data_plane_turn_id = session.turn_id
                committed = self._commit_audio_input(
                    session,
                    realtime_item_id=realtime_item_id,
                    transcript=event.get("transcript"),
                    turn_id=data_plane_turn_id,
                )
                self.emit(
                    self._audio_committed_payload(
                        session,
                        committed=committed,
                        realtime_item_id=realtime_item_id,
                        transcript=event.get("transcript"),
                    )
                )
                if final_payload is not None:
                    await self._start_append(
                        {**final_payload, "duplex_turn_id": data_plane_turn_id},
                        final=True,
                        precreate_response=False,
                        operation_id=commit_reservation.operation_id,
                        retained_committed_payload=final_payload,
                    )
                return
        if event_type == "response.create":
            if self._response_in_progress() or self.tasks.append_tasks or self._stream_request_id is not None:
                if session.active_response_id is None and (
                    session.active_request_id is not None
                    or self.tasks.append_tasks
                    or self._stream_request_id is not None
                ):
                    return
                self._emit_error(
                    "response_already_active", "response.create cannot start while another response is active."
                )
                session.discard_response_options()
                return
            if model_state.committed_audio_payload is not None:
                committed_payload = model_state.committed_audio_payload
                operation_id = model_state.committed_audio_operation_id
                if operation_id is None:
                    operation_id = uuid.uuid4().hex
                    model_state.committed_audio_operation_id = operation_id
                await self._start_append(
                    committed_payload,
                    final=True,
                    precreate_response=True,
                    operation_id=operation_id,
                    retained_committed_payload=committed_payload,
                )
                return
            self._emit_error(
                "response_create_without_input", "Native duplex response.create requires committed audio input."
            )
            session.discard_response_options()
            return
        if not self._response_in_progress():
            commit_reservation = (
                model_state.audio_buffer.prepare_commit(
                    operation_id=uuid.uuid4().hex,
                    chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                )
                if event_type in {"input.commit", "input_audio_buffer.commit"}
                else None
            )
            flushed_buffer_reserved_bytes = (
                model_state.audio_buffer.pending_byte_count if commit_reservation is None else 0
            )
            flushed = (
                commit_reservation.payload
                if commit_reservation is not None
                else model_state.audio_buffer.flush(chunk_period_ms=session.capabilities.chunk_period_ms or 1000)
            )
            if model_state.committed_audio_payload is not None:
                if flushed is not None:
                    flushed = self._merge_audio_payloads(model_state.committed_audio_payload, flushed)
                else:
                    flushed = model_state.committed_audio_payload
            if commit_reservation is not None:
                commit_reservation.commit()
            if flushed is not None:
                if self._should_force_listen_for_short_commit(event, flushed):
                    flushed = dict(flushed)
                    flushed["force_listen"] = True
                model_state.input_since_commit = False
                committed = self._commit_audio_input(
                    session,
                    realtime_item_id=realtime_item_id,
                    transcript=event.get("transcript"),
                )
                self.emit(
                    self._audio_committed_payload(
                        session,
                        committed=committed,
                        realtime_item_id=realtime_item_id,
                        transcript=event.get("transcript"),
                    )
                )
                operation_id = commit_reservation.operation_id if commit_reservation is not None else uuid.uuid4().hex
                reserved_bytes = (
                    commit_reservation.byte_count if commit_reservation is not None else flushed_buffer_reserved_bytes
                )
                model_state.retain_committed_audio(flushed, operation_id=operation_id, reserved_bytes=reserved_bytes)
                if should_create_response:
                    await self._start_append(
                        flushed,
                        final=True,
                        precreate_response=True,
                        operation_id=model_state.committed_audio_operation_id,
                        retained_committed_payload=flushed,
                    )
                else:
                    model_state.deferred_response_create = False
                return
        # Nothing flushed (or a response is still in progress): acknowledge the
        # commit without starting a new response.
        had_uncommitted_audio = (
            model_state.input_since_commit
            or model_state.audio_buffer.has_pending()
            or model_state.committed_audio_payload is not None
            or realtime_validated_audio_commit
        )
        committed = None
        if event_type in {"input_audio_buffer.commit", "input.commit"}:
            model_state.input_since_commit = False
            model_state.speech_since_commit = False
        if event_type != "response.create":
            committed = (
                self._commit_audio_input(
                    session,
                    realtime_item_id=realtime_item_id,
                    transcript=event.get("transcript"),
                )
                if had_uncommitted_audio
                else None
            )
            self.emit(
                self._audio_committed_payload(
                    session,
                    committed=committed,
                    realtime_item_id=realtime_item_id,
                    transcript=event.get("transcript"),
                )
            )
            return
        if committed is not None:
            if isinstance(realtime_item_id, str):
                session.register_history_item(realtime_item_id, committed.message)
            self.emit(self._input_committed_payload(session, committed, realtime_item_id=realtime_item_id))
        # NOTE(refactor): the old generic chat-completion fallback response
        # (`_run_response`) is gone; a native session with a response already in
        # progress only acknowledges the commit here.

    # ------------------------------------------------------------------ #
    # playback.ack (moved from OmniDuplexSessionHandler)                 #
    # ------------------------------------------------------------------ #

    def _handle_playback_ack(self, event: dict[str, object]) -> None:
        session = self.session
        played_ms = event.get("played_ms", event.get("audio_ms", 0))
        committed_ms = event.get("committed_ms")
        if not isinstance(played_ms, int | float):
            self._emit_error("bad_event", "playback.ack requires played_ms")
            return
        try:
            session.touch_lease(DuplexLeaseActivity.PLAYBACK_ACK)
        except Exception:
            pass
        committed_cursor = int(committed_ms) if isinstance(committed_ms, int | float) else int(played_ms)
        item_id = event.get("item_id")
        response_id = event.get("response_id")
        response_id = response_id if isinstance(response_id, str) and response_id else None
        if not isinstance(item_id, str) or not item_id:
            item_id = f"item_{response_id}" if response_id is not None else None
        elif response_id is None and item_id.startswith("item_"):
            response_id = item_id.removeprefix("item_")
        if response_id is None and item_id is None and len(session.pending_history_item_ids) == 1:
            item_id = next(iter(session.pending_history_item_ids))
            if item_id.startswith("item_"):
                response_id = item_id.removeprefix("item_")
        if response_id is None and item_id is None and session.active_response_id is not None:
            response_id = session.active_response_id
            item_id = f"item_{response_id}"
        if response_id is not None:
            expected_item_id = f"item_{response_id}"
            if item_id is None:
                item_id = expected_item_id
            elif item_id != expected_item_id:
                self._emit_error("playback_item_mismatch", "playback.ack item_id must match item_<response_id>.")
                return
            if not session.has_assistant_response_item(response_id, item_id):
                self._emit_error(
                    "playback_item_not_found", f"No assistant response item is registered for {response_id}."
                )
                return
            if session.playback_ack_is_too_late(response_id, item_id):
                self._emit_error(
                    "playback_ack_too_late", "playback.ack arrived after a later user input was committed."
                )
                return
            session.reserve_history_item(item_id)
        elif item_id is not None:
            self._emit_error("playback_item_not_found", "playback.ack requires a response-owned assistant item.")
            return
        hard_truncate = event.get("truncate") is True
        if hard_truncate:
            playback = session.acknowledge_playback(int(played_ms), committed_cursor, response_id=response_id)
            playback = session.truncate_playback_commit(committed_cursor, response_id=response_id)
        else:
            playback = session.acknowledge_playback(int(played_ms), committed_cursor, response_id=response_id)
        committed_history = False
        if isinstance(item_id, str) and item_id:
            expected_item_id = f"item_{response_id}" if response_id is not None else None
            if (
                expected_item_id == item_id
                and item_id not in session.history_item_ids
                and item_id not in session.pending_history_item_ids
            ):
                session.register_history_item(item_id, None)
            committed_history = session.truncate_history_item(
                item_id,
                audio_end_ms=committed_cursor,
                playback=playback,
                hard=hard_truncate,
            )
        elif session.pending_history_item_ids:
            pending_ids = list(session.pending_history_item_ids)
            if len(pending_ids) == 1:
                item_id = pending_ids[0]
                committed_history = session.truncate_history_item(
                    item_id,
                    audio_end_ms=committed_cursor,
                    playback=playback,
                    hard=hard_truncate,
                )
        elif session.active_response_id is not None:
            item_id = f"item_{session.active_response_id}"
            committed_history = session.truncate_history_item(
                item_id,
                audio_end_ms=committed_cursor,
                playback=playback,
                hard=hard_truncate,
            )
        elif session.last_assistant_full_message is not None:
            if item_id is None and session.history_item_ids:
                assistant_item_ids = [
                    known_item_id
                    for known_item_id, message in session.history_item_ids.items()
                    if message.get("role") == "assistant"
                ]
                if len(assistant_item_ids) == 1:
                    item_id = assistant_item_ids[0]
            if isinstance(item_id, str) and item_id:
                committed_history = session.truncate_history_item(
                    item_id,
                    audio_end_ms=committed_cursor,
                    playback=playback,
                    hard=hard_truncate,
                )
        self._emit_events(
            [
                PlaybackAcknowledged(
                    details={
                        "type": "playback.acknowledged",
                        "session_id": session.session_id,
                        "epoch": session.epoch,
                        "item_id": item_id,
                        "played_ms": int(played_ms),
                        "committed_ms": committed_cursor,
                        "truncate": event.get("truncate") is True,
                        "playback": playback.as_dict(),
                        "history_committed": committed_history,
                    }
                )
            ]
        )
        if committed_history and committed_cursor >= max(playback.sent_ms, playback.generated_ms):
            session.release_response_playback(response_id)
            session.release_response_history_snapshot(response_id)


__all__ = ["DuplexAppendTaskMeta", "DuplexSessionRunner", "DuplexSessionTasks"]
