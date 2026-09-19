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
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from vllm.logger import init_logger

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
from vllm_omni.engine.duplex.config import (
    DuplexConfigError,
    DuplexOverlapPolicy,
    DuplexPlaybackCommitPolicy,
    DuplexSessionState,
    ResponseCreateOptions,
)
from vllm_omni.engine.duplex.contracts import (
    DuplexOutputContext,
    DuplexOutputDecision,
    DuplexStagePort,
)
from vllm_omni.engine.duplex.events import (
    DuplexEvent,
    InputCleared,
    SessionExpired,
    SessionHeartbeatAck,
)
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexModelSessionState,
    PcmAppendReservation,
)
from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    discard_pending_input_audio,
    note_input_append,
    resolve_cancel_response,
    resolve_clear_output_audio,
    resolve_commit,
    resolve_create_item,
    resolve_delete_item,
    resolve_truncate_item,
    retrieve_item_events,
)
from vllm_omni.engine.duplex.session import helpers, overlap_policy, playback_ledger
from vllm_omni.engine.duplex.session.append_task import AppendAttempt
from vllm_omni.engine.duplex.session.commit_policy import CommitAction, CommitSnapshot, decide_commit_action
from vllm_omni.engine.duplex.session.context import (
    DuplexRunState,
    DuplexSessionContext,
    DuplexSessionTasks,
    StageOutput,
)
from vllm_omni.engine.duplex.session.control import SessionControl
from vllm_omni.engine.duplex.session.emitter import SessionEmitter
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.lease import DuplexLeaseActivity
from vllm_omni.engine.duplex.session.model_channel import ModelChannel
from vllm_omni.engine.duplex.turn_detection import (
    TurnDetectionResult,
)
from vllm_omni.metrics.stats import StageRequestStats
from vllm_omni.protocol.duplex import convert_input_audio_with_rate

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.outputs import RequestOutput

    from vllm_omni.engine.duplex.session.manager import DuplexSessionManager

logger = init_logger(__name__)

_OffloadT = TypeVar("_OffloadT")


@dataclass(frozen=True, slots=True)
class _Internal:
    """Runner-internal mailbox item (its payload runs through the dict-based handlers)."""

    kind: str
    payload: dict[str, object] = field(default_factory=dict)


_CANCEL_EVENTS = frozenset({"input.cancel", "response.cancel", "barge_in", "output_audio_buffer.clear"})


def compute_silence_continuation_deadline(
    *,
    chunk_period_s: float,
    now: float,
    last_submit: float | None,
    current_deadline: float | None,
) -> tuple[float, float]:
    """Return the silence continuation schedule ``(delay_s, next_silence_deadline)``.

    The first continuation anchors to the latest accepted append's submission
    time (``last_submit + chunk_period_s``), falling back to ``now`` when no
    submission exists. Later continuations advance from the current deadline
    instead of from ``now``, so pipeline processing time does not accumulate
    as timer drift. When the schedule is more than one period overdue it is
    stale: one continuation submits immediately and the schedule restarts
    from ``now`` (``next_silence_deadline = now + chunk_period_s``) instead of
    firing a burst of catch-ups.

    ``delay_s`` is the wait before this continuation and
    ``next_silence_deadline`` the deadline for the following one.
    """
    if current_deadline is None:
        base = last_submit if last_submit is not None else now
        deadline = base + chunk_period_s
    else:
        deadline = current_deadline
    if now - deadline > chunk_period_s:
        deadline = now
    delay_s = max(0.0, deadline - now)
    return delay_s, deadline + chunk_period_s


class DuplexSessionRunner:
    """Owns one ``DuplexEngineSession`` on the orchestrator loop (see module docstring)."""

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
        self._mailbox: asyncio.Queue[DuplexCommand | StageOutput | _Internal] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._worker_stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        #: Flags more than one component reads and writes (see session_context).
        self.run = DuplexRunState()
        self.ctx = DuplexSessionContext(
            session=session,
            model_state=self.model_state,
            plugin=plugin,
            stage_port=stage_port,
            manager=manager,
            tasks=self.tasks,
            run=self.run,
            services=self,
        )
        self.out = SessionEmitter(self.ctx, promote_deferred_overlap=self._promote_deferred_overlap_later)
        self.model = ModelChannel(
            self.ctx,
            self.out,
            close_from_runtime=self._close_from_runtime,
            schedule_silence_continuation=self._schedule_silence_continuation,
            abort_request=self._abort_request_background,
        )
        self.control = SessionControl(
            self.ctx,
            self.out,
            self.model,
            wait_for_append_tail=self._wait_for_append_tail,
        )

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        session = self.session
        if self.out.projector is None:
            default_payload = session.config.extra_body.get("realtime_session_payload")
            self.out.projector = RealtimeProjectionState(
                session_id=session.session_id,
                model=session.config.model,
                default_payload=default_payload if isinstance(default_payload, Mapping) else None,
                initial_session_update=True,
            )
        # Every client speaks the Realtime protocol now; the old runner forced
        # the ACK-only playback ledger for that path.
        session.config.playback_commit_policy = DuplexPlaybackCommitPolicy.ACK_ONLY.value
        self.control.init_turn_detection()
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
            decision = self.model.decide_output(stage_id, output, context)
        consume = decision is not None or stage_id >= context.final_stage_id
        project_intermediate = self.plugin.projects_intermediate_outputs and stage_id == 0
        if not consume and not project_intermediate:
            # Stage0 text without a direct decision feeds the TTS stage as before.
            # Its metrics still have to reach the client: before sessions moved
            # into the engine the orchestrator published them as a standalone
            # ``StageMetricsMessage``, a path session-owned requests no longer
            # take. Hand them to the session instead of dropping them, on the
            # mailbox so they stay ordered with this session's other work.
            snapshot = self.model.stage_metrics_snapshot(stage_id, metrics, output)
            if snapshot is not None and not self.run.closing and self.session.state != DuplexSessionState.CLOSED:
                self._mailbox.put_nowait(_Internal("stage_metrics", {"stage_metrics": snapshot}))
            return False
        if self.run.closing or self.session.state == DuplexSessionState.CLOSED:
            return True
        self._mailbox.put_nowait(
            StageOutput(
                stage_id=stage_id,
                output=output,
                metrics=metrics,
                request_id=request_id,
                context=context,
                decision=decision,
            )
        )
        return consume

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
        return self.run.closed_emitted

    def mark_closed_emitted(self) -> None:
        """Claim the session's one terminal event for the caller.

        The manager emits the deferred terminal itself, after the stage cleanup;
        recording it here stops a late runtime close emitting a second one.
        """
        self.run.closed_emitted = True
        self.run.closed_deferred = False

    @property
    def closing(self) -> bool:
        """Whether an irreversible close has begun (commands and control ops are refused)."""
        return self.run.closing or self.session.state != DuplexSessionState.OPEN

    async def close(self, reason: str, *, emit_closed: bool = True) -> None:
        """Graceful close: cancel work, release the data plane, emit ``session.closed``.

        With ``emit_closed=False`` the manager emits ``session.closed`` itself
        once the stage resources are released, so the event also means "the
        admission slot is free again".
        """
        session = self.session
        if session.state == DuplexSessionState.CLOSED and (self.run.closed_emitted or self.run.closed_deferred):
            await self._stop_worker()
            return
        self._begin_close(reason)
        self.model_state.audio_buffer.clear()
        session.release_buffered_input_bytes()
        self.model_state.input_since_commit = False
        self.model_state.speech_since_commit = False
        self.model_state.clear_committed_audio()
        await self.tasks.cancel_append_tasks()
        self.model.cancel_data_plane_stream()
        await self._cancel_active_response(self.tasks.active_response_task, reason=reason, notify=False)
        self.tasks.active_response_task = None
        self._cleanup_duplex_session_state()
        if not self.run.closed_emitted and not self.run.closed_deferred:
            self.run.close_reason = self.run.close_reason or reason
            if emit_closed:
                self.run.closed_emitted = True
                self.emit({"type": "session.closed", "session_id": session.session_id, "reason": reason})
            else:
                self.run.closed_deferred = True
        session.close()
        await self._stop_worker()

    async def expire(self, reason: str, *, emit_expired: bool = True) -> None:
        """Lease expiry / runtime cleanup: emit ``session.expired`` and tear down.

        With ``emit_expired=False`` the manager emits the event after the stage
        cleanup (see ``close``).
        """
        session = self.session
        self._begin_close(reason)
        if not self.run.closed_emitted:
            if emit_expired:
                # Also when the event was deferred: the manager only emits a
                # deferred terminal for the teardown it drives itself, and an
                # expiry that arrives first (a stage failure racing a wire
                # ``session.close``) takes the runner away from it. Emitting
                # here keeps "every session ends with one terminal event".
                self.run.closed_emitted = True
                self._emit_events([SessionExpired(reason=reason)])
            else:
                self.run.closed_deferred = True
        await self.tasks.cancel_append_tasks()
        self.model.cancel_data_plane_stream()
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
        self.run.closing = True
        self.run.closed_emitted = True
        await self.tasks.cancel_append_tasks()
        self.model.cancel_data_plane_stream()
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

    def spawn(self, coro: Awaitable[None], *, name: str) -> None:
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def offload(self, fn: Callable[..., _OffloadT], *args: object, **kwargs: object) -> _OffloadT:
        loop = self._loop or asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(self.manager.executor, lambda: fn(*args, **kwargs))
        return await loop.run_in_executor(self.manager.executor, fn, *args)

    async def _handle_item(self, item: DuplexCommand | StageOutput | _Internal) -> None:
        session = self.session
        if isinstance(item, StageOutput):
            await self.model.on_stage_output_item(item)
            return
        if isinstance(item, _Internal):
            await self._on_internal(item)
            return
        if self.run.closing or session.state == DuplexSessionState.CLOSED:
            if isinstance(item, Commit):
                session.release_pending_turn()
            elif isinstance(item, AppendAudio):
                session.release_input_bytes(len(item.audio), queued=True)
            return
        await self._on_command(item)

    async def _on_internal(self, item: _Internal) -> None:
        if item.kind == "stage_metrics":
            stage_metrics = item.payload.get("stage_metrics")
            if isinstance(stage_metrics, Mapping):
                self.session.stash_stage_metrics(stage_metrics)
            return
        if item.kind == "promote_deferred_overlap":
            if self.run.closing or self.session.state != DuplexSessionState.OPEN:
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
            self._emit_events(playback_ledger.apply_playback_ack(self.session, payload))
        elif event_type in _CANCEL_EVENTS:
            await self._on_cancel(payload)
        elif event_type == "turn.signal":
            await self.control.on_turn_signal(payload)
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
            session.release_input_bytes(len(command.audio), queued=True)
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
            if resolved.reset_vad:
                self.control.reset_vad()
            if resolved.payload is not None:
                await self._on_commit(resolved.payload)
        elif isinstance(command, CreateResponse):
            await self._on_commit(command.payload())
        elif isinstance(command, ClearInput):
            self._on_clear_input()
        elif isinstance(command, CancelResponse):
            control = resolve_cancel_response(projector, command)
            self._emit_events(control.events)
            for payload in control.payloads:
                await self._on_cancel(payload)
        elif isinstance(command, ClearOutputAudio):
            control = resolve_clear_output_audio(projector, command)
            self._emit_events(control.events)
            for payload in control.payloads:
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
                inner = payload.get("payload")
                normalized: dict[str, object] = dict(inner) if isinstance(inner, Mapping) else {}
                normalized.update(payload)
                normalized["type"] = command.event
                await self._on_cancel(normalized)
            else:
                await self.control.on_turn_signal(payload)
        elif isinstance(command, UpdateSession):
            await self.control.on_session_update(dict(command.patch), realtime_event_id=command.event_id)
        elif isinstance(command, AckPlayback):
            self._emit_events(playback_ledger.apply_playback_ack(self.session, command.payload()))
        elif isinstance(command, Heartbeat):
            self._on_heartbeat(command)
        elif isinstance(command, CreateItem):
            control = resolve_create_item(projector, command)
            self._emit_events(control.events)
            for payload in control.payloads:
                await self._run_internal_payload(payload)
        elif isinstance(command, DeleteItem):
            control = resolve_delete_item(projector, command)
            self._emit_events(control.events)
            for payload in control.payloads:
                await self.control.on_turn_signal(payload)
        elif isinstance(command, TruncateItem):
            control = resolve_truncate_item(projector, command)
            self._emit_events(control.events)
            for payload in control.payloads:
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
        session.release_buffered_input_bytes()
        model_state.input_since_commit = False
        model_state.speech_since_commit = False
        model_state.clear_committed_audio()
        session.cancel_pending_input()
        projector = self.out.projector
        if projector is not None:
            from vllm_omni.engine.duplex.realtime_events import clear_input_buffer

            clear_input_buffer(projector)
        self._emit_events([InputCleared()])

    # ------------------------------------------------------------------ #
    # Emission (delegated to SessionEmitter)                             #
    # ------------------------------------------------------------------ #

    def emit(self, payload: dict[str, object]) -> None:
        """Apply an internal event to the session, project it, and send it."""
        self.out.emit(payload)

    def _emit_events(self, events: list[DuplexEvent]) -> None:
        self.out.emit_events(events)

    def _emit_error(
        self,
        code: str,
        message: str,
        *,
        event_id: object | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.out.emit_error(code, message, event_id=event_id, retryable=retryable)

    def _require_projector(self) -> RealtimeProjectionState:
        return self.out.require_projector()

    def _is_stale_model_output(self, payload: dict[str, object]) -> bool:
        return self.out.is_stale_model_output(payload)

    def _promote_deferred_overlap_later(self, payload: dict[str, object], precreate_response: bool) -> None:
        """Re-enter the mailbox so the promoted turn is handled in command order."""
        self._mailbox.put_nowait(
            _Internal(
                "promote_deferred_overlap",
                {"payload": payload, "precreate_response": precreate_response},
            )
        )

    # ------------------------------------------------------------------ #
    # Session lifecycle helpers                                          #
    # ------------------------------------------------------------------ #

    def _begin_close(self, reason: str) -> None:
        self.run.closing = True
        self.run.close_reason = self.run.close_reason or reason
        self.session.mark_closing()

    def _session_auto_responds(self) -> bool:
        return self.out.auto_responds()

    def _cleanup_duplex_session_state(self) -> None:
        session = self.session
        self.plugin.data_plane.close_session(session.session_id, active_request_id=session.active_request_id)
        self.run.stream_request_id = None

    # ------------------------------------------------------------------ #
    # Append path (was the audio-append branch + start_native_append)    #
    # ------------------------------------------------------------------ #

    async def _barge_in_for_overlap(self, event: dict[str, object], decision: dict[str, object]) -> bool:
        """Interrupt the model so the user's overlapping speech becomes the turn.

        Cancels the active response and the data-plane stream, advances the
        barge-in epoch so stale output is dropped, and reports what was cut.
        Returns whether the append should continue: losing the fence race
        means this append belongs to a superseded epoch and is abandoned.
        """
        session = self.session
        model_state = self.model_state
        event["force_barge_in"] = True
        cancelled_fence = session.fence
        playback_was_active = helpers.assistant_playback_active(self.session)
        model_state.audio_buffer.clear_force_listen()
        session.reset_overlap_speech()
        model_state.input_since_commit = False
        model_state.speech_since_commit = False
        await self.tasks.cancel_append_tasks()
        had_stream = self.run.stream_request_id is not None
        cancel_reason = str(decision.get("cancel_reason") or "barge_in")
        cancelled = await self._cancel_active_response(
            self.tasks.active_response_task,
            reason=cancel_reason,
        )
        had_stream = self.model.cancel_data_plane_stream() or had_stream
        if not cancelled and had_stream:
            old_epoch = session.epoch
            old_response_id = session.active_response_id
            committed_ms = session.playback.committed_ms
            helpers.commit_played_response_history(session, old_response_id, committed_ms)
            new_epoch, old_playback = helpers.advance_barge_in_epoch(session)
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
            helpers.commit_played_response_history(session, session.last_response_id, committed_ms)
            new_epoch, old_playback = helpers.advance_barge_in_epoch(session)
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
            if not await self.model.signal_cancel_fence(cancelled_fence):
                return False
        self.tasks.active_response_task = None
        return True

    async def _on_append_audio(self, event: dict[str, object]) -> None:
        session = self.session
        model_state = self.model_state
        session.mark_user_input_activity()
        audio = event.get("audio") or event.get("data")
        if not isinstance(audio, str):
            self._emit_error("bad_event", "input_audio_buffer.append requires audio")
            return
        if not session.capabilities.supports_barge_in and overlap_policy.event_requests_barge_in(event):
            self._emit_events([helpers.barge_in_unsupported_error()])
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
            converted: tuple[object, object, int | float | None] = await self.offload(
                convert_input_audio_with_rate,
                audio,
                fmt,
                sample_rate_hz=sample_rate_hz,
            )
        except ValueError as exc:
            self._emit_error("bad_event", str(exc))
            return
        audio, fmt, converted_rate = converted
        if converted_rate is not None:
            sample_rate_hz = converted_rate
        if isinstance(fmt, str) and fmt.lower() in {"pcm16", "pcm_s16le", "s16le"}:
            self._emit_error("bad_audio", "input_audio_buffer.append pcm16 audio could not be decoded")
            return
        event["audio"] = audio
        event["format"] = fmt
        event["sample_rate_hz"] = sample_rate_hz
        client_force_listen = bool(event.get("force_listen", False))
        vad_result = await self.control.run_turn_detection(event)
        if (
            vad_result is not None
            and not session.capabilities.supports_core_resumable_request
            and not client_force_listen
        ):
            # VAD's force_listen hint controls native model decoding. Committed
            # turn models already buffer speech; it must not suppress barge-in.
            event.pop("force_listen", None)
        projector = self._require_projector()
        self._emit_events(note_input_append(projector, event, vad_result=vad_result))
        if self.run.closing or session.state != DuplexSessionState.OPEN:
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
        payload["is_speech"] = overlap_policy.input_looks_like_speech(self.session, event, payload)
        auto_responds = self._session_auto_responds()
        defer_append = False
        buffer_overlap_audio = True
        self._mark_pending_silence_superseded()
        overlap_active = helpers.response_in_progress(self.session, self.tasks) and (
            not auto_responds
            or (
                session.capabilities.supports_barge_in
                and (
                    session.config.overlap_policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
                    or overlap_policy.event_requests_barge_in(event)
                )
            )
        )
        if overlap_active:
            decision = overlap_policy.decide(self.session, event, payload, auto_responds=auto_responds)
            self._emit_events([helpers.overlap_decision_event(self.session, decision)])
            action = decision.get("action")
            if action == "drop":
                duration_ms = overlap_policy.input_audio_duration_ms(event, payload)
                self._emit_events(discard_pending_input_audio(projector, duration_ms))
                self._maybe_schedule_vad_commit(vad_result)
                return
            if action == "listen":
                buffer_overlap_audio = bool(decision.get("buffer_audio", True))
                defer_append = bool(decision.get("defer_runtime_append", True))
                if not buffer_overlap_audio and decision.get("preserve_realtime_input") is not True:
                    self._emit_events(
                        discard_pending_input_audio(projector, overlap_policy.input_audio_duration_ms(event, payload))
                    )
                if decision.get("force_listen", True) is True:
                    payload["force_listen"] = True
            else:
                if not await self._barge_in_for_overlap(event, decision):
                    return
                buffer_overlap_audio = True
                defer_append = False
        elif not auto_responds and not overlap_policy.input_looks_like_speech(self.session, event, payload):
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
        if overlap_policy.should_force_listen_for_auto_response_overlap(event, payload, auto_responds=auto_responds):
            payload["force_listen"] = True
        if not buffer_overlap_audio:
            self._maybe_schedule_vad_commit(vad_result)
            return
        session.mark_user_input_activity()
        model_state.input_since_commit = True
        model_state.speech_since_commit = model_state.speech_since_commit or overlap_policy.input_looks_like_speech(
            self.session, event, payload
        )
        raw_audio_bytes = helpers.audio_payload_size_bytes(payload)
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
        append_payload = pcm_reservation.payload
        if append_payload is None:
            self._maybe_schedule_vad_commit(vad_result)
            return
        await self._start_append(append_payload, final=False, pcm_reservation=pcm_reservation)
        self._maybe_schedule_vad_commit(vad_result)

    def _maybe_schedule_vad_commit(self, vad_result: TurnDetectionResult | None) -> None:
        """Server VAD ended the user turn: run the same commit the old translator synthesized."""
        if vad_result is None or not vad_result.should_commit:
            return
        if self._session_auto_responds():
            # A model-native session decides its own turns; server VAD is there
            # to hear the user (speech_started / speech_stopped, barge-in), not
            # to end the turn. Committing at the detector's stop cuts the
            # utterance short: the model listens on that early final unit, the
            # trailing silence lands in a second, near-empty turn, and the
            # client's own commit then has nothing left to answer. The old
            # translator synthesized this commit for turn-based server VAD only.
            return
        command = Commit(final=True, create_response=vad_result.create_response)
        resolved = resolve_commit(self._require_projector(), command)
        self._emit_events(resolved.events)
        if resolved.reset_vad:
            self.control.reset_vad()
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
        on_append_accepted: Callable[[float], None] | None = None,
        before_append: Callable[[], bool] | None = None,
    ) -> asyncio.Task[bool]:
        session = self.session
        model_state = self.model_state
        if not silence_continuation:
            self._mark_pending_silence_superseded()
            if on_append_accepted is None:
                # A real (non-silence) input re-anchors the silence pacing
                # chain: its submission becomes the anchor and the stored
                # deadline is cleared until the next continuation sets one.
                def _reanchor_chain(submit_time: float) -> None:
                    model_state.last_native_submit_monotonic = submit_time
                    model_state.silence_deadline_monotonic = None

                on_append_accepted = _reanchor_chain
        append_epoch = session.epoch
        append_fence = helpers.append_fence(session, payload, epoch=append_epoch)
        append_turn_id = append_fence.turn_id
        request_id = self.ctx.manager.stage_request_id(
            append_fence,
            stage_id=0,
            resumable=session.capabilities.supports_core_resumable_request,
        )
        if final or precreate_response:
            session.bind_request(request_id)
        if precreate_response:
            session.bind_response_turn(append_turn_id)
        if precreate_response and session.active_response_id is None:
            response_id = session.begin_response(turn_id=append_turn_id)
            self.emit(self.model.response_created_payload(response_id, epoch=append_epoch))
        if final and not session.capabilities.supports_core_resumable_request:
            logger.info(
                "Duplex committed turn session=%s request=%s response=%s audio_bytes=%s",
                session.session_id,
                request_id,
                session.active_response_id,
                helpers.audio_payload_size_bytes(payload),
            )
        attempt = AppendAttempt(
            ctx=self.ctx,
            out=self.out,
            model=self.model,
            fail_session=self._fail_session_from_append,
            payload=payload,
            epoch=append_epoch,
            request_id=request_id,
            final=final,
            pcm_reservation=pcm_reservation,
            operation_id=operation_id,
            retained_committed_payload=retained_committed_payload,
            precreated_response_id=session.active_response_id if precreate_response else None,
            on_append_accepted=on_append_accepted,
            before_append=before_append,
        )

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
        task = asyncio.create_task(attempt.run_in_wire_order(predecessor))
        task.add_done_callback(attempt.release_on_failure)
        self.tasks.append_tail = task
        self.tasks.track_append_task(
            task,
            epoch=append_epoch,
            final=final,
            response_bound=final or precreate_response,
        )
        if silence_continuation:
            model_state.pending_silence_task = task
            task.add_done_callback(attempt.clear_pending_silence)
        # Let this wire-order effect start before the next mailbox item can cancel it.
        await asyncio.sleep(0)
        return task

    def _fail_session_from_append(self, reason: str) -> None:
        """An append task died: mark the session closing now, close it out next."""
        self._begin_close(reason)
        self.spawn(self._close_from_runtime(reason), name="duplex-runtime-close")

    async def _wait_for_append_tail(self) -> bool:
        predecessor = self.tasks.append_tail
        if predecessor is None:
            return True
        try:
            return await predecessor
        except asyncio.CancelledError:
            if helpers.task_is_cancelling(asyncio.current_task()):
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
                if helpers.task_is_cancelling(asyncio.current_task()):
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
        chunk_period_s = max(0.0, float(session.capabilities.chunk_period_ms or 1000) / 1000.0)
        # Snapshot the anchor after any wait for pending silence. A real append
        # accepted between the snapshot and this continuation's submission
        # re-anchors the chain; _still_valid() then skips the stale unit.
        anchor = model_state.last_native_submit_monotonic
        # Align the next silence unit to submission_time_N + chunk_period and
        # sleep only the remaining budget. The deadline is stored by the
        # acceptance callback when the append actually submits, so skipped or
        # stale continuations never advance the clock.
        delay_s, next_silence_deadline = compute_silence_continuation_deadline(
            chunk_period_s=chunk_period_s,
            now=time.monotonic(),
            last_submit=anchor,
            current_deadline=model_state.silence_deadline_monotonic,
        )
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        if (
            self.tasks.append_tail is not append_tail
            or ((append_tail is None or append_tail.done()) and self._real_input_waiting())
            or self.model.silence_continuation_is_stale(
                request_id=request_id,
                response_id=response_id,
                response_owned=response_owned,
                expected_epoch=expected_epoch,
                expected_model_turn_id=expected_model_turn_id,
            )
        ):
            return False

        def _still_valid() -> bool:
            return (
                # The anchor changed (a real append was accepted) after this
                # continuation was planned; the unit is outdated.
                model_state.last_native_submit_monotonic == anchor
                and not self._real_input_waiting()
                and not self.model.silence_continuation_is_stale(
                    request_id=request_id,
                    response_id=response_id,
                    response_owned=response_owned,
                    expected_epoch=expected_epoch,
                    expected_model_turn_id=expected_model_turn_id,
                )
            )

        def _on_append_accepted(submit_time: float) -> None:
            # Commit timing state once the runtime accepts the append. If the
            # submission is more than one chunk period past the planned
            # deadline, the schedule is stale: restart from the actual
            # submission. Small delays keep the planned cadence so ordinary
            # jitter does not accumulate as drift.
            model_state.last_native_submit_monotonic = submit_time
            model_state.silence_deadline_monotonic = (
                submit_time + chunk_period_s if submit_time > next_silence_deadline else next_silence_deadline
            )

        model_state.pending_silence_owner_id = owner_id
        task = await self._start_append(
            dict(payload) if isinstance(payload, dict) else {},
            final=False,
            silence_continuation=True,
            on_append_accepted=_on_append_accepted,
            before_append=_still_valid,
        )
        return task is not None

    # ------------------------------------------------------------------ #
    # Runtime-initiated close                                             #
    # ------------------------------------------------------------------ #

    async def _close_from_runtime(self, reason: str) -> None:
        """A runtime failure closed the session from inside the engine."""
        session = self.session
        if session.state == DuplexSessionState.CLOSED:
            return
        self.run.runtime_closed = True
        self._begin_close(reason)
        self._cleanup_duplex_session_state()
        # ``closed_deferred`` means a manager-driven close already promised the
        # terminal after its stage cleanup. Emitting here too would give the
        # session two terminal events: this path can run concurrently, when an
        # append task abandoned by that close fails and calls back in.
        if not self.run.closed_emitted and not self.run.closed_deferred:
            self.run.closed_emitted = True
            self.emit({"type": "session.closed", "session_id": session.session_id, "reason": reason})
        session.close()
        # The manager aborts the stage requests (if any) and frees the
        # admission slot; a session without stage requests must not stay
        # registered until its idle TTL.
        self.manager.close_from_runner(self, reason)

    # ------------------------------------------------------------------ #
    # Cancel / barge-in                                                  #
    # ------------------------------------------------------------------ #

    async def _on_cancel(self, event: dict[str, object]) -> None:
        session = self.session
        model_state = self.model_state
        event_type = str(event.get("type"))
        if event_type == "barge_in" and not session.capabilities.supports_barge_in:
            self._emit_events([helpers.barge_in_unsupported_error()])
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
            has_active_response_work = helpers.response_in_progress(self.session, self.tasks)
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
        playback_was_active = helpers.assistant_playback_active(self.session)
        if event_type in {"input.cancel", "barge_in"}:
            model_state.audio_buffer.clear()
            session.release_buffered_input_bytes()
            model_state.input_since_commit = False
            model_state.speech_since_commit = False
            model_state.clear_committed_audio()
        had_append = await self.tasks.cancel_append_tasks(
            response_bound_only=event_type in {"response.cancel", "output_audio_buffer.clear"},
        )
        if event_type == "response.cancel":
            session.release_input_bytes(model_state.clear_committed_audio())
        had_stream = self.run.stream_request_id is not None
        cancelled = await self._cancel_active_response(self.tasks.active_response_task, reason=cancel_reason)
        had_stream = self.model.cancel_data_plane_stream() or had_stream
        if not cancelled and (had_stream or had_append or had_unbuffered_append):
            old_epoch = session.epoch
            old_response_id = session.active_response_id
            committed_ms = session.playback.committed_ms
            helpers.commit_played_response_history(session, old_response_id, committed_ms)
            new_epoch, old_playback = helpers.advance_barge_in_epoch(session)
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
            helpers.commit_played_response_history(session, session.last_response_id, committed_ms)
            new_epoch, old_playback = helpers.advance_barge_in_epoch(session)
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
            helpers.commit_played_response_history(session, old_response_id, committed_ms)
            new_epoch, old_playback = helpers.advance_barge_in_epoch(session)
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
        if not await self.model.signal_cancel_fence(cancelled_fence):
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
            commit_text=self.model.should_commit_response_to_history(session, old_response_id),
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
        new_epoch, old_playback = helpers.advance_barge_in_epoch(session)
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
                self.model.send_runtime_error("runtime_abort_failed", exc)

    def _cancel_pending_input(self, *, reason: str) -> None:
        session = self.session
        cancelled = session.cancel_pending_input()
        helpers.advance_barge_in_epoch(session)
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

    async def _flush_and_submit_committed_turn(
        self,
        event: dict[str, object],
        *,
        event_type: str,
        realtime_item_id: object,
        should_create_response: bool,
    ) -> bool:
        """Flush the buffered turn and submit it, with no response yet running.

        Returns whether the turn was flushed. An empty buffer falls through to
        the bare acknowledgement at the end of ``_on_commit``.
        """
        session = self.session
        model_state = self.model_state
        commit_reservation = (
            model_state.audio_buffer.prepare_commit(
                operation_id=uuid.uuid4().hex,
                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
            )
            if event_type in {"input.commit", "input_audio_buffer.commit"}
            else None
        )
        flushed_buffer_reserved_bytes = model_state.audio_buffer.pending_byte_count if commit_reservation is None else 0
        flushed = (
            commit_reservation.payload
            if commit_reservation is not None
            else model_state.audio_buffer.flush(chunk_period_ms=session.capabilities.chunk_period_ms or 1000)
        )
        if model_state.committed_audio_payload is not None:
            if flushed is not None:
                flushed = overlap_policy.merge_audio_payloads(model_state.committed_audio_payload, flushed)
            else:
                flushed = model_state.committed_audio_payload
        if commit_reservation is not None:
            commit_reservation.commit()
        if flushed is None:
            return False
        if overlap_policy.should_force_listen_for_short_commit(self.session, event, flushed):
            flushed = dict(flushed)
            flushed["force_listen"] = True
        model_state.input_since_commit = False
        committed = helpers.commit_audio_input(
            session,
            realtime_item_id=realtime_item_id,
            transcript=event.get("transcript"),
        )
        self.emit(
            helpers.audio_committed_payload(
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
        return True

    async def _start_response_from_committed_audio(self) -> None:
        """Answer a client ``response.create``.

        A turn starts from committed input. Audio is the usual kind, but the
        Realtime protocol also lets a client build the turn out of
        ``conversation.item.create`` and ask for a response with no audio at
        all; those items are input too. A model-native session still generates
        per audio unit, so an item-only turn is primed with the model's own
        silence unit -- the one continuations already use -- rather than with
        anything this layer invents.

        Refused only when nothing is waiting, or a response is already running,
        rather than silently producing an empty turn.
        """
        session = self.session
        model_state = self.model_state
        if (
            helpers.response_in_progress(self.session, self.tasks)
            or self.tasks.append_tasks
            or self.run.stream_request_id is not None
        ):
            if session.active_response_id is None and (
                session.active_request_id is not None
                or self.tasks.append_tasks
                or self.run.stream_request_id is not None
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
            session.reset_unanswered_user_items()
            await self._start_append(
                committed_payload,
                final=True,
                precreate_response=True,
                operation_id=operation_id,
                retained_committed_payload=committed_payload,
            )
            return
        if session.unanswered_user_items() and session.capabilities.supports_text_only_turn:
            session.reset_unanswered_user_items()
            await self._start_append({"type": "conversation"}, final=True, precreate_response=True)
            return
        if session.unanswered_user_items():
            # Conversation items are context, not a turn. A model-native model
            # decides to speak from the audio it hears, and there is no audio
            # here: opening a response anyway produces one the model never
            # fills, which the caller only discovers when the session idles
            # out. Text reaches such a model through the seeded opening turn
            # (``DuplexSessionConfig.initial_user_text``) instead.
            self._emit_error(
                "text_only_turn_unsupported",
                "This duplex model answers speech input only. Seed the session with "
                "initial_user_text to ask it in text."
                if session.capabilities.supports_chat_completions
                else "This duplex model answers speech input only.",
            )
            session.reset_unanswered_user_items()
            session.discard_response_options()
            return
        self._emit_error(
            "response_create_without_input",
            "Duplex response.create requires committed audio to answer.",
        )
        session.discard_response_options()

    def _discard_short_overlap_ack(self) -> None:
        """Drop a sub-threshold interjection made while the model is speaking.

        Too short to be a turn, so it neither interrupts the response nor
        becomes one: the buffered audio is discarded and the turn stays with
        the model.
        """
        session = self.session
        model_state = self.model_state
        model_state.audio_buffer.clear()
        session.release_buffered_input_bytes()
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

    def _defer_commit_behind_active_response(
        self,
        event: dict[str, object],
        *,
        realtime_item_id: object,
        should_create_response: bool,
        precreate_response_requested: bool,
    ) -> bool:
        """Retain a commit that arrived while a response is still running.

        The audio is kept so the turn is not lost, and the response it should
        start is remembered; ``_maybe_promote_deferred_overlap`` replays it once
        the active response ends. Returns whether the commit was deferred --
        an empty buffer has nothing to retain and falls through.
        """
        session = self.session
        model_state = self.model_state
        commit_reservation = model_state.audio_buffer.prepare_commit(
            operation_id=uuid.uuid4().hex,
            chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
        )
        deferred_payload = commit_reservation.payload
        if deferred_payload is None:
            commit_reservation.commit()
            return False
        retained_payload: dict[str, object]
        if model_state.committed_audio_payload is not None:
            retained_payload = overlap_policy.merge_audio_payloads(
                model_state.committed_audio_payload,
                deferred_payload,
            )
        else:
            retained_payload = deferred_payload
        model_state.retain_committed_audio(
            retained_payload,
            operation_id=commit_reservation.operation_id,
            reserved_bytes=commit_reservation.byte_count,
        )
        commit_reservation.commit()
        model_state.deferred_response_create = should_create_response
        model_state.deferred_precreate_response = precreate_response_requested
        model_state.input_since_commit = False
        model_state.speech_since_commit = False
        committed = helpers.commit_audio_input(
            session,
            realtime_item_id=realtime_item_id,
            transcript=event.get("transcript"),
        )
        committed_payload = helpers.audio_committed_payload(
            session,
            committed=committed,
            realtime_item_id=realtime_item_id,
            transcript=event.get("transcript"),
        )
        committed_payload["overlap_deferred"] = True
        committed_payload["response_create_deferred"] = should_create_response
        self.emit(committed_payload)
        return True

    async def _commit_and_start_auto_response(self, event: dict[str, object], *, realtime_item_id: object) -> None:
        """Commit the buffered turn and submit it, letting the model answer it."""
        session = self.session
        model_state = self.model_state
        commit_reservation = model_state.audio_buffer.prepare_commit(
            operation_id=uuid.uuid4().hex,
            chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
        )
        final_payload = commit_reservation.payload
        if model_state.committed_audio_payload is not None:
            if final_payload is not None:
                final_payload = overlap_policy.merge_audio_payloads(model_state.committed_audio_payload, final_payload)
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
        committed = helpers.commit_audio_input(
            session,
            realtime_item_id=realtime_item_id,
            transcript=event.get("transcript"),
            turn_id=data_plane_turn_id,
        )
        self.emit(
            helpers.audio_committed_payload(
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

    def _commit_silent_input(self) -> None:
        """Commit a turn the client itself marked as silence: drop it and keep listening."""
        session = self.session
        model_state = self.model_state
        model_state.input_since_commit = False
        model_state.speech_since_commit = False
        model_state.audio_buffer.clear()
        session.release_buffered_input_bytes()
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

    async def _on_commit(self, event: dict[str, object]) -> None:
        session = self.session
        model_state = self.model_state
        event_type = str(event.get("type"))
        realtime_item_id = event.get("realtime_item_id")
        realtime_validated_audio_commit = (
            event_type == "input_audio_buffer.commit" and isinstance(realtime_item_id, str) and bool(realtime_item_id)
        )
        if event_type in {"input.commit", "input_audio_buffer.commit"} and not await self._wait_for_append_tail():
            # resolve_commit has already applied this commit's side effects
            # (speech_stopped emitted, input flags cleared), so returning
            # silently leaves the client with those effects, no
            # input_audio_buffer.committed and no error. The session.update
            # path says so explicitly; this one now does too.
            self._emit_error(
                "commit_aborted",
                "the commit was not applied because the preceding append failed",
                event_id=event.get("realtime_event_id"),
            )
            return
        if event_type == "input_audio_buffer.commit" and event.get("is_speech") is False:
            self._commit_silent_input()
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
            if not has_pending_audio and not self.tasks.append_tasks and self.run.stream_request_id is None:
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
                    response_in_progress=helpers.response_in_progress(self.session, self.tasks),
                    playback_active=helpers.assistant_playback_active(self.session),
                )
            )
            if commit_action is CommitAction.DEFER_ACTIVE_RESPONSE:
                if session.overlap_speech_ms <= session.config.overlap_short_ack_ms:
                    self._discard_short_overlap_ack()
                    return

                if self._defer_commit_behind_active_response(
                    event,
                    realtime_item_id=realtime_item_id,
                    should_create_response=should_create_response,
                    precreate_response_requested=precreate_response_requested,
                ):
                    return
            if commit_action is CommitAction.START_AUTO_RESPONSE:
                await self._commit_and_start_auto_response(event, realtime_item_id=realtime_item_id)
                return
        if event_type == "response.create":
            await self._start_response_from_committed_audio()
            return
        if not helpers.response_in_progress(self.session, self.tasks) and await self._flush_and_submit_committed_turn(
            event,
            event_type=event_type,
            realtime_item_id=realtime_item_id,
            should_create_response=should_create_response,
        ):
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
                helpers.commit_audio_input(
                    session,
                    realtime_item_id=realtime_item_id,
                    transcript=event.get("transcript"),
                )
                if had_uncommitted_audio
                else None
            )
            self.emit(
                helpers.audio_committed_payload(
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
            self.emit(helpers.input_committed_payload(session, committed, realtime_item_id=realtime_item_id))
        # NOTE(refactor): the old generic chat-completion fallback response
        # (`_run_response`) is gone; a native session with a response already in
        # progress only acknowledges the commit here.

    # ------------------------------------------------------------------ #
    # playback.ack (moved from OmniDuplexSessionHandler)                 #
    # ------------------------------------------------------------------ #


__all__ = ["DuplexSessionRunner"]
