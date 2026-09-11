# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-side owner of duplex sessions: admission, lease reaping, command dispatch.

``DuplexOrchestrator`` hosts exactly one manager. The whole session
(``DuplexEngineSession``) lives here, owned by one ``DuplexSessionRunner`` per
session on the orchestrator loop; nothing outside the engine holds session
state.

Control operations (open / close / resume / touch) arrive as correlated RPC
messages and answer through ``result_sink``; session commands arrive one-way
as ``DuplexSessionCommandMessage`` and are pushed onto the runner's ordered
mailbox after backpressure admission; everything a session emits leaves
through ``output_sink`` as ``DuplexSessionEventMessage``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.engine.duplex.commands import AppendAudio, Commit, DuplexCommand
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStagePort,
    DuplexStageRequestContext,
    duplex_resource_request_belongs_to_session,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.events import DuplexEvent, ErrorEvent, SessionClosed, SessionExpired, error_event
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity, DuplexLeaseConfig, DuplexLeaseState
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionError,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexRuntimeConfigError, validate_duplex_plugin_sampling
from vllm_omni.engine.duplex.session import DuplexEngineSession, DuplexFenceMismatchError

if TYPE_CHECKING:
    import janus
    from vllm.config import ModelConfig

    from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
    from vllm_omni.engine.duplex.session_runner import DuplexSessionRunner
    from vllm_omni.engine.messages import EngineQueueMessage

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PendingRequestCleanup:
    session_id: str
    lease_generation: int
    request_ids: tuple[str, ...]
    abort: bool


@dataclass(frozen=True)
class _PendingSessionCleanup:
    """A close/expiry whose stage cleanup has not succeeded yet (retried by the reaper)."""

    kind: str
    session_id: str
    lease_generation: int
    reason: str
    submitted_request_ids: tuple[str, ...]
    reserved_request_ids: tuple[str, ...]


class DuplexSessionManager:
    _CONTROL_TYPES = (
        OpenDuplexSessionMessage,
        CloseDuplexSessionMessage,
        ResumeDuplexSessionMessage,
        TouchDuplexSessionMessage,
    )
    _MESSAGE_TYPES = (*_CONTROL_TYPES, DuplexSessionCommandMessage)

    def __init__(
        self,
        *,
        plugin: DuplexModelPlugin,
        stage_port: DuplexStagePort,
        output_sink: janus.AsyncQueue[EngineQueueMessage],
        result_sink: janus.AsyncQueue[EngineQueueMessage],
        runtime_config: DuplexSessionRuntimeConfig,
        model_config: ModelConfig | None,
        clock: Callable[[], float] | None = None,
        executor: concurrent.futures.ThreadPoolExecutor | None = None,
    ) -> None:
        validate_duplex_plugin_sampling(plugin, sampling_defaults=tuple(stage_port.sampling_defaults()))
        self.plugin = plugin
        self.stage_port = stage_port
        self.model_config = model_config
        self.runtime_config = runtime_config
        self._output_sink = output_sink
        self._result_sink = result_sink
        self._clock = clock or time.monotonic
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="duplex-session",
        )
        self._owns_executor = executor is None
        self._lease_config = DuplexLeaseConfig(
            idle_ttl_s=runtime_config.idle_ttl_s,
            disconnect_grace_s=runtime_config.disconnect_grace_s,
        )
        self.runners: dict[str, DuplexSessionRunner] = {}
        #: Sessions whose close/expiry began but whose stage cleanup has not finalized;
        #: they keep their admission slot until finalized.
        self._closing: dict[str, _PendingSessionCleanup] = {}
        #: Session ids whose ``open`` passed the capacity check and is still awaiting the plugin.
        self._admitting: set[str] = set()
        self._request_index: dict[str, str] = {}
        self._session_control_tails: dict[str, asyncio.Task[None]] = {}
        self._dispatched_control_tasks: set[asyncio.Task[None]] = set()
        self._pending_request_cleanups: dict[tuple[str, int], _PendingRequestCleanup] = {}
        self._request_cleanup_tasks: dict[tuple[str, int], asyncio.Task[None]] = {}
        self._request_cleanups_in_progress: set[tuple[str, int]] = set()
        #: Sessions retained for cleanup retry after their runner is gone.
        self._session_snapshots: dict[str, DuplexEngineSession] = {}

    # ------------------------------------------------------------------ #
    # Views                                                              #
    # ------------------------------------------------------------------ #

    def active_count(self) -> int:
        return len(self.runners)

    def get(self, session_id: str) -> DuplexEngineSession | None:
        runner = self.runners.get(session_id)
        return runner.session if runner is not None else None

    def sessions(self) -> dict[str, DuplexEngineSession]:
        return {session_id: runner.session for session_id, runner in self.runners.items()}

    def runner_for_request_id(self, request_id: str) -> DuplexSessionRunner | None:
        session_id = self._request_index.get(request_id)
        if session_id is None:
            for candidate_id in self.runners:
                if duplex_resource_request_belongs_to_session(request_id, candidate_id):
                    session_id = candidate_id
                    break
        if session_id is None:
            return None
        return self.runners.get(session_id)

    def register_request(self, request_id: str, session_id: str) -> None:
        self._request_index[request_id] = session_id

    def unregister_request(self, request_id: str) -> None:
        self._request_index.pop(request_id, None)

    def _unregister_session_requests(self, session_id: str) -> None:
        for request_id in [rid for rid, sid in self._request_index.items() if sid == session_id]:
            self._request_index.pop(request_id, None)

    # ------------------------------------------------------------------ #
    # Message routing                                                    #
    # ------------------------------------------------------------------ #

    def accepts(self, message: object) -> bool:
        return isinstance(message, self._MESSAGE_TYPES)

    def dispatch(self, message: object) -> None:
        """Route one engine request-queue message without blocking the request handler."""
        if isinstance(message, DuplexSessionCommandMessage):
            self._dispatch_command(message)
            return
        if not isinstance(message, self._CONTROL_TYPES):
            raise TypeError(f"Unsupported duplex control message: {type(message).__name__}")
        session_id = message.session_id

        async def handle_message() -> None:
            await self.handle(message)

        self._run_control(session_id, f"duplex-control-{session_id}-{message.control_id}", handle_message)

    def _run_control(
        self,
        session_id: str,
        name: str,
        operation: Callable[[], Awaitable[None]],
    ) -> asyncio.Task[None]:
        """Run one control operation after the session's previous one (per-session order)."""
        predecessor = self._session_control_tails.get(session_id)

        async def run_ordered() -> None:
            if predecessor is not None:
                try:
                    await predecessor
                except BaseException:
                    pass
            await operation()

        task = asyncio.create_task(run_ordered(), name=name)
        self._session_control_tails[session_id] = task
        self._track_task(task)

        def discard(completed: asyncio.Task[None]) -> None:
            if self._session_control_tails.get(session_id) is completed:
                self._session_control_tails.pop(session_id, None)

        task.add_done_callback(discard)
        return task

    def _track_task(self, task: asyncio.Task[None]) -> None:
        """Keep a background task alive until it finishes and log an escaped exception."""
        self._dispatched_control_tasks.add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self._dispatched_control_tasks.discard(completed)
            if not completed.cancelled() and completed.exception() is not None:
                logger.error("duplex control task %s failed: %r", completed.get_name(), completed.exception())

        task.add_done_callback(done)

    async def handle(self, message: object) -> None:
        if isinstance(message, OpenDuplexSessionMessage):
            await self.open(message)
        elif isinstance(message, CloseDuplexSessionMessage):
            await self.close(message)
        elif isinstance(message, ResumeDuplexSessionMessage):
            await self.resume(message)
        elif isinstance(message, TouchDuplexSessionMessage):
            await self.touch(message)
        elif isinstance(message, DuplexSessionCommandMessage):
            self._dispatch_command(message)
        else:
            raise TypeError(f"Unsupported duplex control message: {type(message).__name__}")

    def _dispatch_command(self, message: DuplexSessionCommandMessage) -> None:
        """Admit one command: identity check, backpressure, then the runner's ordered mailbox.

        Backpressure contract shared with the runner: a ``Commit`` that passes
        admission holds one pending-turn reservation and an ``AppendAudio``
        holds its wire-size byte reservation until the runner dequeues the
        command (the runner then re-reserves the decoded size around its PCM
        reservation), so the mailbox never holds more bytes than the session
        limit even while the worker is busy.
        """
        runner = self.runners.get(message.session_id)
        command = message.command
        if runner is None:
            self._emit_raw(
                message.session_id,
                self._error_event(
                    "unknown_session",
                    f"Unknown or closed duplex session: {message.session_id}",
                    command=command,
                ),
            )
            return
        session = runner.session
        if runner.closing:
            self.emit(
                session,
                self._error_event(
                    "session_closed", f"Duplex session is closing: {session.session_id}", command=command
                ),
            )
            return
        if isinstance(command, AppendAudio):
            limit = int(self.runtime_config.max_pending_input_bytes_per_session)
            if not session.reserve_input_bytes(len(command.audio), limit=limit):
                self.emit(
                    session,
                    self._error_event(
                        "input_backpressure",
                        "Duplex session has too many pending input bytes",
                        command=command,
                    ),
                )
                return
        elif isinstance(command, Commit):
            if not session.reserve_pending_turn(limit=int(self.runtime_config.max_pending_turns_per_session)):
                self.emit(
                    session,
                    self._error_event(
                        "input_backpressure",
                        "Duplex session has too many pending input turns",
                        command=command,
                    ),
                )
                return
        runner.submit(command)

    @staticmethod
    def _error_event(code: str, message: str, *, command: DuplexCommand | None = None) -> ErrorEvent:
        return error_event(code, message, event_id=command.event_id if command is not None else None)

    # ------------------------------------------------------------------ #
    # Emission                                                           #
    # ------------------------------------------------------------------ #

    def emit(self, session: DuplexEngineSession, event: DuplexEvent) -> None:
        """Bind the session identity to one typed event and push it to the engine output queue."""
        self._emit_raw(session.session_id, event, epoch=session.epoch)

    def _emit_raw(
        self,
        session_id: str,
        event: DuplexEvent,
        *,
        epoch: int | None = None,
    ) -> None:
        event = replace(event, session_id=session_id, epoch=epoch)
        message = DuplexSessionEventMessage(session_id=session_id, event=event)
        put_nowait = getattr(self._output_sink, "put_nowait", None)
        if callable(put_nowait):
            put_nowait(message)
        else:  # pragma: no cover - sink without put_nowait
            asyncio.ensure_future(self._output_sink.put(message))

    async def _put_result(
        self,
        message: (
            OpenDuplexSessionMessage
            | CloseDuplexSessionMessage
            | ResumeDuplexSessionMessage
            | TouchDuplexSessionMessage
        ),
        *,
        operation: str,
        ok: bool,
        session: DuplexEngineSession | None = None,
        error: BaseException | None = None,
    ) -> None:
        error_code, error_message, error_retryable = (
            self._control_error(error) if error is not None else (None, None, False)
        )
        result = DuplexControlResultMessage(
            control_id=message.control_id,
            operation=operation,
            session_id=message.session_id,
            ok=ok,
            lease_generation=session.lease_generation if session is not None else None,
            capabilities=session.capabilities if session is not None and ok else None,
            public_session=session.as_public_dict() if session is not None and ok else None,
            error_code=error_code,
            error_message=error_message,
            error_retryable=error_retryable,
        )
        await self._result_sink.put(result)

    @staticmethod
    def _control_error(error: BaseException) -> tuple[str, str, bool]:
        """Map an exception to ``(code, message, retryable)`` for a control result."""
        message = str(error)
        retryable = False
        if isinstance(error, DuplexSessionError):
            code = error.code
            retryable = error.retryable
        elif isinstance(error, DuplexRuntimeConfigError):
            code = error.code
        elif isinstance(error, DuplexFenceMismatchError):
            code = "stale_fence"
        elif isinstance(error, KeyError):
            code = "not_found"
        elif isinstance(error, TypeError | ValueError):
            code = "invalid_argument"
        elif isinstance(error, TimeoutError):
            code = "timeout"
            retryable = True
        else:
            code = "failed_precondition"
        return code, message, retryable

    # ------------------------------------------------------------------ #
    # Sampling / stage request helpers shared with the runner            #
    # ------------------------------------------------------------------ #

    def sampling_params_for(self, session: DuplexEngineSession) -> tuple[object, ...]:
        defaults = tuple(self.stage_port.sampling_defaults())
        configured = self.plugin.configure_sampling_params(
            runtime_config=dict(session.runtime_config),
            defaults=defaults,
        )
        if not isinstance(configured, tuple):
            raise TypeError("duplex plugin must return sampling parameters as a tuple")
        if len(configured) != len(defaults):
            raise ValueError("duplex plugin must return one sampling parameter per stage")
        return configured

    @staticmethod
    def stage_request_id(fence: DuplexFence, *, stage_id: int) -> str:
        return duplex_resource_request_id(fence, f"stage{stage_id}")

    def ensure_stage_request(
        self,
        session: DuplexEngineSession,
        *,
        stage_id: int,
        fence: DuplexFence | None = None,
    ) -> DuplexStageRequestContext | None:
        """Reserve the session's stage request id and register it with the stage port."""
        if stage_id >= self.stage_port.stage_count:
            return None
        effective_fence = fence or session.fence
        request_id = self.stage_request_id(effective_fence, stage_id=stage_id)
        session.reserve_stage_request(stage_id, request_id, fence=effective_fence)
        context = DuplexStageRequestContext(
            request_id=request_id,
            session_id=session.session_id,
            fence=effective_fence,
            stage_id=stage_id,
            final_stage_id=self.stage_port.stage_count - 1,
            config_generation=session.config_generation,
            sampling_params=self.sampling_params_for(session),
            session_config=session.config.as_dict(),
            runtime_config=session.runtime_config,
        )
        self.stage_port.ensure_request(context)
        self.register_request(request_id, session.session_id)
        return context

    # ------------------------------------------------------------------ #
    # Control operations                                                 #
    # ------------------------------------------------------------------ #

    def _admission_count(self) -> int:
        return len(set(self.runners) | set(self._closing) | self._admitting)

    async def open(self, message: OpenDuplexSessionMessage) -> None:
        from vllm_omni.engine.duplex.session_runner import DuplexSessionRunner

        session_id = message.session_id
        session: DuplexEngineSession | None = None
        runner: DuplexSessionRunner | None = None
        # Only the open that put the id in ``_admitting`` may take it out again:
        # a duplicate open must not release the slot the first one is holding.
        holds_admission_slot = False
        try:
            if session_id in self.runners or session_id in self._closing or session_id in self._admitting:
                raise DuplexSessionError(f"Duplex session already exists: {session_id}", code="session_exists")
            max_sessions = int(self.runtime_config.max_sessions)
            if self._admission_count() >= max_sessions:
                raise DuplexSessionError(
                    f"duplex_session_capacity_exhausted: limit={max_sessions}",
                    code="resource_exhausted",
                    retryable=True,
                )
            # Hold the slot across the plugin await: opens of different
            # sessions run concurrently and must not all pass the check above.
            self._admitting.add(session_id)
            holds_admission_slot = True
            config = message.session_config
            self.plugin.validate_client_extra_body(config.extra_body)
            try:
                runtime_config = await self.plugin.prepare_runtime_config(config, model_config=self.model_config)
            except DuplexRuntimeConfigError:
                raise
            except ValueError as exc:
                raise DuplexRuntimeConfigError(str(exc)) from exc
            capabilities = self.plugin.capabilities(max_sessions=max_sessions)
            session = DuplexEngineSession(
                session_id=session_id,
                config=config,
                capabilities=capabilities,
                lease=DuplexLeaseState(config=self._lease_config, generation=0, last_activity=self._clock()),
                _clock=self._clock,
                _runtime_config=dict(runtime_config),
            )
            # Validates the plugin's sampling policy for this runtime config before admission.
            self.sampling_params_for(session)
            session.model_state = self.plugin.create_session_state()
            runner = DuplexSessionRunner(
                session=session,
                plugin=self.plugin,
                stage_port=self.stage_port,
                manager=self,
                model_config=self.model_config,
            )
            self.runners[session_id] = runner
            # Reserve the Stage0 request resource atomically with admission.
            self.ensure_stage_request(session, stage_id=0)
            runner.start()
            await self._put_result(message, operation="open", ok=True, session=session)
        except Exception as exc:
            error_code, _, _ = self._control_error(exc)
            if error_code in {"resource_exhausted", "session_exists"}:
                logger.info("open_duplex_session rejected: %s", exc)
            else:
                logger.exception("open_duplex_session failed: %s", exc)
            if runner is not None and self.runners.get(session_id) is runner:
                self.runners.pop(session_id, None)
                try:
                    await runner.shutdown()
                except Exception:
                    logger.exception("duplex open rollback: runner shutdown failed for %s", session_id)
            if session is not None:
                reserved = session.release_all_requests()
                if reserved:
                    try:
                        await self.stage_port.cleanup(list(reserved))
                    except Exception:
                        logger.warning("duplex open rollback: request cleanup pending for %s", session_id)
                self._unregister_session_requests(session_id)
            await self._put_result(message, operation="open", ok=False, error=exc)
        finally:
            if holds_admission_slot:
                self._admitting.discard(session_id)

    def _require_runner(self, session_id: str) -> DuplexSessionRunner:
        runner = self.runners.get(session_id)
        if runner is None:
            raise DuplexSessionError(f"Unknown duplex session: {session_id}", code="unknown_session")
        if runner.closing:
            raise DuplexSessionError(f"Duplex session is closing: {session_id}", code="session_closed")
        return runner

    async def close(self, message: CloseDuplexSessionMessage) -> None:
        session_id = message.session_id
        runner = self.runners.get(session_id)
        if runner is None:
            # Already closing, or closed and forgotten: ids are never reused
            # (D16), so a repeated close is idempotent rather than an error.
            await self._put_result(message, operation="close", ok=True)
            return
        session = runner.session
        try:
            await self._close_runner(runner, kind="close", reason=message.reason)
        except Exception as exc:
            # The session is closed either way; only the stage cleanup is
            # outstanding and the reaper retries it while the slot stays held.
            logger.warning("close_duplex_session: stage cleanup for %s remains pending: %s", session_id, exc)
        await self._put_result(message, operation="close", ok=True, session=session)

    def close_from_runner(self, runner: DuplexSessionRunner, reason: str) -> None:
        """Finish a close the runner started from its command stream (``session.close``).

        Runs on the orchestrator loop as a tracked task (the runner's worker
        cannot await its own teardown): stage requests are aborted, the
        admission slot is released and ``session.closed`` is emitted last.
        """
        session_id = runner.session.session_id
        if self.runners.get(session_id) is not runner:
            return

        async def close_runner() -> None:
            if self.runners.get(session_id) is runner:
                await self._close_runner(runner, kind="close", reason=reason)

        # Chained on the session's control tail so a control RPC issued after
        # the wire close observes the closed session, never a half-torn one.
        self._run_control(session_id, f"duplex-close-{session_id}", close_runner)

    async def _close_runner(self, runner: DuplexSessionRunner, *, kind: str, reason: str) -> None:
        """Begin an irreversible close, tear down the runner, then release stage resources.

        The session keeps its admission slot (``_closing``) until stage cleanup
        succeeded; a failed cleanup is retried by the reaper. For an explicit
        close the ``session.closed`` event is emitted only after the cleanup
        attempt, so a client that sees it can open a replacement session at
        once (the runner used to emit it before the stage requests were
        aborted, which let a prompt reopen hit ``resource_exhausted``).
        """
        session = runner.session
        submitted = tuple(session.resource_request_ids(submitted=True))
        reserved = tuple(session.resource_request_ids(submitted=False))
        session.begin_close(reason=reason)
        pending = _PendingSessionCleanup(
            kind=kind,
            session_id=session.session_id,
            lease_generation=session.lease_generation,
            reason=reason,
            submitted_request_ids=submitted,
            reserved_request_ids=reserved,
        )
        self._closing[session.session_id] = pending
        if self.runners.get(session.session_id) is runner:
            self.runners.pop(session.session_id, None)
        emit_after_cleanup = kind in {"close", "expired"} and not runner.closed_emitted
        try:
            if kind == "expired":
                await runner.expire(reason, emit_expired=not emit_after_cleanup)
            else:
                await runner.close(reason, emit_closed=not emit_after_cleanup)
        except Exception:
            logger.exception("duplex runner %s failed for session %s", kind, session.session_id)
        session.close()
        try:
            await self._finalize_pending_cleanup(pending, session)
        finally:
            if emit_after_cleanup:
                terminal: DuplexEvent = (
                    SessionExpired(reason=reason) if kind == "expired" else SessionClosed(reason=reason)
                )
                self.emit(session, terminal)

    async def _finalize_pending_cleanup(
        self,
        pending: _PendingSessionCleanup,
        session: DuplexEngineSession | None,
    ) -> None:
        if pending.submitted_request_ids:
            await self.stage_port.cleanup(list(pending.submitted_request_ids), abort=True)
        if pending.reserved_request_ids:
            await self.stage_port.cleanup(list(pending.reserved_request_ids))
        if session is not None:
            session.release_all_requests()
        self._unregister_session_requests(pending.session_id)
        if self._closing.get(pending.session_id) is pending:
            self._closing.pop(pending.session_id, None)
        self._session_snapshots.pop(pending.session_id, None)

    async def resume(self, message: ResumeDuplexSessionMessage) -> None:
        session: DuplexEngineSession | None = None
        try:
            runner = self._require_runner(message.session_id)
            session = runner.session
            try:
                session.resume_lease(expected_lease_generation=message.expected_lease_generation)
            except ValueError as exc:
                raise DuplexSessionError(str(exc), code="session_resume_conflict") from exc
            await self._put_result(message, operation="resume", ok=True, session=session)
        except Exception as exc:
            logger.exception("resume_duplex_session failed: %s", exc)
            await self._put_result(message, operation="resume", ok=False, session=session, error=exc)

    async def touch(self, message: TouchDuplexSessionMessage) -> None:
        session: DuplexEngineSession | None = None
        try:
            runner = self._require_runner(message.session_id)
            session = runner.session
            activity = DuplexLeaseActivity(message.activity)
            if activity is DuplexLeaseActivity.DETACH:
                session.detach_lease()
            else:
                session.touch_lease(activity)
            await self._put_result(message, operation="touch", ok=True, session=session)
        except Exception as exc:
            logger.exception("touch_duplex_session failed: %s", exc)
            await self._put_result(message, operation="touch", ok=False, session=session, error=exc)

    # ------------------------------------------------------------------ #
    # Lease expiry / reaper                                              #
    # ------------------------------------------------------------------ #

    async def reaper_loop(self, shutdown_event: asyncio.Event) -> None:
        interval = float(self.runtime_config.reaper_interval_s)
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
            except TimeoutError:
                try:
                    await self.reap_expired()
                except Exception:
                    logger.exception("[DuplexSessionManager] expiry cleanup failed; retrying on next tick")

    async def reap_expired(self, now: float | None = None) -> int:
        completed = 0
        effective_now = self._clock() if now is None else now
        # Retry request cleanups recorded by the orchestrator error paths.
        for key, request_cleanup in list(self._pending_request_cleanups.items()):
            if key in self._request_cleanups_in_progress:
                continue
            try:
                await self._complete_request_cleanup(key, request_cleanup)
            except Exception as exc:
                logger.warning(
                    "duplex request cleanup remains pending for session %s: %s",
                    request_cleanup.session_id,
                    exc,
                )
                continue
            completed += 1
        # Retry closes/expiries whose stage cleanup failed.
        for session_id, session_cleanup in list(self._closing.items()):
            if session_id in self.runners:
                continue
            if session_cleanup.kind == "request_cleanup" and any(
                key[0] == session_id for key in self._pending_request_cleanups
            ):
                # The stage cleanup is owned by the request-cleanup path (orchestrator
                # or the retry above); ``finalize_closed_sessions`` releases the slot.
                continue
            try:
                await self._finalize_pending_cleanup(session_cleanup, self._session_snapshots.get(session_id))
            except Exception as exc:
                logger.warning(
                    "duplex %s cleanup remains pending for session %s: %s",
                    session_cleanup.kind,
                    session_id,
                    exc,
                )
                continue
            completed += 1
        # Expire leases.
        for session_id, runner in list(self.runners.items()):
            lease = runner.session.lease
            if lease.disconnect_grace_expired(effective_now):
                reason = "disconnect_grace_expired"
            elif lease.idle_expired(effective_now):
                reason = "idle_ttl_expired"
            else:
                continue
            self._session_snapshots[session_id] = runner.session
            try:
                await self._close_runner(runner, kind="expired", reason=reason)
            except Exception as exc:
                logger.warning("duplex expiry cleanup remains pending for session %s: %s", session_id, exc)
                continue
            completed += 1
        return completed

    # ------------------------------------------------------------------ #
    # Request-triggered cleanup (orchestrator error paths)               #
    # ------------------------------------------------------------------ #

    def close_sessions_for_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        cleanup_in_progress: bool = False,
    ) -> dict[str, list[str]]:
        request_id_set = set(request_ids)
        closed: dict[str, list[str]] = {}
        for session_id, runner in list(self.runners.items()):
            session = runner.session
            stale = session.resource_request_ids()
            if request_id_set.isdisjoint(stale):
                continue
            if not session.begin_close(reason="request_cleanup"):
                continue
            closed[session_id] = stale
            key = (session_id, session.lease_generation)
            existing = self._pending_request_cleanups.get(key)
            merged = tuple(dict.fromkeys([*(existing.request_ids if existing is not None else ()), *stale]))
            self._pending_request_cleanups[key] = _PendingRequestCleanup(
                session_id=session_id,
                lease_generation=session.lease_generation,
                request_ids=merged,
                abort=abort or (existing.abort if existing is not None else False),
            )
            if cleanup_in_progress:
                self._request_cleanups_in_progress.add(key)
            # Tear the runner down in the background; the orchestrator owns the stage cleanup.
            self._session_snapshots[session_id] = session
            self._closing[session_id] = _PendingSessionCleanup(
                kind="request_cleanup",
                session_id=session_id,
                lease_generation=session.lease_generation,
                reason="request_cleanup",
                submitted_request_ids=(),
                reserved_request_ids=(),
            )
            self.runners.pop(session_id, None)
            task = asyncio.create_task(runner.expire("request_cleanup"), name=f"duplex-request-cleanup-{session_id}")
            self._track_task(task)
        return closed

    def defer_request_cleanups(self, session_ids: Iterable[str]) -> None:
        session_id_set = set(session_ids)
        active_keys = {key for key in self._request_cleanups_in_progress if key[0] in session_id_set}
        self._request_cleanups_in_progress.difference_update(active_keys)

    def finalize_closed_sessions(self, session_ids: Iterable[str]) -> None:
        session_id_set = set(session_ids)
        for key in list(self._pending_request_cleanups):
            if key[0] not in session_id_set:
                continue
            self._pending_request_cleanups.pop(key, None)
            self._request_cleanups_in_progress.discard(key)
        for session_id in session_id_set:
            session = self._session_snapshots.pop(session_id, None)
            if session is not None and session.lease.terminal_reason is not None:
                session.release_all_requests()
                session.close()
            self._closing.pop(session_id, None)
            self._unregister_session_requests(session_id)

    async def _complete_request_cleanup(
        self,
        key: tuple[str, int],
        pending: _PendingRequestCleanup,
    ) -> None:
        task = self._request_cleanup_tasks.get(key)
        if task is not None and task.done():
            self._request_cleanup_tasks.pop(key, None)
            task = None
        if task is None:
            task = asyncio.create_task(
                self._run_request_cleanup(key, pending),
                name=f"duplex-request-cleanup-{pending.session_id}",
            )
            self._request_cleanup_tasks[key] = task

            def discard(completed: asyncio.Task[None]) -> None:
                if self._request_cleanup_tasks.get(key) is completed:
                    self._request_cleanup_tasks.pop(key, None)

            task.add_done_callback(discard)
        await asyncio.shield(task)

    async def _run_request_cleanup(self, key: tuple[str, int], pending: _PendingRequestCleanup) -> None:
        await self.stage_port.cleanup(list(pending.request_ids), abort=pending.abort)
        if self._pending_request_cleanups.get(key) is pending:
            self._pending_request_cleanups.pop(key, None)
            self._request_cleanups_in_progress.discard(key)
        self.finalize_closed_sessions([pending.session_id])

    # ------------------------------------------------------------------ #
    # Shutdown                                                           #
    # ------------------------------------------------------------------ #

    async def shutdown(self) -> None:
        runners = list(self.runners.values())
        self.runners.clear()
        for runner in runners:
            try:
                await runner.shutdown()
            except Exception:
                logger.exception("duplex runner shutdown failed for %s", runner.session.session_id)
        tasks = tuple(self._dispatched_control_tasks | set(self._request_cleanup_tasks.values()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._session_control_tails.clear()
        self._closing.clear()
        self._session_snapshots.clear()
        self._request_index.clear()
        if self._owns_executor:
            self.executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["DuplexSessionManager"]
