# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Orchestrator-side control plane for experimental duplex sessions.

``Orchestrator`` creates this component only when duplex control is enabled
and supplies a narrow :class:`DuplexStagePort` implementation. Queue messages
are routed through :meth:`DuplexControlPlane.handle`; stage outputs are passed
to :meth:`DuplexControlPlane.decide_output`; request cleanup is reported back
through the control plane so session leases and stage bindings stay coherent.

The module owns duplex session/control algorithms. It deliberately does not
own stage pools, request queues, or OpenAI Realtime protocol state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputContext,
    DuplexOutputDecision,
    DuplexRequestIdentity,
    DuplexRuntimeCapabilities,
    DuplexRuntimeExtension,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    SessionMode,
    duplex_resource_request_id,
)
from vllm_omni.experimental.fullduplex.engine.duplex_session import (
    DuplexAppendReservation,
    DuplexFenceMismatchError,
    DuplexSessionExpiry,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
)
from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity, DuplexLeaseConfig
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexControlError,
    DuplexControlResultMessage,
    DuplexFence,
    DuplexSessionLifecycleMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    SignalDuplexTurnMessage,
    TouchDuplexSessionMessage,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PendingControlCleanup:
    kind: str
    session_id: str
    fence: DuplexFence
    submitted_request_ids: tuple[str, ...]
    reserved_request_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PendingSubmissionCleanup:
    session_id: str
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PendingRequestCleanup:
    session_id: str
    fence: DuplexFence
    lease_generation: int
    request_ids: tuple[str, ...]
    abort: bool


class DuplexResultSink(Protocol):
    async def put(self, message: DuplexControlResultMessage) -> None: ...


class DuplexLifecycleSink(Protocol):
    async def put(self, message: DuplexSessionLifecycleMessage) -> None: ...


class DuplexControlPlane:
    _MESSAGE_TYPES = (
        OpenDuplexSessionMessage,
        AppendDuplexInputMessage,
        SignalDuplexTurnMessage,
        CloseDuplexSessionMessage,
        TouchDuplexSessionMessage,
        ResumeDuplexSessionMessage,
    )

    def __init__(
        self,
        *,
        extension: DuplexRuntimeExtension | None,
        stage_port: DuplexStagePort,
        result_sink: DuplexResultSink,
        lifecycle_sink: DuplexLifecycleSink | None = None,
        lease_config: DuplexLeaseConfig | None = None,
        clock: Callable[[], float] | None = None,
        max_sessions: int | None = None,
        completed_append_limit: int = 256,
    ) -> None:
        self._extension = extension
        self._stage_port = stage_port
        self._result_sink = result_sink
        self._lifecycle_sink = lifecycle_sink
        self._lease_config = lease_config or DuplexLeaseConfig()
        self._sessions = DuplexSessionRuntimeManager(
            clock=clock,
            max_sessions=max_sessions,
            completed_append_limit=completed_append_limit,
        )
        self._pending_expirations: dict[tuple[str, int], DuplexSessionExpiry] = {}
        self._pending_control_cleanups: dict[tuple[str, str, int, int, int, int], _PendingControlCleanup] = {}
        self._control_cleanup_tasks: dict[
            tuple[str, str, int, int, int, int],
            asyncio.Task[None],
        ] = {}
        self._pending_submission_cleanups: dict[str, _PendingSubmissionCleanup] = {}
        self._submission_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._pending_request_cleanups: dict[tuple[str, int, int], _PendingRequestCleanup] = {}
        self._request_cleanup_tasks: dict[tuple[str, int, int], asyncio.Task[None]] = {}
        self._request_cleanups_in_progress: set[tuple[str, int, int]] = set()
        self._session_control_tails: dict[str, asyncio.Task[None]] = {}
        self._dispatched_control_tasks: set[asyncio.Task[None]] = set()

    @property
    def sessions(self) -> DuplexSessionRuntimeManager:
        return self._sessions

    @property
    def pending_submission_cleanup_count(self) -> int:
        return len(self._pending_submission_cleanups)

    def accepts(self, message: object) -> bool:
        return isinstance(message, self._MESSAGE_TYPES)

    async def handle(self, message: object) -> None:
        if isinstance(message, OpenDuplexSessionMessage):
            await self.handle_open(message)
        elif isinstance(message, AppendDuplexInputMessage):
            await self.handle_append(message)
        elif isinstance(message, SignalDuplexTurnMessage):
            await self.handle_signal(message)
        elif isinstance(message, CloseDuplexSessionMessage):
            await self.handle_close(message)
        elif isinstance(message, TouchDuplexSessionMessage):
            await self.handle_touch(message)
        elif isinstance(message, ResumeDuplexSessionMessage):
            await self.handle_resume(message)
        else:
            raise TypeError(f"Unsupported duplex control message: {type(message).__name__}")

    def dispatch(self, message: object) -> None:
        """Schedule one control command without blocking unrelated sessions."""
        if not self.accepts(message):
            raise TypeError(f"Unsupported duplex control message: {type(message).__name__}")
        session_id = message.session_id
        predecessor = self._session_control_tails.get(session_id)

        async def run_ordered() -> None:
            if predecessor is not None:
                await predecessor
            await self.handle(message)

        task = asyncio.create_task(
            run_ordered(),
            name=f"duplex-control-{session_id}-{message.control_id}",
        )
        self._session_control_tails[session_id] = task
        self._dispatched_control_tasks.add(task)

        def discard(completed: asyncio.Task[None]) -> None:
            self._dispatched_control_tasks.discard(completed)
            if self._session_control_tails.get(session_id) is completed:
                self._session_control_tails.pop(session_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)

    async def drain(self) -> None:
        while self._dispatched_control_tasks:
            await asyncio.gather(*tuple(self._dispatched_control_tasks))

    async def shutdown(self) -> None:
        tasks = tuple(
            self._dispatched_control_tasks
            | set(self._control_cleanup_tasks.values())
            | set(self._submission_cleanup_tasks.values())
            | set(self._request_cleanup_tasks.values())
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def coerce_capabilities(raw: dict[str, object]) -> DuplexRuntimeCapabilities:
        input_modes: set[DuplexInputMode] = set()
        values = raw.get("input_modes")
        if values is not None and not isinstance(values, list):
            raise TypeError("duplex input_modes capability must be a list")
        if isinstance(values, list):
            for value in values:
                try:
                    input_modes.add(DuplexInputMode(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"unknown duplex input mode: {value!r}") from exc
        return DuplexRuntimeCapabilities(
            input_modes=input_modes or {DuplexInputMode.TURN_COMMIT_ONLY},
            implementation_level=str(raw.get("implementation_level") or "serving_session_adapter"),
        )

    def sampling_params_for_config(self, runtime_config: dict[str, Any]) -> list[object]:
        defaults = self._stage_port.sampling_defaults()
        if self._extension is None:
            return list(defaults)
        configured = self._extension.configure_sampling_params(
            runtime_config=runtime_config,
            defaults=defaults,
        )
        if not isinstance(configured, tuple):
            raise TypeError("duplex runtime extension must return sampling parameters as a tuple")
        if len(configured) != len(defaults):
            raise ValueError("duplex runtime extension must return one sampling parameter per stage")
        return list(configured)

    @staticmethod
    def stage_request_id(fence: DuplexFence, *, stage_id: int) -> str:
        return duplex_resource_request_id(fence, f"stage{stage_id}")

    def ensure_stage_request(
        self,
        session: DuplexSessionRuntimeState,
        *,
        stage_id: int,
        fence: DuplexFence | None = None,
    ) -> DuplexStageRequestContext | None:
        if stage_id >= self._stage_port.stage_count:
            return None
        effective_fence = fence or session.fence
        request_id = self.stage_request_id(effective_fence, stage_id=stage_id)
        session.reserve_stage_request(stage_id, request_id, fence=effective_fence)
        context = DuplexStageRequestContext(
            request_id=request_id,
            session_id=session.session_id,
            fence=effective_fence,
            stage_id=stage_id,
            final_stage_id=self._stage_port.stage_count - 1,
            config_generation=session.config_generation,
            sampling_params=tuple(self.sampling_params_for_config(session.runtime_config)),
            session_config=session.session_config,
            runtime_config=session.runtime_config,
        )
        self._stage_port.ensure_request(context)
        return context

    async def handle_open(self, message: OpenDuplexSessionMessage) -> None:
        session: DuplexSessionRuntimeState | None = None
        try:
            session_mode = SessionMode(message.session_mode)
            capabilities = self.coerce_capabilities(message.capabilities)
            native_append_modes = capabilities.input_modes - {DuplexInputMode.TURN_COMMIT_ONLY}
            if native_append_modes and self._extension is None:
                raise RuntimeError("duplex_runtime_extension_not_configured")
            session = self.sessions.open_session(
                message.fence,
                capabilities=capabilities,
                session_config=message.session_config,
                runtime_config=message.runtime_config,
                lease_config=self._lease_config,
            )
            request_context = self.ensure_stage_request(session, stage_id=0) if self._extension is not None else None
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="open",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "implementation_level": capabilities.implementation_level,
                            "data_plane_session": True,
                            "session_mode": session_mode.value,
                            "scheduler_request_context": request_context is not None,
                            "request_id": request_context.request_id if request_context is not None else None,
                        },
                    }
                ],
            )
        except Exception as exc:
            if self._control_error(exc).code == "resource_exhausted":
                logger.info("open_duplex_session rejected: %s", exc)
            else:
                logger.exception("open_duplex_session failed: %s", exc)
            if session is not None and self.sessions.get(message.session_id) is session:
                reserved_request_ids = tuple(session.resource_request_ids())
                self.sessions.begin_close_session(session.fence, reason="open_rollback")
                cleanup_key = self._cleanup_key("open_rollback", session.fence)
                pending = _PendingControlCleanup(
                    kind="open_rollback",
                    session_id=session.session_id,
                    fence=session.fence,
                    reserved_request_ids=reserved_request_ids,
                    submitted_request_ids=(),
                )
                self._pending_control_cleanups[cleanup_key] = pending
                try:
                    await self._complete_control_cleanup(cleanup_key, pending)
                except Exception as cleanup_exc:
                    logger.warning(
                        "duplex open rollback remains pending for session %s: %s",
                        session.session_id,
                        cleanup_exc,
                    )
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="open",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
            )

    async def handle_append(self, message: AppendDuplexInputMessage) -> None:
        session: DuplexSessionRuntimeState | None = None
        lease_operation_id = f"append:{message.control_id}"
        operation_started = False
        try:
            session = self.sessions.require(message.session_id)
            await self._complete_pending_submission_cleanup(message.session_id)
            if message.expected_epoch is not None and message.expected_epoch != message.fence.epoch:
                raise ValueError("expected_epoch must match fence.epoch")
            mode = DuplexInputMode(message.mode)
            if message.operation_id is not None:
                completed = session.completed_append(
                    message.operation_id,
                    fence=message.fence,
                    mode=mode,
                    final=message.final,
                )
                if completed is not None:
                    session.touch(message.fence, DuplexLeaseActivity.APPEND)
                    await self.put_result(
                        message.control_id,
                        fence=message.fence,
                        operation="append",
                        session_id=message.session_id,
                        stage_results=completed,
                    )
                    return
            session.begin_operation(message.fence, lease_operation_id)
            operation_started = True
            reservation = session.prepare_append(mode=mode, fence=message.fence)
            stage_results = await self.append_via_data_plane(
                message,
                session=session,
                reservation=reservation,
                mode=mode,
            )
            if message.operation_id is not None:
                session.record_completed_append(
                    message.operation_id,
                    fence=message.fence,
                    mode=mode,
                    final=message.final,
                    stage_results=stage_results,
                )
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="append",
                session_id=message.session_id,
                stage_results=stage_results,
            )
        except Exception as exc:
            logger.exception("append_duplex_input failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="append",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
            )
        finally:
            if session is not None and operation_started and lease_operation_id in session.lease.active_operations:
                session.end_operation(session.fence, lease_operation_id)

    async def append_via_data_plane(
        self,
        message: AppendDuplexInputMessage,
        *,
        session: DuplexSessionRuntimeState,
        reservation: DuplexAppendReservation,
        mode: DuplexInputMode,
    ) -> list[dict[str, object]]:
        if self._extension is None and mode is DuplexInputMode.TURN_COMMIT_ONLY:
            update = session.commit_append(reservation)
            return [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {
                        "supported": True,
                        "data_plane_append": False,
                        "seq": update.seq,
                        "turn_id": update.turn_id,
                        "turn_seq": update.turn_seq,
                        "mode": mode.value,
                    },
                }
            ]
        if self._stage_port.stage_count == 0:
            return [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": False, "error": "duplex_data_plane_has_no_stage"},
                }
            ]
        if self._extension is None:
            raise RuntimeError("duplex_runtime_extension_not_configured")

        stage_id = 0
        request_id = self.stage_request_id(message.fence, stage_id=stage_id)
        existing_binding = session.stage_bindings.get(stage_id)
        already_submitted = existing_binding is not None and existing_binding.request_id == request_id
        request_context = self.ensure_stage_request(
            session,
            stage_id=stage_id,
            fence=message.fence,
        )
        if request_context is None:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        append_plan = self._extension.plan_append(
            request_id=request_id,
            fence=message.fence,
            session_config=dict(request_context.session_config),
            runtime_config=dict(request_context.runtime_config),
            seq=reservation.update.seq,
            turn_seq=reservation.update.turn_seq,
            mode=mode,
            payload=message.payload,
            final=message.final,
            sampling_params=request_context.stage_sampling_params,
        )
        if not isinstance(append_plan, DuplexAppendPlan):
            raise TypeError("duplex runtime extension plan_append() must return DuplexAppendPlan")
        submission = DuplexStageSubmission(
            context=request_context,
            prompt=append_plan.prompt,
            already_submitted=already_submitted,
        )
        submission_result = await self._stage_port.submit(submission)
        try:
            if submission_result.request_id != request_id or submission_result.stage_id != stage_id:
                raise RuntimeError("duplex stage adapter returned a mismatched submission result")
            update = session.commit_append(reservation)
            session.bind_stage_request(stage_id, request_id, fence=message.fence)
        except BaseException:
            self._pending_submission_cleanups[session.session_id] = _PendingSubmissionCleanup(
                session_id=session.session_id,
                request_ids=(request_id,),
            )
            try:
                await self._complete_pending_submission_cleanup(session.session_id)
            except Exception as cleanup_exc:
                logger.warning(
                    "duplex append compensation remains pending for session %s: %s",
                    session.session_id,
                    cleanup_exc,
                )
            raise
        return [
            {
                "stage_id": stage_id,
                "replica_id": submission_result.replica_id,
                "result": {
                    "supported": True,
                    "implementation_level": session.capabilities.implementation_level,
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": request_context.final_stage_id,
                    "seq": update.seq,
                    "turn_id": update.turn_id,
                    "response_seq": message.fence.response_seq,
                    "turn_seq": update.turn_seq,
                    "mode": mode.value,
                    "resumable": True,
                },
            }
        ]

    async def handle_signal(self, message: SignalDuplexTurnMessage) -> None:
        try:
            cancel_events = {"barge_in", "input.cancel", "response.cancel"}
            if message.event not in {*cancel_events, "session.update"}:
                raise ValueError(f"unsupported duplex runtime signal: {message.event}")
            session = self.sessions.require(message.session_id)
            await self._complete_pending_submission_cleanup(message.session_id)
            effective_next_fence = message.next_fence
            if message.event in cancel_events:
                if effective_next_fence is None:
                    raise ValueError(f"{message.event} requires next_fence")
                cleanup_key = self._cleanup_key("cancel", message.fence)
                pending = self._pending_control_cleanups.get(cleanup_key)
                if pending is None:
                    stale_request_ids = session.prepare_cancel_fence(message.fence, effective_next_fence)
                    pending = _PendingControlCleanup(
                        kind="cancel",
                        session_id=message.session_id,
                        fence=message.fence,
                        submitted_request_ids=tuple(stale_request_ids),
                    )
                    self._pending_control_cleanups[cleanup_key] = pending
                await self._complete_control_cleanup(cleanup_key, pending)
            else:
                session.accept_fence(message.fence)
            if message.runtime_config is not None:
                self.sampling_params_for_config(message.runtime_config)
            session.replace_configs(
                session_config=message.session_config,
                runtime_config=message.runtime_config,
            )
            session.touch(session.fence, DuplexLeaseActivity.SIGNAL)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="signal",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "data_plane_signal": True,
                            "event": message.event,
                            "fence": message.fence,
                            "next_fence": effective_next_fence,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("signal_duplex_turn failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="signal",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
            )

    async def handle_close(self, message: CloseDuplexSessionMessage) -> None:
        try:
            await self._complete_pending_submission_cleanup(message.session_id)
            session = self.sessions.get(message.session_id)
            if session is None:
                await self.put_result(
                    message.control_id,
                    fence=message.fence,
                    operation="close",
                    session_id=message.session_id,
                    stage_results=[],
                )
                return
            cleanup_key = self._cleanup_key("close", message.fence)
            pending = self._pending_control_cleanups.get(cleanup_key)
            if pending is None:
                submitted = tuple(session.resource_request_ids(submitted=True))
                reserved = tuple(session.resource_request_ids(submitted=False))
                self.sessions.begin_close_session(message.fence, reason=message.reason)
                pending = _PendingControlCleanup(
                    kind="close",
                    session_id=message.session_id,
                    fence=message.fence,
                    submitted_request_ids=submitted,
                    reserved_request_ids=reserved,
                )
                self._pending_control_cleanups[cleanup_key] = pending
            await self._complete_control_cleanup(cleanup_key, pending)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="close",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "data_plane_close": True,
                            "reason": message.reason,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("close_duplex_session failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="close",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
            )

    async def handle_touch(self, message: TouchDuplexSessionMessage) -> None:
        try:
            session = self.sessions.require(message.session_id)
            activity = DuplexLeaseActivity(message.activity)
            if activity is DuplexLeaseActivity.DETACH:
                session.detach(message.fence)
            else:
                session.touch(message.fence, activity)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="touch",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "activity": activity.value,
                            "lease_generation": session.lease.generation,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("touch_duplex_session failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="touch",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
            )

    async def handle_resume(self, message: ResumeDuplexSessionMessage) -> None:
        try:
            session = self.sessions.require(message.session_id)
            generation = session.resume(
                message.fence,
                expected_lease_generation=message.expected_lease_generation,
            )
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="resume",
                session_id=message.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "lease_generation": generation,
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("resume_duplex_session failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="resume",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
            )

    async def reap_expired(self, now: float | None = None) -> int:
        completed = 0
        for session_id in list(self._pending_submission_cleanups):
            try:
                await self._complete_pending_submission_cleanup(session_id)
            except Exception as exc:
                logger.warning(
                    "duplex append compensation remains pending for session %s: %s",
                    session_id,
                    exc,
                )
        for key, pending in list(self._pending_control_cleanups.items()):
            try:
                await self._complete_control_cleanup(key, pending)
            except Exception as exc:
                logger.warning(
                    "duplex %s cleanup remains pending for session %s: %s",
                    pending.kind,
                    pending.session_id,
                    exc,
                )
        for key, pending in list(self._pending_request_cleanups.items()):
            if key in self._request_cleanups_in_progress:
                continue
            try:
                await self._complete_request_cleanup(key, pending)
            except Exception as exc:
                logger.warning(
                    "duplex request cleanup remains pending for session %s: %s",
                    pending.session_id,
                    exc,
                )
                continue
            completed += 1
        for item in self.sessions.collect_expired(
            now,
            excluded_session_ids=set(self._pending_submission_cleanups),
        ):
            self._pending_expirations.setdefault(
                (item.session_id, item.lease_generation),
                item,
            )
        for key, item in list(self._pending_expirations.items()):
            try:
                if item.submitted_request_ids:
                    await self._stage_port.cleanup(list(item.submitted_request_ids), abort=True)
                if item.reserved_request_ids:
                    await self._stage_port.cleanup(list(item.reserved_request_ids))
                if self._lifecycle_sink is not None:
                    await self._lifecycle_sink.put(
                        DuplexSessionLifecycleMessage(
                            fence=item.fence,
                            session_id=item.session_id,
                            event="expired",
                            reason=item.reason,
                            lease_generation=item.lease_generation,
                            submitted_request_ids=list(item.submitted_request_ids),
                            reserved_request_ids=list(item.reserved_request_ids),
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "duplex expiry cleanup remains pending for session %s: %s",
                    item.session_id,
                    exc,
                )
                continue
            session = self.sessions.get(item.session_id)
            if session is not None and session.lease.generation == item.lease_generation:
                self.sessions.finalize_close_session(session)
            self._pending_expirations.pop(key, None)
            completed += 1
        return completed

    async def _complete_pending_submission_cleanup(self, session_id: str) -> None:
        pending = self._pending_submission_cleanups.get(session_id)
        if pending is None:
            return
        task = self._submission_cleanup_tasks.get(session_id)
        if task is not None and task.done():
            self._submission_cleanup_tasks.pop(session_id, None)
            task = None
        if task is None:
            task = asyncio.create_task(
                self._run_submission_cleanup(pending),
                name=f"duplex-append-compensation-{session_id}",
            )
            self._submission_cleanup_tasks[session_id] = task

            def discard(completed: asyncio.Task[None]) -> None:
                if self._submission_cleanup_tasks.get(session_id) is completed:
                    self._submission_cleanup_tasks.pop(session_id, None)

            task.add_done_callback(discard)
        await asyncio.shield(task)

    async def _run_submission_cleanup(self, pending: _PendingSubmissionCleanup) -> None:
        await self._stage_port.cleanup(list(pending.request_ids), abort=True)
        session = self.sessions.get(pending.session_id)
        if session is not None:
            session.release_request_ids(list(pending.request_ids))
        if self._pending_submission_cleanups.get(pending.session_id) is pending:
            self._pending_submission_cleanups.pop(pending.session_id, None)

    async def _complete_request_cleanup(
        self,
        key: tuple[str, int, int],
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

    async def _run_request_cleanup(
        self,
        key: tuple[str, int, int],
        pending: _PendingRequestCleanup,
    ) -> None:
        await self._stage_port.cleanup(list(pending.request_ids), abort=pending.abort)
        session = self.sessions.get(pending.session_id)
        if (
            session is not None
            and session.fence.incarnation == pending.fence.incarnation
            and session.lease.generation == pending.lease_generation
        ):
            self.sessions.finalize_close_session(session)
        if self._pending_request_cleanups.get(key) is pending:
            self._pending_request_cleanups.pop(key, None)
            self._request_cleanups_in_progress.discard(key)

    @staticmethod
    def _cleanup_key(kind: str, fence: DuplexFence) -> tuple[str, str, int, int, int, int]:
        return (kind, fence.session_id, fence.incarnation, fence.epoch, fence.turn_id, fence.response_seq)

    async def _complete_control_cleanup(
        self,
        key: tuple[str, str, int, int, int, int],
        pending: _PendingControlCleanup,
    ) -> None:
        task = self._control_cleanup_tasks.get(key)
        if task is not None and task.done():
            self._control_cleanup_tasks.pop(key, None)
            task = None
        if task is None:
            task = asyncio.create_task(
                self._run_control_cleanup(key, pending),
                name=f"duplex-{pending.kind}-cleanup-{pending.session_id}",
            )
            self._control_cleanup_tasks[key] = task

            def discard(completed: asyncio.Task[None]) -> None:
                if self._control_cleanup_tasks.get(key) is completed:
                    self._control_cleanup_tasks.pop(key, None)

            task.add_done_callback(discard)
        await asyncio.shield(task)

    async def _run_control_cleanup(
        self,
        key: tuple[str, str, int, int, int, int],
        pending: _PendingControlCleanup,
    ) -> None:
        if pending.submitted_request_ids:
            await self._stage_port.cleanup(list(pending.submitted_request_ids), abort=True)
        if pending.reserved_request_ids:
            await self._stage_port.cleanup(list(pending.reserved_request_ids))
        session = self.sessions.get(pending.session_id)
        if session is not None:
            if pending.kind == "cancel":
                session.release_fence(pending.fence)
            elif pending.kind in {"close", "open_rollback"}:
                self.sessions.finalize_close_session(session)
        self._pending_control_cleanups.pop(key, None)

    @classmethod
    def _iter_result_dicts(cls, result: object):
        if isinstance(result, dict):
            yield result
        elif isinstance(result, list | tuple):
            for item in result:
                yield from cls._iter_result_dicts(item)

    @classmethod
    def _result_counts(cls, stage_results: list[dict[str, object]]) -> tuple[int, int]:
        unsupported_count = 0
        error_count = 0
        for item in stage_results:
            for result in cls._iter_result_dicts(item.get("result")):
                if result.get("supported") is False:
                    unsupported_count += 1
                if result.get("error"):
                    error_count += 1
        return unsupported_count, error_count

    async def put_result(
        self,
        control_id: str,
        *,
        fence: DuplexFence,
        operation: str,
        session_id: str,
        stage_results: list[dict[str, object]],
        error: BaseException | str | None = None,
    ) -> None:
        control_error = self._control_error(error) if error is not None else None
        if error is not None:
            stage_results = [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": False, "error": control_error.message},
                }
            ]
        unsupported_count, error_count = self._result_counts(stage_results)
        session = self.sessions.get(session_id)
        await self._result_sink.put(
            DuplexControlResultMessage(
                control_id=control_id,
                fence=fence,
                operation=operation,
                session_id=session_id,
                ok=error_count == 0 and unsupported_count == 0,
                stage_results=stage_results,
                unsupported_count=unsupported_count,
                error_count=error_count,
                error=control_error,
                accepted_fence=session.fence if session is not None else None,
                lease_generation=session.lease.generation if session is not None else None,
            )
        )

    @staticmethod
    def _control_error(error: BaseException | str) -> DuplexControlError:
        message = str(error)
        if isinstance(error, DuplexFenceMismatchError):
            code = "stale_fence"
            retryable = False
        elif "unknown duplex input mode" in message:
            code = "invalid_capability"
            retryable = False
        elif "duplex_session_capacity_exhausted" in message:
            code = "resource_exhausted"
            retryable = True
        elif isinstance(error, KeyError):
            code = "not_found"
            retryable = False
        elif isinstance(error, (TypeError, ValueError)):
            code = "invalid_argument"
            retryable = False
        elif isinstance(error, TimeoutError):
            code = "timeout"
            retryable = True
        else:
            code = "failed_precondition"
            retryable = False
        return DuplexControlError(code=code, message=message, retryable=retryable)

    def decide_output(
        self,
        stage_id: int,
        output: object,
        context: DuplexOutputContext | None,
    ) -> DuplexOutputDecision | None:
        if context is None or self._extension is None:
            return None
        session = self._sessions.get(context.identity.session_id)
        if session is None:
            return None
        try:
            session.touch(context.identity.fence, DuplexLeaseActivity.MODEL_OUTPUT)
        except RuntimeError:
            return None
        decision = self._extension.decide_output(
            stage_id=stage_id,
            final_stage_id=context.final_stage_id,
            segment_finished=context.segment_finished,
            segment_token_ids=context.segment_token_ids,
            segment_output_metadata=dict(context.segment_output_metadata),
            output=output,
        )
        if decision is not None and not isinstance(decision, DuplexOutputDecision):
            raise TypeError("duplex runtime extension decide_output() must return DuplexOutputDecision or None")
        return decision

    def session_for_identity(self, identity: DuplexRequestIdentity | None) -> DuplexSessionRuntimeState | None:
        if identity is None:
            return None
        return self._sessions.get(identity.session_id)

    def close_sessions_for_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        cleanup_in_progress: bool = False,
    ) -> dict[str, list[str]]:
        closed = self._sessions.close_sessions_for_request_ids(request_ids)
        for session_id, stale_request_ids in closed.items():
            session = self._sessions.get(session_id)
            if session is None:
                continue
            key = (session_id, session.fence.incarnation, session.lease.generation)
            existing = self._pending_request_cleanups.get(key)
            merged_request_ids = tuple(
                dict.fromkeys(
                    [
                        *(existing.request_ids if existing is not None else ()),
                        *stale_request_ids,
                    ]
                )
            )
            self._pending_request_cleanups[key] = _PendingRequestCleanup(
                session_id=session_id,
                fence=session.fence,
                lease_generation=session.lease.generation,
                request_ids=merged_request_ids,
                abort=abort or (existing.abort if existing is not None else False),
            )
            if cleanup_in_progress:
                self._request_cleanups_in_progress.add(key)
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
        self._sessions.finalize_closed_sessions(session_ids)


__all__ = [
    "DuplexControlPlane",
    "DuplexOutputContext",
    "DuplexRequestIdentity",
    "DuplexResultSink",
    "DuplexLifecycleSink",
    "DuplexStagePort",
    "DuplexStageRequestContext",
    "DuplexStageSubmission",
    "DuplexStageSubmissionResult",
]
