# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
import time as _time
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol, TypeGuard

from vllm.logger import init_logger

from vllm_omni.engine.duplex.contracts import (
    DUPLEX_CONTRACT_VERSION,
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
    DuplexTraceEnvelope,
    SessionMode,
    duplex_data_plane_request_info,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity, DuplexLeaseConfig
from vllm_omni.engine.duplex.messages import (
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
from vllm_omni.engine.duplex.session import (
    DuplexAppendReservation,
    DuplexFenceMismatchError,
    DuplexReplayAppend,
    DuplexSessionExpiry,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
    duplex_append_fingerprint,
)

logger = init_logger(__name__)

DuplexCommand = (
    OpenDuplexSessionMessage
    | AppendDuplexInputMessage
    | SignalDuplexTurnMessage
    | CloseDuplexSessionMessage
    | TouchDuplexSessionMessage
    | ResumeDuplexSessionMessage
)


class DuplexControlPreemptedError(RuntimeError):
    """A queued/in-flight control was superseded by cancel or close."""


class DuplexAppendAdmissionError(RuntimeError):
    """A per-session append queue hit its configured hard bound."""


class DuplexOpenAdmissionError(RuntimeError):
    """A session open was rejected by the bounded admission queue."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class DuplexKVRecoveryError(RuntimeError):
    """A deterministic replay failed after the old physical request was lost."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


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


@dataclass
class _PendingSessionOpen:
    message: OpenDuplexSessionMessage
    enqueued_at: float
    sequence: int
    future: asyncio.Future[dict[str, object]]


@dataclass(frozen=True)
class _MaterializedAppend:
    context: DuplexStageRequestContext
    plan: DuplexAppendPlan
    fingerprint: bytes
    replay_append: DuplexReplayAppend | None


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
        max_pending_session_opens: int = 0,
        session_open_queue_timeout_s: float = 30.0,
        session_open_aging_quantum_s: float = 1.0,
        completed_append_limit: int = 256,
        max_pending_appends_per_session: int = 4,
        recovery_max_replay_tokens: int = 8192,
        recovery_max_replay_bytes: int = 64 * 1024 * 1024,
        rollover_trigger_fraction: float = 0.8,
        rollover_retain_tokens: int = 4096,
        metrics: Any = None,
    ) -> None:
        if max_pending_appends_per_session <= 0:
            raise ValueError("max_pending_appends_per_session must be positive")
        if max_pending_session_opens < 0:
            raise ValueError("max_pending_session_opens must be non-negative")
        if session_open_queue_timeout_s <= 0 or session_open_aging_quantum_s <= 0:
            raise ValueError("session open queue timing values must be positive")
        self._extension = extension
        self._stage_port = stage_port
        self._result_sink = result_sink
        self._lifecycle_sink = lifecycle_sink
        self._lease_config = lease_config or DuplexLeaseConfig()
        self._sessions = DuplexSessionRuntimeManager(
            clock=clock,
            max_sessions=max_sessions,
            completed_append_limit=completed_append_limit,
            recovery_max_replay_tokens=recovery_max_replay_tokens,
            recovery_max_replay_bytes=recovery_max_replay_bytes,
            rollover_trigger_fraction=rollover_trigger_fraction,
            rollover_retain_tokens=rollover_retain_tokens,
        )
        self._pending_expirations: dict[tuple[str, int, int], DuplexSessionExpiry] = {}
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
        self._session_control_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._control_task_messages: dict[asyncio.Task[None], object] = {}
        self._control_task_preemption_reasons: dict[asyncio.Task[None], str] = {}
        self._started_control_tasks: set[asyncio.Task[None]] = set()
        self._pending_append_counts: dict[str, int] = {}
        self._max_pending_appends_per_session = max_pending_appends_per_session
        self._max_pending_session_opens = max_pending_session_opens
        self._session_open_queue_timeout_s = session_open_queue_timeout_s
        self._session_open_aging_quantum_s = session_open_aging_quantum_s
        self._pending_session_opens: list[_PendingSessionOpen] = []
        self._next_open_sequence = 0
        self._granted_session_opens = 0
        self._metrics = metrics
        self._dispatched_control_tasks: set[asyncio.Task[None]] = set()
        self._sync_session_metrics()

    @property
    def sessions(self) -> DuplexSessionRuntimeManager:
        return self._sessions

    @property
    def pending_submission_cleanup_count(self) -> int:
        return len(self._pending_submission_cleanups)

    def accepts(self, message: object) -> TypeGuard[DuplexCommand]:
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
        enqueued_at = _time.monotonic()
        is_append = isinstance(message, AppendDuplexInputMessage)
        if is_append and self._pending_append_counts.get(session_id, 0) >= self._max_pending_appends_per_session:

            async def reject_overload() -> None:
                self._metric("observe_duplex_control_queue_wait", "append", 0.0)
                error = DuplexAppendAdmissionError(
                    "duplex_append_admission_exhausted: "
                    f"session={session_id}, limit={self._max_pending_appends_per_session}"
                )
                self._metric("observe_duplex_append", "admission_rejected", 0.0)
                await self.put_result(
                    message.control_id,
                    fence=message.fence,
                    operation="append",
                    session_id=session_id,
                    stage_results=[],
                    error=error,
                )

            task = asyncio.create_task(
                reject_overload(),
                name=f"duplex-control-reject-{session_id}-{message.control_id}",
            )
            self._register_control_task(task, message, count_append=False, update_tail=False)
            return

        preempt_reason = self._preempt_reason(message)
        preempted_tasks: tuple[asyncio.Task[None], ...] = ()
        if preempt_reason is not None:
            preempted_tasks = tuple(self._session_control_tasks.get(session_id, ()))
            for prior_task in preempted_tasks:
                if prior_task.done():
                    continue
                prior_message = self._control_task_messages.get(prior_task)
                if not isinstance(prior_message, AppendDuplexInputMessage):
                    continue
                # A prior cancel may already have interrupted this append and
                # the task may now be publishing its correlated cancellation
                # result.  Cancelling it a second time can interrupt that
                # reply and strand the original RPC waiter forever.  Keep the
                # first preemption reason and let the reply finish.
                if prior_task in self._control_task_preemption_reasons:
                    continue
                self._control_task_preemption_reasons[prior_task] = preempt_reason
                # Task.cancel() before the coroutine's first step bypasses
                # its try/except entirely, losing the correlated reply.
                # An unstarted task publishes the recorded cancellation at
                # entry; only an already-started task needs interruption.
                if prior_task in self._started_control_tasks:
                    prior_task.cancel()
            predecessor = None
        else:
            predecessor = self._session_control_tails.get(session_id)

        async def run_ordered() -> None:
            handle_started = False
            current = asyncio.current_task()
            assert current is not None
            self._started_control_tasks.add(current)
            try:
                if current in self._control_task_preemption_reasons:
                    raise asyncio.CancelledError
                if preempted_tasks:
                    await asyncio.shield(asyncio.gather(*preempted_tasks, return_exceptions=True))
                elif predecessor is not None:
                    # Ordering is not ownership: preempting this append must
                    # not cancel the open/signal/resume it is waiting behind.
                    await asyncio.shield(asyncio.gather(predecessor, return_exceptions=True))
                self._metric(
                    "observe_duplex_control_queue_wait",
                    self._message_operation(message),
                    max(_time.monotonic() - enqueued_at, 0.0),
                )
                handle_started = True
                await self.handle(message)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                reason = self._control_task_preemption_reasons.get(current) if current is not None else None
                if reason is None:
                    raise
                if is_append and not handle_started:
                    self._metric("observe_duplex_append", "cancelled", 0.0)
                await self.put_result(
                    message.control_id,
                    fence=message.fence,
                    operation=self._message_operation(message),
                    session_id=session_id,
                    stage_results=[],
                    error=DuplexControlPreemptedError(
                        f"duplex_control_preempted: operation={self._message_operation(message)}, by={reason}"
                    ),
                )

        task = asyncio.create_task(
            run_ordered(),
            name=f"duplex-control-{session_id}-{message.control_id}",
        )
        self._register_control_task(task, message, count_append=is_append, update_tail=True)

    @staticmethod
    def _message_operation(message: object) -> str:
        if isinstance(message, OpenDuplexSessionMessage):
            return "open"
        if isinstance(message, AppendDuplexInputMessage):
            return "append"
        if isinstance(message, SignalDuplexTurnMessage):
            return "signal"
        if isinstance(message, CloseDuplexSessionMessage):
            return "close"
        if isinstance(message, TouchDuplexSessionMessage):
            return "touch"
        if isinstance(message, ResumeDuplexSessionMessage):
            return "resume"
        return "unknown"

    @staticmethod
    def _preempt_reason(message: object) -> str | None:
        if isinstance(message, CloseDuplexSessionMessage):
            return "close"
        if isinstance(message, SignalDuplexTurnMessage) and message.event in {
            "barge_in",
            "input.cancel",
            "response.cancel",
        }:
            return message.event
        return None

    def _register_control_task(
        self,
        task: asyncio.Task[None],
        message: DuplexCommand,
        *,
        count_append: bool,
        update_tail: bool,
    ) -> None:
        session_id = message.session_id
        if update_tail:
            self._session_control_tails[session_id] = task
            # Only ordered control tasks mutate session state and may need to
            # be preempted by cancel/close. Admission-rejection tasks merely
            # publish a correlated error result; cancelling one can strand its
            # RPC waiter forever.
            self._session_control_tasks.setdefault(session_id, set()).add(task)
        self._control_task_messages[task] = message
        self._dispatched_control_tasks.add(task)
        if count_append:
            self._pending_append_counts[session_id] = self._pending_append_counts.get(session_id, 0) + 1

        def discard(completed: asyncio.Task[None]) -> None:
            self._dispatched_control_tasks.discard(completed)
            self._started_control_tasks.discard(completed)
            self._control_task_preemption_reasons.pop(completed, None)
            self._control_task_messages.pop(completed, None)
            if update_tail:
                session_tasks = self._session_control_tasks.get(session_id)
                if session_tasks is not None:
                    session_tasks.discard(completed)
                    if not session_tasks:
                        self._session_control_tasks.pop(session_id, None)
            if count_append:
                remaining = max(self._pending_append_counts.get(session_id, 1) - 1, 0)
                if remaining:
                    self._pending_append_counts[session_id] = remaining
                else:
                    self._pending_append_counts.pop(session_id, None)
            if self._session_control_tails.get(session_id) is completed:
                self._session_control_tails.pop(session_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)

    async def drain(self) -> None:
        while self._dispatched_control_tasks:
            await asyncio.gather(*tuple(self._dispatched_control_tasks))

    async def shutdown(self) -> None:
        for pending in self._pending_session_opens:
            if not pending.future.done():
                pending.future.set_exception(
                    DuplexOpenAdmissionError("duplex control plane shutdown", reason="shutdown")
                )
        self._pending_session_opens.clear()
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

    def _metric(self, method_name: str, *args: object) -> None:
        method = getattr(self._metrics, method_name, None)
        if not callable(method):
            return
        try:
            method(*args)
        except Exception:
            logger.warning("Failed to publish duplex metric %s", method_name, exc_info=True)

    def _open_capacity_available(self) -> bool:
        limit = self.sessions.max_sessions
        return limit is None or self.sessions.session_count + self._granted_session_opens < limit

    def _admission_status(self, *, mode: str, reason: str, wait_s: float = 0.0) -> dict[str, object]:
        return {
            "decision": "accepted" if mode in {"immediate", "queued"} else "rejected",
            "mode": mode,
            "reason": reason,
            "queue_wait_s": max(wait_s, 0.0),
            "queue_depth": len(self._pending_session_opens),
            "active_sessions": self.sessions.active_session_count,
            "closing_sessions": self.sessions.closing_session_count,
            "capacity_limit": self.sessions.max_sessions,
        }

    def _drain_open_queue(self) -> None:
        if not self._pending_session_opens or self._max_pending_session_opens <= 0:
            return
        now = _time.monotonic()
        while self._pending_session_opens and self._open_capacity_available():
            pending = max(
                self._pending_session_opens,
                key=lambda item: (
                    item.message.admission_priority
                    + max(now - item.enqueued_at, 0.0) / self._session_open_aging_quantum_s,
                    -item.sequence,
                ),
            )
            self._pending_session_opens.remove(pending)
            if pending.future.done():
                continue
            self._granted_session_opens += 1
            pending.future.set_result(
                self._admission_status(
                    mode="queued",
                    reason="capacity_released",
                    wait_s=now - pending.enqueued_at,
                )
            )

    async def _await_open_admission(self, message: OpenDuplexSessionMessage) -> dict[str, object]:
        if self._open_capacity_available():
            return self._admission_status(mode="immediate", reason="capacity_available")
        if self._max_pending_session_opens <= 0:
            raise DuplexOpenAdmissionError(
                "duplex_session_capacity_exhausted: open admission queue is disabled",
                reason="capacity_exhausted",
            )
        if len(self._pending_session_opens) >= self._max_pending_session_opens:
            raise DuplexOpenAdmissionError("duplex_session_admission_queue_full", reason="queue_full")
        loop = asyncio.get_running_loop()
        pending = _PendingSessionOpen(
            message=message,
            enqueued_at=_time.monotonic(),
            sequence=self._next_open_sequence,
            future=loop.create_future(),
        )
        self._next_open_sequence += 1
        self._pending_session_opens.append(pending)
        try:
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout=self._session_open_queue_timeout_s)
        except asyncio.TimeoutError as exc:
            if pending in self._pending_session_opens:
                self._pending_session_opens.remove(pending)
            raise DuplexOpenAdmissionError("duplex_session_admission_queue_timeout", reason="queue_timeout") from exc
        except asyncio.CancelledError:
            if pending in self._pending_session_opens:
                self._pending_session_opens.remove(pending)
            raise
        finally:
            if pending.future.done() and self._granted_session_opens > 0:
                self._granted_session_opens -= 1

    def _sync_session_metrics(self) -> None:
        self._metric(
            "set_duplex_sessions",
            self._sessions.active_session_count,
            self._sessions.closing_session_count,
        )
        active_context = [
            session.context_ledger()
            for session in self._sessions.iter_sessions()
            if session.lease.terminal_reason is None
        ]
        if active_context:
            max_tokens = max(item.context_tokens for item in active_context)
            limits = [item.context_limit for item in active_context if item.context_limit is not None]
            max_limit = max(limits) if limits else 0
            utilizations = [item.context_utilization for item in active_context if item.context_utilization is not None]
            max_utilization = max(utilizations) if utilizations else 0.0
            max_receipts = max(item.completed_append_count for item in active_context)
        else:
            max_tokens = 0
            max_limit = 0
            max_utilization = 0.0
            max_receipts = 0
        self._metric("set_duplex_context", max_tokens, max_limit, max_utilization, max_receipts)
        active_sessions = [
            session
            for session in self._sessions.iter_sessions()
            if session.lease.terminal_reason is None and session.capabilities.prompt_replay
        ]
        self._metric(
            "set_duplex_recovery_state",
            max((session.replay_token_count for session in active_sessions), default=0),
            max((session.replay_byte_count for session in active_sessions), default=0),
            max((session.resource_generation for session in active_sessions), default=0),
        )
        self._drain_open_queue()

    def _record_scheduler_metrics(
        self,
        session_id: str,
        scheduler_metrics: Mapping[str, int | float | bool | str],
        *,
        reset: bool = False,
    ) -> None:
        raw_tokens = scheduler_metrics.get("omni_context_tokens", scheduler_metrics.get("num_prompt_tokens", 0))
        raw_limit = scheduler_metrics.get("omni_context_limit")
        try:
            tokens = max(int(raw_tokens), 0)
            context_limit = int(raw_limit) if raw_limit is not None else None
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid scheduler-native metrics for session %s: %r",
                session_id,
                scheduler_metrics,
            )
            return
        session = self.sessions.get(session_id)
        if session is not None:
            session.update_scheduler_context(tokens=tokens, limit=context_limit)
        self._sync_session_metrics()

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
        raw_contract_version = raw.get("contract_version")
        if raw_contract_version is not None and not isinstance(raw_contract_version, str):
            raise TypeError("duplex contract_version must be a string")
        raw_adapter_id = raw.get("adapter_id")
        if raw_adapter_id is not None and not isinstance(raw_adapter_id, str):
            raise TypeError("duplex adapter_id must be a string")
        raw_runtime_extension_id = raw.get("runtime_extension_id")
        if raw_runtime_extension_id is not None and not isinstance(raw_runtime_extension_id, str):
            raise TypeError("duplex runtime_extension_id must be a string")
        contract_version = raw_contract_version or DUPLEX_CONTRACT_VERSION
        adapter_id = raw_adapter_id or ""
        runtime_extension_id = raw_runtime_extension_id or ""
        raw_stage_count = raw.get("stage_count")
        if raw_stage_count is None:
            stage_count = None
        elif isinstance(raw_stage_count, bool) or not isinstance(raw_stage_count, int):
            raise TypeError("duplex plugin descriptor stage_count must be an integer")
        else:
            stage_count = raw_stage_count
        prompt_replay = raw.get("supports_prompt_replay", False)
        if not isinstance(prompt_replay, bool):
            raise TypeError("duplex supports_prompt_replay must be a boolean")
        capabilities = DuplexRuntimeCapabilities(
            input_modes=input_modes or {DuplexInputMode.TURN_COMMIT_ONLY},
            implementation_level=str(raw.get("implementation_level") or "serving_session_adapter"),
            scheduler_native_append=bool(raw.get("supports_scheduler_native_append", False)),
            prompt_replay=prompt_replay,
            contract_version=contract_version,
            adapter_id=adapter_id,
            runtime_extension_id=runtime_extension_id,
            stage_count=stage_count,
        )
        capabilities.plugin_descriptor()
        return capabilities

    def _validate_plugin_descriptor(self, capabilities: DuplexRuntimeCapabilities) -> None:
        descriptor = capabilities.plugin_descriptor()
        if descriptor is None:
            return
        if descriptor.stage_count != self._stage_port.stage_count:
            raise RuntimeError(
                "duplex_plugin_stage_count_mismatch: "
                f"descriptor={descriptor.stage_count}, engine={self._stage_port.stage_count}"
            )
        if self._extension is None:
            raise RuntimeError("duplex_plugin_runtime_extension_missing")
        extension_id = getattr(self._extension, "runtime_extension_id", None)
        adapter_id = getattr(self._extension, "adapter_id", None)
        if not isinstance(extension_id, str) or not extension_id:
            raise RuntimeError("duplex_plugin_runtime_extension_identity_missing")
        if not isinstance(adapter_id, str) or not adapter_id:
            raise RuntimeError("duplex_plugin_adapter_identity_missing")
        if extension_id != descriptor.runtime_extension_id:
            raise RuntimeError(
                "duplex_plugin_runtime_extension_mismatch: "
                f"descriptor={descriptor.runtime_extension_id!r}, engine={extension_id!r}"
            )
        if adapter_id != descriptor.adapter_id:
            raise RuntimeError(
                f"duplex_plugin_adapter_mismatch: descriptor={descriptor.adapter_id!r}, engine={adapter_id!r}"
            )

    def _validate_engine_capabilities(self, capabilities: DuplexRuntimeCapabilities) -> None:
        if not capabilities.scheduler_native_append:
            if capabilities.implementation_level == "model_native_duplex":
                raise RuntimeError("scheduler_native_append_not_supported_by_engine_stage_0")
            return
        supports_native_append = getattr(self._stage_port, "supports_scheduler_native_append", None)
        if not callable(supports_native_append) or not supports_native_append(0):
            raise RuntimeError("scheduler_native_append_not_supported_by_engine_stage_0")

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
    def stage_request_id(fence: DuplexFence, *, stage_id: int, resource_generation: int = 0) -> str:
        role = f"stage{stage_id}" if resource_generation <= 0 else f"stage{stage_id}g{resource_generation}"
        return duplex_resource_request_id(fence, role)

    def _stage_request_context(
        self,
        session: DuplexSessionRuntimeState,
        *,
        stage_id: int,
        fence: DuplexFence | None = None,
        recovery_replay: bool = False,
        resource_generation: int | None = None,
        trace: DuplexTraceEnvelope | None = None,
    ) -> DuplexStageRequestContext | None:
        if stage_id >= self._stage_port.stage_count:
            return None
        effective_fence = fence or session.fence
        request_id = self.stage_request_id(
            effective_fence,
            stage_id=stage_id,
            resource_generation=(session.resource_generation if resource_generation is None else resource_generation),
        )
        effective_trace = trace or DuplexTraceEnvelope(
            session_id=session.session_id,
            fence=effective_fence,
            event="stage_request",
            request_id=request_id,
            sequence=session.input_seq,
        )
        if effective_trace.request_id != request_id:
            effective_trace = replace(effective_trace, request_id=request_id)
        return DuplexStageRequestContext(
            request_id=request_id,
            session_id=session.session_id,
            fence=effective_fence,
            stage_id=stage_id,
            final_stage_id=self._stage_port.stage_count - 1,
            config_generation=session.config_generation,
            sampling_params=tuple(self.sampling_params_for_config(session.runtime_config)),
            scheduler_native_append=session.capabilities.scheduler_native_append and stage_id == 0,
            recovery_replay=recovery_replay,
            session_config=session.session_config,
            runtime_config=session.runtime_config,
            trace=effective_trace,
        )

    def ensure_stage_request(
        self,
        session: DuplexSessionRuntimeState,
        *,
        stage_id: int,
        fence: DuplexFence | None = None,
        recovery_replay: bool = False,
    ) -> DuplexStageRequestContext | None:
        context = self._stage_request_context(
            session,
            stage_id=stage_id,
            fence=fence,
            recovery_replay=recovery_replay,
        )
        if context is None:
            return None
        session.reserve_stage_request(stage_id, context.request_id, fence=context.fence)
        self._stage_port.ensure_request(context)
        return context

    def _materialize_append(
        self,
        message: AppendDuplexInputMessage,
        *,
        session: DuplexSessionRuntimeState,
        reservation: DuplexAppendReservation,
        mode: DuplexInputMode,
        resource_generation: int,
    ) -> _MaterializedAppend:
        context = self._stage_request_context(
            session,
            stage_id=0,
            fence=message.fence,
            resource_generation=resource_generation,
            trace=DuplexTraceEnvelope(
                session_id=session.session_id,
                fence=message.fence,
                event="append",
                control_id=message.control_id,
                operation_id=message.operation_id,
                sequence=reservation.update.seq,
            ),
        )
        if context is None:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        if self._extension is None:
            raise RuntimeError("duplex_runtime_extension_not_configured")
        plan = self._extension.plan_append(
            request_id=context.request_id,
            fence=message.fence,
            session_config=dict(context.session_config),
            runtime_config=dict(context.runtime_config),
            seq=reservation.update.seq,
            turn_seq=reservation.update.turn_seq,
            mode=mode,
            payload=message.payload,
            final=message.final,
            sampling_params=context.stage_sampling_params,
        )
        if not isinstance(plan, DuplexAppendPlan):
            raise TypeError("duplex runtime extension plan_append() must return DuplexAppendPlan")
        fingerprint = duplex_append_fingerprint(
            mode=mode,
            payload=message.payload,
            final=message.final,
            config_generation=session.config_generation,
            request_metadata={
                "prompt": plan.prompt,
                "session_config": context.session_config,
                "runtime_config": context.runtime_config,
                "sampling_params": context.stage_sampling_params,
            },
        )
        replay_append = None
        if session.capabilities.prompt_replay:
            replay_append = session.prepare_replay_append(
                operation_id=message.operation_id,
                operation_fingerprint=fingerprint,
                prompt=plan.prompt,
            )
        return _MaterializedAppend(
            context=context,
            plan=plan,
            fingerprint=fingerprint,
            replay_append=replay_append,
        )

    async def handle_open(self, message: OpenDuplexSessionMessage) -> None:
        session: DuplexSessionRuntimeState | None = None
        admission: dict[str, object] | None = None
        try:
            if not isinstance(message.admission_priority, int) or isinstance(message.admission_priority, bool):
                raise ValueError("admission_priority must be an integer")
            if not -1000 <= message.admission_priority <= 1000:
                raise ValueError("admission_priority must be between -1000 and 1000")
            session_mode = SessionMode(message.session_mode)
            capabilities = self.coerce_capabilities(message.capabilities)
            self._validate_plugin_descriptor(capabilities)
            self._validate_engine_capabilities(capabilities)
            native_append_modes = capabilities.input_modes - {DuplexInputMode.TURN_COMMIT_ONLY}
            if native_append_modes and self._extension is None:
                raise RuntimeError("duplex_runtime_extension_not_configured")
            admission = await self._await_open_admission(message)
            if admission.get("mode") == "queued":
                self._metric("observe_duplex_control_queue_wait", "open", admission.get("queue_wait_s", 0.0))
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
                admission=admission,
            )
        except Exception as exc:
            if isinstance(exc, DuplexOpenAdmissionError):
                admission = self._admission_status(mode="rejected", reason=exc.reason)
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
                admission=admission,
            )
        finally:
            self._sync_session_metrics()

    async def handle_append(self, message: AppendDuplexInputMessage) -> None:
        session: DuplexSessionRuntimeState | None = None
        lease_operation_id = f"append:{message.control_id}"
        operation_started = False
        started_at = _time.monotonic()
        outcome = "error"
        try:
            session = self.sessions.require(message.session_id)
            await self._complete_pending_submission_cleanup(message.session_id)
            if message.expected_epoch is not None and message.expected_epoch != message.fence.epoch:
                raise ValueError("expected_epoch must match fence.epoch")
            mode = DuplexInputMode(message.mode)
            operation_fingerprint = duplex_append_fingerprint(
                mode=mode,
                payload=message.payload,
                final=message.final,
                config_generation=session.config_generation,
            )
            if session.capabilities.scheduler_native_append and not message.operation_id:
                raise ValueError("scheduler-native append requires a non-empty operation_id")
            if message.operation_id is not None:
                completed = session.completed_append(
                    message.operation_id,
                    fence=message.fence,
                    mode=mode,
                    final=message.final,
                    operation_fingerprint=operation_fingerprint,
                    config_generation=session.config_generation,
                )
                if completed is not None:
                    outcome = "deduplicated"
                    session.touch(message.fence, DuplexLeaseActivity.APPEND)
                    await self.put_result(
                        message.control_id,
                        fence=message.fence,
                        operation="append",
                        session_id=message.session_id,
                        stage_results=self._deduplicated_stage_results(completed),
                        operation_id=message.operation_id,
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
            outcome = "success"
            for stage_result in stage_results:
                result = stage_result.get("result")
                if not isinstance(result, dict):
                    continue
                append_metrics = result.get("append_metrics")
                if isinstance(append_metrics, Mapping):
                    typed_metrics = {
                        str(key): value
                        for key, value in append_metrics.items()
                        if isinstance(value, int | float | bool | str)
                    }
                    self._record_scheduler_metrics(message.session_id, typed_metrics)
                    if bool(typed_metrics.get("deduplicated", False)):
                        outcome = "deduplicated"
            if message.operation_id is not None:
                session.record_completed_append(
                    message.operation_id,
                    fence=message.fence,
                    mode=mode,
                    final=message.final,
                    stage_results=stage_results,
                    operation_fingerprint=operation_fingerprint,
                    config_generation=session.config_generation,
                )
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="append",
                session_id=message.session_id,
                stage_results=stage_results,
                operation_id=message.operation_id,
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            control_error = self._control_error(exc)
            outcome = (
                control_error.code
                if control_error.code
                in {
                    "timeout",
                    "resource_exhausted",
                    "replica_lost",
                    "kv_recovery_failed",
                }
                else "error"
            )
            if control_error.code == "replica_lost" and session is not None:
                try:
                    await self._terminate_replica_lost_session(session)
                except Exception as cleanup_exc:
                    logger.warning(
                        "native duplex replica-loss cleanup remains pending for session %s: %s",
                        session.session_id,
                        cleanup_exc,
                    )
            logger.exception("append_duplex_input failed: %s", exc)
            await self.put_result(
                message.control_id,
                fence=message.fence,
                operation="append",
                session_id=message.session_id,
                stage_results=[],
                error=exc,
                operation_id=message.operation_id,
            )
        finally:
            if session is not None and operation_started and lease_operation_id in session.lease.active_operations:
                session.end_operation(session.fence, lease_operation_id)
            self._metric(
                "observe_duplex_append",
                outcome,
                max(_time.monotonic() - started_at, 0.0),
            )

    async def _terminate_replica_lost_session(self, session: DuplexSessionRuntimeState) -> None:
        submitted = tuple(session.resource_request_ids(submitted=True))
        reserved = tuple(session.resource_request_ids(submitted=False))
        self.sessions.begin_close_session(session.fence, reason="native_kv_replica_lost")
        self._sync_session_metrics()
        cleanup_key = self._cleanup_key("replica_lost", session.fence)
        pending = self._pending_control_cleanups.get(cleanup_key)
        if pending is None:
            pending = _PendingControlCleanup(
                kind="replica_lost",
                session_id=session.session_id,
                fence=session.fence,
                submitted_request_ids=submitted,
                reserved_request_ids=reserved,
            )
            self._pending_control_cleanups[cleanup_key] = pending
            self._metric("inc_duplex_replica_affinity_loss", 1)
        await self._complete_control_cleanup(cleanup_key, pending)

    @staticmethod
    def _deduplicated_stage_results(stage_results: list[dict[str, object]]) -> list[dict[str, object]]:
        deduplicated: list[dict[str, object]] = []
        for stage_result in stage_results:
            copied_stage_result = dict(stage_result)
            result = copied_stage_result.get("result")
            if isinstance(result, dict):
                copied_result = dict(result)
                copied_result["deduplicated"] = True
                append_metrics = copied_result.get("append_metrics")
                if isinstance(append_metrics, dict):
                    copied_result["append_metrics"] = {**append_metrics, "deduplicated": True}
                copied_stage_result["result"] = copied_result
            deduplicated.append(copied_stage_result)
        return deduplicated

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
        materialized = self._materialize_append(
            message,
            session=session,
            reservation=reservation,
            mode=mode,
            resource_generation=session.resource_generation,
        )
        current_prompt_override: dict[str, Any] | None = None
        replay_window_rollover = materialized.replay_append is not None and session.replay_window_rollover_needed(
            materialized.replay_append
        )
        rebuild_reason = session.recovery_reason or "replica_loss" if session.recovery_required else None
        if rebuild_reason is not None or replay_window_rollover:
            if not session.capabilities.prompt_replay:
                raise RuntimeError("duplex prompt replay is not supported by this model")
            target_generation = session.resource_generation + 1
            materialized = self._materialize_append(
                message,
                session=session,
                reservation=reservation,
                mode=mode,
                resource_generation=target_generation,
            )
            candidate_replay = materialized.replay_append
            if candidate_replay is None:
                raise RuntimeError("duplex recovery requires scheduler-native append")
            compact_journal = replay_window_rollover
            retained = list(session.replay_appends)
            if compact_journal:
                retained = session.compacted_replay_appends()
            retained, prepared_replay_prompts, current_prompt_override = self._fit_replay_appends_for_rebuild(
                session,
                retained,
                candidate_replay,
                request_id=materialized.context.request_id,
            )
            compact_journal = compact_journal or retained != list(session.replay_appends)
            await self._rebuild_scheduler_native_session(
                session,
                tuple(retained),
                deadline_monotonic=message.deadline_monotonic,
                reason=("context_rollover" if rebuild_reason is None else rebuild_reason),
                replace_journal=compact_journal,
                prepared_replay_prompts=prepared_replay_prompts,
            )

        request_context = self.ensure_stage_request(session, stage_id=stage_id, fence=message.fence)
        if request_context is None:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        if request_context.request_id != materialized.context.request_id:
            raise RuntimeError("duplex append materialization generation changed before submission")
        request_id = request_context.request_id
        existing_binding = session.stage_bindings.get(stage_id)
        already_submitted = existing_binding is not None and existing_binding.request_id == request_id
        replay_append = materialized.replay_append
        if replay_append is not None and session.replay_append_would_overflow(replay_append):
            raise RuntimeError("duplex_recovery_journal_capacity_exhausted")
        submission = DuplexStageSubmission(
            context=request_context,
            prompt=current_prompt_override or materialized.plan.prompt,
            already_submitted=already_submitted,
            operation_id=message.operation_id,
            operation_fingerprint=materialized.fingerprint,
            deadline_monotonic=message.deadline_monotonic,
            recovery_replay=False,
        )
        try:
            submission_result = await self._stage_port.submit(submission)
        except BaseException:
            # A failed initial admission may have reached EngineCore before its
            # reply was lost. Abort that uncertain request. An update is kept
            # alive instead: retrying the same operation_id is scheduler-
            # deduplicated and preserves its resident KV state.
            if not already_submitted:
                self._pending_submission_cleanups[session.session_id] = _PendingSubmissionCleanup(
                    session_id=session.session_id,
                    request_ids=(request_id,),
                )
                try:
                    await self._complete_pending_submission_cleanup(
                        session.session_id,
                        deadline_monotonic=message.deadline_monotonic,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "duplex initial append compensation remains pending for session %s: %s",
                        session.session_id,
                        cleanup_exc,
                    )
            raise
        try:
            if submission_result.request_id != request_id or submission_result.stage_id != stage_id:
                raise RuntimeError("duplex stage adapter returned a mismatched submission result")
            update = session.commit_append(reservation)
            session.bind_stage_request(stage_id, request_id, fence=message.fence)
            if replay_append is not None:
                session.record_replay_append(replay_append)
        except BaseException:
            self._pending_submission_cleanups[session.session_id] = _PendingSubmissionCleanup(
                session_id=session.session_id,
                request_ids=(request_id,),
            )
            try:
                await self._complete_pending_submission_cleanup(
                    session.session_id,
                    deadline_monotonic=message.deadline_monotonic,
                )
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
                    "trace": asdict(
                        replace(
                            request_context.trace,
                            event="append_submitted",
                            request_id=request_id,
                            sequence=update.seq,
                        )
                    )
                    if request_context.trace is not None
                    else None,
                    "response_seq": message.fence.response_seq,
                    "turn_seq": update.turn_seq,
                    "mode": mode.value,
                    "resumable": True,
                    "append_metrics": dict(submission_result.metrics),
                },
            }
        ]

    def _prepare_recovery_prompt(
        self,
        append: DuplexReplayAppend,
        *,
        request_id: str,
        initial: bool,
        recovery_replay: bool = True,
    ) -> dict[str, Any]:
        """Rebase a prompt, suppressing outputs only for historical replay.

        An empty retained journal also rebases the *live* candidate to restore
        its session prefix. That candidate must retain normal output routing.
        """
        prepare = getattr(self._extension, "prepare_recovery_prompt", None)
        if callable(prepare):
            prepared = prepare(
                prompt=dict(append.prompt),
                request_id=request_id,
                initial=initial,
            )
            if not isinstance(prepared, Mapping):
                raise TypeError("duplex runtime recovery prompt must be a mapping")
            replay_prompt = deepcopy(dict(prepared))
        else:
            replay_prompt = deepcopy(dict(append.prompt))
        model_buffer = replay_prompt.get("model_intermediate_buffer")
        if isinstance(model_buffer, dict):
            model_buffer["request_id"] = request_id
            duplex = model_buffer.get("duplex")
            if isinstance(duplex, dict):
                duplex["recovery_replay"] = recovery_replay
        return replay_prompt

    @staticmethod
    def _prompt_token_count(prompt: Mapping[str, Any]) -> int:
        token_ids = prompt.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("duplex recovery prompt requires prompt_token_ids")
        return len(token_ids)

    def _fit_replay_appends_for_rebuild(
        self,
        session: DuplexSessionRuntimeState,
        replay_appends: list[DuplexReplayAppend],
        candidate: DuplexReplayAppend,
        *,
        request_id: str,
    ) -> tuple[list[DuplexReplayAppend], tuple[dict[str, Any], ...], dict[str, Any] | None]:
        """Trim oldest complete units until a new generation is admissible.

        Journal token counts describe logical append units.  The first unit of
        a new physical request may be larger because a model-specific recovery
        hook restores the session prefix.  Account for that exact materialized
        size and for the live candidate before destroying the old generation.
        """
        retained = list(replay_appends)
        while True:
            logical_tokens = sum(item.token_count for item in retained) + candidate.token_count
            logical_bytes = sum(item.byte_count for item in retained) + candidate.byte_count
            if retained:
                prepared_replay_prompts = tuple(
                    self._prepare_recovery_prompt(
                        append,
                        request_id=request_id,
                        initial=index == 0,
                    )
                    for index, append in enumerate(retained)
                )
                physical_tokens = sum(self._prompt_token_count(prompt) for prompt in prepared_replay_prompts)
                physical_tokens += candidate.token_count
                current_prompt_override = None
            else:
                prepared_replay_prompts = ()
                current_prompt_override = self._prepare_recovery_prompt(
                    candidate,
                    request_id=request_id,
                    initial=True,
                    recovery_replay=False,
                )
                physical_tokens = self._prompt_token_count(current_prompt_override)

            within_journal = (
                logical_tokens <= session.recovery_max_replay_tokens
                and logical_bytes <= session.recovery_max_replay_bytes
            )
            within_context = (
                session.scheduler_context_limit is None or physical_tokens <= session.scheduler_context_limit
            )
            if within_journal and within_context:
                return retained, prepared_replay_prompts, current_prompt_override
            if retained:
                retained.pop(0)
                continue
            if not within_context:
                raise RuntimeError(
                    "streaming_prompt_context_limit_exceeded: "
                    f"projected={physical_tokens}, limit={session.scheduler_context_limit}"
                )
            raise RuntimeError("duplex_recovery_journal_capacity_exhausted")

    async def _rebuild_scheduler_native_session(
        self,
        session: DuplexSessionRuntimeState,
        replay_appends: tuple[DuplexReplayAppend, ...],
        *,
        deadline_monotonic: float | None,
        reason: str | None = None,
        replace_journal: bool = False,
        prepared_replay_prompts: tuple[Mapping[str, Any], ...] | None = None,
    ) -> None:
        """Rebuild one parked scheduler request from committed append units."""
        if not session.capabilities.prompt_replay:
            raise RuntimeError("duplex prompt replay is not supported by this model")
        recovery_reason = reason or session.recovery_reason or "replica_loss"
        old_submitted = tuple(session.resource_request_ids(submitted=True))
        if reason == "context_rollover" and old_submitted:
            wait_replay_safe = getattr(self._stage_port, "wait_replay_safe", None)
            if callable(wait_replay_safe):
                await wait_replay_safe(old_submitted[0], deadline_monotonic=deadline_monotonic)
        if old_submitted:
            await self._stage_port.cleanup(list(old_submitted), abort=True)
            session.release_request_ids(list(old_submitted))

        session.begin_resource_generation(
            replay_appends if replace_journal else session.replay_appends,
            reason=recovery_reason,
        )
        session.mark_recovery_required(recovery_reason)
        self._sync_session_metrics()

        try:
            if prepared_replay_prompts is not None and len(prepared_replay_prompts) != len(replay_appends):
                raise ValueError("duplex recovery prepared prompt count does not match replay journal")
            for index, append in enumerate(replay_appends):
                context = self.ensure_stage_request(
                    session,
                    stage_id=0,
                    fence=session.fence,
                    recovery_replay=True,
                )
                if context is None:
                    raise RuntimeError("duplex_data_plane_has_no_stage")
                result = await self._stage_port.submit(
                    DuplexStageSubmission(
                        context=context,
                        prompt=(
                            dict(prepared_replay_prompts[index])
                            if prepared_replay_prompts is not None
                            else self._prepare_recovery_prompt(
                                append,
                                request_id=context.request_id,
                                initial=index == 0,
                            )
                        ),
                        already_submitted=index > 0,
                        operation_id=append.operation_id,
                        operation_fingerprint=append.operation_fingerprint,
                        deadline_monotonic=deadline_monotonic,
                        recovery_replay=True,
                    )
                )
                if result.request_id != context.request_id or result.stage_id != 0:
                    raise RuntimeError("duplex recovery returned a mismatched submission result")
                session.bind_stage_request(0, context.request_id, fence=session.fence)
                self._record_scheduler_metrics(session.session_id, result.metrics, reset=index == 0)
        except BaseException as exc:
            session.mark_recovery_required(recovery_reason)
            new_request_ids = tuple(session.resource_request_ids())
            if new_request_ids:
                self._pending_submission_cleanups[session.session_id] = _PendingSubmissionCleanup(
                    session_id=session.session_id,
                    request_ids=new_request_ids,
                )
                try:
                    await self._complete_pending_submission_cleanup(
                        session.session_id,
                        deadline_monotonic=deadline_monotonic,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "duplex recovery cleanup remains pending for session %s: %s",
                        session.session_id,
                        cleanup_exc,
                    )
            if isinstance(exc, asyncio.CancelledError):
                raise
            self._metric("inc_duplex_kv_recovery", recovery_reason, "failure", 1)
            retryable = isinstance(exc, (TimeoutError, ConnectionError)) or any(
                marker in str(exc).lower()
                for marker in (
                    "unavailable",
                    "no live replica",
                    "lost its replica binding",
                    "replica is unavailable",
                )
            )
            raise DuplexKVRecoveryError(
                f"duplex_kv_recovery_failed: reason={recovery_reason}: {exc}",
                retryable=retryable,
            ) from exc
        session.mark_recovered()
        self._sync_session_metrics()
        self._metric("inc_duplex_kv_recovery", recovery_reason, "success", 1)

    def prepare_replica_recovery(
        self,
        stage_id: int,
        request_ids: Iterable[str],
        *,
        uncertain_request_ids: Iterable[str] = (),
        allow_recovery: bool = True,
    ) -> tuple[set[str], set[str]]:
        """Classify lost native requests into replay-safe and fail-closed sets."""
        requested = set(request_ids)
        uncertain_operations = set(uncertain_request_ids)
        recoverable: set[str] = set()
        uncertain: set[str] = set()
        for session in self.sessions.iter_sessions():
            affected = requested.intersection(session.resource_request_ids(submitted=True))
            if not affected:
                continue
            recovery_safe = (
                allow_recovery
                and stage_id == 0
                and session.capabilities.scheduler_native_append
                and session.capabilities.prompt_replay
                and session.lease.terminal_reason is None
                and not session.lease.active_operations
                and not affected.intersection(uncertain_operations)
                and bool(session.replay_appends)
            )
            if not recovery_safe:
                uncertain.update(affected)
                continue
            session.release_request_ids(list(affected))
            session.mark_recovery_required("replica_loss")
            recoverable.update(affected)
            self._metric("inc_duplex_replica_affinity_loss", 1)
        return recoverable, uncertain

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
        finally:
            self._sync_session_metrics()

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
                self._sync_session_metrics()
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
        finally:
            self._sync_session_metrics()

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
        for request_key, pending_request in list(self._pending_request_cleanups.items()):
            if request_key in self._request_cleanups_in_progress:
                continue
            try:
                await self._complete_request_cleanup(request_key, pending_request)
            except Exception as exc:
                logger.warning(
                    "duplex request cleanup remains pending for session %s: %s",
                    pending_request.session_id,
                    exc,
                )
                continue
            completed += 1
        for item in self.sessions.collect_expired(
            now,
            excluded_session_ids=set(self._pending_submission_cleanups),
        ):
            self._pending_expirations.setdefault(
                (item.session_id, item.fence.incarnation, item.lease_generation),
                item,
            )
        for expiry_key, item in list(self._pending_expirations.items()):
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
            if (
                session is not None
                and session.fence.incarnation == item.fence.incarnation
                and session.lease.generation == item.lease_generation
            ):
                self.sessions.finalize_close_session(session)
            if self._pending_expirations.get(expiry_key) is item:
                self._pending_expirations.pop(expiry_key, None)
            completed += 1
        self._sync_session_metrics()
        return completed

    async def _complete_pending_submission_cleanup(
        self,
        session_id: str,
        *,
        deadline_monotonic: float | None = None,
    ) -> None:
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
        if deadline_monotonic is None:
            await asyncio.shield(task)
            return
        remaining = deadline_monotonic - _time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"duplex submission cleanup deadline expired for session {session_id}")
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except TimeoutError as exc:
            raise TimeoutError(f"duplex submission cleanup deadline expired for session {session_id}") from exc

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
            self._sync_session_metrics()
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
        # Capture the cleanup owner before any I/O. Another cleanup may retire
        # it and reopen the same logical ID while these awaits are pending.
        session = self.sessions.get(pending.session_id)
        if session is not None and session.fence.incarnation != pending.fence.incarnation:
            session = None
        if pending.submitted_request_ids:
            await self._stage_port.cleanup(list(pending.submitted_request_ids), abort=True)
        if pending.reserved_request_ids:
            await self._stage_port.cleanup(list(pending.reserved_request_ids))
        if session is not None and self.sessions.get(pending.session_id) is session:
            if pending.kind == "cancel":
                session.release_fence(pending.fence)
            elif pending.kind in {"close", "open_rollback", "replica_lost"}:
                if pending.kind == "replica_lost" and self._lifecycle_sink is not None:
                    await self._lifecycle_sink.put(
                        DuplexSessionLifecycleMessage(
                            fence=session.fence,
                            session_id=session.session_id,
                            event="terminated",
                            reason="native_kv_replica_lost",
                            lease_generation=session.lease.generation,
                            submitted_request_ids=list(pending.submitted_request_ids),
                            reserved_request_ids=list(pending.reserved_request_ids),
                        )
                    )
                self.sessions.finalize_close_session(session)
        self._sync_session_metrics()
        if self._pending_control_cleanups.get(key) is pending:
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
        admission: dict[str, object] | None = None,
        operation_id: str | None = None,
    ) -> None:
        control_error = self._control_error(error) if error is not None else None
        if control_error is not None:
            stage_results = [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": False, "error": control_error.message},
                }
            ]
        unsupported_count, error_count = self._result_counts(stage_results)
        session = self.sessions.get(session_id)
        trace = asdict(
            DuplexTraceEnvelope(
                session_id=session_id,
                fence=fence,
                event=operation,
                control_id=control_id,
                operation_id=operation_id,
                request_id=(
                    duplex_data_plane_request_info({"stage_results": stage_results})[0] if stage_results else None
                ),
            )
        )
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
                admission=admission,
                trace=trace,
            )
        )

    @staticmethod
    def _control_error(error: BaseException | str) -> DuplexControlError:
        message = str(error)
        if isinstance(error, DuplexFenceMismatchError):
            code = "stale_fence"
            retryable = False
        elif "unknown duplex input mode" in message or "not_supported_by_engine" in message:
            code = "invalid_capability"
            retryable = False
        elif "duplex_session_capacity_exhausted" in message or isinstance(error, DuplexAppendAdmissionError):
            code = "resource_exhausted"
            retryable = True
        elif isinstance(error, DuplexOpenAdmissionError):
            code = "resource_exhausted"
            retryable = error.reason not in {"queue_timeout"}
        elif "duplex_recovery_journal_" in message:
            code = "resource_exhausted"
            retryable = False
        elif isinstance(error, DuplexKVRecoveryError):
            code = "kv_recovery_failed"
            retryable = error.retryable
        elif "streaming_prompt_context_limit_exceeded" in message:
            code = "resource_exhausted"
            retryable = False
        elif any(
            marker in message
            for marker in (
                "native_duplex_prefill_failed",
                "native_duplex_input_budget_mismatch",
                "native_duplex_input_history_",
            )
        ):
            code = "model_input_error"
            retryable = False
        elif "streaming_prompt_idempotency_capacity_exhausted" in message:
            code = "resource_exhausted"
            retryable = False
        elif "streaming_prompt_idempotency_window_expired" in message:
            code = "idempotency_window_expired"
            retryable = False
        elif "streaming_prompt_uncertain_operation_requires_retry" in message:
            code = "uncertain_operation"
            retryable = True
        elif isinstance(error, DuplexControlPreemptedError):
            code = "cancelled"
            retryable = False
        elif "resident KV cannot be reassigned" in message or "native_kv_replica_lost" in message:
            code = "replica_lost"
            retryable = False
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
        reason: str = "request_cleanup",
    ) -> dict[str, list[str]]:
        closed = self._sessions.close_sessions_for_request_ids(request_ids, reason=reason)
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
        if reason == "native_kv_replica_lost" and closed:
            self._metric("inc_duplex_replica_affinity_loss", len(closed))
        self._sync_session_metrics()
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
        self._sync_session_metrics()


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
