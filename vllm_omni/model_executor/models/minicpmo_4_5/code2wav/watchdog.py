# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""No-progress watchdog for MiniCPM-o Code2Wav operations."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

CODE2WAV_WATCHDOG_TIMEOUT_ENV = "VLLM_OMNI_CODE2WAV_WATCHDOG_TIMEOUT_S"

_MAX_DETAIL_ITEMS = 16
_MAX_DETAIL_TEXT = 256
_MAX_STACK_FRAMES = 64

logger = logging.getLogger(__name__)


def _bounded_value(value: object, *, depth: int = 0) -> object:
    """Return a small, JSON-serializable diagnostic value."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_DETAIL_TEXT]
    if depth >= 2:
        return repr(value)[:_MAX_DETAIL_TEXT]
    if isinstance(value, Mapping):
        return {
            str(key)[:_MAX_DETAIL_TEXT]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_DETAIL_ITEMS]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_bounded_value(item, depth=depth + 1) for item in list(value)[:_MAX_DETAIL_ITEMS]]
    return repr(value)[:_MAX_DETAIL_TEXT]


def _bounded_details(details: Mapping[str, object] | None) -> dict[str, object]:
    if not details:
        return {}
    return {
        str(key)[:_MAX_DETAIL_TEXT]: _bounded_value(value) for key, value in list(details.items())[:_MAX_DETAIL_ITEMS]
    }


@dataclass(frozen=True)
class Code2WavWatchdogReport:
    """One no-progress report emitted for a watched operation."""

    phase: str
    details: Mapping[str, object]
    timeout_s: float
    elapsed_s: float
    owner_thread_id: int
    stack: str


@dataclass
class _ActiveOperation:
    owner_thread_id: int
    phase: str
    details: dict[str, object]
    last_progress_at: float
    reported: bool = False


def _default_reporter(report: Code2WavWatchdogReport) -> None:
    payload = {
        "phase": report.phase,
        "details": dict(report.details),
        "timeout_s": report.timeout_s,
        "elapsed_s": round(report.elapsed_s, 3),
        "owner_thread_id": report.owner_thread_id,
    }
    logger.error(
        "MiniCPM-o Code2Wav made no progress: %s\nBlocked owner-thread stack:\n%s",
        json.dumps(payload, sort_keys=True),
        report.stack,
    )


class Code2WavWatchdog:
    """Report a bounded owner-thread stack when a watched phase stops progressing.

    One daemon monitor thread serves all operations for a model instance. Calls
    to :meth:`phase` reset the current operation's deadline. A timed-out
    operation is reported at most once and is never interrupted by the
    watchdog.
    """

    def __init__(
        self,
        timeout_s: float = 0.0,
        *,
        reporter: Callable[[Code2WavWatchdogReport], None] | None = None,
    ) -> None:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s):
            raise ValueError("Code2Wav watchdog timeout must be finite")
        self.timeout_s = timeout_s
        self._reporter = reporter or _default_reporter
        self._condition = threading.Condition()
        self._local = threading.local()
        self._operations: dict[int, _ActiveOperation] = {}
        self._next_operation_id = 0
        self._closed = False
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._monitor,
                name="minicpmo-code2wav-watchdog",
                daemon=True,
            )
            self._thread.start()

    @classmethod
    def from_environment(
        cls,
        *,
        reporter: Callable[[Code2WavWatchdogReport], None] | None = None,
    ) -> Code2WavWatchdog:
        raw = os.environ.get(CODE2WAV_WATCHDOG_TIMEOUT_ENV)
        if raw is None or not raw.strip():
            return cls(reporter=reporter)
        try:
            timeout_s = float(raw)
        except ValueError as exc:
            raise ValueError(f"{CODE2WAV_WATCHDOG_TIMEOUT_ENV} must be a finite number, got {raw!r}") from exc
        if not math.isfinite(timeout_s):
            raise ValueError(f"{CODE2WAV_WATCHDOG_TIMEOUT_ENV} must be finite, got {raw!r}")
        return cls(timeout_s, reporter=reporter)

    @property
    def enabled(self) -> bool:
        return self.timeout_s > 0

    @contextmanager
    def watch(
        self,
        phase: str,
        details: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        """Watch one operation until the context exits."""
        if not self.enabled:
            yield
            return
        thread_id = threading.get_ident()
        now = time.monotonic()
        operation_id = -1
        with self._condition:
            closed = self._closed
            if not closed:
                operation_id = self._next_operation_id
                self._next_operation_id += 1
                self._operations[operation_id] = _ActiveOperation(
                    owner_thread_id=thread_id,
                    phase=str(phase),
                    details=_bounded_details(details),
                    last_progress_at=now,
                )
                stack = getattr(self._local, "operation_ids", None)
                if stack is None:
                    stack = []
                    self._local.operation_ids = stack
                stack.append(operation_id)
                self._condition.notify_all()
        if closed:
            yield
            return
        try:
            yield
        finally:
            with self._condition:
                self._operations.pop(operation_id, None)
                local_stack = getattr(self._local, "operation_ids", None)
                if local_stack:
                    if local_stack[-1] == operation_id:
                        local_stack.pop()
                    elif operation_id in local_stack:
                        local_stack.remove(operation_id)
                self._condition.notify_all()

    def phase(self, phase: str, details: Mapping[str, object] | None = None) -> None:
        """Record progress and reset the active operation's deadline."""
        if not self.enabled:
            return
        stack = getattr(self._local, "operation_ids", None)
        if not stack:
            return
        with self._condition:
            operation = self._operations.get(stack[-1])
            if operation is None:
                return
            operation.phase = str(phase)
            operation.details = _bounded_details(details)
            operation.last_progress_at = time.monotonic()
            self._condition.notify_all()

    def close(self) -> None:
        """Stop the monitor thread. Safe to call more than once."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._operations.clear()
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, min(self.timeout_s, 5.0)))

    def _monitor(self) -> None:
        while True:
            due: list[tuple[_ActiveOperation, float]] = []
            with self._condition:
                while not self._closed:
                    candidates = [operation for operation in self._operations.values() if not operation.reported]
                    if not candidates:
                        self._condition.wait()
                        continue
                    now = time.monotonic()
                    next_deadline = min(operation.last_progress_at + self.timeout_s for operation in candidates)
                    remaining = next_deadline - now
                    if remaining > 0:
                        self._condition.wait(timeout=remaining)
                        continue
                    for operation in candidates:
                        elapsed = now - operation.last_progress_at
                        if elapsed >= self.timeout_s:
                            operation.reported = True
                            due.append((operation, elapsed))
                    break
                if self._closed:
                    return

            frames = sys._current_frames()
            for operation, elapsed in due:
                frame = frames.get(operation.owner_thread_id)
                stack = "".join(traceback.format_stack(frame, limit=_MAX_STACK_FRAMES)) if frame is not None else ""
                self._reporter(
                    Code2WavWatchdogReport(
                        phase=operation.phase,
                        details=dict(operation.details),
                        timeout_s=self.timeout_s,
                        elapsed_s=elapsed,
                        owner_thread_id=operation.owner_thread_id,
                        stack=stack,
                    )
                )
