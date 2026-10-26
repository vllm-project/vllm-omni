# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The one scheduling seam a session's deadline owners share.

Two per-session owners hold work behind a monotonic deadline: the tool-call
ledger (a result that arrives after ``tool_result_timeout_s``) and the
semantic-VAD gate (a completion verdict that does not arrive before
``max_wait_ms``). Both need the same two things from the loop -- a
``call_later``-shaped registration and a handle they can cancel -- and both must
be driven deterministically by tests instead of sleeping. Nothing here knows
what a deadline means; the owners decide, and a callback only ever hands an
identity back to the session's mailbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

__all__ = ["LoopTimerScheduler", "ScheduledTimer", "TimerScheduler"]


class ScheduledTimer(Protocol):
    """The one thing a deadline owner needs from an armed expiry."""

    def cancel(self) -> None: ...


class TimerScheduler(Protocol):
    """Schedules deadlines; the default is the session loop's ``call_later``.

    The seam exists so tests can drive expiry deterministically instead of
    sleeping: a fake scheduler records the callbacks and fires them on demand.
    """

    def call_later(self, delay_s: float, callback: Callable[[], None]) -> ScheduledTimer: ...


class LoopTimerScheduler:
    """``asyncio.AbstractEventLoop.call_later`` behind :class:`TimerScheduler`."""

    __slots__ = ("_loop",)

    def __init__(self, loop: object) -> None:
        self._loop = loop

    def call_later(self, delay_s: float, callback: Callable[[], None]) -> ScheduledTimer:
        return self._loop.call_later(delay_s, callback)  # type: ignore[attr-defined]
