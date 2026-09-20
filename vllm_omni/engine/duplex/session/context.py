# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared state for :class:`DuplexSessionRunner` and its components.

The runner is a state machine that several components read and write. Splitting
it into modules only helps if the shared part stops being implicit: a component
holding a back-reference to the runner is a mixin with extra steps, and one
holding its own copy of a flag diverges from the others.

So the mutable part is named. ``DuplexRunState`` is the small set of flags that
more than one component touches -- ``grep "\\.run\\."`` finds every mutation --
and ``DuplexSessionContext`` is the read-mostly collaborator bundle everything
is constructed with. Anything a component needs from the runner that is neither
of those is infrastructure, and goes through :class:`RunnerServices`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

    from vllm_omni.engine.duplex.contracts import (
        DuplexOutputContext,
        DuplexOutputDecision,
        DuplexStagePort,
    )
    from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexModelSessionState
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
    from vllm_omni.metrics.stats import StageRequestStats

_OffloadT = TypeVar("_OffloadT")


@dataclass(slots=True)
class DuplexRunState:
    """Mutable per-run flags shared by the runner and its components.

    Every field here is written by one component and read by another; that is
    the criterion for being in this object rather than private to a component.
    """

    #: An irreversible close has begun: commands and control ops are refused.
    closing: bool = False
    #: ``session.closed`` / ``session.expired`` already left the runner.
    closed_emitted: bool = False
    #: The terminal event is deferred to the manager, which emits it after the
    #: stage cleanup so it also means "the admission slot is free again".
    closed_deferred: bool = False
    close_reason: str | None = None
    #: The model runtime reported a close; further data-plane work is pointless.
    runtime_closed: bool = False
    #: Request id of the resumable data-plane stream currently bound to the session.
    stream_request_id: str | None = None


class RunnerServices(Protocol):
    """The only things a component may ask the runner for.

    Deliberately two methods. A component that needs more than task scheduling
    from the runner is reaching for orchestration that belongs in the runner.
    """

    def spawn(self, coro: Awaitable[None], *, name: str) -> None:
        """Run ``coro`` as a tracked background task on the session's loop."""
        ...

    async def offload(self, fn: Callable[..., _OffloadT], *args: object, **kwargs: object) -> _OffloadT:
        """Run a blocking call off the orchestrator loop."""
        ...


@dataclass(slots=True)
class DuplexSessionContext:
    """Collaborators one session's components are built with.

    Read-mostly: the session and the model state are mutated through their own
    APIs, not by rebinding these fields. The mutable flags live in ``run``.
    """

    session: DuplexEngineSession
    model_state: DuplexModelSessionState
    plugin: DuplexModelPlugin
    stage_port: DuplexStagePort
    manager: DuplexSessionManager
    tasks: DuplexSessionTasks
    run: DuplexRunState
    services: RunnerServices


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
# Stage output (the mailbox item the orchestrator hands the runner)           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StageOutput:
    """One stage result, queued for the session that owns the request."""

    stage_id: int
    output: RequestOutput
    metrics: StageRequestStats | None
    request_id: str
    context: DuplexOutputContext
    decision: DuplexOutputDecision | None
