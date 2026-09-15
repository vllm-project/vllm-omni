# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ownership of admitted H3 requests, separate from cancellable HTTP tasks.

Conservative abnormal-path policy: an unknown/fatal exception after engine
dispatch, or an unexpected inner-task cancellation, does not confirm worker quiescence. Retain
those inputs and their admission slots even after ``drain``. Engine shutdown
currently permits deferred worker teardown, so its return cannot authorize file
removal. There is no automatic reclamation of these abnormal retained bundles;
operator cleanup requires independent confirmation that all readers terminated.

The serving boundary distinguishes origin-qualified terminal OmniClientError
validation results from unknown/fatal engine failures. ErrorMessage preserves a
finished, non-streaming final-stage worker rejection without companion branches;
neither the exception class nor a completed abort request supplies that proof.
Those terminal errors,
normal iterator exhaustion, local pre-dispatch failures and post-completion
errors release inputs and admission slots. Cancellation takes precedence over a
terminal-error classification; it never turns an abort acknowledgment into safe
completion.

Guided HTTP ownership requires Python 3.11+ for Task.cancelling(). Older Python
runtimes reject admission before creating a bundle or persisting inputs; empty
or absent timeline guides retain the legacy request lifecycle.
"""

import asyncio
import os
from collections.abc import Coroutine
from dataclasses import dataclass, field
from sys import version_info
from typing import Any

from fastapi import HTTPException
from vllm.logger import init_logger

logger = init_logger(__name__)

# Async job lookup only. Capacity and all tasks (including sync) belong to a handler.
GUIDED_JOBS: dict[str, "GuidedRequestBundle"] = {}


@dataclass(eq=False)
class GuidedRequestBundle:
    owner: "GuidedRequestLifetime"
    descriptors: list[dict[str, Any]] = field(default_factory=list)
    paths: set[str] = field(default_factory=set)
    task: asyncio.Task | None = None
    abandoned: bool = False
    closed: bool = False
    started: bool = False
    engine_started: bool = False
    engine_completed: bool = False
    job_id: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def close(self) -> None:
        self.closed = True
        for path in self.paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Unable to remove guided request input %s", path, exc_info=True)
        self.paths.clear()
        self.owner.bundles.discard(self)
        if self.job_id is not None:
            GUIDED_JOBS.pop(self.job_id, None)

    async def finish_storage(self, work: Coroutine[Any, Any, Any]) -> Any:
        """Keep the caller's storage lock until thread-backed I/O really finishes."""
        try:
            pending = asyncio.create_task(work)
        except BaseException:
            work.close()
            raise
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Cancelling to_thread only cancels its waiter, not its write.
                # Defer cancellation and let the caller discard the saved result.
                self.abandoned = True
        return pending.result()

    def submit(self, work: Coroutine[Any, Any, Any]) -> asyncio.Task:
        if self.task is not None or self.closed:
            work.close()
            raise RuntimeError("Guided request already submitted or closed")
        if self.owner.stopping:
            work.close()
            self.close()
            raise HTTPException(503, "Video server is shutting down.")

        async def run():
            self.started = True
            if self.abandoned:
                work.close()
                return None
            result = await work
            if asyncio.current_task().cancelling():
                self.abandoned = True
            return None if self.abandoned else result

        runner = run()
        try:
            self.task = asyncio.create_task(runner)
        except BaseException:
            runner.close()
            work.close()
            self.close()
            raise

        def completed(task: asyncio.Task) -> None:
            # Cancellation before the coroutine starts also arrives here. Submitted
            # tasks are never cancelled by HTTP/DELETE or the shutdown drain.
            if not task.cancelled():
                task.exception()
            if self.started and (task.cancelled() or task.cancelling()):
                # Also retain if a lower layer swallowed cancellation after an
                # engine abort: normal coroutine return is not then quiescence.
                self.abandoned = True
                logger.error("Guided inner task cancelled unexpectedly; retaining its inputs and reservation")
                return
            if self.engine_started and not self.engine_completed:
                # Async job wrappers may catch the exception and return normally.
                # Only the engine boundary can establish normal completion.
                logger.error("Guided engine completion unconfirmed; retaining its inputs and reservation")
                return
            if task.cancelled():
                work.close()
            self.close()

        self.task.add_done_callback(completed)
        return self.task


class GuidedRequestLifetime:
    def __init__(self) -> None:
        self.bundles: set[GuidedRequestBundle] = set()
        self.stopping = False

    def reserve(self, maximum: int) -> GuidedRequestBundle:
        if version_info < (3, 11):
            raise HTTPException(
                503,
                "Timeline guides require Python 3.11 or newer for cancellation-safe request ownership. "
                "Restart the video server with Python 3.11+ or omit timeline_guides and guide_files.",
            )
        if self.stopping or len(self.bundles) >= maximum:
            raise HTTPException(503, "Timeline guide request capacity is exhausted; retry after active work completes.")
        bundle = GuidedRequestBundle(self)
        self.bundles.add(bundle)
        return bundle

    async def drain(self) -> None:
        """Stop admission and wait for real completion, without sending aborts."""
        self.stopping = True
        for bundle in tuple(self.bundles):
            bundle.abandoned = True
            if bundle.task is None:
                bundle.close()
        tasks = [bundle.task for bundle in self.bundles if bundle.task is not None]
        cancelled = False
        if tasks:
            pending = asyncio.gather(*tasks, return_exceptions=True)
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    # Do not let a second shutdown cancellation exit the engine
                    # context while workers can still be reading these inputs.
                    cancelled = True
        if self.bundles:
            # Engine shutdown can return after a bounded join without terminating
            # every worker. Do not infer a safe cleanup acknowledgment from it.
            logger.error("Retaining %d guided bundles without worker completion confirmation", len(self.bundles))
        if cancelled:
            raise asyncio.CancelledError
