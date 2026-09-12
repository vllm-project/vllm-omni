# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from vllm.logger import init_logger

from vllm_omni.engine.duplex.control_client import DuplexControlRequestError
from vllm_omni.engine.duplex.messages import DuplexFence

logger = init_logger(__name__)


@dataclass
class _OpenAttempt:
    dispatch: asyncio.Task[object]
    close: Callable[[], Awaitable[bool]]
    abandoned: bool = False
    cleanup: asyncio.Task[bool] | None = None
    retry_timer: asyncio.TimerHandle | None = None
    retry_delay_s: float = 1.0


class RuntimeOpenAttempts:
    """Own abandoned opens beyond the lifetime of their connection waiter.

    Dispatch must honor its configured finite RPC timeout. Never cancel it:
    an executor worker could still enqueue open after an immediate close.
    Failed cleanup attempts retain their obligation and one capped-backoff
    timer. Admission is bounded before dispatch, including when clients reconnect
    with fresh IDs and cannot name an unpublished session for an explicit retry.
    """

    def __init__(self, *, max_pending: int = 64, retry_initial_s: float = 1.0, retry_max_s: float = 30.0) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if not math.isfinite(retry_initial_s) or not math.isfinite(retry_max_s):
            raise ValueError("cleanup retry delays must be finite")
        if retry_initial_s <= 0 or retry_max_s < retry_initial_s:
            raise ValueError("cleanup retry delays must be positive and ordered")
        self._max_pending = max_pending
        self._retry_initial_s = retry_initial_s
        self._retry_max_s = retry_max_s
        self._pending: dict[DuplexFence, _OpenAttempt] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def execute(
        self,
        fence: DuplexFence,
        dispatch: Callable[[], Awaitable[object]],
        close: Callable[[], Awaitable[bool]],
        *,
        is_success: Callable[[object], bool] = lambda result: True,
    ) -> object:
        if not await self.retry_pending(fence.session_id):
            raise RuntimeError("Previous runtime open cleanup is still pending")
        if len(self._pending) >= self._max_pending:
            raise DuplexControlRequestError(
                {
                    "operation": "open",
                    "error": {
                        "code": "resource_exhausted",
                        "message": "Runtime open recovery capacity exhausted; retry after cleanup",
                        "retryable": True,
                        "acceptance": "not_accepted",
                    },
                }
            )

        async def run_dispatch() -> object:
            return await dispatch()

        attempt = _OpenAttempt(asyncio.create_task(run_dispatch()), close, retry_delay_s=self._retry_initial_s)
        self._pending[fence] = attempt
        try:
            result = await asyncio.shield(attempt.dispatch)
            if not is_success(result):
                attempt.abandoned = True
                await asyncio.shield(self._start_cleanup(fence, attempt))
                return result
        except (Exception, asyncio.CancelledError) as exc:
            attempt.abandoned = True
            cleanup = self._start_cleanup(fence, attempt)
            if not isinstance(exc, asyncio.CancelledError):
                await asyncio.shield(cleanup)
            raise
        else:
            self._discharge(fence, attempt)
            return result

    def _discharge(self, fence: DuplexFence, attempt: _OpenAttempt) -> None:
        if self._pending.get(fence) is attempt:
            self._pending.pop(fence, None)
        if attempt.retry_timer is not None:
            attempt.retry_timer.cancel()
            attempt.retry_timer = None

    def _schedule_retry(self, fence: DuplexFence, attempt: _OpenAttempt) -> None:
        if self._pending.get(fence) is not attempt or attempt.retry_timer is not None:
            return

        def retry() -> None:
            attempt.retry_timer = None
            if self._pending.get(fence) is attempt:
                self._start_cleanup(fence, attempt)

        attempt.retry_timer = asyncio.get_running_loop().call_later(attempt.retry_delay_s, retry)
        attempt.retry_delay_s = min(self._retry_max_s, attempt.retry_delay_s * 2)

    def _start_cleanup(self, fence: DuplexFence, attempt: _OpenAttempt) -> asyncio.Task[bool]:
        if attempt.cleanup is not None and not attempt.cleanup.done():
            return attempt.cleanup
        if attempt.retry_timer is not None:
            attempt.retry_timer.cancel()
            attempt.retry_timer = None

        async def recover() -> bool:
            try:
                # wait() separates cancellation of dispatch from cancellation
                # of this recovery task on Python 3.10 as well.
                await asyncio.wait((attempt.dispatch,))
                try:
                    result = attempt.dispatch.result()
                    error = result.get("error") if isinstance(result, dict) else None
                    if isinstance(error, dict) and error.get("acceptance") == "not_accepted":
                        self._discharge(fence, attempt)
                        return True
                except DuplexControlRequestError as exc:
                    if exc.acceptance == "not_accepted":
                        self._discharge(fence, attempt)
                        return True
                except (Exception, asyncio.CancelledError):
                    # Dispatch has settled, but only the engine knows whether
                    # input was accepted. Its same-session ordering now makes
                    # the immutable-fence close safe to enqueue.
                    pass
                if not await attempt.close():
                    logger.warning("Abandoned runtime open cleanup remains pending: fence=%s", fence)
                    self._schedule_retry(fence, attempt)
                    return False
            except Exception:
                logger.exception("Abandoned runtime open cleanup failed; obligation retained: fence=%s", fence)
                self._schedule_retry(fence, attempt)
                return False
            self._discharge(fence, attempt)
            return True

        cleanup = asyncio.create_task(recover(), name=f"duplex-abandoned-open-{fence.session_id}")
        attempt.cleanup = cleanup

        def observe(done: asyncio.Task[bool]) -> None:
            if not done.cancelled():
                done.exception()

        cleanup.add_done_callback(observe)
        return cleanup

    async def retry_pending(self, session_id: str) -> bool:
        """Retry this session only; cancellation never cancels the obligation."""
        for fence, attempt in tuple(self._pending.items()):
            if fence.session_id != session_id:
                continue
            if not attempt.abandoned:
                return False
            if not await asyncio.shield(self._start_cleanup(fence, attempt)):
                return False
        return True
