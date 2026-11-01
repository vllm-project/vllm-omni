# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import Future, InvalidStateError
from typing import TypeAlias

from vllm.logger import init_logger

from vllm_omni.engine.messages import (
    EngineQueueMessage,
    ErrorMessage,
)

logger = init_logger(__name__)

RpcCorrelationKey: TypeAlias = tuple[str, str]
RpcWaiter: TypeAlias = queue.Queue[EngineQueueMessage]
RpcAsyncWaiter: TypeAlias = Future[EngineQueueMessage]
_RpcWaiter: TypeAlias = RpcWaiter | RpcAsyncWaiter


class RpcResultRouter:
    """Single consumer that dispatches shared RPC results by correlation ID."""

    def __init__(self, source_queue) -> None:
        self._source_queue = source_queue
        self._pending: dict[RpcCorrelationKey, _RpcWaiter] = {}
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._terminal_error: ErrorMessage | None = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="omni-rpc-result-router",
        )
        self._thread.start()

    def register(self, key: RpcCorrelationKey) -> RpcWaiter:
        waiter: RpcWaiter = queue.Queue(maxsize=1)
        self._register_waiter(key, waiter)
        return waiter

    def register_async(self, key: RpcCorrelationKey) -> RpcAsyncWaiter:
        waiter: RpcAsyncWaiter = Future()
        self._register_waiter(key, waiter)
        return waiter

    def _register_waiter(self, key: RpcCorrelationKey, waiter: _RpcWaiter) -> None:
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError("RPC result router is closed")
            if self._terminal_error is not None:
                raise RuntimeError(self._terminal_error.error)
            if key in self._pending:
                raise RuntimeError(f"duplicate pending RPC correlation key: {key!r}")
            self._pending[key] = waiter

    def unregister(self, key: RpcCorrelationKey, waiter: _RpcWaiter) -> None:
        with self._lock:
            if self._pending.get(key) is waiter:
                self._pending.pop(key, None)

    def close(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._broadcast_error(ErrorMessage(error="RPC result router closed", fatal=True))
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                message = self._source_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as exc:
                if not self._stopped.is_set():
                    logger.exception("RPC result router source queue failed")
                    self._broadcast_error(ErrorMessage(error=str(exc), fatal=True))
                return

            if isinstance(message, ErrorMessage):
                if message.fatal:
                    self._broadcast_error(message)
                else:
                    logger.warning(
                        "Dropping uncorrelated non-fatal RPC error request_id=%s stage_id=%s: %s",
                        message.request_id,
                        message.stage_id,
                        message.error,
                    )
                continue
            key = self._correlation_key(message)
            if key is None:
                logger.warning(
                    "Dropping unexpected RPC result message type=%s",
                    getattr(message, "type", type(message).__name__),
                )
                continue
            with self._lock:
                waiter = self._pending.pop(key, None)
            if waiter is None:
                logger.warning("Dropping late or unknown RPC result correlation_key=%s", key)
                continue
            self._deliver(waiter, message)

    @staticmethod
    def _deliver(waiter: _RpcWaiter, message: EngineQueueMessage) -> None:
        if isinstance(waiter, Future):
            try:
                waiter.set_result(message)
            except InvalidStateError:
                # Cancellation may race with routing after the waiter was
                # removed under the router lock. A cancelled caller must not
                # stop this shared consumer or receive a late reply.
                if not waiter.cancelled():
                    raise
        else:
            waiter.put_nowait(message)

    def _broadcast_error(self, message: ErrorMessage) -> None:
        with self._lock:
            if message.fatal:
                self._terminal_error = message
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            try:
                self._deliver(waiter, message)
            except queue.Full:
                pass

    @staticmethod
    def _correlation_key(message: EngineQueueMessage) -> RpcCorrelationKey | None:
        key = getattr(message, "rpc_correlation_key", None)
        if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(part, str) and part for part in key):
            return None
        return key


class CorrelatedRpcClient:
    """Own request submission and correlated result waiting as one lifecycle."""

    def __init__(self, request_queue, result_queue) -> None:
        self._request_queue = request_queue
        self._router = RpcResultRouter(result_queue)

    def execute(
        self,
        key: RpcCorrelationKey,
        message: EngineQueueMessage,
        *,
        timeout: float | None,
        timeout_message: str,
        block_on_submit: bool = False,
    ) -> EngineQueueMessage:
        waiter = self._router.register(key)
        try:
            if block_on_submit:
                self._request_queue.put(message)
            else:
                self._request_queue.put_nowait(message)
            try:
                result = waiter.get(timeout=timeout)
            except queue.Empty as exc:
                raise TimeoutError(timeout_message) from exc
            if isinstance(result, ErrorMessage):
                raise RuntimeError(result.error)
            return result
        finally:
            self._router.unregister(key, waiter)

    async def execute_async(
        self,
        key: RpcCorrelationKey,
        message: EngineQueueMessage,
        *,
        timeout: float | None,
        timeout_message: str,
    ) -> EngineQueueMessage:
        """Submit without blocking and await a thread-safe correlated future.

        Use the queue's sync face: its async face belongs to the orchestrator
        loop, not necessarily the caller's loop. No executor worker is held
        while awaiting the reply, leaving it available for one-way commands.
        Cancellation only retires the waiter; it cannot retract an admitted
        request. Admission and reply waiting share one timeout budget.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        waiter = self._router.register_async(key)
        try:
            if deadline is not None and deadline <= time.monotonic():
                raise TimeoutError(timeout_message)
            self._request_queue.put_nowait(message)
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(timeout_message)
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(waiter), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(timeout_message) from exc
            if isinstance(result, ErrorMessage):
                raise RuntimeError(result.error)
            return result
        finally:
            self._router.unregister(key, waiter)
            waiter.cancel()

    def close(self) -> None:
        self._router.close()


__all__ = [
    "CorrelatedRpcClient",
    "RpcAsyncWaiter",
    "RpcCorrelationKey",
    "RpcResultRouter",
    "RpcWaiter",
]
