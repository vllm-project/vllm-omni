# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
from typing import TypeAlias

from vllm.logger import init_logger

from vllm_omni.engine.messages import (
    EngineQueueMessage,
    ErrorMessage,
)

logger = init_logger(__name__)

RpcCorrelationKey: TypeAlias = tuple[str, str]
RpcWaiter: TypeAlias = queue.Queue[EngineQueueMessage]


class RpcResultRouter:
    """Single consumer that dispatches shared RPC results by correlation ID."""

    def __init__(self, source_queue) -> None:
        self._source_queue = source_queue
        self._pending: dict[RpcCorrelationKey, RpcWaiter] = {}
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
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError("RPC result router is closed")
            if self._terminal_error is not None:
                raise RuntimeError(self._terminal_error.error)
            if key in self._pending:
                raise RuntimeError(f"duplicate pending RPC correlation key: {key!r}")
            self._pending[key] = waiter
        return waiter

    def unregister(self, key: RpcCorrelationKey, waiter: RpcWaiter) -> None:
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
            waiter.put_nowait(message)

    def _broadcast_error(self, message: ErrorMessage) -> None:
        with self._lock:
            if message.fatal:
                self._terminal_error = message
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            try:
                waiter.put_nowait(message)
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

    def close(self) -> None:
        self._router.close()


__all__ = [
    "CorrelatedRpcClient",
    "RpcCorrelationKey",
    "RpcResultRouter",
    "RpcWaiter",
]
