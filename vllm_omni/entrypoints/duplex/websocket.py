# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WebSocket transport helpers for the duplex Realtime handler.

The ordered session mailbox and the per-session task handles that used to be
defined here now live engine-side (``vllm_omni.engine.duplex.session_runner``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect

#: Largest client frame accepted before it is parsed (base64 audio + video frames).
MAX_EVENT_BYTES = 15 * 1024 * 1024

SendJson = Callable[[dict[str, object]], Awaitable[None]]
CloseFn = Callable[[str], Awaitable[None]]


async def receive_text_with_timeout(websocket: WebSocket, timeout_s: float | None) -> str | None:
    """Receive one text frame; ``None`` on idle timeout. Disconnects propagate."""
    if timeout_s is None or timeout_s <= 0:
        return await websocket.receive_text()
    try:
        return await asyncio.wait_for(websocket.receive_text(), timeout=timeout_s)
    except TimeoutError:
        return None


def attachment_callbacks(websocket: WebSocket) -> tuple[SendJson, CloseFn]:
    """``send`` / ``close`` callables for a ``DuplexSessionAttachmentRegistry`` attachment.

    A send on a socket that already closed surfaces as ``WebSocketDisconnect``
    so the caller's disconnect path runs instead of a generic failure.
    """

    async def attachment_send(payload: dict[str, object]) -> None:
        try:
            await websocket.send_json(payload)
        except RuntimeError as exc:
            message = str(exc)
            if "after sending 'websocket.close'" in message or "response already completed" in message:
                raise WebSocketDisconnect(code=1006) from exc
            raise

    async def attachment_close(reason: str) -> None:
        close = getattr(websocket, "close", None)
        if callable(close):
            with suppress(Exception):
                await close(code=1000, reason=reason)

    return attachment_send, attachment_close


__all__ = ["MAX_EVENT_BYTES", "CloseFn", "SendJson", "attachment_callbacks", "receive_text_with_timeout"]
