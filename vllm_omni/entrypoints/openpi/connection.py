# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WebSocket connection for robot policy inference (OpenPI protocol).

Protocol (compatible with OpenPI policy clients):
    Connect  -> server sends msgpack(PolicyServerConfig fields)
    Infer    -> client sends msgpack(obs), server sends msgpack(ndarray)
    Reset    -> client sends msgpack({endpoint:reset}), server sends msgpack(status)

NumPy serialization is shared in ``vllm_omni.protocol.numpy_msgpack``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Awaitable
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openpi.serving import (
    ServingRealtimeRobotOpenPI,
)
from vllm_omni.protocol.numpy_msgpack import pack as _pack
from vllm_omni.protocol.numpy_msgpack import unpack as _unpack

logger = init_logger(__name__)
_DEFAULT_IDLE_TIMEOUT = 30.0
# Upper bound on the per-connection set of seen session ids.
MAX_TRACKED_SESSIONS = 1024


async def await_policy(operation: Awaitable[Any], receive: asyncio.Task, websocket: WebSocket) -> Any:
    """Cancel in-flight policy work when the peer closes its socket."""
    work = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait((work, receive), return_when=asyncio.FIRST_COMPLETED)
        if work not in done:
            # A failed receive is a transport error: treat it as a disconnect rather than an inference error.
            if receive.exception() is None and receive.result()["type"] != "websocket.disconnect":
                await websocket.close(code=1008, reason="Wait for the policy reply before sending another request")
            raise WebSocketDisconnect()
        return await work
    finally:
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)


class RobotRealtimeConnection:
    """WebSocket connection for robot policy inference."""

    def __init__(
        self,
        websocket: WebSocket,
        serving: ServingRealtimeRobotOpenPI,
        idle_timeout: float | None = _DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self.websocket = websocket
        self.serving = serving
        self._idle_timeout = idle_timeout
        self._session_prefix = "openpi-" + uuid.uuid4().hex + ":"
        self._current_session_id: str | None = None
        # Session ids seen on this connection, most-recently-used last.
        self._seen_sessions: OrderedDict[str, None] = OrderedDict()

    def reset(self) -> None:
        self._current_session_id = None
        self._seen_sessions.clear()

    async def _mark_session_seen(self, session_id: str) -> bool:
        """Record ``session_id``; return True when it is the first sighting.

        A session's first observation is the one that carries ``reset`` to the
        policy, so this needs per-session memory.
        """
        if session_id in self._seen_sessions:
            self._seen_sessions.move_to_end(session_id)
            return False

        self._seen_sessions[session_id] = None
        if len(self._seen_sessions) > MAX_TRACKED_SESSIONS:
            evicted, _ = self._seen_sessions.popitem(last=False)
            await self.serving.close_sessions([evicted])
            logger.debug(
                "Robot OpenPI connection is tracking more than %d sessions; dropped least-recently-used %s",
                MAX_TRACKED_SESSIONS,
                evicted,
            )
        return True

    async def _send_error(self, message: str) -> None:
        await self.websocket.send_bytes(_pack({"type": "error", "message": message}))

    def _unpack_request(self, data: bytes) -> dict[str, Any]:
        obs = _unpack(data)
        if not isinstance(obs, dict):
            raise ValueError("Invalid request payload")
        return obs

    async def handle_connection(self) -> None:
        """Main loop for OpenPI-compatible policy serving."""
        await self.websocket.accept()
        receive: asyncio.Task | None = None

        try:
            # Send model-specific PolicyServerConfig resolved by serving from
            # diffusion od_config.model_config.
            metadata = self.serving.policy_server_config.to_dict()
            if metadata.get("supports_close_session") and not self.serving.has_single_worker():
                await self._send_error("Robot sessions require one stage and one worker replica")
                await self.websocket.close()
                return
            await self.websocket.send_bytes(_pack(metadata))

            receive = asyncio.create_task(self.websocket.receive())
            while True:
                idle_timeout = self._idle_timeout
                if idle_timeout is None:
                    msg = await receive
                else:
                    try:
                        msg = await asyncio.wait_for(
                            asyncio.shield(receive),
                            timeout=idle_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.info("Robot OpenPI connection idle timeout after %.1f seconds", idle_timeout)
                        try:
                            await self.websocket.close()
                        except Exception:
                            logger.debug("Failed to close idle robot OpenPI websocket", exc_info=True)
                        return

                if msg.get("type") == "websocket.disconnect":
                    break

                receive = asyncio.create_task(self.websocket.receive())
                if "bytes" not in msg or not msg["bytes"]:
                    continue

                try:
                    obs = await asyncio.to_thread(self._unpack_request, msg["bytes"])
                except Exception:
                    logger.exception("Invalid robot OpenPI request payload")
                    try:
                        await self._send_error("Invalid request payload")
                    except Exception:
                        break
                    continue

                try:
                    endpoint = obs.pop("endpoint", "infer")

                    if endpoint == "reset":
                        await self.serving.close_sessions(list(self._seen_sessions))
                        self.reset()
                        self.serving.reset(obs)
                        await self.websocket.send_bytes(_pack({"status": "reset successful"}))
                    else:
                        # Client ids are namespaced per connection, so a client that sends none
                        # still gets a connection-private "default" episode.
                        raw_id = obs.get("session_id")
                        session_id = (
                            (self._session_prefix + str(raw_id))
                            if raw_id
                            else (self._current_session_id or self._session_prefix + "default")
                        )
                        if session_id != self._current_session_id:
                            if self._current_session_id is not None:
                                logger.info(
                                    "Robot OpenPI session changed %s -> %s",
                                    self._current_session_id,
                                    session_id,
                                )
                            self._current_session_id = session_id

                        reset = await self._mark_session_seen(session_id)
                        actions = await await_policy(
                            self.serving.infer(obs, session_id=session_id, reset=reset), receive, self.websocket
                        )
                        await self.websocket.send_bytes(_pack(actions))
                except WebSocketDisconnect:
                    break
                except Exception:
                    logger.exception("Error handling request")
                    try:
                        await self._send_error("Internal inference error")
                    except Exception:
                        break

        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Connection error")
        finally:
            if receive is not None:
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
            await self.serving.close_sessions(list(self._seen_sessions))
            self.reset()
