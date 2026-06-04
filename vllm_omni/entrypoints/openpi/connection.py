# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection for robot policy inference (OpenPI protocol).

Protocol (compatible with OpenPI policy clients):
    Connect  -> server sends msgpack(PolicyServerConfig fields)
    Infer    -> client sends msgpack(obs), server sends msgpack(ndarray)
    Reset    -> client sends msgpack({endpoint:reset}), server sends msgpack(status)
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openpi.serving import (
    ServingRealtimeRobotOpenPI,
)

logger = init_logger(__name__)
_DEFAULT_IDLE_TIMEOUT = 30.0
MAX_OPENPI_PAYLOAD_BYTES = 64 * 1024 * 1024


def _get_msgpack_numpy() -> Any:
    try:
        from openpi_client import msgpack_numpy
    except ImportError as exc:
        raise ImportError(
            "The `/v1/realtime/robot/openpi` endpoint requires the optional "
            "`openpi-client` dependency. Install it with `pip install openpi-client`."
        ) from exc

    return msgpack_numpy


def _pack(obj: Any) -> bytes:
    return _get_msgpack_numpy().packb(obj)


def _unpack(data: bytes) -> Any:
    return _get_msgpack_numpy().unpackb(data)


class RobotRealtimeConnection:
    """WebSocket connection for robot policy inference."""

    def __init__(
        self,
        websocket: WebSocket,
        serving: ServingRealtimeRobotOpenPI,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self.websocket = websocket
        self.serving = serving
        self._idle_timeout = idle_timeout
        self._current_session_id: str | None = None
        self._call_count = 0

    def reset(self) -> None:
        self._current_session_id = None
        self._call_count = 0

    async def _send_error(self, message: str) -> None:
        await self.websocket.send_bytes(_pack({"type": "error", "message": message}))

    def _unpack_request(self, data: bytes) -> dict[str, Any]:
        if len(data) > MAX_OPENPI_PAYLOAD_BYTES:
            raise ValueError("OpenPI request payload too large")
        obs = _unpack(data)
        if not isinstance(obs, dict):
            raise ValueError("Invalid request payload")
        return obs

    async def handle_connection(self) -> None:
        """Main loop for OpenPI-compatible policy serving."""
        await self.websocket.accept()

        try:
            # Send model-specific PolicyServerConfig resolved by serving from
            # diffusion od_config.model_config.
            metadata = self.serving.policy_server_config.to_dict()
            await self.websocket.send_bytes(_pack(metadata))

            while True:
                try:
                    msg = await asyncio.wait_for(
                        self.websocket.receive(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.info("Robot OpenPI connection idle timeout after %.1f seconds", self._idle_timeout)
                    try:
                        await self.websocket.close()
                    except Exception:
                        logger.debug("Failed to close idle robot OpenPI websocket", exc_info=True)
                    return

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" not in msg or not msg["bytes"]:
                    continue

                try:
                    obs = self._unpack_request(msg["bytes"])
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
                        self.reset()
                        self.serving.reset(obs)
                        await self.websocket.send_bytes(_pack({"status": "reset successful"}))
                    else:
                        session_id = str(obs.get("session_id") or self._current_session_id or "default")
                        if session_id != self._current_session_id:
                            if self._current_session_id is not None:
                                logger.info(
                                    "Robot OpenPI session changed %s -> %s",
                                    self._current_session_id,
                                    session_id,
                                )
                            self._current_session_id = session_id
                            self._call_count = 0

                        self._call_count += 1
                        actions = await self.serving.infer(
                            obs,
                            session_id=session_id,
                            reset=self._call_count <= 1,
                        )
                        await self.websocket.send_bytes(_pack(actions))
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
