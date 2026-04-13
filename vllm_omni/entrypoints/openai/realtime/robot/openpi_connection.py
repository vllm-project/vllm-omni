# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection for robot policy inference (OpenPI protocol).

Protocol (compatible with DreamZero test_client_AR.py):
    Connect  -> server sends msgpack(PolicyServerConfig fields)
    Infer    -> client sends msgpack(obs), server sends msgpack(ndarray)
    Reset    -> client sends msgpack({endpoint:reset}), server sends "reset successful"
"""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.realtime.robot.openpi_serving import (
    ServingRealtimeRobotOpenPI,
)

logger = init_logger(__name__)


def _pack(obj: Any) -> bytes:
    from openpi_client import msgpack_numpy

    return msgpack_numpy.packb(obj)


def _unpack(data: bytes) -> Any:
    from openpi_client import msgpack_numpy

    return msgpack_numpy.unpackb(data)


class RobotRealtimeConnection:
    """WebSocket connection for robot policy inference."""

    def __init__(
        self,
        websocket: WebSocket,
        serving: ServingRealtimeRobotOpenPI,
    ) -> None:
        self.websocket = websocket
        self.serving = serving

    async def handle_connection(self) -> None:
        """Main loop. Matches DreamZero policy_server.py._handler."""
        await self.websocket.accept()

        try:
            # Send metadata (PolicyServerConfig fields)
            metadata = {
                "image_resolution": (180, 320),
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_stereo_camera": False,
                "needs_session_id": True,
                "action_space": "joint_position",
            }
            await self.websocket.send_bytes(_pack(metadata))

            while True:
                msg = await self.websocket.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" not in msg or not msg["bytes"]:
                    continue

                try:
                    obs = _unpack(msg["bytes"])
                    endpoint = obs.pop("endpoint", "infer")

                    if endpoint == "reset":
                        self.serving.reset(obs)
                        await self.websocket.send_text("reset successful")
                    else:
                        actions = await self.serving.infer(obs)
                        await self.websocket.send_bytes(_pack(actions))
                except Exception:
                    logger.exception("Error handling request")
                    try:
                        await self.websocket.send_text(traceback.format_exc())
                    except Exception:
                        break

        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Connection error")
