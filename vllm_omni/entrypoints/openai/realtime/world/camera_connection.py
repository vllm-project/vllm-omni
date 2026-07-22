# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection for robot policy inference (OpenPI protocol).

Protocol (compatible with DreamZero test_client_AR.py):
    Connect  -> server sends msgpack(PolicyServerConfig fields)
    Infer    -> client sends msgpack(req), server sends msgpack(ndarray)
    Reset    -> client sends msgpack({endpoint:reset}), server sends "reset successful"
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.realtime.world.camera_serving import ServingRealtimeWorldCamera
from vllm_omni.entrypoints.openai.video_api_utils import _normalize_frames

logger = init_logger(__name__)
_DEFAULT_IDLE_TIMEOUT = 300.0
DEFAULT_FRAMES_PER_CHUNK = 4


def _get_msgpack_numpy() -> Any:
    try:
        from openpi_client import msgpack_numpy
    except ImportError as exc:
        raise ImportError(
            "The `/v1/realtime/world/camera` endpoint requires the optional "
            "`openpi-client` dependency. Install it with `pip install openpi-client`."
        ) from exc

    return msgpack_numpy


def _pack(obj: Any) -> bytes:
    return _get_msgpack_numpy().packb(obj)


def _unpack(data: bytes) -> Any:
    return _get_msgpack_numpy().unpackb(data)


class WorldCameraRealtimeConnection:
    """WebSocket connection for world model inference."""

    def __init__(
        self,
        websocket: WebSocket,
        serving: ServingRealtimeWorldCamera,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self.websocket = websocket
        self.serving = serving
        self._idle_timeout = idle_timeout

    async def _send_error(self, message: str) -> None:
        await self.websocket.send_bytes(_pack({"type": "error", "message": message}))

    def _unpack_request(self, data: bytes) -> dict[str, Any]:
        req = _unpack(data)
        if not isinstance(req, dict):
            raise ValueError("Invalid request payload")
        return req

    async def handle_connection(self) -> None:
        """Main loop."""
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
                    logger.info("World Model OpenPI connection idle timeout after %.1f seconds", self._idle_timeout)
                    try:
                        await self.websocket.close()
                    except Exception:
                        logger.debug("Failed to close idle World Model websocket", exc_info=True)
                    return

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" not in msg or not msg["bytes"]:
                    continue

                try:
                    req = self._unpack_request(msg["bytes"])
                except Exception:
                    logger.exception("Invalid world model OpenPI request payload")
                    try:
                        await self._send_error("Invalid request payload")
                    except Exception:
                        break
                    continue

                try:
                    endpoint = req.pop("endpoint", "infer")

                    if endpoint == "reset":
                        self.serving.reset(req)
                        await self.websocket.send_text("reset successful")
                    else:
                        result = await self.serving.infer(req)

                        extra_body: dict = req.get("extra_body", {})

                        frames_per_chunk = extra_body.get("frames_per_chunk", DEFAULT_FRAMES_PER_CHUNK)

                        if (
                            len(result.images) == 1
                            and isinstance(result.images[0], tuple)
                            and len(result.images[0]) == 1
                        ):
                            frames = result.images[0]
                        elif len(result.images) == 1 and isinstance(result.images[0], dict):
                            frames = result.images[0].get("frames") or result.images[0].get("video")
                        else:
                            frames = result.images

                        if len(frames) == 1:
                            frames = frames[0]

                        if isinstance(frames, torch.Tensor):
                            frames = frames.numpy(force=True)

                        frames = _normalize_frames(frames)

                        total = (len(frames) + frames_per_chunk - 1) // frames_per_chunk
                        for i in range(total):
                            chunk = frames[i * frames_per_chunk : (i + 1) * frames_per_chunk]
                            await self.websocket.send_bytes(
                                _pack(
                                    {
                                        "type": "frame",
                                        "index": i,
                                        "total": total,
                                        "video": chunk,
                                    }
                                )
                            )

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
