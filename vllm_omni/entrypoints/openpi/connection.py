# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection for robot policy inference (OpenPI protocol).

Protocol (compatible with OpenPI policy clients):
    Connect  -> server sends msgpack(PolicyServerConfig fields)
    Infer    -> client sends msgpack(obs), server sends msgpack(ndarray)
    Reset    -> client sends msgpack({endpoint:reset}), server sends msgpack(status)

NumPy values use the msgpack-numpy marker mapping:
    ndarray -> {nd: true, type, kind, shape, data}
    scalar  -> {nd: false, type, kind, data}
"""

from __future__ import annotations

import asyncio
from typing import Any

import msgspec
import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openpi.serving import (
    ServingRealtimeRobotOpenPI,
)

logger = init_logger(__name__)
_DEFAULT_IDLE_TIMEOUT = 30.0
MAX_OPENPI_PAYLOAD_BYTES = 64 * 1024 * 1024
_MISSING = object()


def _pack_numpy(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        if not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)
        return {
            b"nd": True,
            b"data": obj.tobytes(),
            b"type": obj.dtype.str,
            b"kind": obj.dtype.kind,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"nd": False,
            b"data": obj.tobytes(),
            b"type": obj.dtype.str,
            b"kind": obj.dtype.kind,
        }
    raise TypeError(f"Unsupported type: {type(obj)!r}")


def _mapping_get(obj: dict[Any, Any], key: str, default: Any = None) -> Any:
    return obj.get(key, obj.get(key.encode(), default))


def _decode_marker_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _unpack_numpy(obj: Any) -> Any:
    if isinstance(obj, dict):
        nd = _mapping_get(obj, "nd", _MISSING)
        dtype = _mapping_get(obj, "type", _MISSING)
        kind = _mapping_get(obj, "kind", _MISSING)
        data = _mapping_get(obj, "data", _MISSING)
        if nd is not _MISSING and dtype is not _MISSING and kind is not _MISSING and data is not _MISSING:
            dtype_obj = np.dtype(_decode_marker_text(dtype))
            kind_text = _decode_marker_text(kind)
            if dtype_obj.kind != kind_text:
                raise ValueError(f"NumPy dtype marker kind mismatch: {dtype_obj.kind!r} != {kind_text!r}")
            if dtype_obj.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {dtype_obj}")

            array = np.frombuffer(data, dtype=dtype_obj).copy()
            if nd:
                shape = _mapping_get(obj, "shape", _MISSING)
                if shape is _MISSING:
                    raise ValueError("NumPy ndarray marker is missing shape")
                return array.reshape(tuple(shape))
            return array[0]
        return {key: _unpack_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_unpack_numpy(value) for value in obj]
    return obj


def _pack(obj: Any) -> bytes:
    return msgspec.msgpack.encode(obj, enc_hook=_pack_numpy)


def _unpack(data: bytes) -> Any:
    return _unpack_numpy(msgspec.msgpack.decode(data))


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
