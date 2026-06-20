# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection for robot policy inference (OpenPI protocol).

Protocol (compatible with OpenPI policy clients):
    Connect  -> server sends msgpack(PolicyServerConfig fields)
    Infer    -> client sends msgpack(obs), server sends msgpack(ndarray)
    Reset    -> client sends msgpack({endpoint:reset}), server sends msgpack(status)

NumPy values use msgpack-numpy marker mappings. Outbound payloads follow the
openpi-client format so official OpenPI clients can decode action responses:
    ndarray -> {__ndarray__: true, dtype, shape, data}
    scalar  -> {__npgeneric__: true, dtype, data}

Inbound payloads accept both the openpi-client markers above and the legacy
vLLM-native markers:
    ndarray -> {nd: true, type, kind, shape, data}
    scalar  -> {nd: false, type, kind, data}
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import msgspec
import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openpi import dreamzero_async_protocol as _dz_async_protocol
from vllm_omni.entrypoints.openpi.dreamzero_async_runtime import (
    DreamZeroAsyncSessionHarness,
)
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
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(f"Unsupported type: {type(obj)!r}")


def _mapping_get(obj: dict[Any, Any], key: str, default: Any = None) -> Any:
    return obj.get(key, obj.get(key.encode(), default))


def _decode_marker_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _is_truthy_marker(value: Any) -> bool:
    return value is True or value == 1


def _decode_vllm_numpy_marker(obj: dict[Any, Any]) -> Any:
    nd = _mapping_get(obj, "nd", _MISSING)
    dtype = _mapping_get(obj, "type", _MISSING)
    kind = _mapping_get(obj, "kind", _MISSING)
    data = _mapping_get(obj, "data", _MISSING)
    if nd is _MISSING or dtype is _MISSING or kind is _MISSING or data is _MISSING:
        return _MISSING

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


def _decode_openpi_numpy_marker(obj: dict[Any, Any]) -> Any:
    if _is_truthy_marker(_mapping_get(obj, "__ndarray__", False)):
        data = _mapping_get(obj, "data", _MISSING)
        dtype = _mapping_get(obj, "dtype", _MISSING)
        shape = _mapping_get(obj, "shape", _MISSING)
        if data is _MISSING or dtype is _MISSING or shape is _MISSING:
            raise ValueError("OpenPI ndarray marker is missing required fields")

        dtype_obj = np.dtype(_decode_marker_text(dtype))
        if dtype_obj.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {dtype_obj}")
        array = np.frombuffer(data, dtype=dtype_obj).copy()
        return array.reshape(tuple(shape))

    if _is_truthy_marker(_mapping_get(obj, "__npgeneric__", False)):
        data = _mapping_get(obj, "data", _MISSING)
        dtype = _mapping_get(obj, "dtype", _MISSING)
        if data is _MISSING or dtype is _MISSING:
            raise ValueError("OpenPI scalar marker is missing required fields")
        dtype_obj = np.dtype(_decode_marker_text(dtype))
        if dtype_obj.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {dtype_obj}")
        return dtype_obj.type(data)

    return _MISSING


def _unpack_numpy(obj: Any) -> Any:
    if isinstance(obj, dict):
        for decoder in (_decode_vllm_numpy_marker, _decode_openpi_numpy_marker):
            decoded = decoder(obj)
            if decoded is not _MISSING:
                return decoded
        return {key: _unpack_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_unpack_numpy(value) for value in obj]
    return obj


def _pack(obj: Any) -> bytes:
    return msgspec.msgpack.encode(obj, enc_hook=_pack_numpy)


def _unpack(data: bytes) -> Any:
    return _unpack_numpy(msgspec.msgpack.decode(data))


class DreamZeroAsyncRealtimeConnection:
    """WebSocket connection for DreamZero async robot control.

    The connection owns protocol handshake/reset/error behavior and delegates
    background observation/forward scheduling to ``DreamZeroAsyncSessionHarness``.
    """

    def __init__(
        self,
        websocket: WebSocket,
        serving: ServingRealtimeRobotOpenPI,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        scheduler: DreamZeroAsyncSessionHarness | None = None,
    ) -> None:
        self.websocket = websocket
        self.serving = serving
        self._idle_timeout = idle_timeout
        self._scheduler = scheduler
        self._session_id: str | None = None
        self._session_epoch = 0
        self._prompt = ""

    async def _send_payload(self, payload: dict[str, Any]) -> None:
        await self.websocket.send_bytes(_dz_async_protocol.pack_message(payload))

    async def _send_error(
        self,
        code: str,
        message: str,
        *,
        session_id: str | None = None,
        session_epoch: int | None = None,
    ) -> None:
        await self._send_payload(
            _dz_async_protocol.make_error(
                code=code,
                message=message,
                session_id=session_id if session_id is not None else self._session_id,
                session_epoch=session_epoch if session_epoch is not None else self._current_epoch_or_none(),
            )
        )

    def _current_epoch_or_none(self) -> int | None:
        return self._session_epoch if self._session_epoch > 0 else None

    def _metadata(self) -> dict[str, Any]:
        return _dz_async_protocol.make_metadata(self.serving.policy_server_config.to_dict())

    def _unpack_request(self, data: bytes) -> dict[str, Any]:
        return _dz_async_protocol.validate_client_message(_dz_async_protocol.unpack_message(data))

    async def handle_connection(self) -> None:
        await self.websocket.accept()
        writer_task: asyncio.Task[None] | None = None

        try:
            await self._send_payload(self._metadata())
            if self._scheduler is not None:
                writer_task = asyncio.create_task(self._writer_loop())
                writer_task.add_done_callback(self._log_writer_task_done)

            while True:
                try:
                    msg = await asyncio.wait_for(
                        self.websocket.receive(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.info("DreamZero async connection idle timeout after %.1f seconds", self._idle_timeout)
                    try:
                        await self.websocket.close()
                    except Exception:
                        logger.debug("Failed to close idle DreamZero async websocket", exc_info=True)
                    return

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" not in msg or not msg["bytes"]:
                    continue

                try:
                    request = self._unpack_request(msg["bytes"])
                except _dz_async_protocol.ProtocolValidationError as exc:
                    await self._send_error(exc.code, exc.message)
                    continue
                except Exception:
                    logger.exception("Invalid DreamZero async request payload")
                    try:
                        await self._send_error("invalid_payload", "Invalid DreamZero async request payload")
                    except Exception:
                        break
                    continue

                try:
                    await self._handle_request(request)
                except Exception:
                    logger.exception("Error handling DreamZero async request")
                    try:
                        await self._send_error("internal_error", "Internal DreamZero async error")
                    except Exception:
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("DreamZero async connection error")
        finally:
            if writer_task is not None:
                writer_task.cancel()
                await asyncio.gather(writer_task, return_exceptions=True)
            if self._scheduler is not None:
                await self._scheduler.close()

    async def _writer_loop(self) -> None:
        if self._scheduler is None:
            return
        while True:
            payload = await self._scheduler.get_outbound()
            logger.debug("Sending DreamZero async outbound message type=%s", payload.get("type"))
            await self._send_payload(payload)

    def _log_writer_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(
                "DreamZero async outbound writer failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _handle_request(self, request: dict[str, Any]) -> None:
        msg_type = request["type"]
        if msg_type == _dz_async_protocol.SESSION_START:
            await self._handle_session_start(request)
            return
        if msg_type == _dz_async_protocol.SESSION_RESET:
            await self._handle_session_reset(request)
            return
        if msg_type == _dz_async_protocol.OBSERVATION_REAL:
            await self._handle_observation_real(request)
            return
        await self._send_error("unknown_message_type", f"Unknown DreamZero async message type: {msg_type}")

    async def _handle_session_start(self, request: dict[str, Any]) -> None:
        if self._session_id is not None:
            await self._send_error("session_already_started", "DreamZero async session is already started")
            return
        self._session_id = request["session_id"]
        self._session_epoch = 1
        self._prompt = request["prompt"]
        if self._scheduler is not None:
            await self._scheduler.start_session(
                session_id=self._session_id,
                session_epoch=self._session_epoch,
                prompt=self._prompt,
            )
        await self._send_payload(_dz_async_protocol.make_session_started(self._session_id, self._session_epoch))

    async def _handle_session_reset(self, request: dict[str, Any]) -> None:
        if not self._matches_active_session(request):
            await self._send_error(
                "stale_session",
                "session_reset does not match the active DreamZero async session",
                session_id=request["session_id"],
                session_epoch=request["session_epoch"],
            )
            return
        self._session_epoch += 1
        if self._scheduler is not None:
            await self._scheduler.reset(session_epoch=self._session_epoch)
        await self._send_payload(
            _dz_async_protocol.make_session_reset_ack(
                self._session_id or request["session_id"],
                self._session_epoch,
            )
        )

    async def _handle_observation_real(self, request: dict[str, Any]) -> None:
        if not self._matches_active_session(request):
            await self._send_error(
                "stale_session",
                "observation_real does not match the active DreamZero async session",
                session_id=request["session_id"],
                session_epoch=request["session_epoch"],
            )
            return
        if self._scheduler is not None:
            await self._scheduler.submit_observation(request)
            return
        await self._send_error(
            "scheduler_unavailable",
            "DreamZero async scheduler is not enabled in this implementation slice",
        )

    def _matches_active_session(self, request: dict[str, Any]) -> bool:
        return self._session_id == request["session_id"] and self._session_epoch == request["session_epoch"]


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

    async def _cleanup_session(self, session_id: str | None) -> None:
        if session_id is None:
            return

        cleanup = getattr(self.serving, "cleanup_dreamzero_session", None)
        if cleanup is None:
            return

        try:
            result = cleanup(session_id)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Failed to clean up DreamZero OpenPI session %s", session_id)

    async def _cleanup_current_session(self) -> None:
        await self._cleanup_session(self._current_session_id)

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
                        await self._cleanup_current_session()
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
                                await self._cleanup_current_session()
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
        finally:
            await self._cleanup_current_session()
