# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero async protocol helpers for robot policy websocket serving."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgspec
import numpy as np

PROTOCOL_NAME = "dreamzero_async"
PROTOCOL_VERSION = 1
LOOKAHEAD_MODE = "real_rebased_one_step_simulated"

SESSION_START = "session_start"
SESSION_STARTED = "session_started"
SESSION_RESET = "session_reset"
SESSION_RESET_ACK = "session_reset_ack"
OBSERVATION_REAL = "observation_real"
ACTION_CHUNK = "action_chunk"
ERROR = "error"

MAX_DREAMZERO_ASYNC_PAYLOAD_BYTES = 64 * 1024 * 1024
_MISSING = object()


class ProtocolValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def pack_message(message: Mapping[str, Any]) -> bytes:
    return msgspec.msgpack.encode(dict(message), enc_hook=_pack_numpy)


def unpack_message(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_DREAMZERO_ASYNC_PAYLOAD_BYTES:
        raise ProtocolValidationError("payload_too_large", "DreamZero async request payload too large")
    payload = _unpack_numpy(msgspec.msgpack.decode(data))
    if not isinstance(payload, dict):
        raise ProtocolValidationError("invalid_payload", "DreamZero async message must be a dictionary")
    return payload


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


def make_metadata(policy_metadata: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(policy_metadata)
    model: dict[str, Any] = {}
    for key in ("action_horizon", "action_dim"):
        if key in metadata:
            model[key] = metadata[key]
    metadata.update(
        {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "lookahead_mode": LOOKAHEAD_MODE,
        }
    )
    if model:
        metadata["model"] = model
    return metadata


def make_session_started(session_id: str, session_epoch: int) -> dict[str, Any]:
    return {
        "type": SESSION_STARTED,
        "session_id": session_id,
        "session_epoch": session_epoch,
        "lookahead_mode": LOOKAHEAD_MODE,
    }


def make_session_reset_ack(session_id: str, session_epoch: int) -> dict[str, Any]:
    return {
        "type": SESSION_RESET_ACK,
        "session_id": session_id,
        "session_epoch": session_epoch,
    }


def make_error(
    *,
    code: str,
    message: str,
    session_id: str | None = None,
    session_epoch: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": ERROR,
        "code": code,
        "message": message,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if session_epoch is not None:
        payload["session_epoch"] = session_epoch
    return payload


def validate_client_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    msg_type = _require_str(payload, "type")
    if msg_type == SESSION_START:
        return validate_session_start(payload)
    if msg_type == OBSERVATION_REAL:
        return validate_observation_real(payload)
    if msg_type == SESSION_RESET:
        return validate_session_reset(payload)
    raise ProtocolValidationError("unknown_message_type", f"Unknown DreamZero async message type: {msg_type}")


def validate_session_start(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_type(payload, SESSION_START)
    protocol_version = _require_int(payload, "protocol_version", minimum=1)
    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolValidationError(
            "unsupported_protocol_version",
            f"Unsupported DreamZero async protocol_version: {protocol_version}",
        )
    session_id = _require_str(payload, "session_id")
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ProtocolValidationError("invalid_field", "session_start.prompt must be a string")
    return {
        "type": SESSION_START,
        "protocol_version": protocol_version,
        "session_id": session_id,
        "prompt": prompt,
    }


def validate_observation_real(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_type(payload, OBSERVATION_REAL)
    robot_obs = payload.get("robot_obs")
    if not isinstance(robot_obs, Mapping):
        raise ProtocolValidationError("invalid_field", "observation_real.robot_obs must be a dictionary")
    return {
        "type": OBSERVATION_REAL,
        "session_id": _require_str(payload, "session_id"),
        "session_epoch": _require_int(payload, "session_epoch", minimum=1),
        "observation_index": _require_int(payload, "observation_index", minimum=1),
        "timestamp_s": _require_number(payload, "timestamp_s"),
        "robot_obs": dict(robot_obs),
    }


def validate_session_reset(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_type(payload, SESSION_RESET)
    return {
        "type": SESSION_RESET,
        "session_id": _require_str(payload, "session_id"),
        "session_epoch": _require_int(payload, "session_epoch", minimum=1),
    }


def validate_action_chunk(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_type(payload, ACTION_CHUNK)
    actions = payload.get("actions")
    if actions is None:
        raise ProtocolValidationError("invalid_field", "action_chunk.actions is required")
    provenance = payload.get("provenance", {})
    monitoring = payload.get("monitoring", {})
    if not isinstance(provenance, Mapping):
        raise ProtocolValidationError("invalid_field", "action_chunk.provenance must be a dictionary")
    if not isinstance(monitoring, Mapping):
        raise ProtocolValidationError("invalid_field", "action_chunk.monitoring must be a dictionary")
    return {
        "type": ACTION_CHUNK,
        "session_id": _require_str(payload, "session_id"),
        "session_epoch": _require_int(payload, "session_epoch", minimum=1),
        "chunk_index": _require_int(payload, "chunk_index", minimum=1),
        "actions": actions,
        "provenance": dict(provenance),
        "monitoring": dict(monitoring),
    }


def validate_error(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_type(payload, ERROR)
    result: dict[str, Any] = {
        "type": ERROR,
        "code": _require_str(payload, "code"),
        "message": _require_str(payload, "message"),
    }
    session_id = payload.get("session_id")
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id:
            raise ProtocolValidationError("invalid_field", "error.session_id must be a non-empty string")
        result["session_id"] = session_id
    session_epoch = payload.get("session_epoch")
    if session_epoch is not None:
        result["session_epoch"] = _require_int(payload, "session_epoch", minimum=1)
    return result


def _require_type(payload: Mapping[str, Any], expected: str) -> None:
    msg_type = _require_str(payload, "type")
    if msg_type != expected:
        raise ProtocolValidationError("invalid_message_type", f"Expected message type {expected}, got {msg_type}")


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolValidationError("invalid_field", f"{key} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError("invalid_field", f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolValidationError("invalid_field", f"{key} must be >= {minimum}")
    return value


def _require_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolValidationError("invalid_field", f"{key} must be a number")
    return float(value)
