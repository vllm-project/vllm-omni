# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NumPy/MessagePack wire markers shared by robot-policy APIs.

Emit OpenPI-compatible markers; accept both those and legacy vLLM markers.
Object, structured and complex dtypes are rejected in both directions.
"""

from typing import Any

import msgpack
import msgspec
import numpy as np

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_MISSING = object()


def _pack_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ("V", "O", "c"):
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

    kind_text = _decode_marker_text(kind)
    # Structured markers carry a dtype descriptor list in `type`, so reject them
    # before `np.dtype()` sees something it cannot parse.
    if kind_text == "V":
        raise ValueError("Unsupported dtype: structured arrays")

    dtype_obj: np.dtype[Any] = np.dtype(_decode_marker_text(dtype))
    if kind_text not in ("", dtype_obj.kind):
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

        dtype_obj: np.dtype[Any] = np.dtype(_decode_marker_text(dtype))
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
        for key, value in obj.items():
            obj[key] = _unpack_numpy(value)
        return obj
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _unpack_numpy(value)
        return obj
    return obj


def pack(obj: Any) -> bytes:
    return msgspec.msgpack.encode(obj, enc_hook=_pack_numpy)


def unpack(data: bytes) -> Any:
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError("Robot request payload too large")
    remaining = 16384

    def container(value: Any) -> Any:
        nonlocal remaining
        remaining -= len(value) + 1
        if remaining < 0:
            raise ValueError("Robot request exceeds container element budget")
        return _unpack_numpy(value) if isinstance(value, dict) else value

    def mapping(pairs: list) -> Any:
        container(pairs)
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate MessagePack map keys")
        return _unpack_numpy(result)

    return msgpack.unpackb(
        data,
        object_pairs_hook=mapping,
        list_hook=container,
        strict_map_key=False,
        max_array_len=4096,
        max_map_len=256,
        max_str_len=65536,
        max_bin_len=MAX_PAYLOAD_BYTES,
        max_ext_len=0,
    )
