# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MessagePack representation of numeric NumPy observations and policy outputs."""

from typing import Any

import msgpack
import numpy as np


def _numeric_dtype(name: str | bytes) -> np.dtype:
    if not isinstance(name, (str, bytes)):
        raise ValueError("Missing numeric dtype")
    dtype = np.dtype(name.decode() if isinstance(name, bytes) else name)
    if dtype.kind not in "buif":
        raise ValueError(f"Unsupported robot tensor dtype {dtype}")
    return dtype


def _encode(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        dtype = _numeric_dtype(value.dtype.str).str
        if isinstance(value, np.ndarray):
            return {b"__ndarray__": True, b"dtype": dtype, b"shape": list(value.shape), b"data": value.tobytes()}
        return {b"__npgeneric__": True, b"dtype": dtype, b"data": value.item()}
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot encode {type(value).__name__} in a robot message")


def _decode(fields: dict) -> Any:
    # Clients may encode marker keys as MessagePack strings or binary data.
    def field(key: str) -> Any:
        return fields.get(key.encode(), fields.get(key))

    array = field("__ndarray__") is True
    scalar = field("__npgeneric__") is True
    # Some clients send numeric arrays with the msgpack-numpy nd/type dialect.
    legacy = isinstance(field("nd"), bool) and field("kind") is not None and field("type") is not None
    if not (array or scalar or legacy):
        return fields
    dtype = _numeric_dtype(field("type") if legacy else field("dtype"))
    if legacy and field("kind") not in ("", b"", dtype.kind, dtype.kind.encode()):
        raise ValueError("NumPy kind does not match dtype")
    data = field("data")
    if scalar:
        if not np.isscalar(data):
            raise ValueError("Invalid numeric scalar")
        return dtype.type(data)
    if not isinstance(data, bytes):
        raise ValueError("Numeric arrays require binary data")
    values = np.frombuffer(data, dtype=dtype)
    if legacy and field("nd") is False:
        return values.reshape(())[()]
    shape = field("shape")
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Invalid numeric array shape")
    return values.reshape(shape).copy()


def pack_message(value: Any) -> bytes:
    """Pack numeric scalars and arrays without converting their dtype or byte order."""
    return msgpack.packb(value, default=_encode, use_bin_type=True, strict_types=True)


def unpack_message(data: bytes) -> Any:
    """Decode an observation, handshake, action chunk or native-output mapping."""
    return msgpack.unpackb(data, raw=False, object_hook=_decode)
