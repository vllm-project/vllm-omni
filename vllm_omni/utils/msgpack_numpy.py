# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NumPy-aware msgpack (de)serialization.

Implements the msgpack-numpy ``__ndarray__`` marker convention used by OpenPI
policy clients, on top of ``msgspec`` (already a dependency). This lets the
server interoperate with those clients, and pass numpy payloads through
serialization boundaries, without depending on the client library.
"""

from typing import Any

import msgspec
import numpy as np


def _enc_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        array = obj if obj.flags.c_contiguous else np.ascontiguousarray(obj)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise NotImplementedError(f"Cannot serialize object of type {type(obj)!r}")


def _as_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if b"__ndarray__" in obj:
            return np.ndarray(
                buffer=bytearray(obj[b"data"]),
                dtype=np.dtype(_as_text(obj[b"dtype"])),
                shape=tuple(obj[b"shape"]),
            )
        if b"__npgeneric__" in obj:
            return np.dtype(_as_text(obj[b"dtype"])).type(obj[b"data"])
        return {key: _decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode(value) for value in obj]
    return obj


def packb(obj: Any) -> bytes:
    """Serialize ``obj`` (with numpy arrays) to msgpack bytes."""
    return msgspec.msgpack.encode(obj, enc_hook=_enc_hook)


def unpackb(data: bytes) -> Any:
    """Deserialize msgpack bytes, reconstructing numpy arrays."""
    return _decode(msgspec.msgpack.decode(data))
