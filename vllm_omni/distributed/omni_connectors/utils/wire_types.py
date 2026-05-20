# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import struct
from typing import Any

import msgspec
import torch

TENSOR_DICT_MAGIC = b"OMTD"
TENSOR_DICT_VERSION = 1
TENSOR_DICT_COMPRESSION_NONE = 0

_HEADER_PREFIX = struct.Struct("<4sII")

_DTYPE_TO_ID: dict[torch.dtype, int] = {
    torch.bool: 1,
    torch.uint8: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.int64: 6,
    torch.float16: 7,
    torch.bfloat16: 8,
    torch.float32: 9,
    torch.float64: 10,
    torch.complex64: 11,
    torch.complex128: 12,
}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}


def is_tensor_dict(obj: Any) -> bool:
    """Return True when *obj* is a dict[str, torch.Tensor]."""
    return isinstance(obj, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items())


def serialize_tensor_dict(tensors: dict[str, torch.Tensor]) -> bytes:
    """Serialize a dict[str, torch.Tensor] without generic object serialization.

    The v1 wire format stores a small msgpack header followed by concatenated
    tensor bytes. A per-tensor compression field is reserved for future use;
    v1 only accepts ``TENSOR_DICT_COMPRESSION_NONE``.
    """
    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0

    for name, tensor in tensors.items():
        dtype_id = _DTYPE_TO_ID.get(tensor.dtype)
        if dtype_id is None:
            raise TypeError(f"Unsupported tensor dtype for tensor-dict wire format: {tensor.dtype}")

        host_tensor = tensor.detach()
        if not host_tensor.is_contiguous():
            host_tensor = host_tensor.contiguous()
        if host_tensor.device.type != "cpu":
            host_tensor = host_tensor.cpu()

        raw = host_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        entries.append(
            {
                "name": name,
                "dtype": dtype_id,
                "shape": list(tensor.shape),
                "offset": offset,
                "nbytes": len(raw),
                "compression": TENSOR_DICT_COMPRESSION_NONE,
            }
        )
        chunks.append(raw)
        offset += len(raw)

    header = msgspec.msgpack.encode({"version": TENSOR_DICT_VERSION, "entries": entries})
    return _HEADER_PREFIX.pack(TENSOR_DICT_MAGIC, TENSOR_DICT_VERSION, len(header)) + header + b"".join(chunks)


def deserialize_tensor_dict(data: bytes | bytearray | memoryview) -> dict[str, torch.Tensor]:
    """Deserialize bytes produced by :func:`serialize_tensor_dict`."""
    view = memoryview(data)
    if len(view) < _HEADER_PREFIX.size:
        raise ValueError("Tensor-dict payload is shorter than the wire header")

    magic, version, header_len = _HEADER_PREFIX.unpack(view[: _HEADER_PREFIX.size])
    if magic != TENSOR_DICT_MAGIC:
        raise ValueError("Tensor-dict payload has an invalid magic value")
    if version != TENSOR_DICT_VERSION:
        raise ValueError(f"Unsupported tensor-dict wire format version: {version}")

    header_start = _HEADER_PREFIX.size
    header_end = header_start + header_len
    if header_end > len(view):
        raise ValueError("Tensor-dict payload has a truncated header")

    header = msgspec.msgpack.decode(view[header_start:header_end])
    if header.get("version") != TENSOR_DICT_VERSION:
        raise ValueError(f"Unsupported tensor-dict header version: {header.get('version')}")

    body = view[header_end:]
    result: dict[str, torch.Tensor] = {}
    for entry in header["entries"]:
        compression = entry.get("compression", TENSOR_DICT_COMPRESSION_NONE)
        if compression != TENSOR_DICT_COMPRESSION_NONE:
            raise ValueError(f"Unsupported tensor-dict compression id: {compression}")

        dtype = _ID_TO_DTYPE.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported tensor-dict dtype id: {entry['dtype']}")

        offset = entry["offset"]
        nbytes = entry["nbytes"]
        raw = body[offset : offset + nbytes]
        if len(raw) != nbytes:
            raise ValueError(f"Tensor-dict payload is truncated for tensor {entry['name']!r}")

        expected_nbytes = torch.empty((), dtype=dtype).element_size()
        for dim in entry["shape"]:
            expected_nbytes *= dim
        if expected_nbytes != nbytes:
            raise ValueError(
                f"Tensor-dict tensor {entry['name']!r} expected {expected_nbytes} bytes, got {nbytes}"
            )

        if nbytes == 0:
            result[entry["name"]] = torch.empty(entry["shape"], dtype=dtype)
        else:
            buffer = bytearray(raw)
            tensor = torch.frombuffer(buffer, dtype=torch.uint8)
            result[entry["name"]] = tensor.view(dtype).reshape(entry["shape"])

    return result
