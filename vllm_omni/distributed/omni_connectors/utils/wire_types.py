# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import struct
import zlib
from typing import Any

import torch

TENSOR_DICT_MAGIC = b"OMDI"
TENSOR_DICT_VERSION = 1
TENSOR_DICT_COMPRESSION_NONE = 0
TENSOR_DICT_LAYOUT_CONTIGUOUS = 0
TENSOR_DICT_ALIGNMENT = 64

_HEADER_PREFIX = struct.Struct("<4sHHII")
_TENSOR_ENTRY_PREFIX = struct.Struct("<H")
_TENSOR_ENTRY_META = struct.Struct("<HBB")
_TENSOR_ENTRY_SUFFIX = struct.Struct("<QQ")

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

    The v1 wire format follows the RFC binary header contract:
    magic/version/header length/count/checksum, per-tensor binary metadata,
    64-byte header padding, then each tensor payload 64-byte aligned.
    """
    entries = bytearray()
    chunks: list[tuple[int, bytes]] = []
    data_offset = 0

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
        aligned_offset = _align(data_offset)
        key = name.encode("utf-8")
        if len(key) > 0xFFFF:
            raise ValueError(f"Tensor key is too long for tensor-dict wire format: {name!r}")
        if tensor.dim() > 0xFF:
            raise ValueError(f"Tensor rank is too large for tensor-dict wire format: {tensor.dim()}")

        entries.extend(_TENSOR_ENTRY_PREFIX.pack(len(key)))
        entries.extend(key)
        entries.extend(
            _TENSOR_ENTRY_META.pack(
                dtype_id,
                tensor.dim(),
                TENSOR_DICT_LAYOUT_CONTIGUOUS,
            )
        )
        for dim in tensor.shape:
            entries.extend(struct.pack("<q", dim))
        entries.extend(_TENSOR_ENTRY_SUFFIX.pack(aligned_offset, len(raw)))
        chunks.append((aligned_offset, raw))
        data_offset = aligned_offset + len(raw)

    header_len = _align(_HEADER_PREFIX.size + len(entries))
    if header_len > 0xFFFF:
        raise ValueError(f"Tensor-dict header is too large: {header_len} bytes")

    header = bytearray(_HEADER_PREFIX.pack(TENSOR_DICT_MAGIC, TENSOR_DICT_VERSION, header_len, len(tensors), 0))
    header.extend(entries)
    header.extend(b"\x00" * (header_len - len(header)))
    checksum = zlib.crc32(header) & 0xFFFFFFFF
    header[: _HEADER_PREFIX.size] = _HEADER_PREFIX.pack(
        TENSOR_DICT_MAGIC,
        TENSOR_DICT_VERSION,
        header_len,
        len(tensors),
        checksum,
    )

    body = bytearray(data_offset)
    for offset, raw in chunks:
        body[offset : offset + len(raw)] = raw
    return bytes(header + body)


def deserialize_tensor_dict(data: bytes | bytearray | memoryview) -> dict[str, torch.Tensor]:
    """Deserialize bytes produced by :func:`serialize_tensor_dict`."""
    view = memoryview(data)
    if len(view) < _HEADER_PREFIX.size:
        raise ValueError("Tensor-dict payload is shorter than the wire header")

    magic, version, header_len, n_tensors, checksum = _HEADER_PREFIX.unpack(view[: _HEADER_PREFIX.size])
    if magic != TENSOR_DICT_MAGIC:
        raise ValueError("Tensor-dict payload has an invalid magic value")
    if version != TENSOR_DICT_VERSION:
        raise ValueError(f"Unsupported tensor-dict wire format version: {version}")

    if header_len < _HEADER_PREFIX.size or header_len % TENSOR_DICT_ALIGNMENT != 0:
        raise ValueError(f"Tensor-dict payload has an invalid header length: {header_len}")
    if header_len > len(view):
        raise ValueError("Tensor-dict payload has a truncated header")

    header = bytearray(view[:header_len])
    header[: _HEADER_PREFIX.size] = _HEADER_PREFIX.pack(magic, version, header_len, n_tensors, 0)
    actual_checksum = zlib.crc32(header) & 0xFFFFFFFF
    if actual_checksum != checksum:
        raise ValueError("Tensor-dict payload header checksum mismatch")

    body = view[header_len:]
    result: dict[str, torch.Tensor] = {}
    cursor = _HEADER_PREFIX.size
    for _ in range(n_tensors):
        if cursor + _TENSOR_ENTRY_PREFIX.size > header_len:
            raise ValueError("Tensor-dict payload has a truncated tensor entry")
        (key_len,) = _TENSOR_ENTRY_PREFIX.unpack(view[cursor : cursor + _TENSOR_ENTRY_PREFIX.size])
        cursor += _TENSOR_ENTRY_PREFIX.size
        if cursor + key_len + _TENSOR_ENTRY_META.size > header_len:
            raise ValueError("Tensor-dict payload has a truncated tensor entry")
        name = bytes(view[cursor : cursor + key_len]).decode("utf-8")
        cursor += key_len
        dtype_id, ndim, layout = _TENSOR_ENTRY_META.unpack(view[cursor : cursor + _TENSOR_ENTRY_META.size])
        cursor += _TENSOR_ENTRY_META.size
        if layout != TENSOR_DICT_LAYOUT_CONTIGUOUS:
            raise ValueError(f"Unsupported tensor-dict layout id: {layout}")
        dtype = _ID_TO_DTYPE.get(dtype_id)
        if dtype is None:
            raise ValueError(f"Unsupported tensor-dict dtype id: {dtype_id}")
        shape: list[int] = []
        shape_bytes = ndim * 8
        if cursor + shape_bytes + _TENSOR_ENTRY_SUFFIX.size > header_len:
            raise ValueError("Tensor-dict payload has a truncated tensor entry")
        for _ in range(ndim):
            (dim,) = struct.unpack("<q", view[cursor : cursor + 8])
            if dim < 0:
                raise ValueError(f"Tensor-dict tensor {name!r} has invalid negative dimension: {dim}")
            shape.append(dim)
            cursor += 8
        offset, nbytes = _TENSOR_ENTRY_SUFFIX.unpack(view[cursor : cursor + _TENSOR_ENTRY_SUFFIX.size])
        cursor += _TENSOR_ENTRY_SUFFIX.size
        if offset % TENSOR_DICT_ALIGNMENT != 0:
            raise ValueError(f"Tensor-dict tensor {name!r} has unaligned data offset: {offset}")
        raw = body[offset : offset + nbytes]
        if len(raw) != nbytes:
            raise ValueError(f"Tensor-dict payload is truncated for tensor {name!r}")

        expected_nbytes = torch.empty((), dtype=dtype).element_size()
        for dim in shape:
            expected_nbytes *= dim
        if expected_nbytes != nbytes:
            raise ValueError(f"Tensor-dict tensor {name!r} expected {expected_nbytes} bytes, got {nbytes}")

        if nbytes == 0:
            result[name] = torch.empty(shape, dtype=dtype)
        else:
            buffer = bytearray(raw)
            tensor = torch.frombuffer(buffer, dtype=torch.uint8)
            result[name] = tensor.view(dtype).reshape(shape)

    return result


def _align(value: int, alignment: int = TENSOR_DICT_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment
