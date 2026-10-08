# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owned CPU tensor trees with a msgpack header and raw tensor regions.

Tensor references use message-scoped msgpack extensions, preserving literal
extensions and native scalar containers without reserved dictionary keys. It never
exposes a view of transport storage to a caller. Unsupported objects retain
the ordinary OmniSerializer path.
"""

from __future__ import annotations

import ctypes
import math
import struct
import uuid
from dataclasses import dataclass

import msgspec
import numpy as np
import torch
from PIL import Image

from .serialization import OmniSerializer

_LENGTH = struct.Struct("<I")
_ENCODER = msgspec.msgpack.Encoder()
_REFERENCE = struct.Struct("<16sI")
_REFERENCE_CODE = 42
_DECODER = msgspec.msgpack.Decoder()


def _align(size: int) -> int:
    return (size + 15) & ~15


class _UnsupportedTreeError(Exception):
    pass


@dataclass
class TensorFrame:
    header: bytes
    tensors: list[torch.Tensor]
    offsets: list[int]
    size: int

    def write(self, buffer: memoryview, start: int) -> None:
        """Copy directly into caller-owned writable storage, while it is locked."""
        _LENGTH.pack_into(buffer, start, len(self.header))
        buffer[start + 4 : start + 4 + len(self.header)] = self.header
        data_start = start + _align(4 + len(self.header))
        # c_char is an anchor only; all bounds are checked by the ring writer.
        anchor = ctypes.c_char.from_buffer(buffer)
        base = ctypes.addressof(anchor) + data_start
        for tensor, offset in zip(self.tensors, self.offsets, strict=True):
            if tensor.nbytes:
                ctypes.memmove(base + offset, tensor.data_ptr(), tensor.nbytes)


def prepare_tensor_frame(payload: object, *, max_bytes: int | None = None) -> TensorFrame | bytes | None:
    """Encode native containers once, or prepare their dense CPU tensor regions.

    Scalar/token lists stay in msgpack's C encoder. Tensor-free messages reuse
    those encoded bytes, rather than constructing and discarding a tagged tree.
    Sources remain caller-owned until the synchronous put() returns.
    """
    tensors: list[torch.Tensor] = []
    offsets: list[int] = []
    descriptors: list[tuple[str, tuple[int, ...], int]] = []
    data_size = 0
    nonce: bytes | None = None

    def encode_tensor(value: object) -> object:
        nonlocal data_size, nonce
        if type(value) is not torch.Tensor:
            # Reuse the common wire representations for mixed model payloads.
            # Large arrays/images keep the ordinary path without first making
            # an extra byte snapshot just to discover that the frame cannot fit.
            if max_bytes is not None:
                if isinstance(value, np.ndarray) and value.nbytes > max_bytes:
                    raise _UnsupportedTreeError
                if isinstance(value, Image.Image) and value.width * value.height * len(value.getbands()) > max_bytes:
                    raise _UnsupportedTreeError
            return OmniSerializer.encoder._enc_hook(value)
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
            raise _UnsupportedTreeError
        offset = _align(data_size)
        index = len(tensors)
        tensors.append(value)
        offsets.append(offset)
        descriptors.append((str(value.dtype).removeprefix("torch."), tuple(value.shape), offset))
        data_size = offset + value.nbytes
        if max_bytes is not None and data_size > max_bytes:
            raise _UnsupportedTreeError
        if nonce is None:
            nonce = uuid.uuid4().bytes
        return msgspec.msgpack.Ext(_REFERENCE_CODE, _REFERENCE.pack(nonce, index))

    try:
        tree = msgspec.msgpack.Encoder(enc_hook=encode_tensor).encode(payload)
        if not tensors:
            return tree
        header = _ENCODER.encode([descriptors, tree, nonce])
    except (_UnsupportedTreeError, TypeError, OverflowError, RecursionError):
        return None
    size = _align(4 + len(header)) + data_size
    if max_bytes is not None and size > max_bytes:
        return None
    # Only materialize non-contiguous/flagged storage after the entire message
    # is known to fit. Unsupported and large frames avoid an unused copy.
    for index, tensor in enumerate(tensors):
        if tensor.is_conj():
            tensor = tensor.resolve_conj()
        if tensor.is_neg():
            tensor = tensor.resolve_neg()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensors[index] = tensor
    return TensorFrame(header, tensors, offsets, size)


def read_tensor_frame(buffer: memoryview) -> object:
    """Decode into independent CPU allocations before returning ring credit."""
    if len(buffer) < 4:
        raise ValueError("truncated tensor frame")
    header_size = _LENGTH.unpack_from(buffer)[0]
    data_start = _align(4 + header_size)
    if data_start > len(buffer):
        raise ValueError("tensor frame header exceeds payload")
    header_view = buffer[4 : 4 + header_size]
    try:
        descriptors, tree, nonce = _DECODER.decode(header_view)
    finally:
        header_view.release()
    # All leaves share one independently owned allocation. This amortizes CPU
    # allocation/copy dispatch across small codec groups and hidden-state leaves.
    data_size = len(buffer) - data_start
    # A small metadata tensor must not keep a large, already consumed hidden
    # state alive. Pack small trees; give large multi-leaf trees separate owners.
    storage = torch.empty(data_size, dtype=torch.uint8) if data_size <= 65536 or len(descriptors) == 1 else None
    anchor = ctypes.c_char.from_buffer(buffer)
    if data_size and storage is not None:
        ctypes.memmove(storage.data_ptr(), ctypes.addressof(anchor) + data_start, data_size)
    tensors = []
    for name, shape, offset in descriptors:
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("unknown tensor frame dtype")
        if any(type(dim) is not int or dim < 0 for dim in shape) or type(offset) is not int or offset < 0:
            raise ValueError("invalid tensor frame region")
        count = math.prod(shape)
        size = count * dtype.itemsize
        start = data_start + offset
        if start + size > len(buffer):
            raise ValueError("tensor frame region exceeds payload")
        if size and storage is not None:
            tensor = storage[offset : offset + size].view(dtype).reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
            if size:
                ctypes.memmove(tensor.data_ptr(), ctypes.addressof(anchor) + start, size)
        tensors.append(tensor)

    def restore_reference(code: int, data: memoryview) -> object:
        if code == _REFERENCE_CODE and len(data) == _REFERENCE.size:
            reference_nonce, index = _REFERENCE.unpack(data)
            if reference_nonce == nonce:
                if index >= len(tensors):
                    raise ValueError("invalid tensor frame reference")
                return tensors[index]
        # Literal user extensions keep their existing msgpack wire semantics.
        return msgspec.msgpack.Ext(code, bytes(data))

    return msgspec.msgpack.decode(tree, ext_hook=restore_reference)
