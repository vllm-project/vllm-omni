# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Versioned tensor frames for the opt-in Mooncake device payload path.

Device tensor bodies stay on device; headers and CPU tensor fields use the host.
Received storage is cloned before the registered pool is released.
This is a value transport: strides and storage aliasing are not preserved.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any

import msgspec
import torch

_MAGIC = b"OMNITNSR"
_PREFIX = struct.Struct("<8sQ")
_ALIGNMENT = 64
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_TENSORS = 4096
_MAX_DEPTH = 64
_DTYPES = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


class UnsupportedTensorPayloadError(ValueError):
    """The payload must use the existing object serializer instead."""


class TensorMetadata(msgspec.Struct):
    dtype: str
    shape: list[int]
    offset: int
    nbytes: int
    cpu: bool


class FrameMetadata(msgspec.Struct):
    version: int
    tree: Any
    tensors: list[TensorMetadata]
    data_size: int


def _align(size: int) -> int:
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def synchronize_stream(device: torch.device) -> None:
    """Wait for copies on the calling thread's current device stream only."""
    if device.type != "cpu":
        torch.get_device_module(device).current_stream(device).synchronize()


@dataclass
class PreparedTensorPayload:
    """An owned frame captured on the producer stream before queue handoff.

    The producer may reuse its outputs after preparation on that same stream.
    Cancellation can drop this object: the frame's allocator tracks the stream
    on which it was allocated. The background sender waits only on this event.
    """

    frame: torch.Tensor
    ready_event: Any = None

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()


def synchronize_device(device: torch.device) -> None:
    """Fence device work, including producers on other streams/threads.

    A device fence is conservative but necessary without a producer event in
    the connector API. The caller must keep inputs immutable until put returns.
    """
    if device.type != "cpu":
        torch.get_device_module(device).synchronize(device)


@dataclass
class TensorPayloadPlan:
    metadata: bytes
    tensors: list[torch.Tensor]
    descriptors: list[TensorMetadata]
    data_size: int

    @property
    def data_offset(self) -> int:
        return _align(_PREFIX.size + len(self.metadata))

    @property
    def size(self) -> int:
        return self.data_offset + self.data_size

    def pack_into(self, destination: torch.Tensor, *, fence_producers: bool = True) -> None:
        """Copy one frame into an allocated uint8 pool slice, then fence it."""
        if destination.dtype != torch.uint8 or destination.numel() != self.size:
            raise ValueError("Invalid tensor payload destination")
        if fence_producers:
            devices = {tensor.device for tensor in self.tensors} | {destination.device}
            for device in devices:
                synchronize_device(device)
        # Clear alignment padding so reused pool contents never leave the host.
        destination.zero_()
        header = bytearray(_PREFIX.pack(_MAGIC, len(self.metadata)) + self.metadata)
        destination[: len(header)].copy_(torch.frombuffer(header, dtype=torch.uint8))
        for tensor, descriptor in zip(self.tensors, self.descriptors):
            if descriptor.nbytes:
                raw = tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
                start = self.data_offset + descriptor.offset
                destination[start : start + descriptor.nbytes].copy_(raw)
        if fence_producers:
            synchronize_stream(destination.device)

    def snapshot(self, device: torch.device) -> PreparedTensorPayload:
        """Capture immutable tensor values on the caller's producer stream."""
        frame = torch.empty(self.size, dtype=torch.uint8, device=device)
        self.pack_into(frame, fence_producers=False)
        event = None
        if device.type != "cpu":
            module = torch.get_device_module(device)
            event = module.Event()
            event.record(module.current_stream(device))
        return PreparedTensorPayload(frame, event)


def plan_tensor_payload(value: Any, pool_device: torch.device) -> TensorPayloadPlan:
    """Describe supported nested values without moving any tensor to CPU."""
    tensors: list[torch.Tensor] = []
    descriptors: list[TensorMetadata] = []
    tensor_ids: dict[int, int] = {}
    ancestors: set[int] = set()
    data_size = 0

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal data_size
        if depth > _MAX_DEPTH:
            raise UnsupportedTensorPayloadError("Tensor payload is too deeply nested")
        if isinstance(item, torch.Tensor):
            dtype = str(item.dtype).removeprefix("torch.")
            if (
                type(item) is not torch.Tensor
                or item.layout != torch.strided
                or dtype not in _DTYPES
                or item.device.type == "meta"
                or item.device not in (torch.device("cpu"), pool_device)
            ):
                raise UnsupportedTensorPayloadError("Unsupported tensor layout, dtype or device")
            if id(item) not in tensor_ids:
                if len(tensors) >= _MAX_TENSORS:
                    raise UnsupportedTensorPayloadError("Too many tensors")
                tensor_ids[id(item)] = len(tensors)
                tensors.append(item)
                descriptors.append(
                    TensorMetadata(dtype, list(item.shape), data_size, item.nbytes, item.device.type == "cpu")
                )
                data_size = _align(data_size + item.nbytes)
            return ["tensor", tensor_ids[id(item)]]
        if type(item) in (dict, list, tuple):
            if id(item) in ancestors:
                raise UnsupportedTensorPayloadError("Cyclic tensor payload")
            ancestors.add(id(item))
            try:
                if type(item) is dict:
                    if any(type(key) is not str for key in item):
                        raise UnsupportedTensorPayloadError("Only string dictionary keys are supported")
                    return ["dict", [[key, visit(val, depth + 1)] for key, val in item.items()]]
                return ["tuple" if type(item) is tuple else "list", [visit(v, depth + 1) for v in item]]
            finally:
                ancestors.remove(id(item))
        if item is None or type(item) in (bool, int, float, str, bytes):
            return ["value", item]
        raise UnsupportedTensorPayloadError(f"Unsupported payload type: {type(item).__name__}")

    tree = visit(value)
    if not tensors:
        raise UnsupportedTensorPayloadError("No tensors in payload")
    try:
        metadata = msgspec.msgpack.encode(FrameMetadata(1, tree, descriptors, data_size))
    except (TypeError, OverflowError) as exc:
        raise UnsupportedTensorPayloadError(str(exc)) from exc
    if len(metadata) > _MAX_METADATA_BYTES:
        raise UnsupportedTensorPayloadError("Tensor payload metadata exceeds 1 MiB")
    return TensorPayloadPlan(metadata, tensors, descriptors, data_size)


def _host_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.cpu().numpy().tobytes()


def decode_tensor_payload(frame: torch.Tensor) -> Any | None:
    """Decode a tensor frame, or return None for a legacy serialized payload.

    Returned device tensors share an owned copy of the packed storage, never
    the registered pool. CPU tensors remain on CPU. All device tensors are
    placed on the receive pool's device, not the sender's device ordinal.
    """
    if frame.dtype != torch.uint8 or frame.ndim != 1:
        raise ValueError("Invalid tensor payload frame")
    if frame.numel() < _PREFIX.size:
        return None
    magic, metadata_size = _PREFIX.unpack(_host_bytes(frame[: _PREFIX.size]))
    if magic != _MAGIC:
        return None
    if not 0 < metadata_size <= _MAX_METADATA_BYTES or _PREFIX.size + metadata_size > frame.numel():
        raise ValueError("Invalid tensor payload metadata size")
    metadata = msgspec.msgpack.decode(
        _host_bytes(frame[_PREFIX.size : _PREFIX.size + metadata_size]),
        type=FrameMetadata,
    )
    if metadata.version != 1:
        raise ValueError("Unsupported tensor payload version")
    data_offset = _align(_PREFIX.size + metadata_size)
    if metadata.data_size < 0 or data_offset + metadata.data_size != frame.numel():
        raise ValueError("Tensor payload size mismatch")
    if not 0 < len(metadata.tensors) <= _MAX_TENSORS:
        raise ValueError("Invalid tensor descriptor count")
    offset = 0
    for desc in metadata.tensors:
        dtype = _DTYPES.get(desc.dtype)
        if dtype is None or len(desc.shape) > _MAX_DEPTH or any(dim < 0 for dim in desc.shape):
            raise ValueError("Invalid tensor dtype or shape")
        size = math.prod(desc.shape) * dtype.itemsize
        if desc.nbytes != size or desc.offset != offset or offset + size > metadata.data_size:
            raise ValueError("Invalid tensor byte range")
        offset = _align(offset + size)
    if offset != metadata.data_size:
        raise ValueError("Invalid tensor data size")

    def restore(node: Any, values: list[Any], depth: int = 0) -> Any:
        if depth > _MAX_DEPTH or not isinstance(node, list) or len(node) != 2:
            raise ValueError("Invalid tensor payload tree")
        tag, content = node
        if tag == "tensor":
            if type(content) is not int or not 0 <= content < len(values):
                raise ValueError("Invalid tensor reference")
            return values[content]
        if tag == "value" and (content is None or type(content) in (bool, int, float, str, bytes)):
            return content
        if tag in ("dict", "list", "tuple") and isinstance(content, list):
            if tag == "dict":
                result = {}
                for pair in content:
                    if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                        raise ValueError("Invalid tensor payload dictionary")
                    result[pair[0]] = restore(pair[1], values, depth + 1)
                return result
            sequence = [restore(child, values, depth + 1) for child in content]
            return tuple(sequence) if tag == "tuple" else sequence
        raise ValueError("Invalid tensor payload tree node")

    # Validate the complete tree before allocating a destination copy.
    restore(metadata.tree, [None] * len(metadata.tensors))
    storage = frame[data_offset:].clone()
    try:
        values = []
        for desc in metadata.tensors:
            raw = storage[desc.offset : desc.offset + desc.nbytes]
            # Interpret CPU fields on CPU: their dtype need not be supported
            # by the receive accelerator (for example float64 metadata).
            if desc.cpu:
                raw = raw.cpu()
            values.append(raw.view(_DTYPES[desc.dtype]).reshape(desc.shape))
        return restore(metadata.tree, values)
    finally:
        # Pool ownership can be released only after the clone has completed.
        synchronize_stream(frame.device)
