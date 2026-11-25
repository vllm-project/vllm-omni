# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Msgspec-safe wire codec for PyTorch CUDA IPC tensor reduction handles.

PyTorch's ``torch.multiprocessing.reductions.reduce_tensor`` emits a 15-tuple
whose elements include Python class objects (``torch.Tensor``,
``torch.UntypedStorage``) and ``torch.dtype`` instances. Cross-process RPC
channels (such as vLLM's msgpack engine-core boundary) reject raw class/dtype
objects unless pickled. This codec maps only the allowlisted PyTorch tensor,
storage, and dtype objects to plain strings so handles round-trip safely over
msgpack without ``pickle`` or ``VLLM_ALLOW_INSECURE_SERIALIZATION``.
"""

from __future__ import annotations

import binascii
from collections.abc import Sequence
from typing import Any

import pybase64 as base64
import torch

IPC_WIRE_TAG = "__omni_cuda_ipc_v1__"
_REBUILD_CUDA_TENSOR_ARITY = 15

_TENSOR_CLS_TO_NAME: dict[type, str] = {
    torch.Tensor: "torch.Tensor",
    torch.nn.Parameter: "torch.nn.Parameter",
}
_NAME_TO_TENSOR_CLS: dict[str, type] = {name: cls for cls, name in _TENSOR_CLS_TO_NAME.items()}

_STORAGE_CLS_TO_NAME: dict[type, str] = {
    torch.UntypedStorage: "torch.UntypedStorage",
    torch.storage.TypedStorage: "torch.storage.TypedStorage",
}
_NAME_TO_STORAGE_CLS: dict[str, type] = {name: cls for cls, name in _STORAGE_CLS_TO_NAME.items()}

_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    getattr(torch, attr): attr for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)
}
_NAME_TO_DTYPE: dict[str, torch.dtype] = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def is_encoded_ipc_handle(value: Any) -> bool:
    """Return True when ``value`` is a wire-encoded CUDA IPC handle envelope."""
    return isinstance(value, dict) and len(value) == 1 and IPC_WIRE_TAG in value


def _check_optional_bytes(value: Any, field_name: str, *, allow_b64_str: bool = False) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if allow_b64_str and isinstance(value, str):
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise ValueError(f"CUDA IPC handle {field_name} has invalid base64 payload") from exc
    raise ValueError(f"CUDA IPC handle {field_name} must be bytes or None, got {type(value).__name__}")


def _encode_optional_bytes(value: Any, field_name: str, *, json_safe: bool) -> bytes | str | None:
    raw = _check_optional_bytes(value, field_name)
    if raw is None or not json_safe:
        return raw
    return base64.b64encode(raw).decode("ascii")


def _check_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"CUDA IPC handle {field_name} must be int, got {type(value).__name__}")
    return value


def _check_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"CUDA IPC handle {field_name} must be bool, got {type(value).__name__}")
    return value


def encode_ipc_handle_args(args: Sequence[Any], *, json_safe: bool = False) -> dict[str, list[Any]]:
    """Encode a 15-tuple of ``rebuild_cuda_tensor`` args into wire-safe primitives."""
    if is_encoded_ipc_handle(args):
        # Validate by round-tripping once rather than blindly trusting input.
        args = decode_ipc_handle_args(args)
    if not isinstance(args, (tuple, list)) or len(args) != _REBUILD_CUDA_TENSOR_ARITY:
        raise ValueError(
            f"Expected {_REBUILD_CUDA_TENSOR_ARITY} rebuild_cuda_tensor args, "
            f"got {type(args).__name__} of length {len(args) if isinstance(args, (tuple, list)) else 'n/a'}"
        )

    (
        tensor_cls,
        tensor_size,
        tensor_stride,
        tensor_offset,
        storage_cls,
        dtype,
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        requires_grad,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = args

    tensor_cls_name = _TENSOR_CLS_TO_NAME.get(tensor_cls)
    if tensor_cls_name is None:
        raise ValueError(f"Unsupported tensor class in CUDA IPC handle: {tensor_cls!r}")
    storage_cls_name = _STORAGE_CLS_TO_NAME.get(storage_cls)
    if storage_cls_name is None:
        raise ValueError(f"Unsupported storage class in CUDA IPC handle: {storage_cls!r}")
    dtype_name = _DTYPE_TO_NAME.get(dtype)
    if dtype_name is None:
        raise ValueError(f"Unsupported dtype in CUDA IPC handle: {dtype!r}")

    if not isinstance(tensor_size, (tuple, list)) or not isinstance(tensor_stride, (tuple, list)):
        raise ValueError("CUDA IPC handle tensor_size and tensor_stride must be sequences of int")

    encoded = [
        tensor_cls_name,
        [_check_int(d, "tensor_size") for d in tensor_size],
        [_check_int(d, "tensor_stride") for d in tensor_stride],
        _check_int(tensor_offset, "tensor_offset"),
        storage_cls_name,
        dtype_name,
        _check_int(storage_device, "storage_device"),
        _encode_optional_bytes(storage_handle, "storage_handle", json_safe=json_safe),
        _check_int(storage_size_bytes, "storage_size_bytes"),
        _check_int(storage_offset_bytes, "storage_offset_bytes"),
        _check_bool(requires_grad, "requires_grad"),
        _encode_optional_bytes(ref_counter_handle, "ref_counter_handle", json_safe=json_safe),
        _check_int(ref_counter_offset, "ref_counter_offset"),
        _encode_optional_bytes(event_handle, "event_handle", json_safe=json_safe),
        _check_bool(event_sync_required, "event_sync_required"),
    ]
    return {IPC_WIRE_TAG: encoded}


def decode_ipc_handle_args(encoded: Any) -> tuple[Any, ...]:
    """Decode a wire-encoded CUDA IPC handle envelope back to ``rebuild_cuda_tensor`` args."""
    if not is_encoded_ipc_handle(encoded):
        raise ValueError(f"Expected encoded CUDA IPC handle dict with key {IPC_WIRE_TAG!r}")
    payload = encoded[IPC_WIRE_TAG]
    if not isinstance(payload, (list, tuple)) or len(payload) != _REBUILD_CUDA_TENSOR_ARITY:
        raise ValueError(
            f"Encoded CUDA IPC handle payload must have {_REBUILD_CUDA_TENSOR_ARITY} elements, "
            f"got {type(payload).__name__}"
        )

    (
        tensor_cls_name,
        tensor_size,
        tensor_stride,
        tensor_offset,
        storage_cls_name,
        dtype_name,
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        requires_grad,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = payload

    if not isinstance(tensor_cls_name, str) or tensor_cls_name not in _NAME_TO_TENSOR_CLS:
        raise ValueError(f"Unsupported tensor class tag in CUDA IPC handle: {tensor_cls_name!r}")
    if not isinstance(storage_cls_name, str) or storage_cls_name not in _NAME_TO_STORAGE_CLS:
        raise ValueError(f"Unsupported storage class tag in CUDA IPC handle: {storage_cls_name!r}")
    if not isinstance(dtype_name, str) or dtype_name not in _NAME_TO_DTYPE:
        raise ValueError(f"Unsupported dtype tag in CUDA IPC handle: {dtype_name!r}")
    if not isinstance(tensor_size, (list, tuple)) or not isinstance(tensor_stride, (list, tuple)):
        raise ValueError("Encoded CUDA IPC handle tensor_size and tensor_stride must be sequences of int")

    return (
        _NAME_TO_TENSOR_CLS[tensor_cls_name],
        tuple(_check_int(d, "tensor_size") for d in tensor_size),
        tuple(_check_int(d, "tensor_stride") for d in tensor_stride),
        _check_int(tensor_offset, "tensor_offset"),
        _NAME_TO_STORAGE_CLS[storage_cls_name],
        _NAME_TO_DTYPE[dtype_name],
        _check_int(storage_device, "storage_device"),
        _check_optional_bytes(storage_handle, "storage_handle", allow_b64_str=True),
        _check_int(storage_size_bytes, "storage_size_bytes"),
        _check_int(storage_offset_bytes, "storage_offset_bytes"),
        _check_bool(requires_grad, "requires_grad"),
        _check_optional_bytes(ref_counter_handle, "ref_counter_handle", allow_b64_str=True),
        _check_int(ref_counter_offset, "ref_counter_offset"),
        _check_optional_bytes(event_handle, "event_handle", allow_b64_str=True),
        _check_bool(event_sync_required, "event_sync_required"),
    )


def _encode_handle_dict(handle_dict: dict[str, Any], *, json_safe: bool = False) -> dict[str, dict[str, list[Any]]]:
    if not isinstance(handle_dict, dict):
        raise ValueError(f"Expected dict mapping GPU UUID to IPC args, got {type(handle_dict).__name__}")
    return {uuid: encode_ipc_handle_args(args, json_safe=json_safe) for uuid, args in handle_dict.items()}


def _decode_handle_dict(handle_dict: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    if not isinstance(handle_dict, dict):
        raise ValueError(f"Expected dict mapping GPU UUID to IPC args, got {type(handle_dict).__name__}")
    decoded: dict[str, tuple[Any, ...]] = {}
    for uuid, entry in handle_dict.items():
        if is_encoded_ipc_handle(entry):
            decoded[uuid] = decode_ipc_handle_args(entry)
        elif isinstance(entry, (tuple, list)):
            decoded[uuid] = tuple(entry)
        else:
            raise ValueError(
                f"Unsupported IPC handle entry for GPU UUID {uuid!r}: expected encoded handle dict or tuple, "
                f"got {type(entry).__name__}"
            )
    return decoded


def encode_ipc_handles(
    handles: list[dict[str, Any]] | dict[str, Any],
    *,
    json_safe: bool = False,
) -> list[dict[str, dict[str, list[Any]]]] | dict[str, dict[str, list[Any]]]:
    """Encode per-parameter or single-buffer IPC handle mappings for msgspec or JSON transport."""
    if isinstance(handles, list):
        return [_encode_handle_dict(item, json_safe=json_safe) for item in handles]
    if isinstance(handles, dict):
        return _encode_handle_dict(handles, json_safe=json_safe)
    raise ValueError(f"Expected list or dict of IPC handles, got {type(handles).__name__}")


def decode_ipc_handles(
    handles: list[dict[str, Any]] | dict[str, Any],
) -> list[dict[str, tuple[Any, ...]]] | dict[str, tuple[Any, ...]]:
    """Decode wire-encoded IPC handle mappings while preserving legacy raw tuples."""
    if isinstance(handles, list):
        return [_decode_handle_dict(item) for item in handles]
    if isinstance(handles, dict):
        return _decode_handle_dict(handles)
    raise ValueError(f"Expected list or dict of IPC handles, got {type(handles).__name__}")
