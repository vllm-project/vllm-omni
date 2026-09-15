# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Versioned, rank-aware CUDA IPC weight sharing primitives.

This module deliberately stops at the process boundary.  It does not know how
to construct or load a model; callers provide the final CUDA tensors and a
manifest describing the tensors they intend to share.

The wire protocol is:

``hello -> manifest -> manifest_ack -> handles -> ready -> heartbeat``

The provider owns the CUDA allocations for the lifetime of every consumer.
Consumers must keep the returned :class:`MappedWeights` alive while using the
mapped tensors and must react to :class:`ProviderUnavailableError` before
issuing more CUDA work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import Any, TypeAlias

logger = logging.getLogger(__name__)

IPC_PROTOCOL_VERSION = 1

ChannelAddress: TypeAlias = str | tuple[str, int]
TensorHandle: TypeAlias = tuple[Any, ...]
TensorRebuilder: TypeAlias = Callable[..., Any]
TensorHandleMapper: TypeAlias = Callable[[TensorHandle, int], TensorHandle]
ProviderExitCallback: TypeAlias = Callable[["ProviderUnavailableError"], None]

_HELLO = "hello"
_MANIFEST = "manifest"
_MANIFEST_ACK = "manifest_ack"
_HANDLES = "handles"
_READY = "ready"
_HEARTBEAT = "heartbeat"
_HEARTBEAT_ACK = "heartbeat_ack"
_CLOSE = "close"
_ERROR = "error"


class WeightSharingError(RuntimeError):
    """Base exception for the weight-sharing protocol."""


class ManifestValidationError(WeightSharingError):
    """Raised when a manifest is malformed or internally inconsistent."""


class ManifestMismatchError(WeightSharingError):
    """Raised when provider and consumer manifests do not describe the same data."""

    def __init__(self, mismatches: Sequence[str] | str) -> None:
        if isinstance(mismatches, str):
            mismatch_list = [mismatches]
        else:
            mismatch_list = list(mismatches)
        self.mismatches = tuple(mismatch_list)
        super().__init__("weight manifest mismatch: " + "; ".join(self.mismatches))


class ProtocolError(WeightSharingError):
    """Raised when a peer violates the IPC handshake or heartbeat protocol."""


class ProtocolTimeoutError(ProtocolError):
    """Raised when a peer does not answer within the configured timeout."""


class ProviderUnavailableError(WeightSharingError):
    """Raised when a consumer can no longer trust its provider process."""


def _as_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{field_name} must be an integer")
    return value


def _as_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{field_name} must be a non-empty string")
    return value


def _as_int_tuple(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ManifestValidationError(f"{field_name} must be a list of integers")
    result = tuple(_as_int(item, f"{field_name}[]") for item in value)
    if any(item < 0 for item in result):
        raise ManifestValidationError(f"{field_name} cannot contain negative values")
    return result


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field_name)


def _storage_identity(tensor: object) -> int | None:
    """Return a process-local storage identity when the tensor exposes one."""
    storage_factory = getattr(tensor, "untyped_storage", None)
    if not callable(storage_factory):
        return None
    storage = storage_factory()
    cdata = getattr(storage, "_cdata", None)
    if isinstance(cdata, int) and cdata:
        return cdata
    return id(storage)


def _storage_nbytes(tensor: object) -> int | None:
    storage_factory = getattr(tensor, "untyped_storage", None)
    if not callable(storage_factory):
        return None
    storage = storage_factory()
    nbytes = getattr(storage, "nbytes", None)
    if not callable(nbytes):
        return None
    value = nbytes()
    return int(value) if isinstance(value, int) else None


@dataclass(frozen=True, slots=True)
class TensorManifestEntry:
    """Metadata required to validate one shared tensor before mapping it.

    ``device_index`` is provider-local diagnostic metadata. CUDA_VISIBLE_DEVICES
    can assign a different ordinal to the same physical GPU in each process, so
    cross-process compatibility is established with ``device_uuid`` instead.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    device_index: int
    device_uuid: str
    parallel_rank: int
    tensor_kind: str = "parameter"
    stride: tuple[int, ...] = ()
    storage_offset: int = 0
    storage_nbytes: int | None = None
    storage_key: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ManifestValidationError("tensor name must be non-empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ManifestValidationError(f"tensor {self.name!r} has a negative shape")
        if self.device_index < 0:
            raise ManifestValidationError(f"tensor {self.name!r} has an invalid device index")
        if not self.device_uuid:
            raise ManifestValidationError(f"tensor {self.name!r} has no physical GPU identity")
        if self.parallel_rank < 0:
            raise ManifestValidationError(f"tensor {self.name!r} has an invalid parallel rank")
        if self.storage_offset < 0:
            raise ManifestValidationError(f"tensor {self.name!r} has an invalid storage offset")
        if self.storage_nbytes is not None and self.storage_nbytes < 0:
            raise ManifestValidationError(f"tensor {self.name!r} has invalid storage size")
        if self.stride and len(self.stride) != len(self.shape):
            raise ManifestValidationError(f"tensor {self.name!r} has inconsistent shape and stride")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "parallel_rank": self.parallel_rank,
            "tensor_kind": self.tensor_kind,
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
        }
        if self.storage_nbytes is not None:
            payload["storage_nbytes"] = self.storage_nbytes
        if self.storage_key is not None:
            payload["storage_key"] = self.storage_key
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> TensorManifestEntry:
        if not isinstance(payload, Mapping):
            raise ManifestValidationError("tensor entry must be an object")
        return cls(
            name=_as_str(payload.get("name"), "tensor.name"),
            shape=_as_int_tuple(payload.get("shape"), "tensor.shape"),
            dtype=_as_str(payload.get("dtype"), "tensor.dtype"),
            device_index=_as_int(payload.get("device_index"), "tensor.device_index"),
            device_uuid=_as_str(payload.get("device_uuid"), "tensor.device_uuid"),
            parallel_rank=_as_int(payload.get("parallel_rank"), "tensor.parallel_rank"),
            tensor_kind=_as_str(payload.get("tensor_kind", "parameter"), "tensor.tensor_kind"),
            stride=_as_int_tuple(payload.get("stride", []), "tensor.stride"),
            storage_offset=_as_int(payload.get("storage_offset", 0), "tensor.storage_offset"),
            storage_nbytes=_optional_int(payload.get("storage_nbytes"), "tensor.storage_nbytes"),
            storage_key=(
                None
                if payload.get("storage_key") is None
                else _as_str(payload.get("storage_key"), "tensor.storage_key")
            ),
        )


@dataclass(frozen=True, slots=True)
class WeightManifest:
    """Versioned identity and tensor metadata for one provider rank."""

    model_id: str
    checkpoint_revision: str
    component: str
    device_index: int
    device_uuid: str
    parallel_rank: int
    parallel_size: int
    tensors: tuple[TensorManifestEntry, ...]
    protocol_version: int = IPC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version <= 0:
            raise ManifestValidationError("protocol_version must be positive")
        for field_name, value in (
            ("model_id", self.model_id),
            ("checkpoint_revision", self.checkpoint_revision),
            ("component", self.component),
            ("device_uuid", self.device_uuid),
        ):
            if not value:
                raise ManifestValidationError(f"{field_name} must be non-empty")
        if self.device_index < 0:
            raise ManifestValidationError("device_index must be non-negative")
        if self.parallel_rank < 0:
            raise ManifestValidationError("parallel_rank must be non-negative")
        if self.parallel_size <= 0 or self.parallel_rank >= self.parallel_size:
            raise ManifestValidationError("parallel rank must be smaller than parallel size")
        names = [entry.name for entry in self.tensors]
        if len(names) != len(set(names)):
            raise ManifestValidationError("tensor names must be unique")
        if not self.tensors:
            raise ManifestValidationError("manifest must contain at least one tensor")
        for entry in self.tensors:
            if entry.device_index != self.device_index:
                raise ManifestValidationError(f"tensor {entry.name!r} has a different device index")
            if entry.device_uuid != self.device_uuid:
                raise ManifestValidationError(f"tensor {entry.name!r} has a different device UUID")
            if entry.parallel_rank != self.parallel_rank:
                raise ManifestValidationError(f"tensor {entry.name!r} has a different parallel rank")
        object.__setattr__(self, "tensors", tuple(sorted(self.tensors, key=lambda entry: entry.name)))

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.tensors)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "checkpoint_revision": self.checkpoint_revision,
            "component": self.component,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "parallel_rank": self.parallel_rank,
            "parallel_size": self.parallel_size,
            "tensors": [entry.to_payload() for entry in self.tensors],
        }

    @classmethod
    def from_payload(cls, payload: object) -> WeightManifest:
        if not isinstance(payload, Mapping):
            raise ManifestValidationError("manifest must be an object")
        tensors_payload = payload.get("tensors")
        if not isinstance(tensors_payload, (list, tuple)):
            raise ManifestValidationError("manifest.tensors must be a list")
        return cls(
            protocol_version=_as_int(payload.get("protocol_version"), "protocol_version"),
            model_id=_as_str(payload.get("model_id"), "model_id"),
            checkpoint_revision=_as_str(payload.get("checkpoint_revision"), "checkpoint_revision"),
            component=_as_str(payload.get("component"), "component"),
            device_index=_as_int(payload.get("device_index"), "device_index"),
            device_uuid=_as_str(payload.get("device_uuid"), "device_uuid"),
            parallel_rank=_as_int(payload.get("parallel_rank"), "parallel_rank"),
            parallel_size=_as_int(payload.get("parallel_size"), "parallel_size"),
            tensors=tuple(TensorManifestEntry.from_payload(entry) for entry in tensors_payload),
        )


def build_weight_manifest(
    tensors: Mapping[str, object],
    *,
    model_id: str,
    checkpoint_revision: str,
    component: str,
    device_index: int,
    device_uuid: str,
    parallel_rank: int,
    parallel_size: int,
    tensor_kinds: Mapping[str, str] | None = None,
) -> WeightManifest:
    """Build a manifest from final tensors without importing PyTorch."""
    if not tensors:
        raise ManifestValidationError("cannot build a manifest from no tensors")

    tensor_kinds = tensor_kinds or {}
    storage_keys: dict[int, str] = {}
    entries: list[TensorManifestEntry] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        shape_value = getattr(tensor, "shape", None)
        if shape_value is None:
            raise ManifestValidationError(f"tensor {name!r} has no shape")
        stride_factory = getattr(tensor, "stride", None)
        stride_value = stride_factory() if callable(stride_factory) else ()
        offset_factory = getattr(tensor, "storage_offset", None)
        storage_offset = int(offset_factory()) if callable(offset_factory) else 0
        identity = _storage_identity(tensor)
        storage_key = None
        if identity is not None:
            storage_key = storage_keys.setdefault(identity, f"storage-{len(storage_keys)}")
        entries.append(
            TensorManifestEntry(
                name=str(name),
                shape=tuple(int(dimension) for dimension in shape_value),
                dtype=str(getattr(tensor, "dtype", "unknown")),
                device_index=device_index,
                device_uuid=device_uuid,
                parallel_rank=parallel_rank,
                tensor_kind=tensor_kinds.get(name, "parameter"),
                stride=tuple(int(value) for value in stride_value),
                storage_offset=storage_offset,
                storage_nbytes=_storage_nbytes(tensor),
                storage_key=storage_key,
            )
        )
    return WeightManifest(
        model_id=model_id,
        checkpoint_revision=checkpoint_revision,
        component=component,
        device_index=device_index,
        device_uuid=device_uuid,
        parallel_rank=parallel_rank,
        parallel_size=parallel_size,
        tensors=tuple(entries),
    )


def get_cuda_device_uuid(device_index: int) -> str:
    """Return a stable physical GPU identity for manifest validation."""
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    raw_uuid = getattr(properties, "uuid", None)
    if isinstance(raw_uuid, bytes):
        raw_uuid = raw_uuid.decode("ascii")
    elif raw_uuid is not None and not isinstance(raw_uuid, str):
        # Recent PyTorch versions expose UUIDs as torch._C._CUuuid objects.
        raw_uuid = str(raw_uuid)
    if not isinstance(raw_uuid, str) or not raw_uuid:
        raise WeightSharingError(f"CUDA device {device_index} did not expose a stable UUID")
    return raw_uuid


def compare_manifests(expected: WeightManifest, actual: WeightManifest) -> tuple[str, ...]:
    """Return all cross-process incompatibilities instead of failing first.

    Device ordinals are intentionally excluded. They are local to a process and
    may differ when CUDA_VISIBLE_DEVICES remaps the same physical GPU. The UUID
    and TP rank are the device-pairing contract.
    """
    mismatches: list[str] = []
    for field_name in (
        "protocol_version",
        "model_id",
        "checkpoint_revision",
        "component",
        "device_uuid",
        "parallel_rank",
        "parallel_size",
    ):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)
        if expected_value != actual_value:
            mismatches.append(f"{field_name}: expected {expected_value!r}, got {actual_value!r}")

    expected_entries = {entry.name: entry for entry in expected.tensors}
    actual_entries = {entry.name: entry for entry in actual.tensors}
    missing = sorted(set(expected_entries) - set(actual_entries))
    extra = sorted(set(actual_entries) - set(expected_entries))
    if missing:
        mismatches.append(f"missing tensors: {missing}")
    if extra:
        mismatches.append(f"unexpected tensors: {extra}")

    for name in sorted(set(expected_entries) & set(actual_entries)):
        expected_entry = expected_entries[name]
        actual_entry = actual_entries[name]
        for field_name in (
            "shape",
            "dtype",
            "device_uuid",
            "parallel_rank",
            "tensor_kind",
            "stride",
            "storage_offset",
        ):
            expected_value = getattr(expected_entry, field_name)
            actual_value = getattr(actual_entry, field_name)
            if expected_value != actual_value:
                mismatches.append(f"tensor {name!r} {field_name}: expected {expected_value!r}, got {actual_value!r}")
        if (
            expected_entry.storage_nbytes is not None
            and actual_entry.storage_nbytes is not None
            and expected_entry.storage_nbytes != actual_entry.storage_nbytes
        ):
            mismatches.append(
                f"tensor {name!r} storage_nbytes: expected {expected_entry.storage_nbytes!r}, "
                f"got {actual_entry.storage_nbytes!r}"
            )
        if (
            expected_entry.storage_key is not None
            and actual_entry.storage_key is not None
            and expected_entry.storage_key != actual_entry.storage_key
        ):
            mismatches.append(
                f"tensor {name!r} storage_key: expected {expected_entry.storage_key!r}, "
                f"got {actual_entry.storage_key!r}"
            )
    return tuple(mismatches)


def _validate_provider_tensor_devices(
    tensors: Mapping[str, object],
    manifest: WeightManifest,
) -> None:
    """Reject tensors whose actual CUDA device differs from the manifest."""
    device_uuids: dict[int, str] = {}
    mismatches: list[str] = []
    for name in manifest.tensor_names:
        tensor = tensors[name]
        if not bool(getattr(tensor, "is_cuda", False)):
            mismatches.append(f"tensor {name!r} is not a CUDA tensor")
            continue
        device = getattr(tensor, "device", None)
        device_index = getattr(device, "index", None)
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            mismatches.append(f"tensor {name!r} has no concrete CUDA device index")
            continue
        if device_index != manifest.device_index:
            mismatches.append(f"tensor {name!r} device_index: expected {manifest.device_index!r}, got {device_index!r}")
            continue
        if device_index not in device_uuids:
            device_uuids[device_index] = get_cuda_device_uuid(device_index)
        device_uuid = device_uuids[device_index]
        if device_uuid != manifest.device_uuid:
            mismatches.append(f"tensor {name!r} device_uuid: expected {manifest.device_uuid!r}, got {device_uuid!r}")
    if mismatches:
        raise ManifestMismatchError(mismatches)


def export_cuda_handles(
    tensors: Mapping[str, object],
    manifest: WeightManifest,
) -> dict[str, TensorHandle]:
    """Export PyTorch CUDA IPC reducer arguments for the manifest tensors."""
    if set(tensors) != set(manifest.tensor_names):
        raise ManifestMismatchError(
            [
                f"tensor names: expected {list(manifest.tensor_names)!r}, got {sorted(tensors)!r}",
            ]
        )

    _validate_provider_tensor_devices(tensors, manifest)

    tensor_kinds = {entry.name: entry.tensor_kind for entry in manifest.tensors}
    current_manifest = build_weight_manifest(
        tensors,
        model_id=manifest.model_id,
        checkpoint_revision=manifest.checkpoint_revision,
        component=manifest.component,
        device_index=manifest.device_index,
        device_uuid=manifest.device_uuid,
        parallel_rank=manifest.parallel_rank,
        parallel_size=manifest.parallel_size,
        tensor_kinds=tensor_kinds,
    )
    mismatches = compare_manifests(manifest, current_manifest)
    if mismatches:
        raise ManifestMismatchError(mismatches)

    from torch.multiprocessing.reductions import reduce_tensor

    handles: dict[str, TensorHandle] = {}
    for name in manifest.tensor_names:
        tensor = tensors[name]
        _, reducer_args = reduce_tensor(tensor)
        handles[name] = tuple(reducer_args)
    return handles


def remap_cuda_handle_device(handle: TensorHandle, device_index: int) -> TensorHandle:
    """Rewrite a PyTorch CUDA reducer handle for a consumer-local ordinal."""
    if device_index < 0:
        raise ValueError("CUDA device index must be non-negative")

    import torch

    device_positions = [
        index for index, value in enumerate(handle) if isinstance(value, torch.device) and value.type == "cuda"
    ]
    if len(device_positions) != 1:
        raise ProtocolError(
            f"CUDA reducer handle must contain exactly one torch.device('cuda', N) value; got {len(device_positions)}"
        )
    remapped = list(handle)
    remapped[device_positions[0]] = torch.device(f"cuda:{device_index}")
    return tuple(remapped)


def rebuild_cuda_tensor(handle: TensorHandle) -> object:
    """Rebuild one CUDA tensor from reducer arguments on the consumer side."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor as rebuild

    return rebuild(*handle)


def _validate_local_manifest_device(manifest: WeightManifest) -> None:
    actual_uuid = get_cuda_device_uuid(manifest.device_index)
    if actual_uuid != manifest.device_uuid:
        raise ManifestMismatchError(
            [
                "consumer device_uuid: "
                f"expected {manifest.device_uuid!r}, got {actual_uuid!r} "
                f"for local cuda:{manifest.device_index}",
            ]
        )


def _validate_mapped_tensors(
    tensors: Mapping[str, object],
    manifest: WeightManifest,
) -> None:
    """Verify reconstructed tensors still match the validated manifest."""
    device_uuids: dict[int, str] = {}
    storage_identities: dict[str, int] = {}
    storage_keys: dict[int, str] = {}
    mismatches: list[str] = []

    for entry in manifest.tensors:
        tensor = tensors[entry.name]
        if not bool(getattr(tensor, "is_cuda", False)):
            mismatches.append(f"tensor {entry.name!r} was not rebuilt on CUDA")
            continue
        device = getattr(tensor, "device", None)
        device_index = getattr(device, "index", None)
        if device_index != manifest.device_index:
            mismatches.append(
                f"tensor {entry.name!r} local device_index: expected {manifest.device_index!r}, got {device_index!r}"
            )
            continue
        if device_index not in device_uuids:
            device_uuids[device_index] = get_cuda_device_uuid(device_index)
        device_uuid = device_uuids[device_index]
        if device_uuid != manifest.device_uuid:
            mismatches.append(
                f"tensor {entry.name!r} local device_uuid: expected {manifest.device_uuid!r}, got {device_uuid!r}"
            )
        shape = tuple(int(dimension) for dimension in getattr(tensor, "shape", ()))
        if shape != entry.shape:
            mismatches.append(f"tensor {entry.name!r} shape: expected {entry.shape!r}, got {shape!r}")
        if str(getattr(tensor, "dtype", "unknown")) != entry.dtype:
            mismatches.append(
                f"tensor {entry.name!r} dtype: expected {entry.dtype!r}, got {getattr(tensor, 'dtype', 'unknown')!r}"
            )
        stride_factory = getattr(tensor, "stride", None)
        stride = tuple(int(value) for value in stride_factory()) if callable(stride_factory) else ()
        if stride != entry.stride:
            mismatches.append(f"tensor {entry.name!r} stride: expected {entry.stride!r}, got {stride!r}")
        offset_factory = getattr(tensor, "storage_offset", None)
        storage_offset = int(offset_factory()) if callable(offset_factory) else 0
        if storage_offset != entry.storage_offset:
            mismatches.append(
                f"tensor {entry.name!r} storage_offset: expected {entry.storage_offset!r}, got {storage_offset!r}"
            )
        storage_nbytes = _storage_nbytes(tensor)
        if entry.storage_nbytes is not None and storage_nbytes != entry.storage_nbytes:
            mismatches.append(
                f"tensor {entry.name!r} storage_nbytes: expected {entry.storage_nbytes!r}, got {storage_nbytes!r}"
            )
        if entry.storage_key is not None:
            identity = _storage_identity(tensor)
            if identity is None:
                mismatches.append(f"tensor {entry.name!r} has no storage identity")
                continue
            previous_identity = storage_identities.setdefault(entry.storage_key, identity)
            if previous_identity != identity:
                mismatches.append(f"tensor {entry.name!r} does not preserve storage alias {entry.storage_key!r}")
            previous_key = storage_keys.setdefault(identity, entry.storage_key)
            if previous_key != entry.storage_key:
                mismatches.append(
                    f"tensor {entry.name!r} unexpectedly aliases storage {previous_key!r} and {entry.storage_key!r}"
                )
    if mismatches:
        raise ManifestMismatchError(mismatches)


@dataclass(frozen=True, slots=True)
class WeightSharingEndpoint:
    """Authenticated loopback endpoint used by one provider and its consumers."""

    address: ChannelAddress
    family: str
    authkey: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.family not in {"AF_UNIX", "AF_INET"}:
            raise ValueError(f"unsupported local IPC family: {self.family!r}")
        if not isinstance(self.authkey, bytes) or not self.authkey:
            raise ValueError("IPC authkey must be non-empty bytes")
        if self.family == "AF_UNIX" and not isinstance(self.address, str):
            raise ValueError("AF_UNIX endpoints require a filesystem path")
        if self.family == "AF_INET":
            if not isinstance(self.address, tuple) or len(self.address) != 2:
                raise ValueError("AF_INET endpoints require a (host, port) tuple")
            host, port = self.address
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("weight sharing endpoints must bind to loopback")
            if not isinstance(port, int) or not 0 <= port <= 65535:
                raise ValueError("AF_INET endpoint port must be between 0 and 65535")

    @classmethod
    def unix(cls, path: str | os.PathLike[str], authkey: bytes) -> WeightSharingEndpoint:
        return cls(str(path), "AF_UNIX", authkey)

    @classmethod
    def tcp(cls, port: int, authkey: bytes, host: str = "127.0.0.1") -> WeightSharingEndpoint:
        return cls((host, port), "AF_INET", authkey)


def _recv_message(connection: Connection, timeout: float | None) -> object:
    if timeout is not None and not connection.poll(timeout):
        raise ProtocolTimeoutError("timed out waiting for IPC peer")
    try:
        return connection.recv()
    except EOFError as exc:
        raise ProviderUnavailableError("IPC peer closed the connection") from exc


def _send_error(connection: Connection, code: str, message: str, details: Sequence[str] = ()) -> None:
    try:
        connection.send({"type": _ERROR, "code": code, "message": message, "details": list(details)})
    except (EOFError, OSError):
        pass


def _error_from_message(message: Mapping[str, object]) -> WeightSharingError:
    code = message.get("code")
    text = message.get("message", "peer rejected the IPC request")
    details = message.get("details")
    if code == "manifest_mismatch":
        if isinstance(details, list) and all(isinstance(item, str) for item in details):
            return ManifestMismatchError(details)
        return ManifestMismatchError(str(text))
    if code == "provider_unavailable":
        return ProviderUnavailableError(str(text))
    return ProtocolError(str(text))


class WeightSharingProvider:
    """Serve one final CUDA allocation set to compatible consumer processes."""

    def __init__(
        self,
        endpoint: WeightSharingEndpoint,
        manifest: WeightManifest,
        tensors: Mapping[str, object],
        *,
        handle_exporter: Callable[
            [Mapping[str, object], WeightManifest], Mapping[str, TensorHandle]
        ] = export_cuda_handles,
        handshake_timeout: float = 10.0,
        heartbeat_interval: float = 1.0,
    ) -> None:
        if handshake_timeout <= 0:
            raise ValueError("handshake_timeout must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if set(tensors) != set(manifest.tensor_names):
            raise ManifestMismatchError([f"tensors do not match manifest tensor names: {sorted(tensors)!r}"])
        self.manifest = manifest
        self.tensors = dict(tensors)
        self._handle_exporter = handle_exporter
        self.handshake_timeout = handshake_timeout
        self.heartbeat_interval = heartbeat_interval
        self._stop_event = Event()
        self._connections: set[Connection] = set()
        self._threads: set[Thread] = set()
        self._connections_lock = Lock()
        self._handle_export_lock = Lock()
        self._listener = Listener(endpoint.address, family=endpoint.family, authkey=endpoint.authkey)
        self._endpoint = WeightSharingEndpoint(
            address=self._listener.address,
            family=endpoint.family,
            authkey=endpoint.authkey,
        )
        self._serve_thread: Thread | None = None
        self._unix_path = (
            Path(self._endpoint.address)
            if self._endpoint.family == "AF_UNIX" and isinstance(self._endpoint.address, str)
            else None
        )

    @property
    def endpoint(self) -> WeightSharingEndpoint:
        """Return the actual bound endpoint, including an allocated TCP port."""
        return self._endpoint

    def serve_once(self) -> None:
        """Accept and serve one consumer synchronously."""
        if self._stop_event.is_set():
            raise ProviderUnavailableError("provider is stopped")
        connection = self._listener.accept()
        with self._connections_lock:
            if self._stop_event.is_set():
                connection.close()
                return
            self._connections.add(connection)
        self._serve_connection(connection)

    def start(self) -> Thread:
        """Start accepting consumers in a daemon thread."""
        if self._serve_thread is not None and self._serve_thread.is_alive():
            return self._serve_thread
        self._serve_thread = Thread(target=self.serve_forever, name="weight-sharing-provider", daemon=True)
        self._serve_thread.start()
        return self._serve_thread

    def serve_forever(self) -> None:
        """Accept compatible consumers until :meth:`stop` is called."""
        while not self._stop_event.is_set():
            try:
                connection = self._listener.accept()
            except (OSError, EOFError):
                if not self._stop_event.is_set():
                    logger.exception("weight-sharing listener stopped unexpectedly")
                return
            with self._connections_lock:
                if self._stop_event.is_set():
                    connection.close()
                    return
                thread = Thread(
                    target=self._serve_connection,
                    args=(connection,),
                    name="weight-sharing-consumer",
                    daemon=True,
                )
                self._connections.add(connection)
                self._threads.add(thread)
                thread.start()

    def stop(self) -> None:
        """Stop accepting consumers and close all active IPC connections."""
        with self._connections_lock:
            if self._stop_event.is_set():
                return
            self._stop_event.set()
            connections = list(self._connections)
            threads = [thread for thread in self._threads if thread is not current_thread()]
        if self._serve_thread is not None and self._serve_thread.is_alive():
            self._wake_listener()
        try:
            self._listener.close()
        except OSError:
            pass
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass
        join_deadline = monotonic() + max(1.0, self.heartbeat_interval * 2)
        if self._serve_thread is not None and self._serve_thread is not current_thread():
            self._serve_thread.join(timeout=max(0.0, join_deadline - monotonic()))
        for thread in threads:
            thread.join(timeout=max(0.0, join_deadline - monotonic()))
        if self._unix_path is not None:
            try:
                if self._unix_path.exists() and stat.S_ISSOCK(self._unix_path.stat().st_mode):
                    self._unix_path.unlink()
            except OSError:
                pass

    def _wake_listener(self) -> None:
        """Unblock the listener thread without starting an authenticated peer."""
        address = self._endpoint.address
        family = socket.AF_INET if self._endpoint.family == "AF_INET" else socket.AF_UNIX
        try:
            with socket.socket(family, socket.SOCK_STREAM) as wake_socket:
                wake_socket.settimeout(1.0)
                wake_socket.connect(address)
        except OSError:
            pass

    def _serve_connection(self, connection: Connection) -> None:
        try:
            hello = _recv_message(connection, self.handshake_timeout)
            if not isinstance(hello, Mapping) or hello.get("type") != _HELLO:
                raise ProtocolError("first IPC message must be hello")
            expected_manifest = WeightManifest.from_payload(hello.get("manifest"))
            mismatches = compare_manifests(expected_manifest, self.manifest)
            if mismatches:
                _send_error(connection, "manifest_mismatch", "provider rejected consumer manifest", mismatches)
                return

            connection.send(
                {
                    "type": _MANIFEST,
                    "manifest": self.manifest.to_payload(),
                    "fingerprint": self.manifest.fingerprint,
                }
            )
            ack = _recv_message(connection, self.handshake_timeout)
            if not isinstance(ack, Mapping) or ack.get("type") != _MANIFEST_ACK:
                raise ProtocolError("expected manifest_ack from consumer")
            if ack.get("fingerprint") != self.manifest.fingerprint:
                raise ManifestMismatchError("manifest fingerprint acknowledgement does not match")

            connection.send({"type": _HANDLES, "handles": self._export_handles()})
            ready_message = _recv_message(connection, self.handshake_timeout)
            if not isinstance(ready_message, Mapping) or ready_message.get("type") != _READY:
                raise ProtocolError("expected ready from consumer")
            self._heartbeat_loop(connection)
        except ManifestMismatchError as exc:
            _send_error(connection, "manifest_mismatch", str(exc), exc.mismatches)
        except ProviderUnavailableError:
            return
        except ProtocolTimeoutError as exc:
            _send_error(connection, "protocol_timeout", str(exc))
        except (ManifestValidationError, ProtocolError) as exc:
            _send_error(connection, "protocol_error", str(exc))
        except (EOFError, OSError):
            return
        finally:
            try:
                connection.close()
            except OSError:
                pass
            with self._connections_lock:
                self._connections.discard(connection)
                self._threads.discard(current_thread())

    def _export_handles(self) -> dict[str, TensorHandle]:
        # PyTorch creates one CUDA IPC ref-counter slot per reduction. Serialize
        # reductions so each consumer receives a fresh, complete handle set.
        with self._handle_export_lock:
            handles = dict(self._handle_exporter(self.tensors, self.manifest))
        if set(handles) != set(self.manifest.tensor_names):
            raise ManifestMismatchError([f"handles do not match manifest tensor names: {sorted(handles)!r}"])
        return handles

    def _heartbeat_loop(self, connection: Connection) -> None:
        sequence = 0
        next_heartbeat = monotonic()
        while not self._stop_event.is_set():
            timeout = max(0.0, next_heartbeat - monotonic())
            if connection.poll(timeout):
                message = connection.recv()
                if not isinstance(message, Mapping):
                    raise ProtocolError("heartbeat message must be an object")
                message_type = message.get("type")
                if message_type == _CLOSE:
                    return
                if message_type != _HEARTBEAT_ACK:
                    raise ProtocolError(f"unexpected message during heartbeat: {message_type!r}")
            if monotonic() >= next_heartbeat:
                sequence += 1
                connection.send({"type": _HEARTBEAT, "sequence": sequence})
                next_heartbeat = monotonic() + self.heartbeat_interval


class MappedWeights(Mapping[str, object]):
    """Consumer-side tensors and the provider liveness monitor."""

    def __init__(
        self,
        tensors: Mapping[str, object],
        manifest: WeightManifest,
        connection: Connection,
        *,
        heartbeat_interval: float,
        on_provider_exit: ProviderExitCallback | None,
    ) -> None:
        self._tensors = dict(tensors)
        self.manifest = manifest
        self._connection = connection
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = max(heartbeat_interval * 3, heartbeat_interval + 0.5)
        self._on_provider_exit = on_provider_exit
        self._last_heartbeat = monotonic()
        self._closed = Event()
        self._dead = Event()
        self._state_lock = Lock()
        self._receive_lock = Lock()
        self._send_lock = Lock()
        self._monitor_thread = Thread(target=self._monitor_loop, name="weight-sharing-monitor", daemon=True)
        self._monitor_thread.start()

    def __getitem__(self, name: str) -> object:
        self._raise_if_dead()
        return self._tensors[name]

    def __iter__(self) -> Iterator[str]:
        self._raise_if_dead()
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    @property
    def provider_alive(self) -> bool:
        return not self._dead.is_set() and not self._closed.is_set()

    def check_liveness(self, timeout: float = 0.0) -> None:
        """Process pending heartbeats and raise if the provider is unavailable."""
        if timeout < 0:
            raise ValueError("liveness timeout cannot be negative")
        deadline = monotonic() + timeout
        acquired = (
            self._receive_lock.acquire(timeout=timeout) if timeout else self._receive_lock.acquire(blocking=False)
        )
        if not acquired:
            self._raise_if_dead()
            self._check_heartbeat_age()
            return
        try:
            self._raise_if_dead()
            if self._closed.is_set():
                raise ProviderUnavailableError("mapped weights are closed")
            try:
                poll_timeout = max(0.0, deadline - monotonic()) if timeout else 0.0
                if timeout and not self._connection.poll(poll_timeout):
                    self._check_heartbeat_age()
                    return
                while self._connection.poll(0):
                    message = self._connection.recv()
                    if not isinstance(message, Mapping):
                        raise ProtocolError("heartbeat message must be an object")
                    message_type = message.get("type")
                    if message_type == _HEARTBEAT:
                        with self._state_lock:
                            self._last_heartbeat = monotonic()
                        self._send({"type": _HEARTBEAT_ACK, "sequence": message.get("sequence")})
                    elif message_type == _ERROR:
                        raise _error_from_message(message)
                    else:
                        raise ProtocolError(f"unexpected provider message: {message_type!r}")
                self._check_heartbeat_age()
            except (EOFError, OSError) as exc:
                raise ProviderUnavailableError("provider closed the IPC connection") from exc
        finally:
            self._receive_lock.release()

    def close(self) -> None:
        """Release the connection; the provider must remain alive until this call."""
        if self._closed.is_set():
            return
        self._closed.set()
        with self._receive_lock:
            try:
                self._send({"type": _CLOSE})
            except (EOFError, OSError, TypeError):
                pass
            try:
                self._connection.close()
            except OSError:
                pass
        if self._monitor_thread is not current_thread():
            self._monitor_thread.join(timeout=max(1.0, self._heartbeat_interval * 2))

    def _monitor_loop(self) -> None:
        while not self._closed.is_set() and not self._dead.is_set():
            try:
                self.check_liveness(timeout=self._heartbeat_interval)
            except ProviderUnavailableError as exc:
                if self._closed.is_set():
                    return
                self._dead.set()
                if self._on_provider_exit is not None:
                    try:
                        self._on_provider_exit(exc)
                    except Exception:
                        logger.exception("provider-exit callback failed")
                return
            except WeightSharingError as exc:
                if self._closed.is_set():
                    return
                self._dead.set()
                if self._on_provider_exit is not None:
                    error = ProviderUnavailableError(str(exc))
                    try:
                        self._on_provider_exit(error)
                    except Exception:
                        logger.exception("provider-exit callback failed")
                return

    def _check_heartbeat_age(self) -> None:
        with self._state_lock:
            age = monotonic() - self._last_heartbeat
        if age > self._heartbeat_timeout:
            raise ProviderUnavailableError(
                f"provider heartbeat expired after {age:.2f}s (limit {self._heartbeat_timeout:.2f}s)"
            )

    def _raise_if_dead(self) -> None:
        if self._dead.is_set():
            raise ProviderUnavailableError("provider is no longer available")

    def _send(self, message: Mapping[str, object]) -> None:
        with self._send_lock:
            self._connection.send(dict(message))


class WeightSharingConsumer:
    """Map provider handles after a complete manifest validation handshake."""

    @classmethod
    def connect(
        cls,
        endpoint: WeightSharingEndpoint,
        expected_manifest: WeightManifest,
        *,
        rebuild: TensorRebuilder = rebuild_cuda_tensor,
        handle_mapper: TensorHandleMapper = remap_cuda_handle_device,
        validate_local_device: bool = True,
        validate_mapped_tensors: bool = True,
        handshake_timeout: float = 10.0,
        heartbeat_interval: float = 1.0,
        on_provider_exit: ProviderExitCallback | None = None,
    ) -> MappedWeights:
        if handshake_timeout <= 0:
            raise ValueError("handshake_timeout must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if validate_local_device:
            _validate_local_manifest_device(expected_manifest)
        connection = Client(endpoint.address, family=endpoint.family, authkey=endpoint.authkey)
        try:
            connection.send({"type": _HELLO, "manifest": expected_manifest.to_payload()})
            manifest_message = _recv_message(connection, handshake_timeout)
            if not isinstance(manifest_message, Mapping):
                raise ProtocolError("manifest response must be an object")
            if manifest_message.get("type") == _ERROR:
                raise _error_from_message(manifest_message)
            if manifest_message.get("type") != _MANIFEST:
                raise ProtocolError("expected manifest from provider")
            provider_manifest = WeightManifest.from_payload(manifest_message.get("manifest"))
            mismatches = compare_manifests(expected_manifest, provider_manifest)
            if mismatches:
                raise ManifestMismatchError(mismatches)
            if manifest_message.get("fingerprint") != provider_manifest.fingerprint:
                raise ProtocolError("provider manifest fingerprint is invalid")
            connection.send({"type": _MANIFEST_ACK, "fingerprint": provider_manifest.fingerprint})

            handles_message = _recv_message(connection, handshake_timeout)
            if not isinstance(handles_message, Mapping):
                raise ProtocolError("handles response must be an object")
            if handles_message.get("type") == _ERROR:
                raise _error_from_message(handles_message)
            if handles_message.get("type") != _HANDLES:
                raise ProtocolError("expected handles from provider")
            handles = handles_message.get("handles")
            if not isinstance(handles, Mapping):
                raise ProtocolError("provider handles must be an object")
            if set(handles) != set(expected_manifest.tensor_names):
                raise ManifestMismatchError([f"handles do not match manifest tensor names: {sorted(handles)!r}"])
            mapped = {
                name: rebuild(handle_mapper(tuple(handles[name]), expected_manifest.device_index))
                for name in expected_manifest.tensor_names
            }
            if validate_mapped_tensors:
                _validate_mapped_tensors(mapped, expected_manifest)
            connection.send({"type": _READY})
            return MappedWeights(
                mapped,
                provider_manifest,
                connection,
                heartbeat_interval=heartbeat_interval,
                on_provider_exit=on_provider_exit,
            )
        except Exception:
            try:
                connection.send({"type": _CLOSE})
            except (EOFError, OSError):
                pass
            connection.close()
            raise


__all__ = [
    "ChannelAddress",
    "IPC_PROTOCOL_VERSION",
    "ManifestMismatchError",
    "ManifestValidationError",
    "MappedWeights",
    "ProtocolError",
    "ProtocolTimeoutError",
    "ProviderUnavailableError",
    "TensorManifestEntry",
    "TensorHandle",
    "TensorHandleMapper",
    "WeightManifest",
    "WeightSharingConsumer",
    "WeightSharingEndpoint",
    "WeightSharingError",
    "WeightSharingProvider",
    "build_weight_manifest",
    "compare_manifests",
    "export_cuda_handles",
    "get_cuda_device_uuid",
    "rebuild_cuda_tensor",
    "remap_cuda_handle_device",
]
