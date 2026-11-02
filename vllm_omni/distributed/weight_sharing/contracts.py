# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only contracts for persistent, rank-paired CUDA weight sharing.

Storage IDs identify PyTorch storages, not CUDA allocations. Each storage is
reduced once per session and imported at most once, even when allocation handles
are equal. None of these descriptors contains tensor data or Python callables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol

import msgspec

if TYPE_CHECKING:
    from .backend import ExportSession, ImportedWeights


class WeightSharingError(RuntimeError):
    def __init__(self, message: str, *, session_id: str = "", poisoned: bool = False):
        super().__init__(message)
        self.session_id = session_id
        self.poisoned = poisoned


class ProtocolError(WeightSharingError):
    pass


class CompatibilityError(WeightSharingError):
    pass


class UnsupportedError(WeightSharingError):
    pass


class ThreadOwnershipError(WeightSharingError):
    pass


class SessionState(str, Enum):
    PREPARING = "PREPARING"
    EXPORTED = "EXPORTED"
    CLAIMED = "CLAIMED"
    ATTACHED = "ATTACHED"
    IMPORTING = "IMPORTING"
    MAPPED = "MAPPED"
    RELEASING = "RELEASING"
    PEER_RELEASED = "PEER_RELEASED"
    CLOSED = "CLOSED"
    POISONED = "POISONED"


class Identity(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    group_id: str
    generation: int
    session_id: str
    provider_id: str
    consumer_id: str
    rank: int
    tp_size: int
    device_uuid: str
    protocol_version: int = 1


@dataclass(frozen=True)
class SessionContext:
    identity: Identity
    device_index: int

    def __post_init__(self) -> None:
        i = self.identity
        if not all((i.group_id, i.session_id, i.provider_id, i.consumer_id, i.device_uuid)):
            raise ValueError("session identity must be nonempty")
        if i.protocol_version != 1 or i.generation < 0 or not 0 <= i.rank < i.tp_size:
            raise ValueError("invalid protocol version, generation or TP rank")
        if self.device_index < 0:
            raise ValueError("device index must be nonnegative")


@dataclass(frozen=True)
class Limits:
    max_bundle_bytes: int = 64 * 1024 * 1024
    max_entries: int = 100_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_bundle_bytes <= (1 << 32) - 1 or self.max_entries < 1:
            raise ValueError("metadata size and entry limits must be positive and bounded")


class TensorSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...] | None = None
    alias_of: str | None = None


class WeightSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    model_id: str
    checkpoint_revision: str
    layout: str
    tensors: tuple[TensorSpec, ...]


class StorageSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    storage_id: str
    nbytes: int


class TensorViewSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    storage_id: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset_bytes: int

    def storage_view_key(self) -> tuple[str, str, tuple[int, ...], tuple[int, ...], int]:
        return self.storage_id, self.dtype, self.shape, self.stride, self.offset_bytes


class WeightManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    spec: WeightSpec
    storages: tuple[StorageSpec, ...]
    views: tuple[TensorViewSpec, ...]


class StorageExport(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    storage_id: str
    codec: str
    device_index: int
    handle: bytes
    storage_nbytes: int
    allocation_offset_bytes: int
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool


class ExportBundle(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One delivery to the consumer named by identity; never replay this bundle."""

    identity: Identity
    import_id: str
    manifest: WeightManifest
    exports: tuple[StorageExport, ...]


class SessionReceipt(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A local outcome relayed by a trusted caller, not an authentication token."""

    identity: Identity
    import_id: str
    op: Literal["imported", "released"]


@dataclass
class PreparedWeights:
    manifest: WeightManifest
    carriers: dict[str, object]
    sources: tuple[object, ...]


class StorageLease(Protocol):
    def expired(self) -> bool: ...

    def close(self) -> None: ...


class CudaIpcCodec(Protocol):
    """All methods execute on the session's CUDA owner thread.

    The backend validates the spec before inspect and the manifest before
    make_views. inspect validates the provider runtime; validate_runtime does
    so for the consumer before importing. synchronize only waits for the GPU.
    """

    def inspect(self, bindings: Mapping[str, object], spec: WeightSpec, context: SessionContext) -> PreparedWeights: ...

    def validate_runtime(self, context: SessionContext) -> None: ...

    def synchronize(self, context: SessionContext) -> None: ...

    def export_storage(self, carrier: object, storage_id: str) -> StorageExport: ...

    def import_storage(self, descriptor: StorageExport, device_index: int) -> object: ...

    def empty_storage(self, device_index: int) -> object: ...

    def make_views(self, manifest: WeightManifest, carriers: Mapping[str, object]) -> dict[str, object]: ...

    def weak_storage(self, carrier: object) -> StorageLease: ...

    def cancel_unconsumed(self, descriptor: StorageExport, device_index: int) -> None: ...


@dataclass(frozen=True)
class SessionSnapshot:
    """Batch state and actual resource progress, without per-storage states.

    Export IDs describe successfully produced/received descriptors, retained as
    metadata after close. Import IDs describe locally held carriers, including
    empty storages. active_storage_id remains set if an operation failed with
    an uncertain outcome; it does not authorize retry or counter cancellation.
    """

    session_id: str
    role: Literal["provider", "consumer"]
    state: SessionState
    storage_count: int
    view_count: int
    shared_nbytes: int
    exported_storage_ids: tuple[str, ...]
    imported_storage_ids: tuple[str, ...]
    active_storage_id: str | None
    error: str = ""


@dataclass(frozen=True)
class CloseResult:
    """CLOSED is local completion; only a provider can confirm peer release.

    A consumer's receipt may be relayed again without repeating CUDA cleanup.
    The caller must deliver it before releasing the provider's resources.
    """

    status: Literal["CLOSED", "HELD", "POISONED"]
    local_complete: bool = False
    peer_confirmed: bool = False
    receipt: bytes | None = None


class WeightSharingBackend(Protocol):
    def prepare_export(
        self, bindings: Mapping[str, object], weight_spec: WeightSpec, context: SessionContext
    ) -> ExportSession: ...

    def claim_export(self, handle_or_session_id: ExportSession | str) -> bytes: ...

    def open_import(self, payload: bytes, expected_spec: WeightSpec, context: SessionContext) -> ImportedWeights: ...

    def confirm_import(self, receipt: bytes) -> None: ...

    def confirm_release(self, receipt: bytes) -> None: ...

    def mark_failed(self, handle_or_session_id: ExportSession | ImportedWeights | str) -> None: ...

    def session_status(self, handle_or_session_id: ExportSession | ImportedWeights | str) -> SessionSnapshot: ...

    def close_import(self, handle_or_session_id: ImportedWeights | str) -> CloseResult: ...

    def release_export(self, handle_or_session_id: ExportSession | str) -> CloseResult: ...


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def dtype_size(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise CompatibilityError(f"unsupported dtype: {dtype}") from exc


def validate_spec(spec: WeightSpec, max_entries: int = 100_000) -> None:
    if not all((spec.model_id, spec.checkpoint_revision, spec.layout)) or not 0 < len(spec.tensors) <= max_entries:
        raise CompatibilityError("weight identity and a bounded, nonempty tensor specification are required")
    names = {t.name for t in spec.tensors}
    if len(names) != len(spec.tensors) or "" in names:
        raise CompatibilityError("tensor names must be nonempty and unique")
    for tensor in spec.tensors:
        dtype_size(tensor.dtype)
        if any(d < 0 for d in tensor.shape) or len(tensor.shape) > 64:
            raise CompatibilityError("invalid tensor dimensions")
        if tensor.stride is not None and (len(tensor.stride) != len(tensor.shape) or any(s < 0 for s in tensor.stride)):
            raise CompatibilityError("invalid tensor stride")
        if tensor.alias_of is not None and (tensor.alias_of not in names or tensor.alias_of == tensor.name):
            raise CompatibilityError("alias must refer to another specified tensor")


def validate_manifest(manifest: WeightManifest, expected: WeightSpec, max_entries: int = 100_000) -> None:
    """Check a produced/received manifest against an already validated spec."""
    if manifest.spec != expected:
        raise CompatibilityError("model, checkpoint, layout or tensor specification differs")
    if len(manifest.storages) > max_entries or len(manifest.views) != len(expected.tensors):
        raise CompatibilityError("invalid storage/view count")
    storages = {s.storage_id: s for s in manifest.storages}
    views = {v.name: v for v in manifest.views}
    if len(storages) != len(manifest.storages) or len(views) != len(manifest.views):
        raise CompatibilityError("duplicate storage/view identity")
    if any(not s.storage_id or s.nbytes < 0 for s in manifest.storages):
        raise CompatibilityError("invalid storage identity or size")
    if set(views) != {t.name for t in expected.tensors} or {v.storage_id for v in views.values()} != set(storages):
        raise CompatibilityError("manifest has missing or unused views/storages")
    for tensor in expected.tensors:
        v = views[tensor.name]
        if (
            v.dtype != tensor.dtype
            or v.shape != tensor.shape
            or (tensor.stride is not None and v.stride != tensor.stride)
        ):
            raise CompatibilityError(f"tensor layout differs: {tensor.name}")
        size = dtype_size(v.dtype)
        if len(v.stride) != len(v.shape) or any(s < 0 for s in v.stride) or v.offset_bytes < 0 or v.offset_bytes % size:
            raise CompatibilityError(f"invalid view: {tensor.name}")
        end = v.offset_bytes
        if all(v.shape):
            end += (1 + sum((d - 1) * s for d, s in zip(v.shape, v.stride))) * size
        if end > storages[v.storage_id].nbytes:
            raise CompatibilityError(f"view exceeds storage: {tensor.name}")
        if tensor.alias_of is not None:
            other = views[tensor.alias_of]
            if v.storage_view_key() != other.storage_view_key():
                raise CompatibilityError(f"alias differs: {tensor.name}")
