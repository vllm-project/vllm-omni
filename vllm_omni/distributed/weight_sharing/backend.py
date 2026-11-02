# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transport-independent weight mappings with owner-thread resource operations.

The trusted caller routes opaque metadata through its control plane, pairs ranks,
reports uncertain delivery/peer failures via mark_failed, and terminates poisoned
sharing groups. This module starts no threads and performs no network I/O.
Keep the backend alive until teardown; never replay a bundle into a replacement
consumer. A lost RPC response is not evidence that CUDA resources were released.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, Thread, current_thread
from typing import Literal, TypeVar
from uuid import uuid4

import msgspec

from .contracts import (
    CloseResult,
    CompatibilityError,
    CudaIpcCodec,
    ExportBundle,
    Limits,
    PreparedWeights,
    ProtocolError,
    SessionContext,
    SessionReceipt,
    SessionSnapshot,
    SessionState,
    StorageExport,
    StorageLease,
    ThreadOwnershipError,
    WeightManifest,
    WeightSharingError,
    WeightSpec,
    validate_manifest,
    validate_spec,
)

_Metadata = TypeVar("_Metadata", ExportBundle, SessionReceipt)


@dataclass
class _Control:
    """CPU metadata shared with read-only status queries."""

    context: SessionContext
    state: SessionState = SessionState.PREPARING
    manifest: WeightManifest | None = None
    exports: list[StorageExport] = field(default_factory=list)
    active_storage_id: str | None = None
    import_id: str = ""
    peer_released: bool = False
    error: str = ""
    lock: Lock = field(default_factory=Lock)


@dataclass(frozen=True)
class ExportSession:
    session_id: str
    manifest: WeightManifest


class ImportedWeights(Mapping[str, object]):
    """Views borrowed by the model; close_import invalidates this mapping.

    External tensor/view/Parameter references must also be removed by the caller.
    Keeping this wrapper alive alone does not prevent close_import from releasing
    its contents.
    """

    def __init__(self, session_id: str, manifest: WeightManifest, tensors: dict[str, object], receipt: bytes):
        self.session_id = session_id
        self.manifest = manifest
        self.receipt = receipt
        self._tensors = tensors
        self._active = True

    def __getitem__(self, name: str) -> object:
        if not self._active:
            raise WeightSharingError("import is closing or closed", session_id=self.session_id)
        return self._tensors[name]

    def __iter__(self) -> Iterator[str]:
        if not self._active:
            raise WeightSharingError("import is closing or closed", session_id=self.session_id)
        return iter(self._tensors)

    def __len__(self) -> int:
        if not self._active:
            raise WeightSharingError("import is closing or closed", session_id=self.session_id)
        return len(self._tensors)


@dataclass
class _Provider:
    control: _Control
    owner: Thread
    prepared: PreparedWeights | None = None


@dataclass
class _Consumer:
    control: _Control
    owner: Thread
    carriers: dict[str, object] = field(default_factory=dict)
    leases: list[StorageLease] = field(default_factory=list)
    result: ImportedWeights | None = None
    local_complete: bool = False


@contextmanager
def _storage_operation(control: _Control, storage_id: str) -> Iterator[None]:
    """Track uncertain work without holding the CPU state lock across CUDA calls."""
    with control.lock:
        control.active_storage_id = storage_id
    yield
    with control.lock:
        control.active_storage_id = None


class TorchCudaIpcBackend:
    """One backend per owner; session_status may be queried by other threads.

    The codec is injectable for independent CPU tests. Production callers should
    use the default TorchCudaIpcCodec, which rejects unsupported devices/ABIs.
    All mutations execute on the owner thread, including control-plane receipts
    and failure reports. No worker, scheduler or process manager is imported here.
    The caller authenticates/routes metadata; identity checks here prevent stale
    or misrouted messages, not a malicious control-plane participant.

    Normal flow (each call runs on its respective owner):
      provider.prepare_export(...) -> export
      provider.claim_export(export) -> bundle bytes
      consumer.open_import(bundle, expected_spec, context) -> weights
      provider.confirm_import(weights.receipt)
      consumer.close_import(weights) -> local result, possibly HELD
      provider.confirm_release(result.receipt)  # only after local completion
      provider.release_export(export)

    Relay bytes, never ImportedWeights or tensors, through the caller's RPCs.
    Receipts can be delivered again; an import with an uncertain outcome cannot.
    """

    def __init__(self, *, codec: CudaIpcCodec | None = None, limits: Limits | None = None):
        if codec is None:
            from .torch_cuda_ipc_codec import TorchCudaIpcCodec

            codec = TorchCudaIpcCodec()
        self._codec = codec
        self._limits = limits or Limits()
        self._sessions: dict[str, _Provider | _Consumer] = {}
        self._registry_lock = Lock()

    def _register(self, record: _Provider | _Consumer) -> None:
        session_id = record.control.context.identity.session_id
        with self._registry_lock:
            if session_id in self._sessions:
                raise WeightSharingError("session ID cannot be reused", session_id=session_id)
            self._sessions[session_id] = record

    def _get(self, handle: ExportSession | ImportedWeights | str, *, owner: bool = True) -> _Provider | _Consumer:
        session_id = handle if isinstance(handle, str) else handle.session_id
        with self._registry_lock:
            record = self._sessions.get(session_id)
        if record is None:
            raise WeightSharingError("unknown session", session_id=session_id)
        if owner and record.owner is not current_thread():
            raise ThreadOwnershipError("CUDA resources belong to another thread", session_id=session_id)
        return record

    @staticmethod
    def _failure(control: _Control, exc: BaseException, *, poisoned: bool = False) -> WeightSharingError:
        with control.lock:
            control.error = type(exc).__name__
            if poisoned:
                control.state = SessionState.POISONED
        return WeightSharingError(
            f"weight sharing failed ({type(exc).__name__}); "
            + ("terminate the sharing group" if poisoned else "inspect the session before cleanup"),
            session_id=control.context.identity.session_id,
            poisoned=poisoned,
        )

    @contextmanager
    def _poison_on_error(self, record: _Provider | _Consumer) -> Iterator[None]:
        """Seal uncertain CUDA/transport failures; retain all resources for teardown."""
        try:
            yield
        except BaseException as exc:
            error = self._failure(record.control, exc, poisoned=True)
            if isinstance(record, _Consumer) and record.result is not None:
                record.result._active = False
            if not isinstance(exc, Exception):
                raise
            raise error from exc

    def _decode(self, payload: bytes, cls: type[_Metadata]) -> _Metadata:
        if type(payload) is not bytes or not 0 < len(payload) <= self._limits.max_bundle_bytes:
            raise ProtocolError("invalid metadata payload size")
        try:
            return msgspec.msgpack.decode(payload, type=cls)
        except msgspec.DecodeError as exc:
            raise ProtocolError("invalid metadata payload") from exc

    @staticmethod
    def _receipt(control: _Control, op: Literal["imported", "released"]) -> bytes:
        return msgspec.msgpack.encode(SessionReceipt(control.context.identity, control.import_id, op))

    def mark_failed(self, handle_or_session_id: ExportSession | ImportedWeights | str) -> None:
        """Report an uncertain control-plane outcome; never infer safe cleanup.

        Invoke on each surviving owner after a peer crash or RPC timeout. No
        automatic peer monitoring remains here. A failed import may not have
        registered a local session if it was rejected during resource-free preflight.
        """
        record = self._get(handle_or_session_id)
        with record.control.lock:
            if record.control.state == SessionState.POISONED:
                return
            record.control.state = SessionState.POISONED
            record.control.error = "control-plane outcome uncertain; terminate sharing group"
        if isinstance(record, _Consumer) and record.result is not None:
            record.result._active = False

    def prepare_export(
        self, bindings: Mapping[str, object], weight_spec: WeightSpec, context: SessionContext
    ) -> ExportSession:
        control = _Control(context)
        record = _Provider(control, current_thread())
        self._register(record)  # Stable identity exists even for partial failures.
        try:
            validate_spec(weight_spec, self._limits.max_entries)
            record.prepared = self._codec.inspect(bindings, weight_spec, context)
            manifest = record.prepared.manifest
            validate_manifest(manifest, weight_spec, self._limits.max_entries)
            self._codec.synchronize(context)
        except Exception as exc:
            raise self._failure(control, exc) from exc
        with control.lock:
            control.manifest = manifest
        # A reducer may create a counter before raising. Preserve every known
        # descriptor and the uncertain storage ID; never guess or retry it.
        with self._poison_on_error(record):
            for storage in manifest.storages:
                if not storage.nbytes:
                    continue
                with _storage_operation(control, storage.storage_id):
                    descriptor = self._codec.export_storage(
                        record.prepared.carriers[storage.storage_id], storage.storage_id
                    )
                    with control.lock:
                        control.exports.append(descriptor)
        with control.lock:
            control.state = SessionState.EXPORTED
        return ExportSession(context.identity.session_id, manifest)

    def claim_export(self, handle_or_session_id: ExportSession | str) -> bytes:
        """Return one bounded bundle, marking exposure BEFORE returning it.

        Even if delivery fails, cancellation is no longer safe. The caller must
        not resend this bundle to another consumer or retry an uncertain import.
        """
        record = self._get(handle_or_session_id)
        if not isinstance(record, _Provider):
            raise WeightSharingError("claim_export requires a provider session")
        control = record.control
        with control.lock:
            if control.state != SessionState.EXPORTED or control.import_id:
                raise ProtocolError("session no longer accepts imports")
            assert control.manifest is not None
            import_id = uuid4().hex
            bundle = ExportBundle(control.context.identity, import_id, control.manifest, tuple(control.exports))
            payload = msgspec.msgpack.encode(bundle)
            if len(payload) > self._limits.max_bundle_bytes:
                raise ProtocolError("metadata exceeds size limit")
            control.import_id = import_id
            control.state = SessionState.CLAIMED
        return payload

    def _confirm(self, payload: bytes, op: Literal["imported", "released"]) -> None:
        receipt = self._decode(payload, SessionReceipt)
        record = self._get(receipt.identity.session_id)
        if not isinstance(record, _Provider):
            raise ProtocolError("confirmation requires a provider session")
        control = record.control
        with control.lock:
            if (
                receipt.identity != control.context.identity
                or not control.import_id
                or receipt.import_id != control.import_id
                or receipt.op != op
            ):
                raise ProtocolError("receipt differs from the claimed session")
            allowed = (SessionState.CLAIMED, SessionState.ATTACHED) if op == "imported" else (SessionState.ATTACHED,)
            # Repeated receipts are harmless; they never repeat device operations
            # or roll back a later state. Bundles, unlike receipts, are single-use.
            if control.state in (SessionState.PEER_RELEASED, SessionState.CLOSED):
                return
            if control.state not in allowed:
                raise ProtocolError("receipt is out of order or session failed")
            if op == "imported":
                control.state = SessionState.ATTACHED
            else:
                control.peer_released = True
                control.state = SessionState.PEER_RELEASED

    def confirm_import(self, receipt: bytes) -> None:
        self._confirm(receipt, "imported")

    def confirm_release(self, receipt: bytes) -> None:
        self._confirm(receipt, "released")

    @staticmethod
    def _validate_exports(manifest: WeightManifest, exports: tuple[StorageExport, ...]) -> None:
        expected = {s.storage_id: s.nbytes for s in manifest.storages if s.nbytes}
        if len(exports) != len(expected) or {d.storage_id for d in exports} != set(expected):
            raise CompatibilityError("exports do not cover exactly the nonempty storages")
        # Reject malformed metadata anywhere in the batch before any rebuild.
        # The codec separately checks the PyTorch-specific ABI and handle format.
        for d in exports:
            if (
                not d.codec
                or not d.handle
                or not d.ref_counter_handle
                or d.storage_nbytes != expected[d.storage_id]
                or d.device_index < 0
                or d.allocation_offset_bytes < 0
                or d.ref_counter_offset < 0
            ):
                raise CompatibilityError("invalid storage export descriptor")

    def open_import(self, payload: bytes, expected_spec: WeightSpec, context: SessionContext) -> ImportedWeights:
        # Reject replay before touching CUDA; _register rechecks after preflight.
        with self._registry_lock:
            if context.identity.session_id in self._sessions:
                raise WeightSharingError("session ID cannot be reused", session_id=context.identity.session_id)
        # Local preflight creates no IPC references; failures need no session cleanup.
        validate_spec(expected_spec, self._limits.max_entries)
        bundle = self._decode(payload, ExportBundle)
        if bundle.identity != context.identity or not bundle.import_id:
            raise CompatibilityError("bundle identity differs from the expected rank pair")
        manifest = bundle.manifest
        validate_manifest(manifest, expected_spec, self._limits.max_entries)
        self._validate_exports(manifest, bundle.exports)
        self._codec.validate_runtime(context)
        self._codec.synchronize(context)
        control = _Control(
            context,
            state=SessionState.IMPORTING,
            import_id=bundle.import_id,
            manifest=manifest,
            exports=list(bundle.exports),
        )
        record = _Consumer(control, current_thread())
        self._register(record)  # A successful or uncertain import can never be reopened.
        with self._poison_on_error(record):
            for descriptor in bundle.exports:
                with _storage_operation(control, descriptor.storage_id):
                    carrier = self._codec.import_storage(descriptor, context.device_index)
                    with control.lock:
                        record.carriers[descriptor.storage_id] = carrier
                    record.leases.append(self._codec.weak_storage(carrier))
                    del carrier
            for storage in manifest.storages:
                if not storage.nbytes:
                    carrier = self._codec.empty_storage(context.device_index)
                    with control.lock:
                        record.carriers[storage.storage_id] = carrier
                    del carrier
            tensors = self._codec.make_views(manifest, record.carriers)
            self._codec.synchronize(context)
            record.result = ImportedWeights(
                context.identity.session_id, manifest, tensors, self._receipt(control, "imported")
            )
            with control.lock:
                control.state = SessionState.MAPPED
            return record.result

    def close_import(self, handle_or_session_id: ImportedWeights | str) -> CloseResult:
        record = self._get(handle_or_session_id)
        if not isinstance(record, _Consumer):
            raise WeightSharingError("close_import requires a consumer session")
        control = record.control
        if control.state == SessionState.POISONED:
            return CloseResult("POISONED", record.local_complete)
        if control.state == SessionState.CLOSED:
            return CloseResult("CLOSED", True, receipt=self._receipt(control, "released"))
        with control.lock:
            control.state = SessionState.RELEASING
        with self._poison_on_error(record):
            if record.result is None:
                raise WeightSharingError("import has not completed")
            record.result._active = False
            self._codec.synchronize(control.context)
            record.result._tensors.clear()
            with control.lock:
                carriers, record.carriers = record.carriers, {}
            carriers.clear()  # Destroy CUDA owners outside the state lock.
            if any(not lease.expired() for lease in record.leases):
                # Local aliases may be dropped later; the caller can retry close.
                return CloseResult("HELD")
            for lease in record.leases:
                lease.close()
            record.leases.clear()
            record.local_complete = True
            with control.lock:
                control.state = SessionState.CLOSED
            return CloseResult("CLOSED", True, receipt=self._receipt(control, "released"))

    def release_export(self, handle_or_session_id: ExportSession | str) -> CloseResult:
        record = self._get(handle_or_session_id)
        if not isinstance(record, _Provider):
            raise WeightSharingError("release_export requires a provider session")
        control = record.control
        with control.lock:
            if control.state == SessionState.CLOSED:
                return CloseResult("CLOSED", True, control.peer_released)
            if control.state == SessionState.POISONED:
                return CloseResult("POISONED")
            # An empty-weight bundle still binds an import that must be closed.
            if control.import_id and not control.peer_released:
                return CloseResult("HELD")
            cancellable = control.exports if not control.import_id else ()
            control.state = SessionState.RELEASING
        with self._poison_on_error(record):
            for descriptor in cancellable:
                with _storage_operation(control, descriptor.storage_id):
                    self._codec.cancel_unconsumed(descriptor, control.context.device_index)
            record.prepared = None
            with control.lock:
                control.state = SessionState.CLOSED
            return CloseResult("CLOSED", True, control.peer_released)

    def session_status(self, handle_or_session_id: ExportSession | ImportedWeights | str) -> SessionSnapshot:
        record = self._get(handle_or_session_id, owner=False)
        control = record.control
        with control.lock:
            manifest = control.manifest
            return SessionSnapshot(
                session_id=control.context.identity.session_id,
                role="provider" if isinstance(record, _Provider) else "consumer",
                state=control.state,
                storage_count=len(manifest.storages) if manifest else 0,
                view_count=len(manifest.views) if manifest else 0,
                shared_nbytes=sum(s.nbytes for s in manifest.storages) if manifest else 0,
                exported_storage_ids=tuple(d.storage_id for d in control.exports),
                imported_storage_ids=tuple(record.carriers) if isinstance(record, _Consumer) else (),
                active_storage_id=control.active_storage_id,
                error=control.error,
            )
