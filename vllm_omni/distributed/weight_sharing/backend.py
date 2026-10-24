# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Persistent weight mappings with CPU-only control and owner-thread cleanup.

Closing a socket is never evidence that an IPC mapping has been released.
The caller must keep this backend alive, stop model use before close_import,
and terminate the sharing group when a snapshot is POISONED. Sessions use one
connection from import through release; a failed session cannot reconnect.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from threading import Event, Lock, Thread, current_thread
from uuid import uuid4

from .channel import FramedConnection, Limits, SessionServer, connect
from .contracts import (
    CloseResult,
    CompatibilityError,
    CudaIpcCodec,
    Message,
    MessageOp,
    PreparedWeights,
    ProtocolError,
    SessionContext,
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


@dataclass
class _Control:
    """Only CPU metadata is reachable from the socket handler."""

    context: SessionContext
    state: SessionState = SessionState.PREPARING
    manifest: WeightManifest | None = None
    exports: list[StorageExport] = field(default_factory=list)
    active_storage_id: str | None = None
    import_id: str = ""
    closing: bool = False
    peer_released: bool = False
    error: str = ""
    lock: Lock = field(default_factory=Lock)
    release_complete: Event = field(default_factory=Event)
    message: partial[Message] = field(init=False)

    def __post_init__(self) -> None:
        identity = self.context.identity
        self.message = partial(Message, session_id=identity.session_id, generation=identity.generation)


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

    def __init__(self, session_id: str, manifest: WeightManifest, tensors: dict[str, object]):
        self.session_id = session_id
        self.manifest = manifest
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
        return len(self._tensors)


@dataclass
class _Provider:
    control: _Control
    owner: Thread
    prepared: PreparedWeights | None = None
    server: SessionServer | None = None


@dataclass
class _Consumer:
    control: _Control
    owner: Thread
    carriers: dict[str, object] = field(default_factory=dict)
    leases: list[StorageLease] = field(default_factory=list)
    result: ImportedWeights | None = None
    connection: FramedConnection | None = None
    local_complete: bool = False
    peer_confirmed: bool = False


@contextmanager
def _storage_operation(control: _Control, storage_id: str) -> Iterator[None]:
    """Track uncertain work without holding the CPU state lock across CUDA calls."""
    with control.lock:
        control.active_storage_id = storage_id
    yield
    with control.lock:
        control.active_storage_id = None


def _receive(
    control: _Control, connection: FramedConnection, op: MessageOp | None = None, *, wait_for_message: bool = False
) -> Message:
    message = connection.recv(wait_for_message=wait_for_message)
    identity = control.context.identity
    if message.session_id != identity.session_id or message.generation != identity.generation:
        raise ProtocolError("message belongs to another session or generation")
    if message.op == "error":
        raise ProtocolError(message.error or "peer rejected the operation")
    if op is not None and (message.op != op or message.import_id != control.import_id):
        raise ProtocolError(f"expected {op} for the active import")
    return message


def _serve(control: _Control, connection: FramedConnection) -> None:
    """One CPU-only handshake and release; never reconnect or replay handles."""
    claimed = False
    try:
        hello = _receive(control, connection)
        if (
            hello.op != "hello"
            or not hello.import_id
            or hello.identity != control.context.identity
            or not hmac.compare_digest(hello.token, control.context.token)
        ):
            raise ProtocolError("peer identity or authentication does not match")
        with control.lock:
            if control.closing or control.import_id or control.state != SessionState.EXPORTED:
                raise ProtocolError("session no longer accepts imports")
            manifest = control.manifest
            control.import_id = hello.import_id
            claimed = True
        connection.send(control.message("manifest", manifest=manifest))
        _receive(control, connection, "get_bundle")
        with control.lock:
            if control.closing:
                raise ProtocolError("session is closing")
            # Claiming the batch already prohibits cancellation, even if send fails.
            exports = tuple(control.exports)
        connection.send(control.message("bundle", import_id=control.import_id, exports=exports))
        # CUDA import and model use can outlive the per-message IO timeout.
        _receive(control, connection, "imported", wait_for_message=True)
        with control.lock:
            control.state = SessionState.ATTACHED
        connection.send(control.message("accepted", import_id=control.import_id))
        _receive(control, connection, "release", wait_for_message=True)
        with control.lock:
            control.state = SessionState.RELEASING
        connection.send(control.message("released", import_id=control.import_id))
        with control.lock:
            control.peer_released = True
            control.state = SessionState.PEER_RELEASED
    except (OSError, EOFError, WeightSharingError) as exc:
        if claimed:
            # Transport loss never proves that mappings or counters were released.
            with control.lock:
                control.state = SessionState.POISONED
                control.error = f"{type(exc).__name__}; terminate sharing group"
        else:
            try:
                connection.send(control.message("error", error=str(exc)))
            except (OSError, EOFError, WeightSharingError):
                pass
    finally:
        if claimed:
            # Owner shutdown must not interrupt the final release ACK.
            control.release_complete.set()


class TorchCudaIpcBackend:
    """One backend per owner; session_status may be queried by other threads.

    The codec is injectable for independent CPU tests. Production callers should
    use the default TorchCudaIpcCodec, which rejects unsupported devices/ABIs.
    No worker, scheduler, KV connector or process manager is imported here.
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
            if isinstance(record, _Consumer):
                if record.result is not None:
                    record.result._active = False
                self._close_connection(record)
            if not isinstance(exc, Exception):
                raise
            raise error from exc

    @staticmethod
    def _close_connection(record: _Consumer) -> None:
        if record.connection is not None:
            record.connection.close()
            record.connection = None

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
        # Only CPU metadata is captured by this callback, never tensor owners.
        record.server = SessionServer(context.endpoint, self._limits, lambda connection: _serve(control, connection))
        try:
            record.server.start()
        except Exception as exc:
            raise self._failure(control, exc) from exc
        return ExportSession(context.identity.session_id, manifest)

    @staticmethod
    def _exchange(record: _Consumer, op: MessageOp, expected_reply: MessageOp) -> None:
        connection = record.connection
        if connection is None:
            raise ProtocolError("import connection is closed")
        control = record.control
        connection.send(control.message(op, import_id=control.import_id))
        _receive(control, connection, expected_reply)

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

    def open_import(self, endpoint: str, expected_spec: WeightSpec, context: SessionContext) -> ImportedWeights:
        if endpoint != context.endpoint:
            raise CompatibilityError("endpoint differs from the session bootstrap")
        control = _Control(context, state=SessionState.IMPORTING, import_id=uuid4().hex)
        record = _Consumer(control, current_thread())
        self._register(record)  # A successful or failed session can never be reopened.
        with self._poison_on_error(record):
            validate_spec(expected_spec, self._limits.max_entries)
            self._codec.validate_runtime(context)
            self._codec.synchronize(context)
            connection = connect(endpoint, self._limits)
            record.connection = connection
            connection.send(
                control.message("hello", import_id=control.import_id, identity=context.identity, token=context.token)
            )
            reply = _receive(control, connection)
            if reply.op != "manifest" or reply.manifest is None:
                raise ProtocolError("expected manifest")
            manifest = reply.manifest
            validate_manifest(manifest, expected_spec, self._limits.max_entries)
            with control.lock:
                control.manifest = manifest
            connection.send(control.message("get_bundle", import_id=control.import_id))
            bundle = _receive(control, connection, "bundle")
            self._validate_exports(manifest, bundle.exports)
            with control.lock:
                control.exports = list(bundle.exports)
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
            record.result = ImportedWeights(context.identity.session_id, manifest, tensors)
            self._exchange(record, "imported", "accepted")
            with control.lock:
                control.state = SessionState.MAPPED
            return record.result

    def close_import(self, handle_or_session_id: ImportedWeights | str) -> CloseResult:
        record = self._get(handle_or_session_id)
        if not isinstance(record, _Consumer):
            raise WeightSharingError("close_import requires a consumer session")
        control = record.control
        if control.state == SessionState.POISONED:
            return CloseResult("POISONED", record.local_complete, record.peer_confirmed)
        if control.state == SessionState.CLOSED:
            return CloseResult("CLOSED", record.local_complete, record.peer_confirmed)
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
                # Local aliases may be dropped later; retain the same connection.
                return CloseResult("HELD")
            for lease in record.leases:
                lease.close()
            record.leases.clear()
            record.local_complete = True
            self._exchange(record, "release", "released")
            record.peer_confirmed = True
            self._close_connection(record)
            with control.lock:
                control.state = SessionState.CLOSED
            return CloseResult("CLOSED", True, True)

    def release_export(self, handle_or_session_id: ExportSession | str) -> CloseResult:
        record = self._get(handle_or_session_id)
        if not isinstance(record, _Provider):
            raise WeightSharingError("release_export requires a provider session")
        control = record.control
        with control.lock:
            awaiting_ack = control.state == SessionState.RELEASING and bool(control.import_id)
        if awaiting_ack:
            control.release_complete.wait(self._limits.timeout)
        with control.lock:
            if control.state == SessionState.CLOSED:
                return CloseResult("CLOSED", True, control.peer_released)
            if control.state == SessionState.POISONED:
                return CloseResult("POISONED")
            control.closing = True
            # An empty-weight bundle still binds an import that must be closed.
            if control.import_id and not control.peer_released:
                return CloseResult("HELD")
            cancellable = control.exports if not control.import_id else ()
            control.state = SessionState.RELEASING
        with self._poison_on_error(record):
            for descriptor in cancellable:
                with _storage_operation(control, descriptor.storage_id):
                    self._codec.cancel_unconsumed(descriptor, control.context.device_index)
            # Stop/join outside the state lock; the CPU thread never owns sources.
            if record.server is not None:
                record.server.stop()
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
