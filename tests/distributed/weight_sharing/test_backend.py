# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive the real backend directly: no sockets, server threads or orchestrator."""

import importlib
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import msgspec
import pytest

from tests.distributed.weight_sharing.helpers import FakeCodec, FakeStorage, FakeTensor, make_context, make_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Runtime:
    def __init__(self, package):
        self.api = importlib.import_module(f"{package.__name__}.contracts")
        self.backend = importlib.import_module(f"{package.__name__}.backend")
        self.codec = FakeCodec(self.api)
        self.provider = self.backend.TorchCudaIpcBackend(codec=self.codec)
        self.consumer = self.backend.TorchCudaIpcBackend(codec=self.codec)
        self.context = make_context(self.api)

    def publish(self, *, two_storages=False):
        bindings, spec = make_weights(self.api, two_storages=two_storages)
        return bindings, spec, self.provider.prepare_export(bindings, spec, self.context)

    def attach(self, spec, exported):
        bundle = self.provider.claim_export(exported)
        imported = self.consumer.open_import(bundle, spec, self.context)
        self.provider.confirm_import(imported.receipt)
        return bundle, imported

    def finish(self, exported, imported):
        result = self.consumer.close_import(imported)
        assert result.status == "CLOSED" and result.local_complete and not result.peer_confirmed
        assert result.receipt
        self.provider.confirm_release(result.receipt)
        result = self.provider.release_export(exported)
        assert result.status == "CLOSED" and result.local_complete and result.peer_confirmed


@pytest.fixture
def r(weight_sharing):
    return Runtime(weight_sharing)


@pytest.mark.parametrize("two_storages", [False, True])
def test_roundtrip_views_receipts_and_local_close_are_distinct(r, two_storages):
    threads = set(threading.enumerate())
    bindings, spec, exported = r.publish(two_storages=two_storages)
    sources = [weakref.ref(value.storage) for value in bindings.values()]
    del bindings
    bundle, imported = r.attach(spec, exported)
    assert type(bundle) is bytes and type(imported.receipt) is bytes
    assert imported["first"].storage is imported["first_alias"].storage
    assert (imported["first"].storage is imported["second"].storage) is not two_storages
    assert imported["second"].offset_bytes == 16
    assert imported["first"].storage.data == list(range(8))
    assert len(r.codec.exports) == 1 + two_storages
    assert r.provider.session_status(exported).state == r.api.SessionState.ATTACHED
    result = r.consumer.close_import(imported)
    assert result.status == "CLOSED" and not result.peer_confirmed
    assert r.provider.release_export(exported).status == "HELD"
    assert all(source() is not None for source in sources)
    calls = list(r.codec.calls)
    assert r.consumer.close_import(imported) == result
    assert r.codec.calls == calls
    r.provider.confirm_release(result.receipt)
    assert r.provider.release_export(exported).status == "CLOSED"
    assert all(source() is None for source in sources)
    assert not r.codec.cancellations
    for read in (lambda: imported["first"], lambda: iter(imported), lambda: len(imported)):
        with pytest.raises(r.api.WeightSharingError):
            read()
    assert {owner for _, owner in r.codec.calls} == {threading.get_ident()}
    assert set(threading.enumerate()) == threads


def test_external_alias_holds_import_and_provider(r):
    _, spec, exported = r.publish()
    _, imported = r.attach(spec, exported)
    external = imported["first"]
    result = r.consumer.close_import(imported)
    assert result == r.api.CloseResult("HELD")
    assert r.provider.release_export(exported).status == "HELD"
    assert not r.codec.leases[0].closed
    del external
    r.finish(exported, imported)
    assert r.codec.leases[0].closed


def test_unclaimed_export_cancels_once_and_cannot_be_claimed_after_close(r):
    _, _, exported = r.publish(two_storages=True)
    assert r.provider.release_export(exported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"
    assert r.codec.cancellations == ["storage-0", "storage-1"]
    with pytest.raises(r.api.ProtocolError):
        r.provider.claim_export(exported)


def test_claim_marks_exposure_before_delivery_and_cannot_repeat(r):
    _, spec, exported = r.publish()
    bundle = r.provider.claim_export(exported)
    assert r.provider.session_status(exported).state == r.api.SessionState.CLAIMED
    with pytest.raises(r.api.ProtocolError):
        r.provider.claim_export(exported)
    assert r.provider.release_export(exported).status == "HELD"
    assert not r.codec.cancellations
    imported = r.consumer.open_import(bundle, spec, r.context)
    r.provider.confirm_import(imported.receipt)
    r.finish(exported, imported)


def test_no_reimport_before_or_after_close(r):
    bindings, spec, exported = r.publish()
    bundle, imported = r.attach(spec, exported)
    for closed in (False, True):
        if closed:
            r.finish(exported, imported)
        with pytest.raises(r.api.WeightSharingError, match="reused"):
            r.consumer.open_import(bundle, spec, r.context)
        with pytest.raises(r.api.WeightSharingError, match="reused"):
            r.provider.prepare_export(bindings, spec, r.context)
    assert len(r.codec.imports) == len(r.codec.exports) == 1


def test_receipts_are_idempotent_but_release_cannot_precede_import_confirmation(r):
    _, spec, exported = r.publish()
    imported = r.consumer.open_import(r.provider.claim_export(exported), spec, r.context)
    result = r.consumer.close_import(imported)
    with pytest.raises(r.api.ProtocolError, match="out of order"):
        r.provider.confirm_release(result.receipt)
    r.provider.confirm_import(imported.receipt)
    r.provider.confirm_import(imported.receipt)
    r.provider.confirm_release(result.receipt)
    for closed in (False, True):
        if closed:
            r.provider.release_export(exported)
        before = r.provider.session_status(exported)
        r.provider.confirm_release(result.receipt)
        r.provider.confirm_import(imported.receipt)
        assert r.provider.session_status(exported) == before
    assert not r.codec.cancellations


@pytest.mark.parametrize("mismatch", ["generation", "device_uuid", "rank", "provider_id", "consumer_id", "spec"])
def test_bundle_mismatch_rejected_without_creating_consumer_resources(r, mismatch):
    _, spec, exported = r.publish()
    payload = r.provider.claim_export(exported)
    context = r.context
    if mismatch == "spec":
        spec = msgspec.structs.replace(spec, checkpoint_revision="other")
    else:
        changes = {mismatch: 2 if mismatch == "generation" else "other"}
        if mismatch == "rank":
            changes = {"rank": 1, "tp_size": 2}
        context = make_context(r.api, **changes)
    with pytest.raises(r.api.CompatibilityError):
        r.consumer.open_import(payload, spec, context)
    with pytest.raises(r.api.WeightSharingError, match="unknown session"):
        r.consumer.session_status(context.identity.session_id)
    assert not r.codec.imports
    # A local rejection does not automatically notify the provider.
    assert r.provider.release_export(exported).status == "HELD"
    r.provider.mark_failed(exported)
    assert r.provider.release_export(exported).status == "POISONED"
    assert not r.codec.cancellations


@pytest.mark.parametrize(
    "field", ["generation", "device_uuid", "rank", "provider_id", "consumer_id", "import_id", "op"]
)
def test_bad_receipt_cannot_release_or_change_session(r, field):
    _, spec, exported = r.publish()
    _, imported = r.attach(spec, exported)
    receipt = msgspec.msgpack.decode(imported.receipt, type=r.api.SessionReceipt)
    if field in ("import_id", "op"):
        receipt = msgspec.structs.replace(receipt, **{field: "other" if field == "import_id" else "released"})
    else:
        identity = msgspec.structs.replace(
            receipt.identity, **{field: 99 if field in ("rank", "generation") else "other"}
        )
        receipt = msgspec.structs.replace(receipt, identity=identity)
    before = r.provider.session_status(exported)
    with pytest.raises(r.api.ProtocolError):
        r.provider.confirm_import(msgspec.msgpack.encode(receipt))
    assert r.provider.session_status(exported) == before
    r.finish(exported, imported)


@pytest.mark.parametrize("fault", ["delivery", "import_receipt", "release_receipt"])
def test_uncertain_control_outcome_requires_explicit_failure_and_retains_resources(r, fault):
    bindings, spec, exported = r.publish(two_storages=True)
    sources = [weakref.ref(value.storage) for value in bindings.values()]
    del bindings
    payload = r.provider.claim_export(exported)
    imported = None
    if fault != "delivery":
        imported = r.consumer.open_import(payload, spec, r.context)
        if fault == "release_receipt":
            r.provider.confirm_import(imported.receipt)
            assert r.consumer.close_import(imported).local_complete
    assert r.provider.release_export(exported).status == "HELD"
    r.provider.mark_failed(exported)
    r.provider.mark_failed(exported)
    assert r.provider.release_export(exported).status == "POISONED"
    if imported is not None:
        r.consumer.mark_failed(imported)
        result = r.consumer.close_import(imported)
        assert result.status == "POISONED" and result.receipt is None
        assert result.local_complete == (fault == "release_receipt")
        with pytest.raises(r.api.ProtocolError):
            r.provider.confirm_import(imported.receipt)
        if fault != "release_receipt":
            assert all(not lease.expired() for lease in r.codec.leases)
    assert all(source() is not None for source in sources)
    assert not r.codec.cancellations


@pytest.mark.parametrize("operation", ["export", "import", "cancel"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_partial_device_failure_keeps_active_storage_and_prevents_retry(r, operation, error_type):
    bindings, spec = make_weights(r.api, two_storages=True)
    r.codec.failures[operation] = (2, error_type("injected device failure"))
    backend = r.consumer if operation == "import" else r.provider
    if operation == "export":
        invoke = partial(r.provider.prepare_export, bindings, spec, r.context)
    else:
        exported = r.provider.prepare_export(bindings, spec, r.context)
        if operation == "import":
            payload = r.provider.claim_export(exported)
            invoke = partial(r.consumer.open_import, payload, spec, r.context)
        else:
            invoke = partial(r.provider.release_export, exported)
    expected = r.api.WeightSharingError if error_type is RuntimeError else error_type
    with pytest.raises(expected):
        invoke()
    session_id = r.context.identity.session_id
    snapshot = backend.session_status(session_id)
    assert snapshot.state == r.api.SessionState.POISONED
    assert snapshot.active_storage_id == "storage-1"
    if operation == "import":
        assert snapshot.imported_storage_ids == ("storage-0",)
        assert all(not lease.expired() for lease in r.codec.leases)
        assert r.consumer.close_import(session_id).status == "POISONED"
        with pytest.raises(r.api.WeightSharingError, match="reused"):
            invoke()
        r.provider.mark_failed(exported)
    else:
        assert r.provider.release_export(session_id).status == "POISONED"
        if operation == "export":
            assert snapshot.exported_storage_ids == ("storage-0",)
    assert len(r.codec.cancellations) == (2 if operation == "cancel" else 0)
    assert backend.session_status(session_id) == snapshot


@pytest.mark.parametrize("operation", ["validate_runtime", "synchronize"])
def test_consumer_preflight_failure_needs_no_cleanup(r, monkeypatch, operation):
    _, spec, exported = r.publish()
    payload = r.provider.claim_export(exported)

    def fail(*args):
        raise r.api.UnsupportedError("preflight failed")

    monkeypatch.setattr(r.codec, operation, fail)
    with pytest.raises(r.api.UnsupportedError):
        r.consumer.open_import(payload, spec, r.context)
    with pytest.raises(r.api.WeightSharingError, match="unknown session"):
        r.consumer.close_import(r.context.identity.session_id)
    assert not r.codec.imports


@pytest.mark.parametrize("operation", ["make_views", "weak_storage", "synchronize"])
def test_failure_after_rebuild_retains_imports(r, monkeypatch, operation):
    _, spec, exported = r.publish()
    payload = r.provider.claim_export(exported)
    original = getattr(r.codec, operation)

    def fail(*args):
        if r.codec.imports:
            raise RuntimeError("post-import failure")
        return original(*args)

    monkeypatch.setattr(r.codec, operation, fail)
    with pytest.raises(r.api.WeightSharingError) as error:
        r.consumer.open_import(payload, spec, r.context)
    assert error.value.poisoned
    snapshot = r.consumer.session_status(r.context.identity.session_id)
    assert snapshot.imported_storage_ids == ("storage-0",)
    assert r.consumer.close_import(snapshot.session_id).status == "POISONED"


def test_lease_close_failure_cannot_produce_release_receipt(r, monkeypatch):
    _, spec, exported = r.publish()
    _, imported = r.attach(spec, exported)

    def fail():
        raise RuntimeError("mapping close failed")

    monkeypatch.setattr(r.codec.leases[0], "close", fail)
    with pytest.raises(r.api.WeightSharingError) as error:
        r.consumer.close_import(imported)
    assert error.value.poisoned
    result = r.consumer.close_import(imported)
    assert result.status == "POISONED" and not result.local_complete and result.receipt is None
    assert r.provider.release_export(exported).status == "HELD"


@pytest.mark.parametrize(
    "operation", ["claim_export", "release_export", "confirm_import", "confirm_release", "mark_failed", "close_import"]
)
def test_mutations_require_owner_thread_but_status_does_not(r, operation):
    _, spec, exported = r.publish()
    if operation in ("claim_export", "release_export", "mark_failed"):
        target, argument = r.provider, exported
    else:
        _, imported = r.attach(spec, exported)
        target, argument = (r.consumer, imported) if operation == "close_import" else (r.provider, imported.receipt)
        if operation == "confirm_release":
            argument = r.consumer.close_import(imported).receipt
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(r.api.ThreadOwnershipError):
            executor.submit(getattr(target, operation), argument).result(timeout=3)
        snapshot = executor.submit(r.provider.session_status, exported).result(timeout=3)
    assert snapshot.state != r.api.SessionState.POISONED


def test_empty_storage_still_requires_explicit_peer_confirmation(r):
    spec = r.api.WeightSpec("model", "revision", "layout", (r.api.TensorSpec("empty", "float32", (0,)),))
    exported = r.provider.prepare_export({"empty": FakeTensor(FakeStorage(0), shape=(0,))}, spec, r.context)
    _, imported = r.attach(spec, exported)
    assert imported["empty"].storage.nbytes == 0 and not r.codec.exports
    assert r.provider.release_export(exported).status == "HELD"
    r.finish(exported, imported)


def test_oversized_export_can_be_cancelled_because_no_bundle_was_returned(r):
    _, _, exported = r.publish()
    r.provider._limits = r.api.Limits(max_bundle_bytes=1)
    with pytest.raises(r.api.ProtocolError, match="size limit"):
        r.provider.claim_export(exported)
    assert r.provider.session_status(exported).state == r.api.SessionState.EXPORTED
    assert r.provider.release_export(exported).status == "CLOSED"
    assert r.codec.cancellations == ["storage-0"]


@pytest.mark.parametrize("payload", [b"", b"invalid", b"\x81", b"x" * 2049])
def test_malformed_metadata_is_rejected_before_import(r, payload):
    _, spec = make_weights(r.api)
    r.consumer._limits = r.api.Limits(max_bundle_bytes=2048)
    with pytest.raises(r.api.ProtocolError):
        r.consumer.open_import(payload, spec, r.context)
    assert not r.codec.imports


@pytest.mark.parametrize(
    "changes", [{"handle": b""}, {"storage_nbytes": 1}, {"allocation_offset_bytes": -1}, {"ref_counter_handle": b""}]
)
def test_all_descriptors_validated_before_any_import(r, changes):
    _, spec, exported = r.publish(two_storages=True)
    bundle = msgspec.msgpack.decode(r.provider.claim_export(exported), type=r.api.ExportBundle)
    bad = msgspec.structs.replace(bundle.exports[1], **changes)
    payload = msgspec.msgpack.encode(msgspec.structs.replace(bundle, exports=(bundle.exports[0], bad)))
    with pytest.raises(r.api.CompatibilityError):
        r.consumer.open_import(payload, spec, r.context)
    assert not r.codec.imports


def _cpu_provider(control):
    from tests.distributed.weight_sharing.helpers import load_weight_sharing, publish_to_parent, release_from_parent

    runtime = Runtime(load_weight_sharing())
    _, _, exported = runtime.publish(two_storages=True)
    publish_to_parent(runtime.provider, exported, control)
    release_from_parent(runtime.provider, exported, control)


def _cpu_consumer(control):
    from tests.distributed.weight_sharing.helpers import load_weight_sharing, receive

    runtime = Runtime(load_weight_sharing())
    _, spec = make_weights(runtime.api, two_storages=True)
    # FakeCodec has no device memory: seed its local stand-in for the two handles.
    # The real CUDA tests use the same harness with actual reducer/rebuilder calls.
    runtime.codec.payloads = {f"storage-{i}".encode(): (32, list(range(8))) for i in range(2)}
    imported = runtime.consumer.open_import(receive(control), spec, runtime.context)
    control.send(imported.receipt)
    receive(control, "attached")
    assert imported["first"].storage.data == list(range(8))
    external = imported["second"]
    assert runtime.consumer.close_import(imported).status == "HELD"
    del external
    result = runtime.consumer.close_import(imported)
    assert result.local_complete and not result.peer_confirmed
    control.send(result.receipt)


def test_spawned_metadata_relay_without_omni_or_device_runtime():
    from tests.distributed.weight_sharing.helpers import relay_import, relay_release, spawn_workers

    with spawn_workers((_cpu_provider, ()), (_cpu_consumer, ())) as workers:
        (_, provider), (_, consumer) = workers
        relay_import(provider, consumer)
        relay_release(provider, consumer)


def test_provider_preflight_failure_remains_queryable_and_cancellable(r, monkeypatch):
    bindings, spec = make_weights(r.api)

    def fail(*args):
        raise r.api.UnsupportedError("unsupported layout")

    monkeypatch.setattr(r.codec, "inspect", fail)
    with pytest.raises(r.api.WeightSharingError) as error:
        r.provider.prepare_export(bindings, spec, r.context)
    assert not error.value.poisoned
    assert r.provider.session_status(error.value.session_id).error == "UnsupportedError"
    with pytest.raises(r.api.ProtocolError):
        r.provider.claim_export(error.value.session_id)
    assert r.provider.release_export(error.value.session_id).status == "CLOSED"
    assert not r.codec.exports and not r.codec.cancellations


def test_status_query_during_device_operation_does_not_hold_control_lock(r):
    snapshots = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        r.codec.before_export = lambda: snapshots.append(
            executor.submit(r.provider.session_status, r.context.identity.session_id).result(timeout=3)
        )
        _, _, exported = r.publish()
    assert snapshots[0].active_storage_id == "storage-0"
    assert r.provider.session_status(exported).active_storage_id is None
    r.provider.release_export(exported)


def test_failure_notification_does_not_erase_original_device_diagnostic(r):
    r.codec.failures["export"] = (1, RuntimeError("failed reducer"))
    with pytest.raises(r.api.WeightSharingError):
        r.publish()
    session_id = r.context.identity.session_id
    snapshot = r.provider.session_status(session_id)
    r.provider.mark_failed(session_id)
    assert r.provider.session_status(session_id) == snapshot
