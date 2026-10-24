# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib
import struct
import threading
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType

import msgspec
import pytest

from tests.distributed.weight_sharing.helpers import (
    FakeCodec,
    FakeStorage,
    FakeTensor,
    SocketPairTransport,
    make_context,
    make_weights,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Runtime:
    def __init__(self, package: ModuleType):
        self.api = importlib.import_module(f"{package.__name__}.contracts")
        self.backend = importlib.import_module(f"{package.__name__}.backend")
        self.channel = importlib.import_module(f"{package.__name__}.channel")
        self.transport = SocketPairTransport(self.channel)
        self.codec = FakeCodec(self.api)
        limits = self.channel.Limits(timeout=3)
        self.provider = self.backend.TorchCudaIpcBackend(codec=self.codec, limits=limits)
        self.consumer = self.backend.TorchCudaIpcBackend(codec=self.codec, limits=limits)
        self.context = make_context(self.api)

    def publish(self, *, two_storages: bool = False):
        bindings, spec = make_weights(self.api, two_storages=two_storages)
        return bindings, spec, self.provider.prepare_export(bindings, spec, self.context)

    def assert_poisoned_pair(self, exported):
        assert self.consumer.close_import(self.context.identity.session_id).status == "POISONED"
        assert self.transport.servers[self.context.endpoint].handled.wait(timeout=3)
        assert self.provider.release_export(exported).status == "POISONED"
        assert not self.codec.cancellations

    @contextmanager
    def failing_operation(self, operation, error_type):
        error = (error_type or self.api.WeightSharingError)("injected storage operation failure")
        self.codec.failures[operation] = (2, error)
        expected = self.api.WeightSharingError if isinstance(error, Exception) else type(error)
        with pytest.raises(expected) as caught:
            yield
        if isinstance(error, Exception):
            assert caught.value.__cause__ is error
            assert caught.value.session_id == self.context.identity.session_id
            assert caught.value.poisoned
        else:
            assert caught.value is error


@pytest.fixture
def r(weight_sharing: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Iterator[Runtime]:
    runtime = Runtime(weight_sharing)
    monkeypatch.setattr(runtime.backend, "SessionServer", runtime.transport.server)
    monkeypatch.setattr(runtime.backend, "connect", runtime.transport.connect)
    yield runtime
    runtime.transport.close()


def test_unique_storage_export_and_unexposed_release(r):
    bindings, spec = make_weights(r.api)
    owner = threading.get_ident()
    exported = r.provider.prepare_export(bindings, spec, r.context)
    snapshot = r.provider.session_status(exported)
    assert snapshot.state == r.api.SessionState.EXPORTED
    assert (snapshot.storage_count, snapshot.view_count, snapshot.shared_nbytes) == (1, 3, 32)
    assert len(r.codec.exports) == 1
    assert r.provider.release_export(exported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"
    assert snapshot.exported_storage_ids == ("storage-0",)
    assert snapshot.imported_storage_ids == ()
    assert snapshot.active_storage_id is None
    assert r.codec.cancellations == ["storage-0"]
    assert {thread for _, thread in r.codec.calls} == {owner}


def test_roundtrip_preserves_views_and_close_is_idempotent(r):
    bindings, spec = make_weights(r.api)
    replies = []
    r.transport.before_send = lambda message: replies.append(message.op)
    exported = r.provider.prepare_export(bindings, spec, r.context)
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    server = r.transport.servers[r.context.endpoint]
    assert r.provider.session_status(exported).error == ""
    assert r.consumer.session_status(imported).error == ""
    assert imported["first"].storage is imported["second"].storage
    assert imported["first"].storage is imported["first_alias"].storage
    assert imported["second"].offset_bytes == 16
    assert imported["first"].storage.data is bindings["first"].storage.data
    assert r.provider.release_export(exported).status == "HELD"
    assert r.consumer.close_import(imported).status == "CLOSED"
    assert r.consumer.close_import(imported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"
    assert len(server.connections) == 1
    assert replies == ["manifest", "bundle", "accepted", "released"]
    assert r.provider.session_status(exported).error == ""
    assert not r.codec.cancellations
    provider_status = r.provider.session_status(exported)
    consumer_status = r.consumer.session_status(imported)
    assert provider_status.state == consumer_status.state == r.api.SessionState.CLOSED
    assert provider_status.exported_storage_ids == consumer_status.exported_storage_ids == ("storage-0",)
    assert consumer_status.imported_storage_ids == ()
    assert provider_status.active_storage_id is consumer_status.active_storage_id is None
    assert [name for name, _ in r.codec.calls].count("validate_runtime") == 1
    assert {thread for _, thread in r.codec.calls} == {threading.get_ident()}


def test_consumer_runtime_rejection_precedes_connection_and_cuda_operations(r, monkeypatch):
    bindings, spec, exported = r.publish()
    r.codec.calls.clear()

    def reject_runtime(context):
        r.codec._called("validate_runtime")
        raise r.api.UnsupportedError("injected unsupported runtime")

    monkeypatch.setattr(r.codec, "validate_runtime", reject_runtime)
    with pytest.raises(r.api.WeightSharingError, match="UnsupportedError"):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert r.codec.calls == [("validate_runtime", threading.get_ident())]
    assert not r.transport.servers[r.context.endpoint].connections
    assert r.provider.release_export(exported).status == "CLOSED"


def test_import_identity_cannot_be_reopened(r):
    bindings, spec, exported = r.publish()
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    server = r.transport.servers[r.context.endpoint]
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert len(server.connections) == len(r.codec.imports) == 1
    assert r.consumer.close_import(imported).status == "CLOSED"
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert len(server.connections) == len(r.codec.imports) == 1
    assert r.provider.release_export(exported).status == "CLOSED"


def test_provider_shutdown_waits_until_release_ack_is_sent(r, monkeypatch):
    bindings, spec, exported = r.publish()
    server = r.transport.servers[r.context.endpoint]
    control = r.provider._get(exported).control
    ack_blocked = threading.Event()
    owner_waiting = threading.Event()
    allow_ack = threading.Event()
    original_wait = control.release_complete.wait
    original_stop = server.stop

    def delay_release_ack(message):
        if message.op == "released":
            ack_blocked.set()
            if not allow_ack.wait(timeout=3):
                raise r.channel.ChannelError("test did not allow release acknowledgement")

    def observe_owner_wait(timeout=None):
        owner_waiting.set()
        return original_wait(timeout)

    def stop_after_ack():
        assert control.release_complete.is_set(), "Server stopped during release acknowledgement"
        assert control.peer_released
        original_stop()

    def import_and_close():
        imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
        return r.consumer.close_import(imported)

    def unblock_after_owner_waits():
        assert owner_waiting.wait(timeout=3), "Provider did not wait for release acknowledgement"
        allow_ack.set()

    r.transport.before_send = delay_release_ack
    with monkeypatch.context() as patch:
        patch.setattr(control.release_complete, "wait", observe_owner_wait)
        patch.setattr(server, "stop", stop_after_ack)
        with ThreadPoolExecutor(max_workers=2) as executor:
            consumer_result = executor.submit(import_and_close)
            try:
                assert ack_blocked.wait(timeout=3), "Consumer did not request release"
                unblocker = executor.submit(unblock_after_owner_waits)
                assert r.provider.release_export(exported).status == "CLOSED"
                unblocker.result(timeout=3)
                assert consumer_result.result(timeout=3).status == "CLOSED"
            finally:
                allow_ack.set()
    assert owner_waiting.is_set()
    assert r.provider.session_status(exported).state == r.api.SessionState.CLOSED
    assert r.consumer.session_status(r.context.identity.session_id).state == r.api.SessionState.CLOSED


def test_close_waits_for_external_view_reference(r):
    bindings, spec, exported = r.publish()
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    held_view = imported["second"]
    result = r.consumer.close_import(imported)
    assert result.status == "HELD"
    assert not result.local_complete
    assert r.provider.release_export(exported).status == "HELD"
    del held_view
    assert r.consumer.close_import(imported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"
    assert all(lease.closed for lease in r.codec.leases)


def test_only_owner_can_release_but_other_threads_can_read_status(r):
    bindings, spec, exported = r.publish()
    with ThreadPoolExecutor(max_workers=1) as executor:
        snapshot = executor.submit(r.provider.session_status, exported).result(timeout=3)
        assert snapshot.state == r.api.SessionState.EXPORTED
        with pytest.raises(r.api.ThreadOwnershipError):
            executor.submit(r.provider.release_export, exported).result(timeout=3)
    assert not r.codec.cancellations
    assert r.provider.release_export(exported).status == "CLOSED"


def test_export_does_not_hold_state_lock_during_codec_call(r):
    bindings, spec = make_weights(r.api)
    snapshots = []
    completed = threading.Event()

    def read_snapshot():
        snapshots.append(r.provider.session_status(r.context.identity.session_id))
        completed.set()

    def before_export():
        reader = threading.Thread(target=read_snapshot, daemon=True)
        reader.start()
        assert completed.wait(timeout=3), "Status lock was held across the reducer call"
        reader.join(timeout=3)

    r.codec.before_export = before_export
    exported = r.provider.prepare_export(bindings, spec, r.context)
    assert snapshots[0].state == r.api.SessionState.PREPARING
    assert snapshots[0].active_storage_id == "storage-0"
    assert r.provider.release_export(exported).status == "CLOSED"


@pytest.mark.parametrize("failure_mode", ["send", "truncated", "invalid", "handle", "device"])
def test_failed_bundle_poisoned_without_retry_or_cancellation(r, failure_mode):
    bindings, spec = make_weights(r.api, two_storages=True)
    source_ref = weakref.ref(bindings["first"].storage)
    exported = r.provider.prepare_export(bindings, spec, r.context)
    del bindings
    exposed = []
    server = r.transport.servers[r.context.endpoint]

    def fail_bundle(message):
        if message.op != "bundle":
            return None
        exposed.append(r.provider.session_status(exported))
        changes: dict[str, dict[str, object]] = {
            "invalid": {"storage_nbytes": 999},
            "handle": {"handle": b""},
            "device": {"device_index": -1},
        }
        if failure_mode in changes:
            descriptor = msgspec.structs.replace(message.exports[1], **changes[failure_mode])
            return msgspec.structs.replace(message, exports=(message.exports[0], descriptor))
        if failure_mode == "truncated":
            # A valid header followed by an incomplete MessagePack payload.
            server.connections[-1]._socket.sendall(struct.pack("!I", 128) + b"\x81")
            server.connections[-1].close()
        raise r.channel.ChannelError("injected bundle delivery failure")

    r.transport.before_send = fail_bundle
    with pytest.raises(r.api.WeightSharingError) as failure:
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert failure.value.poisoned
    assert len(exposed) == 1
    assert exposed[0].exported_storage_ids == ("storage-0", "storage-1")
    assert exposed[0].active_storage_id is None
    assert not r.codec.imports
    r.assert_poisoned_pair(exported)
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert len(server.connections) == 1
    assert source_ref() is not None


def test_lost_import_ack_poisoned_and_rebuild_is_not_repeated(r):
    bindings, spec, exported = r.publish()
    server = r.transport.servers[r.context.endpoint]

    def lose_ack(message):
        if message.op == "accepted":
            raise r.channel.ChannelError("injected lost import acknowledgement")

    r.transport.before_send = lose_ack
    with pytest.raises(r.api.WeightSharingError) as failure:
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert failure.value.poisoned
    assert server.handled.wait(timeout=3)
    assert r.consumer.session_status(r.context.identity.session_id).state == r.api.SessionState.POISONED
    assert len(r.codec.imports) == 1
    assert all(not lease.expired() for lease in r.codec.leases)
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert len(server.connections) == len(r.codec.imports) == 1
    r.assert_poisoned_pair(exported)


def test_lost_release_ack_poisoned_without_reconnecting(r):
    bindings, spec, exported = r.publish()
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    server = r.transport.servers[r.context.endpoint]

    def lose_ack(message):
        if message.op == "released":
            raise r.channel.ChannelError("injected lost release acknowledgement")

    r.transport.before_send = lose_ack
    with pytest.raises(r.api.WeightSharingError) as failure:
        r.consumer.close_import(imported)
    assert failure.value.poisoned
    assert server.handled.wait(timeout=3)
    calls = list(r.codec.calls)
    assert r.consumer.close_import(imported).status == "POISONED"
    assert r.codec.calls == calls
    assert len(server.connections) == 1
    assert r.provider.release_export(exported).status == "POISONED"


def test_disconnect_is_not_a_release_and_cannot_be_resumed(r):
    bindings, spec = make_weights(r.api)
    source_ref = weakref.ref(bindings["first"].storage)
    exported = r.provider.prepare_export(bindings, spec, r.context)
    del bindings
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    server = r.transport.servers[r.context.endpoint]
    server.connections[0].close()
    assert server.handled.wait(timeout=3)
    assert r.provider.session_status(exported).state == r.api.SessionState.POISONED
    assert r.provider.release_export(exported).status == "POISONED"
    assert source_ref() is not None
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    with pytest.raises(r.api.WeightSharingError) as failure:
        r.consumer.close_import(imported)
    assert failure.value.poisoned
    assert len(server.connections) == 1
    assert not r.codec.cancellations
    replacement = r.backend.TorchCudaIpcBackend(codec=r.codec)
    with pytest.raises(r.api.WeightSharingError) as rejected:
        replacement.open_import(r.context.endpoint, spec, r.context)
    assert rejected.value.poisoned
    assert len(r.codec.imports) == 1
    assert r.provider.session_status(exported).state == r.api.SessionState.POISONED


def test_attached_session_can_idle_beyond_operation_timeout(r):
    limits = r.channel.Limits(timeout=0.2)
    r.provider._limits = r.consumer._limits = limits
    bindings, spec, exported = r.publish()
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    server = r.transport.servers[r.context.endpoint]
    assert not server.handled.wait(timeout=0.4), "Healthy idle session was disconnected"
    assert r.provider.session_status(exported).state == r.api.SessionState.ATTACHED
    assert r.consumer.close_import(imported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"


@pytest.mark.parametrize("mismatch", ["generation", "device_uuid", "rank", "token", "spec"])
def test_handshake_mismatch_never_imports_storage(r, mismatch):
    bindings, spec, exported = r.publish()
    context = r.context
    expected = spec
    if mismatch == "generation":
        context = make_context(r.api, generation=2)
    elif mismatch == "device_uuid":
        context = make_context(r.api, device_uuid="GPU-other")
    elif mismatch == "rank":
        context = make_context(r.api, rank=1, tp_size=2)
    elif mismatch == "token":
        context = replace(context, token=b"invalid-token-long-enough")
    else:
        expected = msgspec.structs.replace(spec, checkpoint_revision="other-revision")
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(context.endpoint, expected, context)
    assert r.transport.servers[r.context.endpoint].handled.wait(timeout=3)
    assert not r.codec.imports
    assert r.provider.session_status(exported).exported_storage_ids == ("storage-0",)
    if mismatch == "spec":
        # Authentication already bound the session before manifest checking.
        assert r.provider.release_export(exported).status == "POISONED"
        assert not r.codec.cancellations
    else:
        assert r.provider.session_status(exported).state == r.api.SessionState.EXPORTED
        assert r.provider.release_export(exported).status == "CLOSED"
        assert len(r.codec.cancellations) == 1


def test_empty_storage_has_no_export_descriptor_but_still_requires_peer_release(r):
    bindings = {"empty": FakeTensor(FakeStorage(0), shape=(0,))}
    spec = r.api.WeightSpec("model", "revision", "layout", (r.api.TensorSpec("empty", "float32", (0,)),))
    exported = r.provider.prepare_export(bindings, spec, r.context)
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert imported["empty"].storage.nbytes == 0
    assert not r.codec.exports
    snapshot = r.consumer.session_status(imported)
    assert snapshot.exported_storage_ids == ()
    assert snapshot.imported_storage_ids == ("storage-0",)
    assert r.provider.release_export(exported).status == "HELD"
    assert r.consumer.close_import(imported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"


def test_listener_startup_failure_is_queryable_and_unexposed_counters_cancel(r, monkeypatch):
    bindings, spec = make_weights(r.api)
    original = r.transport.server

    def cannot_listen(*args):
        server = original(*args)

        def fail_start():
            raise r.channel.ChannelError("injected listener startup failure")

        server.start = fail_start
        return server

    monkeypatch.setattr(r.backend, "SessionServer", cannot_listen)
    with pytest.raises(r.api.WeightSharingError) as failure:
        r.provider.prepare_export(bindings, spec, r.context)
    assert failure.value.session_id == r.context.identity.session_id
    assert r.provider.release_export(failure.value.session_id).status == "CLOSED"
    assert len(r.codec.cancellations) == 1


def test_close_from_wrong_thread_does_not_invalidate_import(r):
    bindings, spec, exported = r.publish()
    imported = r.consumer.open_import(r.context.endpoint, spec, r.context)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(r.api.ThreadOwnershipError):
            executor.submit(r.consumer.close_import, imported).result(timeout=3)
    assert imported["first"].shape == (4,)
    assert r.consumer.close_import(imported).status == "CLOSED"
    assert r.provider.release_export(exported).status == "CLOSED"


@pytest.mark.parametrize("error_type", [None, KeyboardInterrupt, SystemExit], ids=["error", "interrupt", "exit"])
def test_partial_export_failure_is_queryable_and_cannot_retry(r, error_type):
    bindings, spec = make_weights(r.api, two_storages=True)
    with r.failing_operation("export", error_type):
        r.provider.prepare_export(bindings, spec, r.context)
    session_id = r.context.identity.session_id
    snapshot = r.provider.session_status(session_id)
    assert snapshot.state == r.api.SessionState.POISONED
    assert snapshot.exported_storage_ids == ("storage-0",)
    assert snapshot.imported_storage_ids == ()
    assert snapshot.active_storage_id == "storage-1"
    # A throwing reducer may have installed another counter before failing.
    assert r.provider.release_export(session_id).status == "POISONED"
    assert len(r.codec.exports) == 1
    assert not r.codec.cancellations
    with pytest.raises(r.api.WeightSharingError):
        r.provider.prepare_export(bindings, spec, r.context)
    assert len(r.codec.exports) == 1


@pytest.mark.parametrize("error_type", [None, KeyboardInterrupt, SystemExit], ids=["error", "interrupt", "exit"])
def test_partial_import_failure_retains_resources_and_cannot_reconsume(r, error_type):
    bindings, spec, exported = r.publish(two_storages=True)
    source_refs = tuple(weakref.ref(tensor.storage) for tensor in bindings.values())
    del bindings
    with r.failing_operation("import", error_type):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    snapshot = r.consumer.session_status(r.context.identity.session_id)
    assert snapshot.state == r.api.SessionState.POISONED
    assert snapshot.exported_storage_ids == ("storage-0", "storage-1")
    assert snapshot.imported_storage_ids == ("storage-0",)
    assert snapshot.active_storage_id == "storage-1"
    assert len(r.codec.imports) == 2
    assert all(not lease.expired() for lease in r.codec.leases)
    with pytest.raises(r.api.WeightSharingError):
        r.consumer.open_import(r.context.endpoint, spec, r.context)
    assert len(r.codec.imports) == 2
    # A disconnect cannot establish whether the failed rebuild used its counter.
    r.assert_poisoned_pair(exported)
    assert all(reference() is not None for reference in source_refs)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_cancel_interrupt_is_not_repeated(r, error_type):
    bindings, spec, exported = r.publish(two_storages=True)
    with r.failing_operation("cancel", error_type):
        r.provider.release_export(exported)
    snapshot = r.provider.session_status(exported)
    assert snapshot.state == r.api.SessionState.POISONED
    assert snapshot.exported_storage_ids == ("storage-0", "storage-1")
    assert snapshot.active_storage_id == "storage-1"
    assert len(r.codec.cancellations) == 2
    assert r.provider.release_export(exported).status == "POISONED"
    assert len(r.codec.cancellations) == 2
