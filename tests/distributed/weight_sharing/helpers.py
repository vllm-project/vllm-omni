# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load the independent backend without importing unrelated serving packages."""

from __future__ import annotations

import importlib.util
import multiprocessing
import sys
import threading
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest


def load_weight_sharing() -> ModuleType:
    """Import the actual package under a private, test-only namespace.

    The enclosing ``vllm_omni`` package initializes model configuration and the
    distributed package imports connectors. Those are intentionally outside
    this backend's CPU unit tests; no fake ``vllm_omni`` modules are installed.
    """
    name = "_weight_sharing_test_package"
    if name in sys.modules:
        return sys.modules[name]
    directory = Path(__file__).resolve().parents[3] / "vllm_omni" / "distributed" / "weight_sharing"
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for loaded_name in tuple(sys.modules):
            if loaded_name == name or loaded_name.startswith(f"{name}."):
                del sys.modules[loaded_name]
        raise
    return module


def make_tensor_specs(api, bindings, aliases=None):
    aliases = aliases or {}
    return tuple(
        api.TensorSpec(
            name,
            str(tensor.dtype).removeprefix("torch."),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            aliases.get(name),
        )
        for name, tensor in bindings.items()
    )


def make_context(api, **identity_changes):
    identity = {
        "group_id": "group",
        "generation": 1,
        "session_id": "session",
        "provider_id": "ar-rank-0",
        "consumer_id": "dit-rank-0",
        "rank": 0,
        "tp_size": 1,
        "device_uuid": "GPU-test",
    }
    identity.update(identity_changes)
    return api.SessionContext(api.Identity(**identity), 0)


@dataclass
class FakeStorage:
    """A CPU token with independent importer lifetime, not a CUDA emulator."""

    nbytes: int
    data: list[int] = field(default_factory=list)


@dataclass
class FakeTensor:
    storage: FakeStorage
    shape: tuple[int, ...] = (4,)
    stride: tuple[int, ...] = (1,)
    offset_bytes: int = 0
    dtype: str = "float32"


class FakeLease:
    def __init__(self, storage: FakeStorage):
        self.reference = weakref.ref(storage)
        self.closed = False

    def expired(self) -> bool:
        return self.reference() is None

    def close(self) -> None:
        self.closed = True


class FakeCodec:
    """Replace only GPU primitives while exercising the actual backend.

    The backend still owns its real locks, registry, batch transitions and
    receipt checks. Import creates a distinct lifetime token whose payload
    references the provider's data, so retaining a returned view prevents close.
    """

    def __init__(self, api):
        self.api = api
        self.calls: list[tuple[str, int]] = []
        self.exports: list[object] = []
        self.imports: list[str] = []
        self.cancellations: list[str] = []
        self.leases: list[FakeLease] = []
        self.payloads: dict[bytes, tuple[int, list[int]]] = {}
        self.failures: dict[str, tuple[int, BaseException]] = {}
        self.before_export: Callable[[], None] | None = None

    def _called(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def _fail(self, operation: str, attempt: int) -> None:
        failure = self.failures.get(operation)
        if failure is not None and failure[0] == attempt:
            raise failure[1]

    def inspect(self, bindings, spec, context):
        self._called("inspect")
        storage_ids: dict[int, str] = {}
        carriers = {}
        views = []
        for name, tensor in bindings.items():
            key = id(tensor.storage)
            if key not in storage_ids:
                storage_ids[key] = f"storage-{len(storage_ids)}"
                carriers[storage_ids[key]] = tensor.storage
            views.append(
                self.api.TensorViewSpec(
                    name,
                    storage_ids[key],
                    tensor.dtype,
                    tensor.shape,
                    tensor.stride,
                    tensor.offset_bytes,
                )
            )
        manifest = self.api.WeightManifest(
            spec,
            tuple(self.api.StorageSpec(key, value.nbytes) for key, value in carriers.items()),
            tuple(views),
        )
        return self.api.PreparedWeights(manifest, carriers, tuple(bindings.values()))

    def synchronize(self, context):
        self._called("synchronize")

    def validate_runtime(self, context):
        self._called("validate_runtime")

    def export_storage(self, carrier, storage_id):
        self._called("export_storage")
        if self.before_export is not None:
            self.before_export()
        self._fail("export", len(self.exports) + 1)
        handle = storage_id.encode()
        self.payloads[handle] = (carrier.nbytes, carrier.data)
        descriptor = self.api.StorageExport(
            storage_id, "test-cpu", 0, handle, carrier.nbytes, 0, b"counter", 0, b"", False
        )
        self.exports.append(descriptor)
        return descriptor

    def import_storage(self, descriptor, device_index):
        self._called("import_storage")
        self.imports.append(descriptor.storage_id)
        self._fail("import", len(self.imports))
        nbytes, data = self.payloads[descriptor.handle]
        return FakeStorage(nbytes, data)

    def empty_storage(self, device_index):
        self._called("empty_storage")
        return FakeStorage(0)

    def make_views(self, manifest, carriers):
        self._called("make_views")
        return {
            view.name: FakeTensor(carriers[view.storage_id], view.shape, view.stride, view.offset_bytes, view.dtype)
            for view in manifest.views
        }

    def weak_storage(self, carrier):
        self._called("weak_storage")
        lease = FakeLease(carrier)
        self.leases.append(lease)
        return lease

    def cancel_unconsumed(self, descriptor, device_index):
        self._called("cancel_unconsumed")
        self.cancellations.append(descriptor.storage_id)
        self._fail("cancel", len(self.cancellations))


def make_weights(api, *, two_storages: bool = False):
    storage = FakeStorage(32, list(range(8)))
    second = FakeStorage(32, list(range(8))) if two_storages else storage
    bindings = {
        "first": FakeTensor(storage),
        "second": FakeTensor(second, offset_bytes=16),
        "first_alias": FakeTensor(storage),
    }
    spec = api.WeightSpec(
        "test-model",
        "revision-1",
        "tp1-final",
        (
            api.TensorSpec("first", "float32", (4,), (1,)),
            api.TensorSpec("second", "float32", (4,), (1,)),
            api.TensorSpec("first_alias", "float32", (4,), (1,), alias_of="first"),
        ),
    )
    return bindings, spec


def receive(pipe, expected=None, timeout=45):
    assert pipe.poll(timeout), f"worker timed out waiting for {expected!r}"
    message = pipe.recv()
    if expected is not None:
        assert message == expected
    return message


def publish_to_parent(backend, exported, control):
    """Device owners expose bytes only; the parent relays the control plane."""
    control.send(backend.claim_export(exported))
    backend.confirm_import(receive(control))
    control.send("attached")


def release_from_parent(backend, exported, control):
    backend.confirm_release(receive(control))
    assert backend.release_export(exported).status == "CLOSED"
    control.send("closed")


def relay_import(provider, consumer):
    consumer.send(receive(provider))
    provider.send(receive(consumer))
    receive(provider, "attached")
    consumer.send("attached")


def relay_release(provider, consumer):
    provider.send(receive(consumer))
    receive(provider, "closed")


@contextmanager
def spawn_workers(*workers):
    """Start (target, args[, env]) workers, appending their control pipe argument."""
    context = multiprocessing.get_context("spawn")
    running = []
    try:
        for worker in workers:
            target, args = worker[:2]
            parent, child = context.Pipe()
            process = context.Process(target=target, args=(*args, child))
            running.append((process, parent))
            try:
                with pytest.MonkeyPatch.context() as patch:
                    for key, value in (worker[2] if len(worker) == 3 else {}).items():
                        patch.setenv(key, value)
                    process.start()
            finally:
                child.close()
        yield running
        for process, _ in running:
            process.join(timeout=45)
            assert process.exitcode == 0, f"{process.name}: exit={process.exitcode}"
    finally:
        survivors = []
        # These tests start providers before their consumers. Preserve ownership
        # ordering even when an assertion or communication timeout interrupts them.
        for process, pipe in reversed(running):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if process.is_alive():
                    survivors.append(process.name)
                else:
                    process.close()
            pipe.close()
        assert not survivors, f"workers survived cleanup: {survivors}"
