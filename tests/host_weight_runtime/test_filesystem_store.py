# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lifecycle and concurrency tests for the local filesystem store."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import threading
import time
import warnings
from collections.abc import Sequence
from dataclasses import replace
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest
import torch

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    ArtifactInventoryState,
    BuildRequest,
    CapacityPolicy,
    ComponentIdentity,
    CoordinationScope,
    FailureCode,
    HostWeightError,
    HostWeightLease,
    IntegrityPolicy,
    ProducerIdentity,
    ProductionMetadata,
    ProductionSourceMode,
    ResolutionStage,
    RuntimeWeightLayout,
    StorageClass,
    StorageDomainPolicy,
    StorageScope,
    StoreResult,
    StoreStatus,
    TensorKind,
    TensorWriteSpec,
    ValidationLevel,
    WeightArtifactIdentity,
    WeightProductionSpec,
    WeightRepresentation,
    WeightSourceIdentity,
)
from vllm_omni.host_weight_runtime.filesystem import (
    FilesystemHostWeightStore,
    detect_filesystem_type,
    detect_storage_class,
)
from vllm_omni.host_weight_runtime.filesystem import store as filesystem_store_module
from vllm_omni.host_weight_runtime.filesystem.locks import FileLock

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Spawning imports the complete vLLM package and exceeded 10 seconds on a loaded
# CI worker. Keep process-bootstrap bounds separate from the short store
# deadlines exercised by these tests.
_PROCESS_START_TIMEOUT_SECONDS = 60.0
_PROCESS_COMPLETION_TIMEOUT_SECONDS = 60.0
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 5.0


def _identity(*, tp_rank: int = 0) -> WeightArtifactIdentity:
    return WeightArtifactIdentity(
        schema_version=1,
        source=WeightSourceIdentity("test-org/test-model", "0123456789abcdef", "source-files-sha256"),
        component=ComponentIdentity("transformer", "diffusion.dit"),
        representation=WeightRepresentation("runtime-bf16", "torch.bfloat16"),
        layout=RuntimeWeightLayout(
            "final-module-layout",
            tensor_parallel_size=2,
            tensor_parallel_rank=tp_rank,
        ),
        adaptation=AdaptationIdentity(),
        producer=ProducerIdentity(
            "test.final-layout",
            "1",
            "implementation-sha256",
            "test-producer-v1",
            "test-restorer-v1",
        ),
    )


def _weight(offset: int = 0) -> torch.Tensor:
    return torch.arange(offset, offset + 12, dtype=torch.float32).to(torch.bfloat16).reshape(4, 3)


def _specs() -> tuple[TensorWriteSpec, ...]:
    return (
        TensorWriteSpec("layer.weight", (4, 3), torch.bfloat16),
        TensorWriteSpec("layer.scale", (2,), torch.float32, kind=TensorKind.BUFFER, role="scale"),
    )


def _open_fd_targets() -> tuple[str, ...]:
    targets = []
    for path in Path("/proc/self/fd").iterdir():
        with contextlib.suppress(OSError):
            targets.append(os.readlink(path))
    return tuple(targets)


class FakeProducer:
    def __init__(
        self,
        identity: WeightArtifactIdentity,
        *,
        counter_path: Path | None = None,
        delay_seconds: float = 0.0,
        started: object | None = None,
        release: object | None = None,
        write_mode: str = "rows",
        weight_offset: int = 0,
        coordination_scope: CoordinationScope = CoordinationScope.SINGLE_PROCESS,
    ) -> None:
        self._spec = WeightProductionSpec(
            producer_id="test.final-layout",
            outputs=(identity,),
            source_mode=ProductionSourceMode.FINALIZED_MODEL,
            coordination_scope=coordination_scope,
        )
        self.counter_path = counter_path
        self.delay_seconds = delay_seconds
        self.started = started
        self.release = release
        self.write_mode = write_mode
        self.weight_offset = weight_offset

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: object) -> ProductionMetadata:
        if self.counter_path is not None:
            fd = os.open(self.counter_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_CLOEXEC, 0o600)
            try:
                os.write(fd, f"{os.getpid()}\n".encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        if self.started is not None:
            self.started.set()  # type: ignore[attr-defined]
        if self.release is not None and not self.release.wait(timeout=10):  # type: ignore[attr-defined]
            raise TimeoutError("test producer was not released")
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

        with writer.open_tensor_file("weights.safetensors", _specs()) as output:  # type: ignore[attr-defined]
            if self.write_mode == "rows":
                output.write_rows("layer.weight", 0, _weight(self.weight_offset)[:2])
                output.write_rows("layer.weight", 2, _weight(self.weight_offset)[2:])
            elif self.write_mode == "complete":
                output.write_tensor("layer.weight", _weight(self.weight_offset))
            elif self.write_mode == "incomplete":
                output.write_rows("layer.weight", 0, _weight(self.weight_offset)[:2])
            elif self.write_mode == "overlap":
                output.write_rows("layer.weight", 0, _weight(self.weight_offset)[:3])
                output.write_rows("layer.weight", 2, _weight(self.weight_offset)[2:])
            else:  # pragma: no cover - test helper guard
                raise AssertionError(f"unknown write mode {self.write_mode}")
            output.write_tensor("layer.scale", torch.tensor([1.0, 2.0]))
        return ProductionMetadata("test-producer-v1", "test-restorer-v1")


class CrashProducer(FakeProducer):
    def produce(self, writer: object) -> ProductionMetadata:
        output = writer.open_tensor_file("weights.safetensors", _specs())  # type: ignore[attr-defined]
        output.write_rows("layer.weight", 0, _weight()[:2])
        os._exit(17)


class NonCooperativeStore(FilesystemHostWeightStore):
    """Test double representing a publisher that does not join the build lock."""

    def _build_lock_path(self, key: str) -> Path:
        return self.locks_dir / f"{key}.noncooperative-build.lock"

    def _remove_stale_temps_locked(self, _key: str) -> None:
        return


class InitialLookupBarrierStore(FilesystemHostWeightStore):
    """Make both test workers observe the initial miss before either builds."""

    def __init__(
        self,
        domain: StorageDomainPolicy,
        capacity: CapacityPolicy,
        integrity: IntegrityPolicy,
        *,
        initial_lookup_barrier: object,
    ) -> None:
        super().__init__(domain, capacity, integrity)
        self._initial_lookup_barrier = initial_lookup_barrier
        self._lookup_count = 0

    def lookup(
        self,
        identity: WeightArtifactIdentity,
        *,
        validation: ValidationLevel,
        deadline: float | None = None,
    ) -> StoreResult:
        result = super().lookup(identity, validation=validation, deadline=deadline)
        self._lookup_count += 1
        if self._lookup_count == 1:
            self._initial_lookup_barrier.wait(timeout=10)  # type: ignore[attr-defined]
        return result


class PausingOpenLeaseStore(FilesystemHostWeightStore):
    """Pause after opening a lease so another lookup can publish a deny marker."""

    def __init__(
        self,
        domain: StorageDomainPolicy,
        capacity: CapacityPolicy,
        integrity: IntegrityPolicy,
        *,
        lease_opened: threading.Event,
        resume_lookup: threading.Event,
    ) -> None:
        super().__init__(domain, capacity, integrity)
        self._lease_opened = lease_opened
        self._resume_lookup = resume_lookup

    def _open_lease(
        self,
        identity: WeightArtifactIdentity,
        entry_path: Path,
        artifact_lock: FileLock,
        validation: ValidationLevel,
    ) -> HostWeightLease:
        lease = super()._open_lease(identity, entry_path, artifact_lock, validation)
        self._lease_opened.set()
        if not self._resume_lookup.wait(5):
            lease.close()
            raise TimeoutError("test did not resume paused lookup")
        return lease


def _make_store(
    root: Path,
    *,
    capacity: CapacityPolicy | None = None,
    integrity: IntegrityPolicy | None = None,
) -> FilesystemHostWeightStore:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    domain = StorageDomainPolicy(root=root, storage_class=detect_storage_class(root))
    return FilesystemHostWeightStore(domain, capacity or CapacityPolicy(), integrity or IntegrityPolicy())


def _build_process(root: str, counter: str, initial_lookup_barrier: object, result_queue: object) -> None:
    try:
        identity = _identity()
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        store = InitialLookupBarrierStore(
            StorageDomainPolicy(root=root_path, storage_class=detect_storage_class(root_path)),
            CapacityPolicy(),
            IntegrityPolicy(),
            initial_lookup_barrier=initial_lookup_barrier,
        )
        result = store.get_or_build(
            BuildRequest(identity),
            FakeProducer(identity, counter_path=Path(counter), delay_seconds=0.25),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 15,
        )
        fingerprint = result.lease.provenance.backing_fingerprint if result.lease is not None else None
        if result.lease is not None:
            result.lease.close()
        result_queue.put((result.status.value, fingerprint, None))  # type: ignore[attr-defined]
    except BaseException as exc:
        result_queue.put(("error", None, repr(exc)))  # type: ignore[attr-defined]


def _controlled_build_process(root: str, started: object, release: object, result_queue: object) -> None:
    try:
        identity = _identity()
        store = _make_store(Path(root))
        result = store.get_or_build(
            BuildRequest(identity),
            FakeProducer(identity, started=started, release=release),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 15,
        )
        if result.lease is not None:
            result.lease.close()
        result_queue.put((result.status.value, None))  # type: ignore[attr-defined]
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))  # type: ignore[attr-defined]


def _crash_build_process(root: str) -> None:
    identity = _identity()
    store = _make_store(Path(root))
    store.get_or_build(
        BuildRequest(identity),
        CrashProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 15,
    )


def _join_processes(
    processes: Sequence[BaseProcess],
    timeout: float = _PROCESS_COMPLETION_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    timed_out = [process for process in processes if process.is_alive()]
    if not timed_out:
        return
    timed_out_workers = ", ".join(f"{process.name}(pid={process.pid})" for process in timed_out)
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(_PROCESS_CLEANUP_TIMEOUT_SECONDS)
    survivors = [process for process in timed_out if process.is_alive()]
    for process in survivors:
        process.kill()
        process.join(_PROCESS_CLEANUP_TIMEOUT_SECONDS)
    unstoppable = [process for process in survivors if process.is_alive()]
    assert not unstoppable, "timed-out test worker processes could not be stopped"
    pytest.fail(f"test worker processes exceeded the {timeout:g}-second group deadline: {timed_out_workers}")


def test_build_lookup_share_one_published_backing_and_lease_blocks_cleanup(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")

    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.status is StoreStatus.BUILT
    assert built.lease is not None
    assert torch.equal(built.lease.tensors["layer.weight"], _weight())
    assert torch.equal(built.lease.tensors["layer.scale"], torch.tensor([1.0, 2.0]))
    assert built.lease.mapped_regions
    assert all(region.address > 0 and region.length > 0 and region.read_only for region in built.lease.mapped_regions)
    with pytest.raises(TypeError, match="process-local"):
        built.lease.__reduce_ex__(4)

    second = store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM)
    assert second.status is StoreStatus.HIT
    assert second.lease is not None
    assert second.lease.provenance.publication_id == built.lease.provenance.publication_id
    assert second.lease.provenance.backing_fingerprint == built.lease.provenance.backing_fingerprint
    payload = store.artifacts_dir / identity.key / "weights.safetensors"
    assert any(str(payload) in target for target in _open_fd_targets())

    artifact = store.inspect_artifact(identity, validation=ValidationLevel.FULL_CHECKSUM)
    assert artifact.status is StoreStatus.HIT
    assert artifact.manifest is not None and artifact.manifest.identity == identity
    assert artifact.publication_id == built.lease.provenance.publication_id
    assert artifact.artifact_lock_active
    domain = store.inspect_domain()
    assert domain.domain_uuid == built.lease.provenance.domain_uuid
    assert domain.filesystem_type == detect_filesystem_type(store.root)
    assert any(
        item.artifact_key == identity.key and item.state is ArtifactInventoryState.READY and item.size_bytes > 0
        for item in domain.inventory
    )

    active = store.cleanup(identity)
    assert active is not None and active.code is FailureCode.ACTIVE_LEASE
    second.lease.close()
    built.lease.close()
    built.lease.close()
    assert not built.lease.tensors
    assert not any(str(payload) in target for target in _open_fd_targets())

    assert store.cleanup(identity) is None
    assert store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM).status is StoreStatus.MISS


def test_concurrent_lease_close_waits_for_resource_teardown(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None
    lease = built.lease
    teardown_started = threading.Event()
    allow_teardown = threading.Event()
    second_started = threading.Event()
    second_returned = threading.Event()
    errors: list[BaseException] = []

    def blocking_closer() -> None:
        teardown_started.set()
        if not allow_teardown.wait(5):
            raise TimeoutError("test did not release lease teardown")

    def close_lease(*, started: threading.Event | None = None, returned: threading.Event | None = None) -> None:
        if started is not None:
            started.set()
        try:
            lease.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if returned is not None:
                returned.set()

    lease._resource_closers = (*lease._resource_closers, blocking_closer)
    first = threading.Thread(target=close_lease)
    second = threading.Thread(target=close_lease, kwargs={"started": second_started, "returned": second_returned})
    first.start()
    assert teardown_started.wait(5)
    second.start()
    assert second_started.wait(5)
    assert not second_returned.wait(0.2)

    allow_teardown.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert second_returned.is_set()
    assert not errors


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-specific flock ownership regression")
def test_forked_child_close_does_not_release_parent_lease_lock(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions run in the parent
        try:
            built.lease.close()
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, child_status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(child_status) == 0

    active = store.cleanup(identity)
    assert active is not None and active.code is FailureCode.ACTIVE_LEASE
    built.lease.close()
    assert store.cleanup(identity) is None


@pytest.mark.parametrize("write_mode", ["incomplete", "overlap"])
def test_failed_streaming_writes_never_publish_partial_artifacts(tmp_path: Path, write_mode: str) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")

    result = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity, write_mode=write_mode),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )

    assert result.status is StoreStatus.FAILED
    assert result.failure is not None and result.failure.code is FailureCode.PRODUCER_FAILED
    assert not list(store.tmp_dir.iterdir())
    assert not list(store.artifacts_dir.iterdir())


@pytest.mark.parametrize(
    "dtype",
    [
        dtype
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
        if (dtype := getattr(torch, name, None)) is not None
    ],
)
def test_available_float8_formats_round_trip_through_streaming_store(tmp_path: Path, dtype: torch.dtype) -> None:
    identity = replace(
        _identity(),
        representation=WeightRepresentation(name="runtime-fp8", dtype=str(dtype)),
    )
    expected = torch.zeros((2, 4), dtype=dtype)

    class Float8Producer:
        spec = WeightProductionSpec(
            producer_id="test.final-layout",
            outputs=(identity,),
            source_mode=ProductionSourceMode.FINALIZED_MODEL,
        )

        def produce(self, writer: object) -> ProductionMetadata:
            spec = TensorWriteSpec("weight", tuple(expected.shape), dtype)
            with writer.open_tensor_file("weights.safetensors", (spec,)) as output:  # type: ignore[attr-defined]
                output.write_tensor("weight", expected)
            return ProductionMetadata("test-producer-v1", "test-restorer-v1")

    store = _make_store(tmp_path / "store")
    result = store.get_or_build(
        BuildRequest(identity),
        Float8Producer(),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )

    assert result.status is StoreStatus.BUILT
    assert result.lease is not None
    actual = result.lease.tensors["weight"]
    assert actual.dtype is dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    result.lease.close()


def test_lookup_treats_orphan_deny_marker_as_miss(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    deny = store.deny_dir / f"{identity.key}.json"
    deny.write_text("{}", encoding="utf-8")

    result = store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM)

    assert result.status is StoreStatus.MISS


def test_overlapping_weaker_lookup_cannot_return_after_artifact_is_denied(tmp_path: Path) -> None:
    identity = _identity()
    root = tmp_path / "store"
    store = _make_store(root)
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None
    built.lease.close()

    artifact_dir = store.artifacts_dir / identity.key
    payload = artifact_dir / "weights.safetensors"
    os.chmod(artifact_dir, 0o755)
    os.chmod(payload, 0o644)
    with payload.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(payload, 0o444)
    os.chmod(artifact_dir, 0o555)

    lease_opened = threading.Event()
    resume_lookup = threading.Event()
    overlapping = PausingOpenLeaseStore(
        StorageDomainPolicy(root=root, storage_class=detect_storage_class(root)),
        CapacityPolicy(),
        IntegrityPolicy(),
        lease_opened=lease_opened,
        resume_lookup=resume_lookup,
    )
    weak_results: list[StoreResult] = []

    def weak_lookup() -> None:
        weak_results.append(overlapping.lookup(identity, validation=ValidationLevel.MANIFEST_AND_METADATA))

    thread = threading.Thread(target=weak_lookup)
    thread.start()
    try:
        assert lease_opened.wait(5)
        strong = store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM)
        assert strong.status is StoreStatus.INVALID
        assert (store.deny_dir / f"{identity.key}.json").is_file()
    finally:
        resume_lookup.set()
        thread.join(5)

    assert not thread.is_alive()
    assert len(weak_results) == 1
    weak = weak_results[0]
    assert weak.status is StoreStatus.INVALID
    assert weak.failure is not None and weak.failure.code is FailureCode.ARTIFACT_DENIED


def test_full_checksum_denies_quarantines_and_rebuilds_corrupt_payload(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None
    built.lease.close()

    artifact_dir = store.artifacts_dir / identity.key
    payload = artifact_dir / "weights.safetensors"
    os.chmod(artifact_dir, 0o755)
    os.chmod(payload, 0o644)
    with payload.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(payload, 0o444)
    os.chmod(artifact_dir, 0o555)

    metadata_only = store.lookup(identity, validation=ValidationLevel.MANIFEST_AND_METADATA)
    assert metadata_only.status is StoreStatus.HIT
    assert metadata_only.lease is not None
    metadata_only.lease.close()

    inspected = store.inspect_artifact(identity, validation=ValidationLevel.FULL_CHECKSUM)
    assert inspected.status is StoreStatus.INVALID
    assert inspected.failure is not None and inspected.failure.code is FailureCode.CHECKSUM_MISMATCH
    assert not (store.deny_dir / f"{identity.key}.json").exists()

    invalid = store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM)
    assert invalid.status is StoreStatus.INVALID
    assert invalid.failure is not None and invalid.failure.code is FailureCode.CHECKSUM_MISMATCH
    assert (store.deny_dir / f"{identity.key}.json").is_file()

    rebuilt = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert rebuilt.status is StoreStatus.BUILT
    assert rebuilt.lease is not None
    assert torch.equal(rebuilt.lease.tensors["layer.weight"], _weight())
    rebuilt.lease.close()
    assert not (store.deny_dir / f"{identity.key}.json").exists()
    assert len(list(store.quarantine_dir.iterdir())) == 1


def test_malformed_safetensors_is_a_typed_invalid_artifact(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None
    built.lease.close()

    artifact_dir = store.artifacts_dir / identity.key
    payload = artifact_dir / "weights.safetensors"
    os.chmod(artifact_dir, 0o755)
    os.chmod(payload, 0o644)
    with payload.open("r+b") as handle:
        handle.write(b"\xff" * 8)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(payload, 0o444)
    os.chmod(artifact_dir, 0o555)

    result = store.lookup(identity, validation=ValidationLevel.MANIFEST_AND_METADATA)

    assert result.status is StoreStatus.INVALID
    assert result.failure is not None and result.failure.code is FailureCode.MANIFEST_CORRUPT
    assert (store.deny_dir / f"{identity.key}.json").is_file()


@pytest.mark.parametrize("schema_field", ["producer_schema", "restorer_schema"])
def test_lookup_rejects_manifest_schema_that_differs_from_identity(
    tmp_path: Path,
    schema_field: str,
) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None
    built.lease.close()

    artifact_dir = store.artifacts_dir / identity.key
    manifest_path = artifact_dir / "manifest.json"
    marker_path = artifact_dir / "READY.json"
    manifest_document = json.loads(manifest_path.read_bytes())
    marker_document = json.loads(marker_path.read_bytes())
    manifest_document[schema_field] = "incompatible-schema"
    manifest_bytes = json.dumps(
        manifest_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    marker_document["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    marker_bytes = json.dumps(
        marker_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    os.chmod(artifact_dir, 0o755)
    for path, contents in ((manifest_path, manifest_bytes), (marker_path, marker_bytes)):
        os.chmod(path, 0o644)
        path.write_bytes(contents)
        os.chmod(path, 0o444)
    os.chmod(artifact_dir, 0o555)

    result = store.lookup(identity, validation=ValidationLevel.MANIFEST_AND_METADATA)

    assert result.status is StoreStatus.INVALID
    assert result.failure is not None and result.failure.code is FailureCode.MANIFEST_INCOMPATIBLE
    assert (store.deny_dir / f"{identity.key}.json").is_file()


def test_noncooperative_conflicting_publication_is_reported_without_replacement(tmp_path: Path) -> None:
    identity = _identity()
    root = tmp_path / "store"
    store = _make_store(root)
    started = threading.Event()
    release = threading.Event()
    thread_result: list[StoreResult] = []

    def build_expected_content() -> None:
        thread_result.append(
            store.get_or_build(
                BuildRequest(identity),
                FakeProducer(identity, started=started, release=release),
                validation=ValidationLevel.FULL_CHECKSUM,
                deadline=time.monotonic() + 10,
            )
        )

    builder = threading.Thread(target=build_expected_content)
    builder.start()
    try:
        assert started.wait(timeout=5)
        competing = NonCooperativeStore(store.domain_policy, store.capacity_policy, store.integrity_policy)
        published = competing.get_or_build(
            BuildRequest(identity),
            FakeProducer(identity, weight_offset=100),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 10,
        )
        assert published.status is StoreStatus.BUILT
        assert published.lease is not None
        published.lease.close()
    finally:
        release.set()
        builder.join(timeout=10)
    assert not builder.is_alive()
    assert len(thread_result) == 1
    collision = thread_result[0]
    assert collision.status is StoreStatus.FAILED
    assert collision.failure is not None and collision.failure.code is FailureCode.IDENTITY_COLLISION

    winner = store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM)
    assert winner.status is StoreStatus.HIT
    assert winner.lease is not None
    assert torch.equal(winner.lease.tensors["layer.weight"], _weight(100))
    winner.lease.close()
    assert len(list(store.quarantine_dir.iterdir())) == 1


def test_payload_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    built = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert built.lease is not None
    built.lease.close()

    artifact_dir = store.artifacts_dir / identity.key
    payload = artifact_dir / "weights.safetensors"
    outside = tmp_path / "outside.safetensors"
    os.chmod(artifact_dir, 0o755)
    payload.rename(outside)
    payload.symlink_to(outside)
    os.chmod(artifact_dir, 0o555)

    result = store.lookup(identity, validation=ValidationLevel.MANIFEST_AND_METADATA)
    assert result.status is StoreStatus.INVALID
    assert result.failure is not None and result.failure.code is FailureCode.MANIFEST_CORRUPT


def test_coordination_deadline_does_not_claim_to_cancel_an_active_producer(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    started = time.monotonic()

    result = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity, delay_seconds=0.15),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 0.02,
    )

    assert time.monotonic() - started >= 0.1
    assert result.status is StoreStatus.BUILT
    assert result.lease is not None
    result.lease.close()


@pytest.mark.parallel
def test_multiprocess_contention_runs_one_builder_and_joins_same_artifact(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    root = tmp_path / "store"
    _make_store(root)
    counter = tmp_path / "producer-count"
    initial_lookup_barrier = ctx.Barrier(2)
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_build_process,
            args=(str(root), str(counter), initial_lookup_barrier, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    _join_processes(processes)

    results = [result_queue.get(timeout=3) for _ in processes]
    assert all(process.exitcode == 0 for process in processes)
    assert sorted(item[0] for item in results) == [StoreStatus.BUILT.value, StoreStatus.JOINED.value]
    assert all(item[2] is None for item in results)
    assert len({item[1] for item in results}) == 1
    assert counter.read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parallel
def test_reader_sees_miss_during_build_and_waiter_has_bounded_timeout(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    root = tmp_path / "store"
    store = _make_store(root)
    identity = _identity()
    started = ctx.Event()
    release = ctx.Event()
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_controlled_build_process,
        args=(str(root), started, release, result_queue),
    )
    process.start()
    try:
        assert started.wait(timeout=_PROCESS_START_TIMEOUT_SECONDS)
        assert store.lookup(identity, validation=ValidationLevel.FULL_CHECKSUM).status is StoreStatus.MISS
        timed_out = store.get_or_build(
            BuildRequest(identity),
            FakeProducer(identity),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 0.1,
        )
        assert timed_out.status is StoreStatus.TIMEOUT
        assert timed_out.failure is not None and timed_out.failure.code is FailureCode.ACTIVE_BUILD_TIMEOUT
    finally:
        release.set()
        _join_processes([process])

    status, error = result_queue.get(timeout=3)
    assert process.exitcode == 0
    assert (status, error) == (StoreStatus.BUILT.value, None)


@pytest.mark.parallel
def test_crashed_builder_leaves_only_invisible_temp_and_next_builder_recovers(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    root = tmp_path / "store"
    store = _make_store(root)
    process = ctx.Process(target=_crash_build_process, args=(str(root),))
    process.start()
    _join_processes([process])

    assert process.exitcode == 17
    assert list(store.tmp_dir.iterdir())
    assert not list(store.artifacts_dir.iterdir())

    identity = _identity()
    recovered = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert recovered.status is StoreStatus.BUILT
    assert recovered.lease is not None
    recovered.lease.close()
    assert not list(store.tmp_dir.iterdir())


def test_capacity_and_enospc_failures_are_typed_and_clean_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    limited = _make_store(tmp_path / "limited", capacity=CapacityPolicy(max_artifact_bytes=64))
    too_large = limited.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert too_large.status is StoreStatus.FAILED
    assert too_large.failure is not None and too_large.failure.code is FailureCode.ARTIFACT_TOO_LARGE
    assert not list(limited.tmp_dir.iterdir())

    no_space = _make_store(tmp_path / "enospc")

    def raise_enospc(_fd: int, _offset: int, _length: int) -> None:
        raise OSError(errno.ENOSPC, "injected no space")

    monkeypatch.setattr(os, "posix_fallocate", raise_enospc)
    failed = no_space.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert failed.status is StoreStatus.FAILED
    assert failed.failure is not None and failed.failure.code is FailureCode.ENOSPC
    assert not list(no_space.tmp_dir.iterdir())
    assert not list(no_space.artifacts_dir.iterdir())


@pytest.mark.parametrize(
    ("failure_errno", "expected_stage", "expected_code"),
    [
        (errno.ENOSPC, ResolutionStage.CAPACITY, FailureCode.ENOSPC),
        (errno.EDQUOT, ResolutionStage.CAPACITY, FailureCode.ENOSPC),
        (errno.EACCES, ResolutionStage.DOMAIN, FailureCode.DOMAIN_UNAVAILABLE),
    ],
)
def test_staging_directory_failures_are_typed_and_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_errno: int,
    expected_stage: ResolutionStage,
    expected_code: FailureCode,
) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")
    original_mkdir = Path.mkdir

    def fail_staging_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path.parent == store.tmp_dir:
            raise OSError(failure_errno, "injected staging directory failure")
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fail_staging_mkdir)
    result = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )

    assert result.status is StoreStatus.FAILED
    assert result.failure is not None
    assert result.failure.stage is expected_stage
    assert result.failure.code is expected_code
    assert not list(store.tmp_dir.iterdir())
    assert not list(store.artifacts_dir.iterdir())


def test_store_and_free_space_capacity_limits_fail_before_publication(tmp_path: Path) -> None:
    identity = _identity()
    store_limited = _make_store(tmp_path / "store-limit", capacity=CapacityPolicy(max_store_bytes=1))
    store_result = store_limited.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert store_result.status is StoreStatus.FAILED
    assert store_result.failure is not None and store_result.failure.code is FailureCode.STORE_LIMIT_EXCEEDED
    assert not list(store_limited.artifacts_dir.iterdir())

    free_root = tmp_path / "free-limit"
    free_root.mkdir(mode=0o700)
    unavailable_reserve = shutil.disk_usage(free_root).free + 1
    free_limited = _make_store(free_root, capacity=CapacityPolicy(min_free_bytes=unavailable_reserve))
    free_result = free_limited.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert free_result.status is StoreStatus.FAILED
    assert free_result.failure is not None and free_result.failure.code is FailureCode.INSUFFICIENT_CAPACITY
    assert not list(free_limited.artifacts_dir.iterdir())


def test_domain_capacity_policy_is_authoritative(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _make_store(root, capacity=CapacityPolicy(max_store_bytes=4096))

    with pytest.raises(HostWeightError, match="capacity policy differs"):
        _make_store(root, capacity=CapacityPolicy(max_store_bytes=8192))


@pytest.mark.parametrize("filesystem_type", ["nfs4", "cifs", "lustre", "ceph", "unknownfs"])
def test_remote_and_unknown_filesystems_cannot_back_the_local_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filesystem_type: str,
) -> None:
    monkeypatch.setattr(filesystem_store_module, "detect_filesystem_type", lambda _path: filesystem_type)
    root = tmp_path / "store"

    with pytest.raises(ValueError, match="not node-local|not recognized"):
        detect_storage_class(root)
    with pytest.raises(HostWeightError) as error:
        FilesystemHostWeightStore(
            StorageDomainPolicy(root=root, storage_class=StorageClass.DISK),
            CapacityPolicy(),
            IntegrityPolicy(),
        )

    assert error.value.failure.code is FailureCode.DOMAIN_POLICY_MISMATCH
    assert not root.exists()


def test_trusted_store_rejects_group_or_world_writable_root(tmp_path: Path) -> None:
    root = tmp_path / "untrusted"
    root.mkdir(mode=0o700)
    root.chmod(0o777)

    with pytest.raises(HostWeightError) as error:
        FilesystemHostWeightStore(
            StorageDomainPolicy(root=root, storage_class=detect_storage_class(root)),
            CapacityPolicy(),
            IntegrityPolicy(),
        )

    assert error.value.failure.code is FailureCode.DOMAIN_UNAVAILABLE


def test_unimplemented_coordination_and_fs_verity_return_typed_failures(tmp_path: Path) -> None:
    identity = _identity()
    store = _make_store(tmp_path / "store")

    unsupported_producer = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity, coordination_scope=CoordinationScope.GROUP_COHORT),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert unsupported_producer.status is StoreStatus.FAILED
    assert unsupported_producer.failure is not None
    assert unsupported_producer.failure.code is FailureCode.PRODUCER_UNSUPPORTED

    fs_verity = store.lookup(identity, validation=ValidationLevel.FS_VERITY)
    assert fs_verity.status is StoreStatus.FAILED
    assert fs_verity.failure is not None and fs_verity.failure.code is FailureCode.UNSUPPORTED


def test_unimplemented_numa_enforcement_and_background_scrub_fail_explicitly(tmp_path: Path) -> None:
    numa_root = tmp_path / "numa"
    numa_root.mkdir(mode=0o700)
    with pytest.raises(HostWeightError) as numa_error:
        FilesystemHostWeightStore(
            StorageDomainPolicy(
                root=numa_root,
                scope=StorageScope.NUMA,
                storage_class=detect_storage_class(numa_root),
                numa_prefault=True,
            ),
            CapacityPolicy(),
            IntegrityPolicy(),
        )
    assert numa_error.value.failure.code is FailureCode.UNSUPPORTED

    with pytest.raises(HostWeightError) as scrub_error:
        _make_store(tmp_path / "scrub", integrity=IntegrityPolicy(background_scrub=True))
    assert scrub_error.value.failure.code is FailureCode.UNSUPPORTED
