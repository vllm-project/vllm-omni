# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU process and policy coverage for domain-wide producer admission."""

from __future__ import annotations

import errno
import json
import multiprocessing as mp
import os
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

import pytest
import torch

from tests.host_weight_runtime.test_filesystem_store import FakeProducer, _identity, _make_store, _publish_test_artifact
from vllm_omni.host_weight_runtime import (
    BuildRequest,
    CapacityPolicy,
    FailureCode,
    HostWeightError,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    ProductionMetadata,
    ResolutionOutcome,
    RuntimeMode,
    StoreResult,
    StoreStatus,
    TensorWriteSpec,
    ValidationLevel,
    WaitPolicy,
    WeightArtifactIdentity,
    WeightProductionSpec,
)
from vllm_omni.host_weight_runtime.filesystem.locks import FileLock, lock_is_active
from vllm_omni.host_weight_runtime.protocols import ArtifactWriter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class AdmissionProducer:
    def __init__(self, identity: WeightArtifactIdentity, control: Connection | None = None) -> None:
        self._spec = FakeProducer(identity).spec
        self.control = control

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        action = "produce"
        if self.control is not None:
            self.control.send("entered")
            if not self.control.poll(60):
                raise TimeoutError("test did not release producer")
            action = self.control.recv()
        spec = TensorWriteSpec("payload", (65536,), torch.bfloat16)
        with writer.open_tensor_file("weights.safetensors", (spec,)) as output:
            output.write_tensor("payload", torch.arange(65536, dtype=torch.float32).to(torch.bfloat16))
            if action == "crash":
                os._exit(27)
        return ProductionMetadata("test-producer-v1", "test-restorer-v1")


def _admission_process(root: Path, policy: CapacityPolicy, rank: int, control: Connection) -> None:
    try:
        store = _make_store(root, capacity=policy)
        identity = _identity(tp_rank=rank)
        original_lookup = store.lookup
        first_lookup = True

        def notify_lookup(
            identity: WeightArtifactIdentity, *, validation: ValidationLevel, deadline: float | None = None
        ) -> StoreResult:
            nonlocal first_lookup
            result = original_lookup(identity, validation=validation, deadline=deadline)
            if first_lookup:
                # Prove the caller observed a miss before the parent lets the
                # active producer publish, independent of process scheduling.
                control.send("looked_up")
                first_lookup = False
            return result

        store.lookup = notify_lookup  # type: ignore[method-assign]
        control.send("ready")
        assert control.poll(60) and control.recv() == "start"
        result = store.get_or_build(
            BuildRequest(identity),
            AdmissionProducer(identity, control),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 30,
        )
        if result.lease is not None:
            result.lease.close()
        control.send((result.status.value, result.failure.code.value if result.failure is not None else None))
    finally:
        control.close()


@pytest.mark.parametrize("admission", ["concurrent", "serialized"])
@pytest.mark.parametrize("same_key", [False, True])
def test_cross_identity_admission_and_capacity(
    tmp_path: Path, admission: Literal["concurrent", "serialized"], same_key: bool
) -> None:
    policy = CapacityPolicy(max_store_bytes=192 * 1024, build_admission=admission)
    store = _make_store(tmp_path / "store", capacity=policy)
    ctx = mp.get_context("spawn")
    pairs = [ctx.Pipe() for _ in range(2)]
    processes = [
        ctx.Process(target=_admission_process, args=(store.root, policy, 0 if same_key else rank, pair[1]))
        for rank, pair in enumerate(pairs)
    ]
    try:
        for process, pair in zip(processes, pairs, strict=True):
            process.start()
            pair[1].close()
        first, second = [pair[0] for pair in pairs]
        for connection in (first, second):
            assert connection.poll(60) and connection.recv() == "ready"
        first.send("start")
        assert first.poll(10) and first.recv() == "looked_up"
        assert first.poll(10) and first.recv() == "entered"
        second.send("start")
        assert second.poll(10) and second.recv() == "looked_up"
        if admission == "concurrent" and not same_key:
            assert second.poll(10) and second.recv() == "entered"
        else:
            assert not second.poll(0.2), "a second producer entered while coordination should exclude it"
        first.send("produce")
        assert first.poll(30) and first.recv() == (StoreStatus.BUILT.value, None)
        if same_key:
            assert second.poll(30) and second.recv() == (StoreStatus.JOINED.value, None)
        else:
            if admission == "serialized":
                assert second.poll(10) and second.recv() == "entered"
            second.send("produce")
            assert second.poll(30)
            assert second.recv() == (StoreStatus.FAILED.value, FailureCode.STORE_LIMIT_EXCEEDED.value)
        assert policy.max_store_bytes is not None
        assert store.inspect_domain().store_bytes < policy.max_store_bytes
        assert not list(store.tmp_dir.iterdir())
        assert {path.name for path in store.artifacts_dir.iterdir()} == {_identity(tp_rank=0).key}
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(5)
                process.close()
        for pair in pairs:
            for connection in pair:
                connection.close()


def test_admission_timeout_preserves_owner_and_warm_hit_bypasses_lock(tmp_path: Path) -> None:
    policy = CapacityPolicy(max_store_bytes=512 * 1024, build_admission="serialized")
    store = _make_store(tmp_path / "store", capacity=policy)
    identity, _ = _publish_test_artifact(store)
    other = _identity(tp_rank=1)
    lock_path = store.locks_dir / "domain-build.lock"
    with FileLock(lock_path, exclusive=True, deadline=None):
        hit = store.get_or_build(
            BuildRequest(identity),
            FakeProducer(identity),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 0.02,
        )
        assert hit.status is StoreStatus.HIT and hit.lease is not None
        hit.lease.close()
        timeout = store.get_or_build(
            BuildRequest(other),
            FakeProducer(other),
            validation=ValidationLevel.FULL_CHECKSUM,
            deadline=time.monotonic() + 0.02,
        )
        assert timeout.status is StoreStatus.TIMEOUT
        assert timeout.failure is not None and timeout.failure.code is FailureCode.ACTIVE_BUILD_TIMEOUT
        assert timeout.failure.retryable
        assert lock_is_active(lock_path)
    _publish_test_artifact(store, other)


def test_producer_failure_releases_admission(tmp_path: Path) -> None:
    store = _make_store(
        tmp_path / "store", capacity=CapacityPolicy(max_store_bytes=16384, build_admission="serialized")
    )
    identity = _identity()
    failed = store.get_or_build(
        BuildRequest(identity),
        FakeProducer(identity, write_mode="incomplete"),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 5,
    )
    assert failed.status is StoreStatus.FAILED
    assert not lock_is_active(store.locks_dir / "domain-build.lock")
    _publish_test_artifact(store, identity)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(RuntimeMode.PREFERRED, ResolutionOutcome.CANONICAL_FALLBACK), (RuntimeMode.REQUIRED, ResolutionOutcome.FAILED)],
)
def test_admission_timeout_obeys_runtime_mode(tmp_path: Path, mode: RuntimeMode, expected: ResolutionOutcome) -> None:
    policy = CapacityPolicy(max_store_bytes=16384, build_admission="serialized")
    store = _make_store(tmp_path / "store", capacity=policy)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=mode, domain=store.domain_policy, capacity=policy, wait=WaitPolicy(0.01))
    )
    identity = _identity()
    with FileLock(store.locks_dir / "domain-build.lock", exclusive=True, deadline=None):
        resolution = runtime.resolve(identity, producer=FakeProducer(identity))
    assert resolution.report.outcome is expected
    failure = resolution.report.attempts[-1].failure
    assert failure is not None and failure.code is FailureCode.ACTIVE_BUILD_TIMEOUT


def test_crashed_producer_releases_admission_and_same_key_retry_recovers(tmp_path: Path) -> None:
    policy = CapacityPolicy(max_store_bytes=192 * 1024, build_admission="serialized")
    store = _make_store(tmp_path / "store", capacity=policy)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_admission_process, args=(store.root, policy, 0, child))
    process.start()
    child.close()
    try:
        assert parent.poll(60) and parent.recv() == "ready"
        parent.send("start")
        assert parent.poll(10) and parent.recv() == "looked_up"
        assert parent.poll(10) and parent.recv() == "entered"
        parent.send("crash")
        process.join(10)
        assert process.exitcode == 27
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        process.close()
        parent.close()
    assert not lock_is_active(store.locks_dir / "domain-build.lock")
    assert list(store.tmp_dir.iterdir())
    identity = _identity(tp_rank=0)
    recovered = store.get_or_build(
        BuildRequest(identity),
        AdmissionProducer(identity),
        validation=ValidationLevel.FULL_CHECKSUM,
        deadline=time.monotonic() + 10,
    )
    assert recovered.status is StoreStatus.BUILT and recovered.lease is not None
    assert torch.equal(recovered.lease.tensors["payload"], torch.arange(65536, dtype=torch.float32).to(torch.bfloat16))
    recovered.lease.close()
    assert not list(store.tmp_dir.iterdir())


def test_admission_release_error_preserves_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(
        tmp_path / "store", capacity=CapacityPolicy(max_store_bytes=16384, build_admission="serialized")
    )
    original_close = FileLock.close

    def fail_admission_close(lock: FileLock) -> None:
        original_close(lock)
        if lock.path.name == "domain-build.lock":
            raise OSError(errno.EIO, "injected admission release failure")

    monkeypatch.setattr(FileLock, "close", fail_admission_close)
    _publish_test_artifact(store)
    assert not lock_is_active(store.locks_dir / "domain-build.lock")


def test_admission_policy_is_authoritative_and_legacy_document_is_unchanged(tmp_path: Path) -> None:
    concurrent = CapacityPolicy(max_store_bytes=16384)
    serialized = CapacityPolicy(max_store_bytes=16384, build_admission="serialized")
    legacy = _make_store(tmp_path / "legacy", capacity=concurrent)
    path = legacy.root / "domain-policy.json"
    original = path.read_bytes()
    assert json.loads(original) == {
        "schema_version": 1,
        "policy_version": 1,
        "max_artifact_bytes": None,
        "max_store_bytes": 16384,
        "min_free_bytes": 0,
        "eviction": "none",
    }
    with pytest.raises(HostWeightError, match="incompatible schema"):
        _make_store(legacy.root, capacity=serialized)
    assert path.read_bytes() == original
    strict = _make_store(tmp_path / "serialized", capacity=serialized)
    document = json.loads((strict.root / "domain-policy.json").read_bytes())
    assert document["schema_version"] == 2 and document["build_admission"] == "serialized"
    assert _make_store(strict.root, capacity=serialized).domain_uuid == strict.domain_uuid
    with pytest.raises(HostWeightError, match="incompatible schema"):
        _make_store(strict.root, capacity=concurrent)


def test_admission_policy_validation() -> None:
    with pytest.raises(ValueError, match="requires max_store_bytes"):
        CapacityPolicy(build_admission="serialized")
    with pytest.raises(ValueError, match="must be concurrent or serialized"):
        CapacityPolicy(build_admission="unknown")  # type: ignore[arg-type]
