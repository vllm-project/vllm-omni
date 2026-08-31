# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime resolution and publication-policy tests for Host Weight Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    ComponentIdentity,
    FailureCode,
    HostWeightFailure,
    HostWeightLease,
    HostWeightLeaseCarrier,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    IntegrityPolicy,
    LookupPhase,
    PostLoadPublicationOutcome,
    PostLoadPublicationReport,
    ProducerIdentity,
    ProductionMetadata,
    ProductionPolicy,
    ProductionSourceMode,
    RemoteImportPolicy,
    RemoteOnMiss,
    ResolutionOutcome,
    ResolutionReport,
    ResolutionStage,
    RuntimeMode,
    RuntimeWeightLayout,
    StorageDomainPolicy,
    StoreResult,
    StoreStatus,
    TensorWriteSpec,
    ValidationLevel,
    WeightArtifactIdentity,
    WeightProductionSpec,
    WeightRepresentation,
    WeightSourceIdentity,
)
from vllm_omni.host_weight_runtime.filesystem import FilesystemHostWeightStore, detect_storage_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _identity() -> WeightArtifactIdentity:
    return WeightArtifactIdentity(
        schema_version=1,
        source=WeightSourceIdentity("test-org/test-model", "0123456789abcdef", "source-files-sha256"),
        component=ComponentIdentity("transformer", "diffusion.dit"),
        representation=WeightRepresentation("runtime-bf16", "torch.bfloat16"),
        layout=RuntimeWeightLayout("final-module-layout"),
        adaptation=AdaptationIdentity(),
        producer=ProducerIdentity(
            "test.final-layout",
            "1",
            "implementation-sha256",
            "test-producer-v1",
            "test-restorer-v1",
        ),
    )


class CountingProducer:
    def __init__(
        self,
        identity: WeightArtifactIdentity,
        *,
        lookup_phase: LookupPhase = LookupPhase.PRE_LOAD_SAFE,
    ) -> None:
        self.calls = 0
        self._spec = WeightProductionSpec(
            producer_id="test.final-layout",
            outputs=(identity,),
            source_mode=ProductionSourceMode.FINALIZED_MODEL,
            lookup_phase=lookup_phase,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: object) -> ProductionMetadata:
        self.calls += 1
        spec = TensorWriteSpec("weight", (2, 2), torch.bfloat16)
        with writer.open_tensor_file("weights.safetensors", (spec,)) as output:  # type: ignore[attr-defined]
            output.write_tensor("weight", torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2))
        return ProductionMetadata(
            producer_schema="test-producer-v1",
            restorer_schema="test-restorer-v1",
            format_metadata=CanonicalJson.from_value({"source": "test"}),
        )


def _domain(root: Path) -> StorageDomainPolicy:
    return StorageDomainPolicy(root=root, storage_class=detect_storage_class(root.parent))


def test_disabled_mode_does_not_probe_or_create_configured_root(tmp_path: Path) -> None:
    root = tmp_path / "must-not-exist"
    runtime = HostWeightRuntime.from_config(HostWeightRuntimeConfig(mode=RuntimeMode.DISABLED, domain=_domain(root)))

    resolution = runtime.resolve()

    assert resolution.report.outcome is ResolutionOutcome.CANONICAL_DIRECT
    assert resolution.report.attempts == ()
    assert resolution.lease is None
    assert not root.exists()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RuntimeMode.PREFERRED, ResolutionOutcome.CANONICAL_FALLBACK),
        (RuntimeMode.REQUIRED, ResolutionOutcome.FAILED),
    ],
)
def test_local_miss_uses_configured_fallback_policy(
    tmp_path: Path,
    mode: RuntimeMode,
    expected: ResolutionOutcome,
) -> None:
    runtime = HostWeightRuntime.from_config(HostWeightRuntimeConfig(mode=mode, domain=_domain(tmp_path / mode.value)))

    resolution = runtime.resolve(_identity())

    assert resolution.report.outcome is expected
    assert resolution.lease is None
    assert [attempt.result.value for attempt in resolution.report.attempts] == ["miss", "skipped"]
    expected_action = "canonical_fallback" if mode is RuntimeMode.PREFERRED else "fail_startup"
    assert [attempt.action.value for attempt in resolution.report.attempts] == [expected_action, expected_action]


def test_local_production_then_exact_warm_hit_emits_one_terminal_report_each(tmp_path: Path) -> None:
    reports: list[ResolutionReport] = []
    identity = _identity()
    producer = CountingProducer(identity)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / "store")),
        observer=reports.append,
    )

    cold = runtime.resolve(identity, producer=producer)
    assert cold.report.outcome is ResolutionOutcome.LOCAL_PRODUCTION
    assert cold.lease is not None
    assert torch.equal(
        cold.lease.tensors["weight"],
        torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2),
    )
    cold.lease.close()
    assert cold.report.attempts[0].action.value == "try_producer"

    warm = runtime.resolve(identity)
    assert warm.report.outcome is ResolutionOutcome.LOCAL_HIT
    assert warm.lease is not None
    warm.lease.close()

    assert producer.calls == 1
    assert reports == [cold.report, warm.report]
    assert all(report.resolution_id for report in reports)


def test_lease_carrier_is_single_take_and_process_local(tmp_path: Path) -> None:
    identity = _identity()
    producer = CountingProducer(identity)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / "store"))
    )
    resolution = runtime.resolve(identity, producer=producer)
    assert resolution.lease is not None

    carrier = HostWeightLeaseCarrier(resolution.lease)
    with pytest.raises(TypeError, match="cannot be serialized"):
        carrier.__reduce_ex__(4)

    lease = carrier.take()
    assert lease is resolution.lease
    with pytest.raises(RuntimeError, match="already taken"):
        carrier.take()
    carrier.close()  # ownership has moved; the carrier must not close twice.
    assert not lease.closed
    lease.close()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RuntimeMode.PREFERRED, ResolutionOutcome.FAILED),
        (RuntimeMode.REQUIRED, ResolutionOutcome.FAILED),
    ],
)
def test_remote_required_is_explicitly_unsupported_and_skips_local_production(
    tmp_path: Path,
    mode: RuntimeMode,
    expected: ResolutionOutcome,
) -> None:
    identity = _identity()
    producer = CountingProducer(identity)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=mode,
            domain=_domain(tmp_path / mode.value),
            remote=RemoteImportPolicy(
                on_local_miss=RemoteOnMiss.REQUIRE,
                providers=("future-mooncake",),
            ),
        )
    )

    resolution = runtime.resolve(identity, producer=producer)

    assert resolution.report.outcome is expected
    assert producer.calls == 0
    assert [attempt.stage.value for attempt in resolution.report.attempts] == ["lookup", "remote"]
    assert resolution.report.attempts[-1].failure is not None
    assert resolution.report.attempts[-1].failure.code.value == "unsupported"
    assert resolution.report.attempts[-1].action.value == "fail_startup"


def test_observer_failure_cannot_change_resolution_outcome(tmp_path: Path) -> None:
    def broken_observer(_report: object) -> None:
        raise RuntimeError("metrics backend unavailable")

    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / "store")),
        observer=broken_observer,
    )

    resolution = runtime.resolve(_identity())

    assert resolution.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK


def test_preload_resolution_does_not_run_postload_only_producer(tmp_path: Path) -> None:
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / "store"))
    )

    resolution = runtime.resolve(identity, producer=producer)

    assert resolution.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK
    assert producer.calls == 0
    assert resolution.report.attempts[0].action.value == "canonical_fallback"


def test_postload_publication_closes_lease_and_only_warms_future_resolution(
    tmp_path: Path,
) -> None:
    publications: list[PostLoadPublicationReport] = []
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            production=ProductionPolicy(
                allow_local_build=False,
                allow_post_load_publish=True,
            ),
        ),
        publication_observer=publications.append,
    )

    preload = runtime.resolve(identity, producer=producer)
    assert preload.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK
    assert producer.calls == 0

    publication = runtime.publish_after_load(identity, producer=producer)
    assert publication.outcome is PostLoadPublicationOutcome.PUBLISHED
    assert publication.identity_digest == identity.key

    already_present = runtime.publish_after_load(identity, producer=producer)
    assert already_present.outcome is PostLoadPublicationOutcome.ALREADY_PRESENT

    warm = runtime.resolve(identity)
    assert warm.report.outcome is ResolutionOutcome.LOCAL_HIT
    assert warm.lease is not None
    assert torch.equal(
        warm.lease.tensors["weight"],
        torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2),
    )
    warm.lease.close()
    assert producer.calls == 1
    assert publications == [publication, already_present]
    assert isinstance(runtime.store, FilesystemHostWeightStore)
    assert runtime.store.cleanup(identity) is None


def test_postload_publication_policy_skip_does_not_run_producer(tmp_path: Path) -> None:
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / "store"))
    )

    publication = runtime.publish_after_load(identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.POLICY_DISABLED
    assert producer.calls == 0


def test_postload_publication_reports_lease_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )
    assert runtime.store is not None

    class FailingLease:
        def close(self) -> None:
            raise RuntimeError("injected lease cleanup failure")

    def return_failing_lease(*_args: object, **_kwargs: object) -> StoreResult:
        return StoreResult(StoreStatus.BUILT, lease=cast(HostWeightLease, FailingLease()))

    monkeypatch.setattr(runtime.store, "get_or_build", return_failing_lease)

    publication = runtime.publish_after_load(identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.FAILED
    assert publication.failure is not None
    assert publication.failure.stage is ResolutionStage.LIFECYCLE
    assert publication.failure.code is FailureCode.PUBLICATION_FAILED
    assert publication.failure.details.to_value() == {
        "exception_type": "RuntimeError",
        "store_outcome": "built",
    }


def test_disabled_postload_publication_does_not_probe_configured_root(tmp_path: Path) -> None:
    root = tmp_path / "must-not-exist"
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.DISABLED,
            domain=_domain(root),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )

    publication = runtime.publish_after_load()

    assert publication.outcome is PostLoadPublicationOutcome.RUNTIME_DISABLED
    assert not root.exists()


def test_postload_publication_rejects_preload_producer(tmp_path: Path) -> None:
    identity = _identity()
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )

    with pytest.raises(ValueError, match="POST_LOAD_ONLY"):
        runtime.publish_after_load(identity, producer=CountingProducer(identity))


def test_postload_publication_failure_is_reported_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )
    assert runtime.store is not None
    failure = HostWeightFailure(
        stage=ResolutionStage.CAPACITY,
        code=FailureCode.ENOSPC,
        retryable=True,
        message="injected post-load publication failure",
        details=CanonicalJson.empty(),
    )

    def fail_publication(*_args: object, **_kwargs: object) -> StoreResult:
        return StoreResult(StoreStatus.FAILED, failure=failure)

    monkeypatch.setattr(runtime.store, "get_or_build", fail_publication)
    publication = runtime.publish_after_load(identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.FAILED
    assert publication.failure is failure
    assert producer.calls == 0


def test_postload_publication_reports_store_initialization_failure(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(root),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )

    publication = runtime.publish_after_load(identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.FAILED
    assert publication.failure is not None
    assert publication.failure.code is FailureCode.DOMAIN_UNAVAILABLE
    assert publication.failure.retryable
    assert producer.calls == 0


def test_postload_publication_maps_joined_and_closes_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )
    assert runtime.store is not None

    class ClosingLease:
        closed = False

        def close(self) -> None:
            self.closed = True

    lease = ClosingLease()

    def return_joined_lease(*_args: object, **_kwargs: object) -> StoreResult:
        return StoreResult(StoreStatus.JOINED, lease=cast(HostWeightLease, lease))

    monkeypatch.setattr(runtime.store, "get_or_build", return_joined_lease)

    publication = runtime.publish_after_load(identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.JOINED
    assert lease.closed


def test_postload_publication_synthesizes_failure_for_unexpected_store_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    producer = CountingProducer(identity, lookup_phase=LookupPhase.POST_LOAD_ONLY)
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            production=ProductionPolicy(allow_post_load_publish=True),
        )
    )
    assert runtime.store is not None

    def return_unexpected_miss(*_args: object, **_kwargs: object) -> StoreResult:
        return StoreResult(StoreStatus.MISS)

    monkeypatch.setattr(runtime.store, "get_or_build", return_unexpected_miss)

    publication = runtime.publish_after_load(identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.FAILED
    assert publication.failure is not None
    assert publication.failure.stage is ResolutionStage.PRODUCTION
    assert publication.failure.code is FailureCode.PRODUCER_FAILED
    assert not publication.failure.retryable


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RuntimeMode.PREFERRED, ResolutionOutcome.CANONICAL_FALLBACK),
        (RuntimeMode.REQUIRED, ResolutionOutcome.FAILED),
    ],
)
def test_retryable_store_initialization_failure_obeys_runtime_mode(
    tmp_path: Path,
    mode: RuntimeMode,
    expected: ResolutionOutcome,
) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")
    runtime = HostWeightRuntime.from_config(HostWeightRuntimeConfig(mode=mode, domain=_domain(root)))

    resolution = runtime.resolve(_identity())

    assert resolution.report.outcome is expected
    assert resolution.lease is None
    assert len(resolution.report.attempts) == 1
    assert resolution.report.attempts[0].failure is not None
    assert resolution.report.attempts[0].failure.code.value == "domain_unavailable"
    assert resolution.report.attempts[0].failure.retryable


@pytest.mark.parametrize(
    ("stage", "code"),
    [
        (ResolutionStage.PRODUCTION, FailureCode.PRODUCER_UNSUPPORTED),
        (ResolutionStage.VALIDATION, FailureCode.IDENTITY_COLLISION),
        (ResolutionStage.LIFECYCLE, FailureCode.PUBLICATION_FAILED),
    ],
)
def test_nonretryable_post_initialization_failure_cannot_fall_back_in_preferred_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: ResolutionStage,
    code: FailureCode,
) -> None:
    identity = _identity()
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / code.value))
    )
    assert runtime.store is not None

    def fail_production(
        _request: object,
        _producer: object,
        *,
        validation: object,
        deadline: float,
    ) -> StoreResult:
        del validation, deadline
        return StoreResult(
            StoreStatus.FAILED,
            failure=HostWeightFailure(
                stage=stage,
                code=code,
                retryable=False,
                message="injected nonretryable production failure",
                details=CanonicalJson.empty(),
            ),
        )

    monkeypatch.setattr(runtime.store, "get_or_build", fail_production)
    resolution = runtime.resolve(identity, producer=CountingProducer(identity))

    assert resolution.report.outcome is ResolutionOutcome.FAILED
    assert resolution.report.attempts[-1].action.value == "fail_startup"
    assert resolution.report.attempts[-1].failure is not None
    assert resolution.report.attempts[-1].failure.code is code


def test_retryable_post_initialization_failure_can_fall_back_in_preferred_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(tmp_path / "retryable"))
    )
    assert runtime.store is not None

    def fail_production(
        _request: object,
        _producer: object,
        *,
        validation: object,
        deadline: float,
    ) -> StoreResult:
        del validation, deadline
        return StoreResult(
            StoreStatus.FAILED,
            failure=HostWeightFailure(
                stage=ResolutionStage.CAPACITY,
                code=FailureCode.ENOSPC,
                retryable=True,
                message="injected recoverable capacity failure",
                details=CanonicalJson.empty(),
            ),
        )

    monkeypatch.setattr(runtime.store, "get_or_build", fail_production)
    resolution = runtime.resolve(identity, producer=CountingProducer(identity))

    assert resolution.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK
    assert resolution.report.attempts[-1].action.value == "canonical_fallback"


def test_nonretryable_store_configuration_failure_remains_visible_in_preferred_mode(tmp_path: Path) -> None:
    root = tmp_path / "untrusted"
    root.mkdir(mode=0o700)
    root.chmod(0o777)
    runtime = HostWeightRuntime.from_config(HostWeightRuntimeConfig(mode=RuntimeMode.PREFERRED, domain=_domain(root)))

    resolution = runtime.resolve(_identity())

    assert resolution.report.outcome is ResolutionOutcome.FAILED
    assert resolution.report.attempts[0].action.value == "fail_startup"
    assert resolution.report.attempts[0].failure is not None
    assert not resolution.report.attempts[0].failure.retryable


def test_unsupported_integrity_policy_fails_instead_of_disguising_configuration_error(tmp_path: Path) -> None:
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=_domain(tmp_path / "store"),
            integrity=IntegrityPolicy(local_lookup=ValidationLevel.FS_VERITY),
        )
    )

    resolution = runtime.resolve(_identity())

    assert resolution.report.outcome is ResolutionOutcome.FAILED
    assert resolution.report.attempts[0].failure is not None
    assert resolution.report.attempts[0].failure.code.value == "unsupported"
