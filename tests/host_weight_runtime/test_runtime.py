# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolution-policy tests for the loader-adjacent Host Weight Runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    ComponentIdentity,
    FailureCode,
    HostWeightFailure,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    IntegrityPolicy,
    LookupPhase,
    ProducerIdentity,
    ProductionMetadata,
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
from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

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
