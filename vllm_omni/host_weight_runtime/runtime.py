# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Policy-driven exact resolution over HostWeightStore contracts."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from .config import HostWeightRuntimeConfig, RemoteOnMiss, RuntimeMode
from .errors import FailureCode, HostWeightError, HostWeightFailure, ResolutionAction, ResolutionStage
from .identity import CanonicalJson, WeightArtifactIdentity
from .lease import HostWeightLease
from .outcomes import (
    AttemptResult,
    HostWeightResolution,
    ResolutionAttempt,
    ResolutionOutcome,
    ResolutionReport,
    StoreResult,
    StoreStatus,
)
from .protocols import BuildRequest, HostWeightStore, LookupPhase, WeightProducer

logger = logging.getLogger(__name__)


class HostWeightRuntime:
    """Resolve one exact representation without owning model restoration."""

    def __init__(
        self,
        config: HostWeightRuntimeConfig,
        store: HostWeightStore | None = None,
        *,
        observer: Callable[[ResolutionReport], None] | None = None,
        _initialization_failure: HostWeightFailure | None = None,
    ) -> None:
        if observer is not None and not callable(observer):
            raise ValueError("Host Weight Runtime observer must be callable")
        if config.mode is RuntimeMode.DISABLED and (store is not None or _initialization_failure is not None):
            raise ValueError("disabled Host Weight Runtime must not initialize a store")
        if config.mode is not RuntimeMode.DISABLED and (store is None) == (_initialization_failure is None):
            raise ValueError("enabled Host Weight Runtime requires exactly one store or initialization failure")
        self.config = config
        self.store = store
        self._initialization_failure = _initialization_failure
        self._observer = observer

    @classmethod
    def from_config(
        cls,
        config: HostWeightRuntimeConfig,
        *,
        observer: Callable[[ResolutionReport], None] | None = None,
    ) -> HostWeightRuntime:
        """Construct lazily so disabled mode performs no filesystem probe."""
        if config.mode is RuntimeMode.DISABLED:
            return cls(config, observer=observer)
        assert config.domain is not None
        from .filesystem import FilesystemHostWeightStore  # noqa: PLC0415

        try:
            store = FilesystemHostWeightStore(config.domain, config.capacity, config.integrity)
        except HostWeightError as exc:
            return cls(config, observer=observer, _initialization_failure=exc.failure)
        return cls(config, store, observer=observer)

    @property
    def enabled(self) -> bool:
        return self.config.mode is not RuntimeMode.DISABLED

    def resolve(
        self,
        identity: WeightArtifactIdentity | None = None,
        *,
        producer: WeightProducer | None = None,
    ) -> HostWeightResolution:
        started = time.monotonic()
        if not self.enabled:
            return self._finish(
                ResolutionReport(
                    resolution_id=uuid.uuid4().hex,
                    outcome=ResolutionOutcome.CANONICAL_DIRECT,
                    attempts=(),
                    elapsed_seconds=time.monotonic() - started,
                )
            )
        if identity is None:
            raise ValueError("enabled Host Weight Runtime requires an exact artifact identity")
        if self._initialization_failure is not None:
            failure = self._initialization_failure
            action = self._action_for_failure(failure)
            return self._finish_terminal(
                (
                    ResolutionAttempt(
                        stage=failure.stage,
                        result=AttemptResult.ERROR,
                        action=action,
                        elapsed_seconds=time.monotonic() - started,
                        failure=failure,
                    ),
                ),
                started,
            )
        assert self.store is not None
        deadline = started + self.config.wait.coordination_timeout_seconds
        attempts: list[ResolutionAttempt] = []

        lookup_started = time.monotonic()
        local = self.store.lookup(
            identity,
            validation=self.config.integrity.local_lookup,
            deadline=deadline,
        )
        if local.status is StoreStatus.HIT:
            assert local.lease is not None
            attempts.append(
                ResolutionAttempt(
                    stage=ResolutionStage.LOOKUP,
                    result=AttemptResult.HIT,
                    action=ResolutionAction.RETURN_LEASE,
                    elapsed_seconds=time.monotonic() - lookup_started,
                )
            )
            return self._finish(
                ResolutionReport(
                    resolution_id=local.lease.provenance.resolution_id,
                    outcome=ResolutionOutcome.LOCAL_HIT,
                    attempts=tuple(attempts),
                    elapsed_seconds=time.monotonic() - started,
                ),
                lease=local.lease,
            )

        local_action = self._next_after_local(local, producer)
        attempts.append(
            self._store_attempt(
                ResolutionStage.LOOKUP,
                local,
                time.monotonic() - lookup_started,
                action=local_action,
            )
        )
        if local.status in {StoreStatus.TIMEOUT, StoreStatus.FAILED}:
            return self._finish_terminal(tuple(attempts), started)

        producer_allowed = self._producer_allowed(producer)
        if self.config.remote.on_local_miss is not RemoteOnMiss.DISABLED:
            remote_failure = HostWeightFailure(
                stage=ResolutionStage.REMOTE,
                code=FailureCode.UNSUPPORTED,
                retryable=False,
                message="remote artifact providers are modeled but not implemented",
                details=CanonicalJson.empty(),
            )
            attempts.append(
                ResolutionAttempt(
                    stage=ResolutionStage.REMOTE,
                    result=AttemptResult.SKIPPED,
                    action=self._action_for_failure(remote_failure),
                    elapsed_seconds=0.0,
                    failure=remote_failure,
                )
            )
            return self._finish_terminal(tuple(attempts), started)

        if producer_allowed:
            assert producer is not None
            production_started = time.monotonic()
            produced = self.store.get_or_build(
                BuildRequest(identity),
                producer,
                validation=self.config.integrity.local_lookup,
                deadline=deadline,
            )
            if produced.status in {StoreStatus.HIT, StoreStatus.BUILT, StoreStatus.JOINED}:
                assert produced.lease is not None
                attempts.append(
                    ResolutionAttempt(
                        stage=ResolutionStage.PRODUCTION,
                        result=AttemptResult.SUCCESS,
                        action=ResolutionAction.RETURN_LEASE,
                        elapsed_seconds=time.monotonic() - production_started,
                    )
                )
                return self._finish(
                    ResolutionReport(
                        resolution_id=produced.lease.provenance.resolution_id,
                        outcome=ResolutionOutcome.LOCAL_PRODUCTION,
                        attempts=tuple(attempts),
                        elapsed_seconds=time.monotonic() - started,
                    ),
                    lease=produced.lease,
                )
            attempts.append(
                self._store_attempt(
                    ResolutionStage.PRODUCTION,
                    produced,
                    time.monotonic() - production_started,
                    action=self._terminal_action_for_store_result(produced),
                )
            )
        else:
            unavailable = HostWeightFailure(
                stage=ResolutionStage.PRODUCTION,
                code=FailureCode.PRODUCER_UNAVAILABLE,
                retryable=True,
                message="no allowed producer can build the exact requested representation",
                details=CanonicalJson.empty(),
            )
            attempts.append(
                ResolutionAttempt(
                    stage=ResolutionStage.PRODUCTION,
                    result=AttemptResult.SKIPPED,
                    action=self._action_for_failure(unavailable),
                    elapsed_seconds=0.0,
                    failure=unavailable,
                )
            )

        return self._finish_terminal(tuple(attempts), started)

    def _mode_terminal_action(self) -> ResolutionAction:
        if self.config.mode is RuntimeMode.PREFERRED:
            return ResolutionAction.CANONICAL_FALLBACK
        return ResolutionAction.FAIL_STARTUP

    def _action_for_failure(self, failure: HostWeightFailure | None) -> ResolutionAction:
        if failure is None:
            raise ValueError("failed store results require a typed failure")
        if self.config.mode is RuntimeMode.PREFERRED and failure.retryable:
            return ResolutionAction.CANONICAL_FALLBACK
        return ResolutionAction.FAIL_STARTUP

    def _terminal_action_for_store_result(self, result: StoreResult) -> ResolutionAction:
        if result.status in {StoreStatus.MISS, StoreStatus.INVALID}:
            return self._mode_terminal_action()
        return self._action_for_failure(result.failure)

    def _next_after_local(
        self,
        result: StoreResult,
        producer: WeightProducer | None,
    ) -> ResolutionAction:
        if result.status in {StoreStatus.TIMEOUT, StoreStatus.FAILED}:
            return self._terminal_action_for_store_result(result)
        if self.config.remote.on_local_miss is not RemoteOnMiss.DISABLED:
            return ResolutionAction.TRY_REMOTE
        if self._producer_allowed(producer):
            if result.status is StoreStatus.INVALID:
                return ResolutionAction.QUARANTINE_THEN_PRODUCE
            return ResolutionAction.TRY_PRODUCER
        return self._mode_terminal_action()

    def _producer_allowed(self, producer: WeightProducer | None) -> bool:
        return (
            producer is not None
            and self.config.production.allow_local_build
            and producer.spec.lookup_phase is LookupPhase.PRE_LOAD_SAFE
        )

    @staticmethod
    def _store_attempt(
        stage: ResolutionStage,
        result: StoreResult,
        elapsed: float,
        *,
        action: ResolutionAction,
    ) -> ResolutionAttempt:
        if result.status is StoreStatus.MISS:
            return ResolutionAttempt(
                stage=stage,
                result=AttemptResult.MISS,
                action=action,
                elapsed_seconds=elapsed,
            )
        if result.status is StoreStatus.TIMEOUT:
            attempt_result = AttemptResult.TIMEOUT
        elif result.status is StoreStatus.INVALID:
            attempt_result = AttemptResult.ERROR
        else:
            attempt_result = AttemptResult.ERROR
        return ResolutionAttempt(
            stage=stage,
            result=attempt_result,
            action=action,
            elapsed_seconds=elapsed,
            failure=result.failure,
        )

    def _finish_terminal(
        self,
        attempts: tuple[ResolutionAttempt, ...],
        started: float,
    ) -> HostWeightResolution:
        action = attempts[-1].action
        if action is ResolutionAction.CANONICAL_FALLBACK:
            outcome = ResolutionOutcome.CANONICAL_FALLBACK
        elif action is ResolutionAction.FAIL_STARTUP:
            outcome = ResolutionOutcome.FAILED
        else:
            raise ValueError(f"resolution cannot terminate with action {action.value}")
        return self._finish(
            ResolutionReport(
                resolution_id=uuid.uuid4().hex,
                outcome=outcome,
                attempts=attempts,
                elapsed_seconds=time.monotonic() - started,
            )
        )

    def _finish(
        self,
        report: ResolutionReport,
        *,
        lease: HostWeightLease | None = None,
    ) -> HostWeightResolution:
        if self._observer is not None:
            try:
                self._observer(report)
            except Exception:
                logger.exception(
                    "Host Weight Runtime observer failed",
                    extra={"resolution_id": report.resolution_id},
                )
        return HostWeightResolution(report=report, lease=lease)


__all__ = ["HostWeightRuntime"]
