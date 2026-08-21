# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed store and end-to-end resolution outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .errors import HostWeightFailure, ResolutionAction, ResolutionStage

if TYPE_CHECKING:
    from .lease import HostWeightLease


class StoreStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    BUILT = "built"
    JOINED = "joined"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True)
class StoreResult:
    status: StoreStatus
    lease: HostWeightLease | None = None
    failure: HostWeightFailure | None = None

    def __post_init__(self) -> None:
        has_lease = self.status in {StoreStatus.HIT, StoreStatus.BUILT, StoreStatus.JOINED}
        if has_lease != (self.lease is not None):
            raise ValueError(f"store result {self.status.value} has inconsistent lease ownership")
        if self.status in {StoreStatus.HIT, StoreStatus.MISS, StoreStatus.BUILT, StoreStatus.JOINED}:
            if self.failure is not None:
                raise ValueError(f"successful store result {self.status.value} must not carry a failure")
        if self.status in {StoreStatus.INVALID, StoreStatus.TIMEOUT, StoreStatus.FAILED} and self.failure is None:
            raise ValueError(f"store result {self.status.value} requires a typed failure")


class AttemptResult(str, Enum):
    HIT = "hit"
    MISS = "miss"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class ResolutionOutcome(str, Enum):
    LOCAL_HIT = "local_hit"
    REMOTE_IMPORT = "remote_import"
    LOCAL_PRODUCTION = "local_production"
    CANONICAL_DIRECT = "canonical_direct"
    CANONICAL_FALLBACK = "canonical_fallback"
    FAILED = "failed"


@dataclass(frozen=True)
class ResolutionAttempt:
    stage: ResolutionStage
    result: AttemptResult
    action: ResolutionAction
    elapsed_seconds: float
    failure: HostWeightFailure | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("resolution attempt elapsed_seconds must be finite and non-negative")
        if self.result in {AttemptResult.HIT, AttemptResult.MISS, AttemptResult.SUCCESS} and self.failure is not None:
            raise ValueError(f"resolution attempt {self.result.value} must not carry a failure")
        if self.result in {AttemptResult.TIMEOUT, AttemptResult.ERROR} and self.failure is None:
            raise ValueError(f"resolution attempt {self.result.value} requires a typed failure")


@dataclass(frozen=True)
class ResolutionReport:
    resolution_id: str
    outcome: ResolutionOutcome
    attempts: tuple[ResolutionAttempt, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.resolution_id:
            raise ValueError("resolution report requires a resolution_id")
        if not isinstance(self.attempts, tuple):
            raise ValueError("resolution report attempts must be an immutable tuple")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("resolution report elapsed_seconds must be finite and non-negative")


@dataclass(frozen=True)
class HostWeightResolution:
    report: ResolutionReport
    lease: HostWeightLease | None = None

    def __post_init__(self) -> None:
        lease_outcomes = {
            ResolutionOutcome.LOCAL_HIT,
            ResolutionOutcome.REMOTE_IMPORT,
            ResolutionOutcome.LOCAL_PRODUCTION,
        }
        if (self.report.outcome in lease_outcomes) != (self.lease is not None):
            raise ValueError("resolution outcome has inconsistent lease ownership")


__all__ = [
    "AttemptResult",
    "HostWeightResolution",
    "ResolutionAttempt",
    "ResolutionOutcome",
    "ResolutionReport",
    "StoreResult",
    "StoreStatus",
]
