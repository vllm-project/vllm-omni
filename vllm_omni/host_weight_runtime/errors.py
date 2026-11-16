# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stable Host Weight Runtime failures and actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .identity import CanonicalJson


class ResolutionStage(str, Enum):
    CONFIGURATION = "configuration"
    DOMAIN = "domain"
    LOOKUP = "lookup"
    VALIDATION = "validation"
    REMOTE = "remote"
    PRODUCTION = "production"
    CAPACITY = "capacity"
    RESTORATION = "restoration"
    LIFECYCLE = "lifecycle"
    CANONICAL_LOADING = "canonical_loading"


class FailureCode(str, Enum):
    INVALID_CONFIG = "invalid_config"
    COHORT_POLICY_MISMATCH = "cohort_policy_mismatch"
    DOMAIN_POLICY_MISMATCH = "domain_policy_mismatch"
    DOMAIN_UNAVAILABLE = "domain_unavailable"
    STORAGE_CLASS_MISMATCH = "storage_class_mismatch"
    NUMA_PLACEMENT_INVALID = "numa_placement_invalid"
    LOCAL_MISS = "local_miss"
    ARTIFACT_DENIED = "artifact_denied"
    ACTIVE_BUILD_TIMEOUT = "active_build_timeout"
    MANIFEST_INCOMPATIBLE = "manifest_incompatible"
    MANIFEST_CORRUPT = "manifest_corrupt"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    IDENTITY_COLLISION = "identity_collision"
    REMOTE_MISS = "remote_miss"
    REMOTE_UNAVAILABLE = "remote_unavailable"
    REMOTE_AUTH_FAILED = "remote_auth_failed"
    REMOTE_TIMEOUT = "remote_timeout"
    REMOTE_ARTIFACT_INVALID = "remote_artifact_invalid"
    PRODUCER_UNAVAILABLE = "producer_unavailable"
    PRODUCER_UNSUPPORTED = "producer_unsupported"
    PRODUCER_FAILED = "producer_failed"
    PRODUCTION_TIMEOUT = "production_timeout"
    ARTIFACT_TOO_LARGE = "artifact_too_large"
    STORE_LIMIT_EXCEEDED = "store_limit_exceeded"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    ENOSPC = "enospc"
    RESTORER_UNAVAILABLE = "restorer_unavailable"
    RESTORE_PREFLIGHT_FAILED = "restore_preflight_failed"
    RESTORE_COMMIT_FAILED = "restore_commit_failed"
    ACTIVE_LEASE = "active_lease"
    PUBLICATION_FAILED = "publication_failed"
    QUARANTINE_FAILED = "quarantine_failed"
    CANONICAL_SOURCE_FAILED = "canonical_source_failed"
    UNSUPPORTED = "unsupported"


class ResolutionAction(str, Enum):
    RETURN_LEASE = "return_lease"
    TRY_REMOTE = "try_remote"
    TRY_PRODUCER = "try_producer"
    QUARANTINE_THEN_PRODUCE = "quarantine_then_produce"
    CANONICAL_FALLBACK = "canonical_fallback"
    FAIL_STARTUP = "fail_startup"
    SKIP = "skip"


@dataclass(frozen=True)
class HostWeightFailure:
    stage: ResolutionStage
    code: FailureCode
    retryable: bool
    message: str
    details: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ResolutionStage) or not isinstance(self.code, FailureCode):
            raise ValueError("host weight failure stage and code must use their typed enums")
        if not isinstance(self.retryable, bool):
            raise ValueError("host weight failure retryable must be a boolean")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("host weight failure requires a message")
        if not isinstance(self.details, CanonicalJson):
            raise ValueError("host weight failure details must use CanonicalJson")


class HostWeightError(RuntimeError):
    """Internal exception carrying a stable, policy-neutral failure."""

    def __init__(self, failure: HostWeightFailure):
        super().__init__(failure.message)
        self.failure = failure


__all__ = [
    "FailureCode",
    "HostWeightError",
    "HostWeightFailure",
    "ResolutionAction",
    "ResolutionStage",
]
