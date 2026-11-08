# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed, read-only snapshots for local Host Weight Store inspection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .config import ValidationLevel
from .errors import HostWeightFailure
from .identity import WeightArtifactIdentity
from .manifest import WeightManifest
from .outcomes import StoreStatus


class ArtifactInventoryState(str, Enum):
    READY = "ready"
    DENIED = "denied"
    QUARANTINED = "quarantined"
    TEMPORARY = "temporary"


@dataclass(frozen=True)
class ArtifactInventoryEntry:
    artifact_key: str
    state: ArtifactInventoryState
    storage_name: str
    size_bytes: int
    reason: str | None = None


@dataclass(frozen=True)
class DomainInspection:
    domain_uuid: str
    scope: str
    domain_id: str
    storage_class: str
    filesystem_type: str
    policy_version: int
    store_bytes: int
    filesystem_free_bytes: int
    max_artifact_bytes: int | None
    max_store_bytes: int | None
    min_free_bytes: int
    inventory: tuple[ArtifactInventoryEntry, ...]


@dataclass(frozen=True)
class ArtifactInspection:
    artifact_key: str
    status: StoreStatus
    size_bytes: int
    manifest: WeightManifest | None
    publication_id: str | None
    backing_fingerprint: str | None
    failure: HostWeightFailure | None
    build_lock_active: bool
    artifact_lock_active: bool


class HostWeightInspector(Protocol):
    def inspect_domain(self) -> DomainInspection: ...

    def inspect_artifact(
        self,
        identity: WeightArtifactIdentity,
        *,
        validation: ValidationLevel,
    ) -> ArtifactInspection: ...


__all__ = [
    "ArtifactInspection",
    "ArtifactInventoryEntry",
    "ArtifactInventoryState",
    "DomainInspection",
    "HostWeightInspector",
]
