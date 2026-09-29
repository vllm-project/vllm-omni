# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed policy for the loader-adjacent Host Weight Runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RuntimeMode(str, Enum):
    DISABLED = "disabled"
    PREFERRED = "preferred"
    REQUIRED = "required"


class StorageScope(str, Enum):
    NODE = "node"
    NUMA = "numa"


class StorageClass(str, Enum):
    DISK = "disk"
    TMPFS = "tmpfs"


class ValidationLevel(str, Enum):
    MANIFEST_AND_METADATA = "manifest_and_metadata"
    FULL_CHECKSUM = "full_checksum"
    FS_VERITY = "fs_verity"


class RemoteOnMiss(str, Enum):
    DISABLED = "disabled"
    TRY = "try"
    REQUIRE = "require"


@dataclass(frozen=True)
class StorageDomainPolicy:
    root: Path
    scope: StorageScope = StorageScope.NODE
    domain_id: str = "node"
    storage_class: StorageClass = StorageClass.DISK
    numa_prefault: bool = False
    verify_numa_residency: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not str(self.root):
            raise ValueError("host weight store root must not be empty")
        if not isinstance(self.scope, StorageScope) or not isinstance(self.storage_class, StorageClass):
            raise ValueError("storage scope and class must use their typed enum values")
        if not isinstance(self.domain_id, str) or not self.domain_id:
            raise ValueError("host weight store domain_id must not be empty")
        if not isinstance(self.numa_prefault, bool) or not isinstance(self.verify_numa_residency, bool):
            raise ValueError("NUMA placement options must be booleans")
        if self.scope is StorageScope.NODE and (self.numa_prefault or self.verify_numa_residency):
            raise ValueError("NUMA placement options require scope='numa'")


@dataclass(frozen=True)
class CapacityPolicy:
    max_artifact_bytes: int | None = None
    max_store_bytes: int | None = None
    min_free_bytes: int = 0
    eviction: str = "none"

    def __post_init__(self) -> None:
        for name, value in (
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("max_store_bytes", self.max_store_bytes),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be positive when configured")
        if isinstance(self.min_free_bytes, bool) or not isinstance(self.min_free_bytes, int) or self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must not be negative")
        if not isinstance(self.eviction, str) or self.eviction != "none":
            raise ValueError("automatic host weight eviction is not implemented")


@dataclass(frozen=True)
class IntegrityPolicy:
    local_lookup: ValidationLevel = ValidationLevel.MANIFEST_AND_METADATA
    require_trusted_permissions: bool = True
    background_scrub: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.local_lookup, ValidationLevel):
            raise ValueError("local lookup validation must use ValidationLevel")
        if not isinstance(self.require_trusted_permissions, bool) or not isinstance(self.background_scrub, bool):
            raise ValueError("integrity switches must be booleans")


@dataclass(frozen=True)
class RemoteImportPolicy:
    on_local_miss: RemoteOnMiss = RemoteOnMiss.DISABLED
    providers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.on_local_miss, RemoteOnMiss):
            raise ValueError("remote on-miss policy must use RemoteOnMiss")
        if not isinstance(self.providers, tuple):
            raise ValueError("remote providers must be an immutable tuple")
        if self.on_local_miss is not RemoteOnMiss.DISABLED and not self.providers:
            raise ValueError("remote providers are required when remote import is enabled")
        if any(not isinstance(provider, str) or not provider for provider in self.providers):
            raise ValueError("remote provider identifiers must not be empty")


@dataclass(frozen=True)
class ProductionPolicy:
    allow_local_build: bool = True
    allow_post_load_publish: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_local_build, bool) or not isinstance(self.allow_post_load_publish, bool):
            raise ValueError("production switches must be booleans")


@dataclass(frozen=True)
class WaitPolicy:
    coordination_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        timeout = self.coordination_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("coordination_timeout_seconds must be positive")


@dataclass(frozen=True)
class HostWeightRuntimeConfig:
    mode: RuntimeMode = RuntimeMode.DISABLED
    domain: StorageDomainPolicy | None = None
    remote: RemoteImportPolicy = field(default_factory=RemoteImportPolicy)
    production: ProductionPolicy = field(default_factory=ProductionPolicy)
    capacity: CapacityPolicy = field(default_factory=CapacityPolicy)
    wait: WaitPolicy = field(default_factory=WaitPolicy)
    integrity: IntegrityPolicy = field(default_factory=IntegrityPolicy)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("only Host Weight Runtime config schema_version=1 is supported")
        if not isinstance(self.mode, RuntimeMode):
            raise ValueError("runtime mode must use RuntimeMode")
        if self.domain is not None and not isinstance(self.domain, StorageDomainPolicy):
            raise ValueError("runtime domain must use StorageDomainPolicy")
        for name, value, expected in (
            ("remote", self.remote, RemoteImportPolicy),
            ("production", self.production, ProductionPolicy),
            ("capacity", self.capacity, CapacityPolicy),
            ("wait", self.wait, WaitPolicy),
            ("integrity", self.integrity, IntegrityPolicy),
        ):
            if not isinstance(value, expected):
                raise ValueError(f"runtime {name} policy has the wrong type")
        if self.mode is not RuntimeMode.DISABLED and self.domain is None:
            raise ValueError("enabled Host Weight Runtime requires a storage domain")


__all__ = [
    "CapacityPolicy",
    "HostWeightRuntimeConfig",
    "IntegrityPolicy",
    "ProductionPolicy",
    "RemoteImportPolicy",
    "RemoteOnMiss",
    "RuntimeMode",
    "StorageClass",
    "StorageDomainPolicy",
    "StorageScope",
    "ValidationLevel",
    "WaitPolicy",
]
