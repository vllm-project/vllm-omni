# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Versioned immutable manifests for runtime-ready host weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import cast

import regex as re

from .identity import CanonicalJson, JsonValue, WeightArtifactIdentity, canonical_json

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestValidationError(ValueError):
    """Raised when a published manifest violates its schema."""


class TensorKind(str, Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{name} must be a non-empty string")
    return value


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{name} must be an integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_safe_file_name(value: object) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError("artifact file name must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or path.name != value or value in {".", ".."}:
        raise ManifestValidationError(f"artifact file name must be one safe relative basename: {value!r}")
    return value


def _require_int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{name} must be an array")
    return tuple(_require_int(f"{name}[{index}]", item) for index, item in enumerate(value))


def _exact_keys(value: dict[str, object], expected: set[str], kind: str) -> None:
    keys = set(value)
    if keys != expected:
        raise ManifestValidationError(
            f"{kind} fields differ: missing={sorted(expected - keys)}, unexpected={sorted(keys - expected)}"
        )


@dataclass(frozen=True)
class TensorManifestEntry:
    name: str
    kind: TensorKind
    role: str
    file_name: str
    storage_key: str
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    nbytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_text("tensor.name", self.name)
        _require_text("tensor.storage_key", self.storage_key)
        _require_text("tensor.role", self.role)
        _require_text("tensor.dtype", self.dtype)
        if not isinstance(self.kind, TensorKind):
            raise ManifestValidationError("tensor kind must use TensorKind")
        _require_safe_file_name(self.file_name)
        if not isinstance(self.shape, tuple) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in self.shape
        ):
            raise ManifestValidationError(f"tensor {self.name!r} has a negative shape dimension")
        if (
            not isinstance(self.stride, tuple)
            or len(self.stride) != len(self.shape)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.stride)
        ):
            raise ManifestValidationError(f"tensor {self.name!r} has an invalid stride")
        if _require_int(f"tensors[{self.name!r}].nbytes", self.nbytes) < 0:
            raise ManifestValidationError(f"tensor {self.name!r} has negative nbytes")
        _require_sha256(f"tensors[{self.name!r}].sha256", self.sha256)

    def semantic_dict(self) -> dict[str, JsonValue]:
        """Return content identity independent of physical file sharding."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "stride": list(self.stride),
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "role": self.role,
            "file_name": self.file_name,
            "storage_key": self.storage_key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "stride": list(self.stride),
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TensorManifestEntry:
        expected = {
            "name",
            "kind",
            "role",
            "file_name",
            "storage_key",
            "shape",
            "dtype",
            "stride",
            "nbytes",
            "sha256",
        }
        _exact_keys(value, expected, "tensor manifest entry")
        kind_value = _require_text("tensor.kind", value["kind"])
        try:
            kind = TensorKind(kind_value)
        except ValueError as exc:
            raise ManifestValidationError(f"unknown tensor kind {value['kind']!r}") from exc
        return cls(
            name=_require_text("tensor.name", value["name"]),
            kind=kind,
            role=_require_text("tensor.role", value["role"]),
            file_name=_require_safe_file_name(value["file_name"]),
            storage_key=_require_text("tensor.storage_key", value["storage_key"]),
            shape=_require_int_tuple("tensor.shape", value["shape"]),
            dtype=_require_text("tensor.dtype", value["dtype"]),
            stride=_require_int_tuple("tensor.stride", value["stride"]),
            nbytes=_require_int("tensor.nbytes", value["nbytes"]),
            sha256=_require_sha256("tensor.sha256", value["sha256"]),
        )


@dataclass(frozen=True)
class FileManifestEntry:
    file_name: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _require_safe_file_name(self.file_name)
        if _require_int(f"files[{self.file_name!r}].size", self.size) < 0:
            raise ManifestValidationError(f"artifact file {self.file_name!r} has negative size")
        _require_sha256(f"files[{self.file_name!r}].sha256", self.sha256)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"file_name": self.file_name, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> FileManifestEntry:
        _exact_keys(value, {"file_name", "size", "sha256"}, "file manifest entry")
        return cls(
            file_name=_require_safe_file_name(value["file_name"]),
            size=_require_int("file.size", value["size"]),
            sha256=_require_sha256("file.sha256", value["sha256"]),
        )


@dataclass(frozen=True)
class WeightManifest:
    schema_version: int
    artifact_key: str
    identity: WeightArtifactIdentity
    producer_schema: str
    restorer_schema: str
    tensors: tuple[TensorManifestEntry, ...]
    files: tuple[FileManifestEntry, ...]
    artifact_content_sha256: str
    format_metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        if _require_int("manifest.schema_version", self.schema_version) < 1:
            raise ManifestValidationError("manifest schema_version must be positive")
        _require_sha256("artifact_key", self.artifact_key)
        _require_sha256("artifact_content_sha256", self.artifact_content_sha256)
        if not isinstance(self.identity, WeightArtifactIdentity):
            raise ManifestValidationError("manifest identity must use WeightArtifactIdentity")
        if self.artifact_key != self.identity.key:
            raise ManifestValidationError("manifest artifact_key does not match its semantic identity")
        _require_text("manifest.producer_schema", self.producer_schema)
        _require_text("manifest.restorer_schema", self.restorer_schema)
        if (
            not isinstance(self.tensors, tuple)
            or not isinstance(self.files, tuple)
            or not self.tensors
            or not self.files
            or any(not isinstance(entry, TensorManifestEntry) for entry in self.tensors)
            or any(not isinstance(entry, FileManifestEntry) for entry in self.files)
        ):
            raise ManifestValidationError("an artifact must contain at least one tensor and one file")
        if not isinstance(self.format_metadata, CanonicalJson):
            raise ManifestValidationError("manifest format_metadata must use CanonicalJson")
        tensor_names = [entry.name for entry in self.tensors]
        if tensor_names != sorted(tensor_names) or len(set(tensor_names)) != len(tensor_names):
            raise ManifestValidationError("manifest tensors must have unique names in canonical order")
        file_names = [entry.file_name for entry in self.files]
        if file_names != sorted(file_names) or len(set(file_names)) != len(file_names):
            raise ManifestValidationError("manifest files must have unique names in canonical order")
        available_files = set(file_names)
        if any(entry.file_name not in available_files for entry in self.tensors):
            raise ManifestValidationError("tensor manifest references an undeclared artifact file")
        storage_bindings = [(entry.file_name, entry.storage_key) for entry in self.tensors]
        if len(set(storage_bindings)) != len(storage_bindings):
            raise ManifestValidationError("manifest tensor storage bindings must be unique")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def total_tensor_bytes(self) -> int:
        return sum(entry.nbytes for entry in self.tensors)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_key": self.artifact_key,
            "identity": self.identity.to_dict(),
            "producer_schema": self.producer_schema,
            "restorer_schema": self.restorer_schema,
            "tensors": [entry.to_dict() for entry in self.tensors],
            "files": [entry.to_dict() for entry in self.files],
            "artifact_content_sha256": self.artifact_content_sha256,
            "format_metadata": self.format_metadata.to_value(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WeightManifest:
        expected = {
            "schema_version",
            "artifact_key",
            "identity",
            "producer_schema",
            "restorer_schema",
            "tensors",
            "files",
            "artifact_content_sha256",
            "format_metadata",
        }
        _exact_keys(value, expected, "weight manifest")
        identity = value["identity"]
        if not isinstance(identity, dict):
            raise ManifestValidationError("manifest identity must be an object")
        tensors = value["tensors"]
        files = value["files"]
        if not isinstance(tensors, list) or not all(isinstance(item, dict) for item in tensors):
            raise ManifestValidationError("manifest tensors must be an array of objects")
        if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
            raise ManifestValidationError("manifest files must be an array of objects")
        return cls(
            schema_version=_require_int("manifest.schema_version", value["schema_version"]),
            artifact_key=_require_sha256("manifest.artifact_key", value["artifact_key"]),
            identity=WeightArtifactIdentity.from_dict(cast(dict[str, object], identity)),
            producer_schema=_require_text("manifest.producer_schema", value["producer_schema"]),
            restorer_schema=_require_text("manifest.restorer_schema", value["restorer_schema"]),
            tensors=tuple(TensorManifestEntry.from_dict(cast(dict[str, object], item)) for item in tensors),
            files=tuple(FileManifestEntry.from_dict(cast(dict[str, object], item)) for item in files),
            artifact_content_sha256=_require_sha256(
                "manifest.artifact_content_sha256", value["artifact_content_sha256"]
            ),
            format_metadata=CanonicalJson.from_value(value["format_metadata"]),
        )


@dataclass(frozen=True)
class PublicationMarker:
    schema_version: int
    publication_id: str
    manifest_sha256: str
    artifact_content_sha256: str
    integrity_mode: str

    def __post_init__(self) -> None:
        if _require_int("publication.schema_version", self.schema_version) < 1:
            raise ManifestValidationError("publication schema_version must be positive")
        _require_text("publication.publication_id", self.publication_id)
        _require_text("publication.integrity_mode", self.integrity_mode)
        _require_sha256("publication.manifest_sha256", self.manifest_sha256)
        _require_sha256("publication.artifact_content_sha256", self.artifact_content_sha256)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "manifest_sha256": self.manifest_sha256,
            "artifact_content_sha256": self.artifact_content_sha256,
            "integrity_mode": self.integrity_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> PublicationMarker:
        expected = {
            "schema_version",
            "publication_id",
            "manifest_sha256",
            "artifact_content_sha256",
            "integrity_mode",
        }
        _exact_keys(value, expected, "publication marker")
        return cls(
            schema_version=_require_int("publication.schema_version", value["schema_version"]),
            publication_id=_require_text("publication.publication_id", value["publication_id"]),
            manifest_sha256=_require_sha256("publication.manifest_sha256", value["manifest_sha256"]),
            artifact_content_sha256=_require_sha256(
                "publication.artifact_content_sha256", value["artifact_content_sha256"]
            ),
            integrity_mode=_require_text("publication.integrity_mode", value["integrity_mode"]),
        )


__all__ = [
    "FileManifestEntry",
    "ManifestValidationError",
    "PublicationMarker",
    "TensorKind",
    "TensorManifestEntry",
    "WeightManifest",
]
