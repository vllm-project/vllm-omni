# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deterministic semantic identity for runtime-ready host weights."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class IdentityValidationError(ValueError):
    """Raised when an artifact identity is not deterministic JSON."""


def _validate_json(value: object, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityValidationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, list | tuple):
        return [_validate_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityValidationError(f"{path} contains a non-string mapping key")
            normalized[key] = _validate_json(item, f"{path}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    raise IdentityValidationError(f"{path} contains unsupported value type {type(value).__qualname__}")


def canonical_json(value: object) -> bytes:
    """Return one stable JSON encoding, rejecting process-specific objects."""
    return json.dumps(
        _validate_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class CanonicalJson:
    """Immutable snapshot of deterministic JSON metadata."""

    encoded: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.encoded, bytes):
            raise IdentityValidationError("canonical metadata must be immutable bytes")
        try:
            value = json.loads(self.encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IdentityValidationError("canonical metadata is not valid JSON") from exc
        if canonical_json(value) != self.encoded:
            raise IdentityValidationError("canonical metadata does not use the required stable encoding")

    @classmethod
    def from_value(cls, value: object) -> CanonicalJson:
        return cls(canonical_json(value))

    @classmethod
    def empty(cls) -> CanonicalJson:
        return cls(b"{}")

    def to_value(self) -> JsonValue:
        return json.loads(self.encoded)


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityValidationError(f"{name} must not be empty")
    return value


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentityValidationError(f"{name} must be an integer")
    return value


def _require_metadata(name: str, value: object) -> None:
    if not isinstance(value, CanonicalJson):
        raise IdentityValidationError(f"{name} must use CanonicalJson")


def _require_exact_keys(value: dict[str, object], expected: set[str], kind: str) -> None:
    keys = set(value)
    if keys != expected:
        raise IdentityValidationError(
            f"{kind} fields differ: missing={sorted(expected - keys)}, unexpected={sorted(keys - expected)}"
        )


@dataclass(frozen=True)
class WeightSourceIdentity:
    model_id: str
    revision: str
    fingerprint: str
    metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        _require_text("source.model_id", self.model_id)
        _require_text("source.revision", self.revision)
        _require_text("source.fingerprint", self.fingerprint)
        _require_metadata("source.metadata", self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata.to_value(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WeightSourceIdentity:
        _require_exact_keys(value, {"model_id", "revision", "fingerprint", "metadata"}, "source identity")
        return cls(
            model_id=_require_text("source.model_id", value["model_id"]),
            revision=_require_text("source.revision", value["revision"]),
            fingerprint=_require_text("source.fingerprint", value["fingerprint"]),
            metadata=CanonicalJson.from_value(value["metadata"]),
        )


@dataclass(frozen=True)
class ComponentIdentity:
    name: str
    ownership: str
    metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        _require_text("component.name", self.name)
        _require_text("component.ownership", self.ownership)
        _require_metadata("component.metadata", self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "ownership": self.ownership, "metadata": self.metadata.to_value()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ComponentIdentity:
        _require_exact_keys(value, {"name", "ownership", "metadata"}, "component identity")
        return cls(
            name=_require_text("component.name", value["name"]),
            ownership=_require_text("component.ownership", value["ownership"]),
            metadata=CanonicalJson.from_value(value["metadata"]),
        )


@dataclass(frozen=True)
class WeightRepresentation:
    name: str
    dtype: str
    metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        _require_text("representation.name", self.name)
        _require_text("representation.dtype", self.dtype)
        _require_metadata("representation.metadata", self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "dtype": self.dtype, "metadata": self.metadata.to_value()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WeightRepresentation:
        _require_exact_keys(value, {"name", "dtype", "metadata"}, "weight representation")
        return cls(
            name=_require_text("representation.name", value["name"]),
            dtype=_require_text("representation.dtype", value["dtype"]),
            metadata=CanonicalJson.from_value(value["metadata"]),
        )


@dataclass(frozen=True)
class RuntimeWeightLayout:
    name: str
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    sequence_parallel_size: int = 1
    sequence_parallel_backend: str = "none"
    metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        _require_text("layout.name", self.name)
        tensor_parallel_size = _require_int("layout.tensor_parallel_size", self.tensor_parallel_size)
        tensor_parallel_rank = _require_int("layout.tensor_parallel_rank", self.tensor_parallel_rank)
        sequence_parallel_size = _require_int("layout.sequence_parallel_size", self.sequence_parallel_size)
        if tensor_parallel_size < 1:
            raise IdentityValidationError("tensor_parallel_size must be positive")
        if not 0 <= tensor_parallel_rank < tensor_parallel_size:
            raise IdentityValidationError("tensor_parallel_rank must be within tensor_parallel_size")
        if sequence_parallel_size < 1:
            raise IdentityValidationError("sequence_parallel_size must be positive")
        _require_text("layout.sequence_parallel_backend", self.sequence_parallel_backend)
        _require_metadata("layout.metadata", self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "sequence_parallel_backend": self.sequence_parallel_backend,
            "metadata": self.metadata.to_value(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> RuntimeWeightLayout:
        expected = {
            "name",
            "tensor_parallel_size",
            "tensor_parallel_rank",
            "sequence_parallel_size",
            "sequence_parallel_backend",
            "metadata",
        }
        _require_exact_keys(value, expected, "runtime weight layout")
        return cls(
            name=_require_text("layout.name", value["name"]),
            tensor_parallel_size=_require_int("layout.tensor_parallel_size", value["tensor_parallel_size"]),
            tensor_parallel_rank=_require_int("layout.tensor_parallel_rank", value["tensor_parallel_rank"]),
            sequence_parallel_size=_require_int("layout.sequence_parallel_size", value["sequence_parallel_size"]),
            sequence_parallel_backend=_require_text(
                "layout.sequence_parallel_backend", value["sequence_parallel_backend"]
            ),
            metadata=CanonicalJson.from_value(value["metadata"]),
        )


@dataclass(frozen=True)
class AdaptationIdentity:
    kind: str = "base"
    fingerprint: str | None = None
    metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        _require_text("adaptation.kind", self.kind)
        if self.kind != "base" and not self.fingerprint:
            raise IdentityValidationError("non-base adaptation requires a fingerprint")
        if self.fingerprint is not None and (not isinstance(self.fingerprint, str) or not self.fingerprint):
            raise IdentityValidationError("adaptation fingerprint must be a non-empty string or null")
        _require_metadata("adaptation.metadata", self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "fingerprint": self.fingerprint, "metadata": self.metadata.to_value()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> AdaptationIdentity:
        _require_exact_keys(value, {"kind", "fingerprint", "metadata"}, "adaptation identity")
        fingerprint = value["fingerprint"]
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise IdentityValidationError("adaptation fingerprint must be a string or null")
        return cls(
            kind=_require_text("adaptation.kind", value["kind"]),
            fingerprint=fingerprint,
            metadata=CanonicalJson.from_value(value["metadata"]),
        )


@dataclass(frozen=True)
class ProducerIdentity:
    producer_id: str
    version: str
    implementation_fingerprint: str
    manifest_schema: str
    restorer_schema: str

    def __post_init__(self) -> None:
        for name, value in (
            ("producer.producer_id", self.producer_id),
            ("producer.version", self.version),
            ("producer.implementation_fingerprint", self.implementation_fingerprint),
            ("producer.manifest_schema", self.manifest_schema),
            ("producer.restorer_schema", self.restorer_schema),
        ):
            _require_text(name, value)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "producer_id": self.producer_id,
            "version": self.version,
            "implementation_fingerprint": self.implementation_fingerprint,
            "manifest_schema": self.manifest_schema,
            "restorer_schema": self.restorer_schema,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ProducerIdentity:
        expected = {
            "producer_id",
            "version",
            "implementation_fingerprint",
            "manifest_schema",
            "restorer_schema",
        }
        _require_exact_keys(value, expected, "producer identity")
        return cls(
            producer_id=_require_text("producer.producer_id", value["producer_id"]),
            version=_require_text("producer.version", value["version"]),
            implementation_fingerprint=_require_text(
                "producer.implementation_fingerprint", value["implementation_fingerprint"]
            ),
            manifest_schema=_require_text("producer.manifest_schema", value["manifest_schema"]),
            restorer_schema=_require_text("producer.restorer_schema", value["restorer_schema"]),
        )


@dataclass(frozen=True)
class WeightArtifactIdentity:
    schema_version: int
    source: WeightSourceIdentity
    component: ComponentIdentity
    representation: WeightRepresentation
    layout: RuntimeWeightLayout
    adaptation: AdaptationIdentity
    producer: ProducerIdentity

    def __post_init__(self) -> None:
        if _require_int("identity.schema_version", self.schema_version) < 1:
            raise IdentityValidationError("identity schema_version must be positive")
        for name, value, expected in (
            ("source", self.source, WeightSourceIdentity),
            ("component", self.component, ComponentIdentity),
            ("representation", self.representation, WeightRepresentation),
            ("layout", self.layout, RuntimeWeightLayout),
            ("adaptation", self.adaptation, AdaptationIdentity),
            ("producer", self.producer, ProducerIdentity),
        ):
            if not isinstance(value, expected):
                raise IdentityValidationError(f"identity {name} has the wrong contract type")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "component": self.component.to_dict(),
            "representation": self.representation.to_dict(),
            "layout": self.layout.to_dict(),
            "adaptation": self.adaptation.to_dict(),
            "producer": self.producer.to_dict(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def key(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WeightArtifactIdentity:
        expected = {
            "schema_version",
            "source",
            "component",
            "representation",
            "layout",
            "adaptation",
            "producer",
        }
        _require_exact_keys(value, expected, "weight artifact identity")
        source = value["source"]
        component = value["component"]
        representation = value["representation"]
        layout = value["layout"]
        adaptation = value["adaptation"]
        producer = value["producer"]
        if any(
            not isinstance(section, dict)
            for section in (source, component, representation, layout, adaptation, producer)
        ):
            raise IdentityValidationError("identity sections must be JSON objects")
        return cls(
            schema_version=_require_int("identity.schema_version", value["schema_version"]),
            source=WeightSourceIdentity.from_dict(cast(dict[str, object], source)),
            component=ComponentIdentity.from_dict(cast(dict[str, object], component)),
            representation=WeightRepresentation.from_dict(cast(dict[str, object], representation)),
            layout=RuntimeWeightLayout.from_dict(cast(dict[str, object], layout)),
            adaptation=AdaptationIdentity.from_dict(cast(dict[str, object], adaptation)),
            producer=ProducerIdentity.from_dict(cast(dict[str, object], producer)),
        )


__all__ = [
    "AdaptationIdentity",
    "CanonicalJson",
    "ComponentIdentity",
    "IdentityValidationError",
    "JsonValue",
    "ProducerIdentity",
    "RuntimeWeightLayout",
    "WeightArtifactIdentity",
    "WeightRepresentation",
    "WeightSourceIdentity",
    "canonical_json",
]
