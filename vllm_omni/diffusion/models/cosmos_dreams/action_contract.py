# SPDX-License-Identifier: Apache-2.0
"""Strict, self-verifying Cosmos-Dreams action contract."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

ACTION_SCHEMA_VERSION = 2
NORMALIZER_SCHEMA_VERSION = 1
RAW_ACTION_DIM = 29
MODEL_ACTION_DIM = 64
ACTION_TOKENS_PER_FRAME = 4
NUM_EMBODIMENT_DOMAINS = 32
AGIBOT_DOMAIN_ID = 15
RANGE_FLOOR = 1e-8
LAYOUT_ID = "agibot_backward_framewise_rot6d_v1"
ACTION_NORMALIZERS_RELATIVE_DIR = "projects/cosmos3/cosmos3/datasets/action/normalizers"
SHARED_AGIBOT_STATS = "agibot_backward_framewise_rot6d.json"
LEGACY_GEAR_STATS = "agibot_gear_gripper_backward_framewise_rot6d.json"

_SUPPORTED_DATASETS: dict[str, tuple[str, frozenset[str]]] = {
    "AgiBotWorldBetaDataset": ("agibotworld", frozenset({SHARED_AGIBOT_STATS})),
    "AgibotGEARGripperDataset": (
        "agibot_gear_gripper",
        frozenset({SHARED_AGIBOT_STATS, LEGACY_GEAR_STATS}),
    ),
    "AgibotGEARGripperExtDataset": (
        "agibot_gear_gripper_ext",
        frozenset({SHARED_AGIBOT_STATS, LEGACY_GEAR_STATS}),
    ),
}


def float32_value(value: float) -> float:
    """Round a finite scalar to runtime normalization precision."""

    result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    if not math.isfinite(result):
        raise ValueError(f"Cosmos-Dreams action-contract value must be finite, got {value!r}.")
    return 0.0 if result == 0.0 else result


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return float32_value(value)
    return value


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Hash semantic JSON with the producer's float32 canonicalization."""

    encoded = json.dumps(
        _canonicalize(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionLayoutField(_StrictModel):
    name: str
    offset: StrictInt = Field(ge=0)
    size: StrictInt = Field(gt=0)
    unit: str
    representation: str | None = None
    closed_value: StrictFloat | None = None
    open_value: StrictFloat | None = None


class ActionLayout(_StrictModel):
    id: Literal["agibot_backward_framewise_rot6d_v1"]
    pose_convention: Literal["backward_framewise"]
    delta_equation: Literal["T_i^-1 @ T_{i+1}"]
    rotation_representation: Literal["rot6d_columns"]
    fields: tuple[ActionLayoutField, ...]

    @model_validator(mode="after")
    def validate_target_layout(self) -> ActionLayout:
        expected = (
            ("head_translation", 0, 3, "meter", None, None, None),
            ("head_rotation", 3, 6, "dimensionless", "rot6d_columns", None, None),
            ("right_translation", 9, 3, "meter", None, None, None),
            ("right_rotation", 12, 6, "dimensionless", "rot6d_columns", None, None),
            ("right_gripper", 18, 1, "open_fraction", None, 0.0, 1.0),
            ("left_translation", 19, 3, "meter", None, None, None),
            ("left_rotation", 22, 6, "dimensionless", "rot6d_columns", None, None),
            ("left_gripper", 28, 1, "open_fraction", None, 0.0, 1.0),
        )
        actual = tuple(
            (
                field.name,
                field.offset,
                field.size,
                field.unit,
                field.representation,
                field.closed_value,
                field.open_value,
            )
            for field in self.fields
        )
        if actual != expected:
            raise ValueError(f"Cosmos-Dreams action layout must exactly match {LAYOUT_ID!r}.")
        return self


class ActionPadding(_StrictModel):
    stage: Literal["after_normalization"]
    value: Literal[0.0]


class AffineTransform(_StrictModel):
    type: Literal["affine"]
    offset: tuple[StrictFloat, ...]
    scale: tuple[StrictFloat, ...]
    forward_clamp: Literal[False]

    @model_validator(mode="after")
    def validate_parameters(self) -> AffineTransform:
        if len(self.offset) != RAW_ACTION_DIM or len(self.scale) != RAW_ACTION_DIM:
            raise ValueError(
                "Cosmos-Dreams normalizer offset/scale lengths must equal raw_action_dim=29, "
                f"got {len(self.offset)} and {len(self.scale)}."
            )
        canonical_offset = tuple(float32_value(value) for value in self.offset)
        canonical_scale = tuple(float32_value(value) for value in self.scale)
        if canonical_offset != self.offset or canonical_scale != self.scale:
            raise ValueError("Cosmos-Dreams normalizer offset/scale must be encoded at float32 precision.")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("Cosmos-Dreams normalizer scales must be strictly positive.")
        return self


class QuantileRotDerivation(_StrictModel):
    statistics_block: Literal["global_raw"]
    low_key: Literal["q01"]
    high_key: Literal["q99"]
    range_floor: StrictFloat

    @model_validator(mode="after")
    def validate_range_floor(self) -> QuantileRotDerivation:
        if self.range_floor != float32_value(RANGE_FLOOR):
            raise ValueError("Cosmos-Dreams quantile_rot range_floor must equal the training value 1e-8.")
        return self


class SourceProvenance(_StrictModel):
    path: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def validate_artifact_paths(self) -> SourceProvenance:
        allowed_paths = {
            f"{ACTION_NORMALIZERS_RELATIVE_DIR}/{filename}"
            for _, filenames in _SUPPORTED_DATASETS.values()
            for filename in filenames
        }
        if self.path not in allowed_paths:
            raise ValueError(f"Cosmos-Dreams normalizer source path is not a supported target artifact: {self.path!r}.")
        expected_artifact_path = f"cosmos_dreams_action_sources/{self.sha256}.json"
        if self.artifact_path != expected_artifact_path:
            raise ValueError(
                "Cosmos-Dreams normalizer artifact_path must be content-addressed under "
                f"cosmos_dreams_action_sources/: expected {expected_artifact_path!r}."
            )
        return self


class TrainingConfigProvenance(_StrictModel):
    experiment: str = Field(min_length=1)
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ResolvedDatasetDescriptor(_StrictModel):
    dataset_class: Literal[
        "AgiBotWorldBetaDataset",
        "AgibotGEARGripperDataset",
        "AgibotGEARGripperExtDataset",
    ]
    embodiment: Literal[
        "agibotworld",
        "agibot_gear_gripper",
        "agibot_gear_gripper_ext",
    ]
    method: Literal["quantile_rot"]
    apply_forward_clamp: Literal[False]
    pose_convention: Literal["backward_framewise"]
    rotation_format: Literal["rot6d"]
    stats_filename: Literal[
        "agibot_backward_framewise_rot6d.json",
        "agibot_gear_gripper_backward_framewise_rot6d.json",
    ]

    @model_validator(mode="after")
    def validate_association(self) -> ResolvedDatasetDescriptor:
        expected_embodiment, allowed_stats = _SUPPORTED_DATASETS[self.dataset_class]
        if self.embodiment != expected_embodiment or self.stats_filename not in allowed_stats:
            raise ValueError(
                "Cosmos-Dreams training dataset descriptor has an invalid class/embodiment/statistics association."
            )
        return self


class ResolvedTrainingConfigExcerpt(_StrictModel):
    datasets: tuple[ResolvedDatasetDescriptor, ...]
    experiment: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_datasets(self) -> ResolvedTrainingConfigExcerpt:
        if not self.datasets:
            raise ValueError("Cosmos-Dreams resolved training config must contain datasets.")
        return self


class QuantileRotNormalizerContract(_StrictModel):
    schema_version: Literal[1]
    method: Literal["quantile_rot"]
    transform: AffineTransform
    derivation: QuantileRotDerivation
    source: SourceProvenance
    training_config: TrainingConfigProvenance
    transform_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def behavioral_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "transform": self.transform.model_dump(mode="json"),
            "derivation": self.derivation.model_dump(mode="json"),
        }

    @model_validator(mode="after")
    def verify_transform_hash(self) -> QuantileRotNormalizerContract:
        expected = canonical_sha256(self.behavioral_payload())
        if self.transform_sha256 != expected:
            raise ValueError(
                "Cosmos-Dreams normalizer transform_sha256 does not match its behavioral payload: "
                f"expected {expected}, got {self.transform_sha256}."
            )
        return self


class CosmosDreamsActionSchema(_StrictModel):
    schema_version: Literal[2]
    action_tokens_per_frame: Literal[4]
    raw_action_dim: Literal[29]
    model_action_dim: Literal[64]
    num_embodiment_domains: Literal[32]
    default_embodiment: str = Field(min_length=1)
    embodiment_to_domain: dict[str, StrictInt]
    layout: ActionLayout
    padding: ActionPadding
    training_config_excerpt: ResolvedTrainingConfigExcerpt
    normalizers: dict[str, QuantileRotNormalizerContract]
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def behavioral_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_tokens_per_frame": self.action_tokens_per_frame,
            "raw_action_dim": self.raw_action_dim,
            "model_action_dim": self.model_action_dim,
            "num_embodiment_domains": self.num_embodiment_domains,
            "default_embodiment": self.default_embodiment,
            "embodiment_to_domain": dict(sorted(self.embodiment_to_domain.items())),
            "layout": self.layout.model_dump(mode="json", exclude_none=True),
            "padding": self.padding.model_dump(mode="json"),
            "normalizer_sha256_by_embodiment": {
                name: normalizer.transform_sha256 for name, normalizer in sorted(self.normalizers.items())
            },
        }

    @property
    def digest(self) -> str:
        """Digest the complete conditioning payload, including provenance."""

        return canonical_sha256(self.model_dump(mode="json", exclude_none=True))

    def validate_temporal_compression_factor(self, temporal_compression_factor: int) -> None:
        """Validate the action-token cadence against the model's VAE cadence."""

        if self.action_tokens_per_frame != temporal_compression_factor:
            raise ValueError(
                "Cosmos-Dreams action_tokens_per_frame must equal temporal_compression_factor; "
                f"got {self.action_tokens_per_frame} and {temporal_compression_factor}"
            )

    @model_validator(mode="after")
    def verify_target_contract(self) -> CosmosDreamsActionSchema:
        if not self.embodiment_to_domain:
            raise ValueError("Cosmos-Dreams action contract must declare at least one embodiment.")
        if set(self.embodiment_to_domain) != set(self.normalizers):
            raise ValueError("Cosmos-Dreams normalizers must define exactly one entry per declared embodiment.")
        if self.default_embodiment not in self.normalizers:
            raise ValueError("Cosmos-Dreams default_embodiment must name one declared embodiment.")
        invalid_domains = {
            name: domain_id for name, domain_id in self.embodiment_to_domain.items() if domain_id != AGIBOT_DOMAIN_ID
        }
        if invalid_domains:
            raise ValueError(
                f"Cosmos-Dreams target action contract supports only domain {AGIBOT_DOMAIN_ID}, got {invalid_domains}."
            )
        descriptor_by_embodiment: dict[str, ResolvedDatasetDescriptor] = {}
        for descriptor in self.training_config_excerpt.datasets:
            previous = descriptor_by_embodiment.get(descriptor.embodiment)
            if previous is not None and previous.stats_filename != descriptor.stats_filename:
                raise ValueError(
                    f"Cosmos-Dreams embodiment {descriptor.embodiment!r} resolves to conflicting statistics files."
                )
            descriptor_by_embodiment[descriptor.embodiment] = descriptor
        descriptor_embodiments = set(descriptor_by_embodiment)
        if descriptor_embodiments != set(self.normalizers):
            raise ValueError("Cosmos-Dreams training dataset descriptors must cover exactly the declared normalizers.")
        resolved_sha256 = canonical_sha256(self.training_config_excerpt.model_dump(mode="json"))
        repository_revisions: set[str] = set()
        for embodiment, normalizer in self.normalizers.items():
            if normalizer.training_config.experiment != self.training_config_excerpt.experiment:
                raise ValueError(f"Cosmos-Dreams normalizer {embodiment!r} disagrees with the training experiment.")
            if normalizer.training_config.resolved_sha256 != resolved_sha256:
                raise ValueError(
                    f"Cosmos-Dreams normalizer {embodiment!r} has the wrong resolved training-config hash."
                )
            if normalizer.source.repository_revision != normalizer.training_config.repository_revision:
                raise ValueError(f"Cosmos-Dreams normalizer {embodiment!r} mixes source and training revisions.")
            repository_revisions.add(normalizer.training_config.repository_revision)
            expected_source_path = (
                f"{ACTION_NORMALIZERS_RELATIVE_DIR}/{descriptor_by_embodiment[embodiment].stats_filename}"
            )
            if normalizer.source.path != expected_source_path:
                raise ValueError(
                    f"Cosmos-Dreams normalizer {embodiment!r} source does not match its resolved dataset config."
                )
        if len(repository_revisions) != 1:
            raise ValueError("Cosmos-Dreams normalizers must come from one pinned repository revision.")
        expected = canonical_sha256(self.behavioral_payload())
        if self.contract_sha256 != expected:
            raise ValueError(
                "Cosmos-Dreams action contract_sha256 does not match its behavioral payload: "
                f"expected {expected}, got {self.contract_sha256}."
            )
        return self

    def resolve_embodiment(self, name: str | None, domain_id: int | None) -> str:
        """Resolve the normalizer independently of the model's domain embedding."""

        if name is None or not str(name).strip():
            embodiment = self.default_embodiment
        else:
            embodiment = str(name).strip().lower()
        if embodiment not in self.normalizers:
            raise ValueError(f"Unknown Cosmos-Dreams embodiment {name!r}; expected one of {sorted(self.normalizers)}.")
        expected_domain = self.embodiment_to_domain[embodiment]
        if domain_id is not None and int(domain_id) != expected_domain:
            raise ValueError(
                "Cosmos-Dreams embodiment/domain mismatch: "
                f"{embodiment!r} requires domain_id={expected_domain}, got {domain_id}."
            )
        return embodiment
