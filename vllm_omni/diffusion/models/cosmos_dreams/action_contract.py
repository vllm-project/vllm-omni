# SPDX-License-Identifier: Apache-2.0
"""Strict, self-verifying Cosmos-Dreams action contract."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

ACTION_SCHEMA_VERSION = 3
NORMALIZER_SCHEMA_VERSION = 1
AGIBOT_RAW_ACTION_DIM = 29
CAMERA_RAW_ACTION_DIM = 9
MODEL_ACTION_DIM = 64
ACTION_TOKENS_PER_FRAME = 4
NUM_EMBODIMENT_DOMAINS = 32
AGIBOT_DOMAIN_ID = 15
CAMERA_DOMAIN_ID = 2
RANGE_FLOOR = 1e-8
LAYOUT_ID = "agibot_backward_framewise_rot6d_v1"
CAMERA_LAYOUT_ID = "camera_pose_backward_framewise_rot6d_v1"
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
    id: Literal[
        "agibot_backward_framewise_rot6d_v1",
        "camera_pose_backward_framewise_rot6d_v1",
    ]
    pose_convention: Literal["backward_framewise"]
    delta_equation: Literal["T_i^-1 @ T_{i+1}"]
    rotation_representation: Literal["rot6d_columns"]
    fields: tuple[ActionLayoutField, ...]

    @model_validator(mode="after")
    def validate_target_layout(self) -> ActionLayout:
        if self.id == LAYOUT_ID:
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
        else:
            expected = (
                ("camera_translation", 0, 3, "meter", None, None, None),
                ("camera_rotation", 3, 6, "dimensionless", "rot6d_columns", None, None),
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
            raise ValueError(f"Cosmos-Dreams action layout must exactly match {self.id!r}.")
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
        if not self.offset or len(self.offset) != len(self.scale):
            raise ValueError("Cosmos-Dreams normalizer offset/scale must have equal non-zero lengths.")
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


class CameraResolvedDatasetDescriptor(_StrictModel):
    dataset_class: Literal["CameraDatasetSharded"]
    embodiment: Literal["camera_pose"]
    method: Literal["pose_scale"]
    apply_forward_clamp: Literal[False]
    mode: Literal["forward_dynamics"]
    pose_convention: Literal["backward_framewise"]
    rotation_format: Literal["rot6d"]
    translation_scale: StrictFloat = Field(gt=0)
    rotation_scale: StrictFloat = Field(gt=0)

    @model_validator(mode="after")
    def validate_scale_precision(self) -> CameraResolvedDatasetDescriptor:
        if self.translation_scale != float32_value(self.translation_scale) or self.rotation_scale != float32_value(
            self.rotation_scale
        ):
            raise ValueError("Cosmos-Dreams camera action scale factors must be encoded at float32 precision.")
        return self


ActionResolvedDatasetDescriptor = Annotated[
    ResolvedDatasetDescriptor | CameraResolvedDatasetDescriptor,
    Field(discriminator="dataset_class"),
]


class ResolvedTrainingConfigExcerpt(_StrictModel):
    datasets: tuple[ActionResolvedDatasetDescriptor, ...]
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
        if len(self.transform.offset) != AGIBOT_RAW_ACTION_DIM:
            raise ValueError(
                "Cosmos-Dreams quantile_rot offset/scale lengths must equal raw_action_dim=29, "
                f"got {len(self.transform.offset)} and {len(self.transform.scale)}."
            )
        expected = canonical_sha256(self.behavioral_payload())
        if self.transform_sha256 != expected:
            raise ValueError(
                "Cosmos-Dreams normalizer transform_sha256 does not match its behavioral payload: "
                f"expected {expected}, got {self.transform_sha256}."
            )
        return self


class PoseScaleDerivation(_StrictModel):
    translation_scale: StrictFloat = Field(gt=0)
    rotation_scale: StrictFloat = Field(gt=0)

    @model_validator(mode="after")
    def validate_scale_precision(self) -> PoseScaleDerivation:
        if self.translation_scale != float32_value(self.translation_scale) or self.rotation_scale != float32_value(
            self.rotation_scale
        ):
            raise ValueError("Cosmos-Dreams pose_scale factors must be encoded at float32 precision.")
        return self


class PoseScaleNormalizerContract(_StrictModel):
    schema_version: Literal[1]
    method: Literal["pose_scale"]
    transform: AffineTransform
    derivation: PoseScaleDerivation
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
    def verify_transform(self) -> PoseScaleNormalizerContract:
        if len(self.transform.offset) != CAMERA_RAW_ACTION_DIM:
            raise ValueError(
                "Cosmos-Dreams pose_scale offset/scale lengths must equal raw_action_dim=9, "
                f"got {len(self.transform.offset)} and {len(self.transform.scale)}."
            )
        expected_offset = (0.0,) * CAMERA_RAW_ACTION_DIM
        expected_scale = (float32_value(1.0 / self.derivation.translation_scale),) * 3 + (
            float32_value(1.0 / self.derivation.rotation_scale),
        ) * (CAMERA_RAW_ACTION_DIM - 3)
        if self.transform.offset != expected_offset or self.transform.scale != expected_scale:
            raise ValueError(
                "Cosmos-Dreams pose_scale transform does not match its translation_scale/rotation_scale derivation."
            )
        expected_hash = canonical_sha256(self.behavioral_payload())
        if self.transform_sha256 != expected_hash:
            raise ValueError(
                "Cosmos-Dreams normalizer transform_sha256 does not match its behavioral payload: "
                f"expected {expected_hash}, got {self.transform_sha256}."
            )
        return self


ActionNormalizerContract = Annotated[
    QuantileRotNormalizerContract | PoseScaleNormalizerContract,
    Field(discriminator="method"),
]


class CosmosDreamsEmbodimentContract(_StrictModel):
    """Per-embodiment raw action semantics for schema v3."""

    domain_id: StrictInt = Field(ge=0, lt=NUM_EMBODIMENT_DOMAINS)
    raw_action_dim: Literal[9, 29]
    layout: ActionLayout
    normalizer: ActionNormalizerContract

    @model_validator(mode="after")
    def verify_layout_family(self) -> CosmosDreamsEmbodimentContract:
        camera_variant = self.layout.id == CAMERA_LAYOUT_ID
        expected_domain_id = CAMERA_DOMAIN_ID if camera_variant else AGIBOT_DOMAIN_ID
        expected_raw_action_dim = CAMERA_RAW_ACTION_DIM if camera_variant else AGIBOT_RAW_ACTION_DIM
        expected_normalizer_type = PoseScaleNormalizerContract if camera_variant else QuantileRotNormalizerContract
        if self.domain_id != expected_domain_id:
            raise ValueError(
                f"Cosmos-Dreams layout {self.layout.id!r} requires domain_id={expected_domain_id}, "
                f"got {self.domain_id}."
            )
        if self.raw_action_dim != expected_raw_action_dim:
            raise ValueError(
                f"Cosmos-Dreams layout {self.layout.id!r} requires raw_action_dim={expected_raw_action_dim}, "
                f"got {self.raw_action_dim}."
            )
        if not isinstance(self.normalizer, expected_normalizer_type):
            expected_method = "pose_scale" if camera_variant else "quantile_rot"
            raise ValueError(f"Cosmos-Dreams layout {self.layout.id!r} requires method={expected_method!r}.")
        return self


class CosmosDreamsActionSchema(_StrictModel):
    """Per-embodiment action contract for action-conditioned checkpoints."""

    schema_version: Literal[3]
    action_tokens_per_frame: Literal[4]
    model_action_dim: Literal[64]
    num_embodiment_domains: Literal[32]
    default_embodiment: str = Field(min_length=1)
    embodiments: dict[str, CosmosDreamsEmbodimentContract]
    padding: ActionPadding
    training_config_excerpt: ResolvedTrainingConfigExcerpt
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def behavioral_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_tokens_per_frame": self.action_tokens_per_frame,
            "model_action_dim": self.model_action_dim,
            "num_embodiment_domains": self.num_embodiment_domains,
            "default_embodiment": self.default_embodiment,
            "embodiments": {
                name: {
                    "domain_id": contract.domain_id,
                    "raw_action_dim": contract.raw_action_dim,
                    "layout": contract.layout.model_dump(mode="json", exclude_none=True),
                    "normalizer_sha256": contract.normalizer.transform_sha256,
                }
                for name, contract in sorted(self.embodiments.items())
            },
            "padding": self.padding.model_dump(mode="json"),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude_none=True))

    @property
    def embodiment_to_domain(self) -> dict[str, int]:
        return {name: contract.domain_id for name, contract in self.embodiments.items()}

    @property
    def normalizers(self) -> dict[str, ActionNormalizerContract]:
        return {name: contract.normalizer for name, contract in self.embodiments.items()}

    @property
    def raw_action_dim(self) -> int:
        """Compatibility view of the default embodiment's raw width."""

        return self.embodiments[self.default_embodiment].raw_action_dim

    def validate_temporal_compression_factor(self, temporal_compression_factor: int) -> None:
        if self.action_tokens_per_frame != temporal_compression_factor:
            raise ValueError(
                "Cosmos-Dreams action_tokens_per_frame must equal temporal_compression_factor; "
                f"got {self.action_tokens_per_frame} and {temporal_compression_factor}"
            )

    @model_validator(mode="after")
    def verify_target_contract(self) -> CosmosDreamsActionSchema:
        if not self.embodiments:
            raise ValueError("Cosmos-Dreams action contract must declare at least one embodiment.")
        if self.default_embodiment not in self.embodiments:
            raise ValueError("Cosmos-Dreams default_embodiment must name one declared embodiment.")

        descriptor_by_embodiment: dict[
            str,
            ResolvedDatasetDescriptor | CameraResolvedDatasetDescriptor,
        ] = {}
        for descriptor in self.training_config_excerpt.datasets:
            previous = descriptor_by_embodiment.get(descriptor.embodiment)
            if previous is not None and previous != descriptor:
                raise ValueError(f"Cosmos-Dreams embodiment {descriptor.embodiment!r} has conflicting semantics.")
            descriptor_by_embodiment[descriptor.embodiment] = descriptor
        if set(descriptor_by_embodiment) != set(self.embodiments):
            raise ValueError("Cosmos-Dreams embodiment contracts must exactly cover the resolved dataset descriptors.")

        resolved_sha256 = canonical_sha256(self.training_config_excerpt.model_dump(mode="json"))
        repository_revisions: set[str] = set()
        for embodiment, contract in self.embodiments.items():
            descriptor = descriptor_by_embodiment[embodiment]
            camera_variant = isinstance(descriptor, CameraResolvedDatasetDescriptor)
            if camera_variant != (contract.layout.id == CAMERA_LAYOUT_ID):
                raise ValueError(
                    f"Cosmos-Dreams embodiment {embodiment!r} layout disagrees with its dataset descriptor."
                )
            normalizer = contract.normalizer
            if normalizer.training_config.experiment != self.training_config_excerpt.experiment:
                raise ValueError(f"Cosmos-Dreams normalizer {embodiment!r} disagrees with the training experiment.")
            if normalizer.training_config.resolved_sha256 != resolved_sha256:
                raise ValueError(
                    f"Cosmos-Dreams normalizer {embodiment!r} has the wrong resolved training-config hash."
                )
            repository_revisions.add(normalizer.training_config.repository_revision)
            if isinstance(normalizer, QuantileRotNormalizerContract):
                if normalizer.source.repository_revision != normalizer.training_config.repository_revision:
                    raise ValueError(f"Cosmos-Dreams normalizer {embodiment!r} mixes source and training revisions.")
                if not isinstance(descriptor, ResolvedDatasetDescriptor):
                    raise ValueError("Cosmos-Dreams quantile_rot normalizer requires an AgiBot dataset descriptor.")
                expected_source_path = f"{ACTION_NORMALIZERS_RELATIVE_DIR}/{descriptor.stats_filename}"
                if normalizer.source.path != expected_source_path:
                    raise ValueError(
                        f"Cosmos-Dreams normalizer {embodiment!r} source does not match its resolved dataset config."
                    )
            else:
                if not isinstance(descriptor, CameraResolvedDatasetDescriptor):
                    raise ValueError("Cosmos-Dreams pose_scale normalizer requires a camera dataset descriptor.")
                if (
                    normalizer.derivation.translation_scale != descriptor.translation_scale
                    or normalizer.derivation.rotation_scale != descriptor.rotation_scale
                ):
                    raise ValueError(
                        f"Cosmos-Dreams normalizer {embodiment!r} disagrees with its resolved camera scale config."
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
        if name is None or not str(name).strip():
            candidates = [
                embodiment
                for embodiment, contract in self.embodiments.items()
                if domain_id is not None and contract.domain_id == int(domain_id)
            ]
            if domain_id is None or self.default_embodiment in candidates:
                embodiment = self.default_embodiment
            elif len(candidates) == 1:
                embodiment = candidates[0]
            elif not candidates:
                raise ValueError(f"No Cosmos-Dreams embodiment uses domain_id={domain_id}.")
            else:
                raise ValueError(
                    f"Cosmos-Dreams domain_id={domain_id} is ambiguous across {sorted(candidates)}; "
                    "supply an embodiment name."
                )
        else:
            embodiment = str(name).strip().lower()
        if embodiment not in self.embodiments:
            raise ValueError(f"Unknown Cosmos-Dreams embodiment {name!r}; expected one of {sorted(self.embodiments)}.")
        expected_domain = self.embodiments[embodiment].domain_id
        if domain_id is not None and int(domain_id) != expected_domain:
            raise ValueError(
                "Cosmos-Dreams embodiment/domain mismatch: "
                f"{embodiment!r} requires domain_id={expected_domain}, got {domain_id}."
            )
        return embodiment

    def raw_action_dim_for(self, embodiment: str) -> int:
        resolved = self.resolve_embodiment(embodiment, None)
        return self.embodiments[resolved].raw_action_dim

    def layout_for(self, embodiment: str) -> ActionLayout:
        resolved = self.resolve_embodiment(embodiment, None)
        return self.embodiments[resolved].layout

    def normalizer_for(self, embodiment: str) -> ActionNormalizerContract:
        resolved = self.resolve_embodiment(embodiment, None)
        return self.embodiments[resolved].normalizer
