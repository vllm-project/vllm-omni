# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams action contract as consumed by inference.

The exported artifact also records how it was produced: the training
experiment, the dataset classes it was resolved from, and the repository
revision and content hash of each normalizer source file. Whether the exporter
assembled those correctly is the exporter's concern, and serving cannot act on
the answer, so this module accepts those blocks without interpreting them and
validates only what changes model output: the affine transform applied to raw
actions, the embodiment/domain table, and the raw action widths.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

AGIBOT_RAW_ACTION_DIM = 29
CAMERA_RAW_ACTION_DIM = 9
NUM_EMBODIMENT_DOMAINS = 32
AGIBOT_DOMAIN_ID = 15
# Training-time floor on a quantile range; the runtime normalizer warns when a
# scale sits close enough to it to indicate a degenerate channel.
RANGE_FLOOR = 1e-8


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


# Exporter-recorded provenance. Declared so that ``extra="forbid"`` still
# accepts a real artifact, but deliberately left opaque: serving neither reads
# these blocks nor includes them in any hashed payload, so a newer exporter can
# extend them without breaking model load.
Provenance = Mapping[str, Any]


class ActionLayoutField(_StrictModel):
    name: str
    offset: StrictInt = Field(ge=0)
    size: StrictInt = Field(gt=0)
    unit: str
    representation: str | None = None
    closed_value: StrictFloat | None = None
    open_value: StrictFloat | None = None


class ActionLayout(_StrictModel):
    """Field map of one raw action row.

    Retained as a typed model because ``contract_sha256`` is computed over its
    ``exclude_none`` dump; the runtime itself never reads individual fields.
    """

    id: Literal[
        "agibot_backward_framewise_rot6d_v1",
        "legacy_yam_fk_backward_framewise_rot6d_v1",
        "camera_pose_backward_framewise_rot6d_v1",
    ]
    pose_convention: Literal["backward_framewise"]
    delta_equation: Literal["T_i^-1 @ T_{i+1}"]
    rotation_representation: Literal["rot6d_columns"]
    fields: tuple[ActionLayoutField, ...]


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


class QuantileRotNormalizerContract(_StrictModel):
    schema_version: Literal[1]
    method: Literal["quantile_rot"]
    transform: AffineTransform
    derivation: QuantileRotDerivation
    source: Provenance
    training_config: Provenance
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


class PoseScaleDerivation(_StrictModel):
    translation_scale: StrictFloat = Field(gt=0)
    rotation_scale: StrictFloat = Field(gt=0)


class PoseScaleNormalizerContract(_StrictModel):
    schema_version: Literal[1]
    method: Literal["pose_scale"]
    transform: AffineTransform
    derivation: PoseScaleDerivation
    training_config: Provenance
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
    """Per-embodiment raw action semantics."""

    domain_id: StrictInt = Field(ge=0, lt=NUM_EMBODIMENT_DOMAINS)
    raw_action_dim: Literal[9, 20, 29]
    layout: ActionLayout
    normalizer: ActionNormalizerContract

    @model_validator(mode="after")
    def verify_normalizer_width(self) -> CosmosDreamsEmbodimentContract:
        # A normalizer narrower or wider than the declared raw width would
        # accept a request-shaped action and then fail inside normalize().
        if len(self.normalizer.transform.offset) != self.raw_action_dim:
            raise ValueError(
                f"Cosmos-Dreams normalizer dimension must equal raw_action_dim={self.raw_action_dim}, "
                f"got {len(self.normalizer.transform.offset)}."
            )
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
    training_config_excerpt: Provenance | None = None
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
        """Identity of the whole contract, provenance included."""

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
