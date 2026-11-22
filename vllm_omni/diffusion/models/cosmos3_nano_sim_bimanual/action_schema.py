# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Schema-5 Bimanual action contract consumed by inference.

Serving validates the prepared 59D layout, domain table, padding, and source
provenance recorded by the exporter. Training details remain opaque because
they do not change runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

UNIFIED_ACTION_DIM = 59
NUM_EMBODIMENT_DOMAINS = 32


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionInputContract(_StrictModel):
    action_space: Literal["normalized_unified_v1"]
    action_dim: Literal[59]
    domain_routing: Literal["scalar_or_per_action_row"]
    validity: Literal["masked_to_zero_after_normalization_by_source"]
    runtime_normalization: Literal[False]
    model_mode: Literal["forward_dynamics"]


class ActionLayoutField(_StrictModel):
    name: str = Field(min_length=1)
    offset: StrictInt = Field(ge=0)
    size: StrictInt = Field(gt=0)


class ActionLayout(_StrictModel):
    id: Literal["unified_v1"]
    pose_convention: Literal["backward_chunk_anchored_16f"]
    rotation_representation: Literal["rot6d_columns"]
    fields: tuple[ActionLayoutField, ...]


class ActionNormalizerSource(_StrictModel):
    artifact_path: str = Field(min_length=1)
    path: str = Field(min_length=1)
    pose_convention_authority: str = Field(min_length=1)
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ActionNormalizer(_StrictModel):
    method: Literal["global_asinh_unified_v1"]
    runtime_application: Literal[False]
    source: ActionNormalizerSource


class ActionPadding(_StrictModel):
    stage: Literal["after_normalization_and_validity_mask"]
    value: Literal[0.0]


class EmbodimentContract(_StrictModel):
    domain_id: StrictInt = Field(ge=0, lt=NUM_EMBODIMENT_DOMAINS)
    input_action_dim: Literal[59]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class Cosmos3NanoSimBimanualActionSchema(_StrictModel):
    """Prepared canonical actions with explicit scalar or per-row domain routing."""

    mode: Literal["action"]
    schema_version: Literal[5]
    action_tokens_per_frame: Literal[4]
    model_action_dim: Literal[64]
    num_embodiment_domains: Literal[32]
    default_embodiment: str = Field(min_length=1)
    embodiments: dict[str, EmbodimentContract]
    padding: ActionPadding
    input_contract: ActionInputContract
    layout: ActionLayout
    normalizer: ActionNormalizer
    training_config_excerpt: dict[str, Any]
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_contract(self) -> Cosmos3NanoSimBimanualActionSchema:
        payload = self.model_dump(mode="json", exclude={"mode", "contract_sha256"})
        if canonical_sha256(payload) != self.contract_sha256:
            raise ValueError("Bimanual action contract_sha256 mismatch")
        expected_fields = (
            ("ego_pose", 0, 9),
            ("right_wrist_pose", 9, 9),
            ("right_fingertips", 18, 15),
            ("right_gripper_width_m", 33, 1),
            ("left_wrist_pose", 34, 9),
            ("left_fingertips", 43, 15),
            ("left_gripper_width_m", 58, 1),
        )
        actual_fields = tuple((field.name, field.offset, field.size) for field in self.layout.fields)
        if actual_fields != expected_fields:
            raise ValueError("Unsupported unified_v1 action field layout")
        if not self.embodiments:
            raise ValueError("Bimanual action contract must declare at least one embodiment")
        if self.default_embodiment not in self.embodiments:
            raise ValueError("Missing default embodiment")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    @property
    def embodiment_to_domain(self) -> dict[str, int]:
        return {name: entry.domain_id for name, entry in self.embodiments.items()}

    @property
    def input_action_dim(self) -> int:
        return UNIFIED_ACTION_DIM

    def validate_temporal_compression_factor(self, factor: int) -> None:
        if factor != self.action_tokens_per_frame:
            raise ValueError("Action token count disagrees with temporal compression")

    def resolve_embodiment(self, name: str | None, domain_id: int | None) -> str:
        if name is not None:
            if name not in self.embodiments:
                raise ValueError(f"Unknown embodiment {name!r}")
            if domain_id is not None and self.embodiment_to_domain[name] != domain_id:
                raise ValueError("Embodiment/domain mismatch")
            return name
        if domain_id is None:
            return self.default_embodiment
        candidates = [name for name, value in self.embodiment_to_domain.items() if value == domain_id]
        if not candidates:
            raise ValueError(f"Unknown domain {domain_id}")
        return self.default_embodiment if self.default_embodiment in candidates else sorted(candidates)[0]
