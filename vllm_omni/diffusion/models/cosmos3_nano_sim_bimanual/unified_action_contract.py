# SPDX-License-Identifier: Apache-2.0
"""Unified production actions, already normalized by input preparation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import canonical_sha256


class UnifiedActionConditioning(BaseModel):
    """Prepared canonical actions with explicit scalar or per-row domain routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["action"]
    schema_version: Literal[5]
    action_tokens_per_frame: Literal[4]
    model_action_dim: Literal[64]
    num_embodiment_domains: Literal[32]
    default_embodiment: str
    embodiments: dict[str, dict[str, int]]
    padding: dict[str, Any]
    input_contract: dict[str, Any]
    layout: dict[str, Any]
    normalizer: dict[str, Any]
    training_config_excerpt: dict[str, Any]
    contract_sha256: str

    @model_validator(mode="after")
    def validate_contract(self) -> UnifiedActionConditioning:
        payload = self.model_dump(exclude={"mode", "contract_sha256"})
        if canonical_sha256(payload) != self.contract_sha256:
            raise ValueError("Unified contract_sha256 mismatch")
        if self.input_contract != {
            "action_space": "normalized_unified_v1",
            "action_dim": 59,
            "domain_routing": "scalar_or_per_action_row",
            "validity": "masked_to_zero_after_normalization_by_source",
            "runtime_normalization": False,
            "model_mode": "forward_dynamics",
        }:
            raise ValueError("Unsupported unified input contract")
        if (
            self.layout.get("id") != "unified_v1"
            or self.layout.get("pose_convention") != "backward_chunk_anchored_16f"
            or self.layout.get("rotation_representation") != "rot6d_columns"
        ):
            raise ValueError("Unsupported unified action layout")
        if (
            self.normalizer.get("method") != "global_asinh_unified_v1"
            or self.normalizer.get("runtime_application") is not False
        ):
            raise ValueError("Unified inputs must already be normalized")
        if self.padding != {"stage": "after_normalization_and_validity_mask", "value": 0.0}:
            raise ValueError("Unsupported unified padding")
        if self.default_embodiment not in self.embodiments:
            raise ValueError("Missing default embodiment")
        for entry in self.embodiments.values():
            if (
                set(entry) != {"domain_id", "input_action_dim"}
                or entry["input_action_dim"] != 59
                or not 0 <= entry["domain_id"] < self.num_embodiment_domains
            ):
                raise ValueError("Invalid unified embodiment")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump())

    @property
    def embodiment_to_domain(self) -> dict[str, int]:
        return {name: entry["domain_id"] for name, entry in self.embodiments.items()}

    @property
    def normalizers(self) -> dict[str, Any]:
        return {}

    @property
    def raw_action_dim(self) -> int:
        return 59

    @property
    def inference_camera_profile(self) -> None:
        return None

    def raw_action_dim_for(self, embodiment: str) -> int:
        self.resolve_embodiment(embodiment, None)
        return 59

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
