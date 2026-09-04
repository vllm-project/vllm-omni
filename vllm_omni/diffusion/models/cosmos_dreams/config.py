# SPDX-License-Identifier: Apache-2.0
"""Strict schema-v1 artifact and deployment configuration for Cosmos-Dreams."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import CosmosDreamsActionSchema
from vllm_omni.diffusion.models.cosmos_dreams.control_contract import (
    CosmosDreamsActionConditioning,
    CosmosDreamsConditioning,
    CosmosDreamsControlVideoConditioning,
    parse_cosmos_dreams_conditioning,
)

COSMOS_DREAMS_SCHEMA_VERSION = 1
COSMOS_DREAMS_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "checkpoint_iteration",
        "checkpoint_hash",
        "chunk_size",
        "window_frames",
        "sink_frames",
        "text_cache_max_len",
        "attention_mode",
        "video_temporal_causal",
        "latent_patch_size",
        "vae_spatial_compression_factor",
        "temporal_compression_factor",
        "fixed_step_sampler_config",
        "conditioning",
        "temporal_modality_margin",
        "unified_3d_mrope_reset_spatial_ids",
        "base_fps",
        "enable_fps_modulation",
    }
)
_DEPLOY_ARTIFACT_ENVELOPES = frozenset(
    {
        "cosmos_dreams",
        "causal_manifest",
        "interactive_config",
        "diffusion_expert_config",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    params = getattr(value, "params", None)
    return params if isinstance(params, dict) else {}


def _deployment_roots(config: Any) -> list[tuple[str, dict[str, Any]]]:
    roots: list[tuple[str, dict[str, Any]]] = []
    for attr in ("custom_pipeline_args", "model_config"):
        value = _mapping(getattr(config, attr, None))
        if value:
            roots.append((attr, value))
    return roots


def deploy_option(config: Any, key: str, default: Any = None) -> Any:
    """Read one deployment option without inspecting artifact envelopes."""

    direct = getattr(config, key, None)
    if direct is not None:
        return direct
    for _, root in _deployment_roots(config):
        if root.get(key) is not None:
            return root[key]
    return default


def _exported_artifact_source(config: Any) -> dict[str, Any]:
    """Return the sole supported artifact source in transformer configuration."""

    transformer_config = _mapping(getattr(config, "tf_model_config", None))
    return _mapping(transformer_config.get("cosmos_dreams"))


def _validate_deploy_layout(config: Any) -> None:
    """Keep signed artifact data out of deployment-owned configuration roots."""

    for attr, root in _deployment_roots(config):
        envelopes = sorted(_DEPLOY_ARTIFACT_ENVELOPES & set(root))
        if envelopes:
            raise ValueError(
                f"Cosmos-Dreams artifact envelopes are not supported in {attr}: {envelopes}. "
                "The artifact must come from tf_model_config['cosmos_dreams']."
            )
        artifact_fields = sorted(COSMOS_DREAMS_ARTIFACT_FIELDS & set(root))
        if artifact_fields:
            raise ValueError(
                f"Cosmos-Dreams artifact fields must not be placed at deploy root {attr}: {artifact_fields}."
            )


def _strict_int(value: Any, name: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Cosmos-Dreams artifact {name} must be an integer, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"Cosmos-Dreams artifact {name} must be positive, got {value}")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Cosmos-Dreams artifact {name} must be a boolean, got {value!r}")
    return value


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Cosmos-Dreams artifact {name} must be numeric, got {value!r}")
    return float(value)


def _parse_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    fixed_step = artifact["fixed_step_sampler_config"]
    if not isinstance(fixed_step, dict):
        raise ValueError("Cosmos-Dreams artifact fixed_step_sampler_config must be an object")
    raw_t_list = fixed_step.get("t_list")
    if not isinstance(raw_t_list, list):
        raise ValueError("Cosmos-Dreams artifact fixed_step_sampler_config.t_list must be a list")
    t_list = tuple(_strict_float(value, "fixed_step_sampler_config.t_list entry") for value in raw_t_list)
    base_fps = _strict_float(artifact["base_fps"], "base_fps")
    return {
        "schema_version": _strict_int(artifact["schema_version"], "schema_version"),
        "chunk_size": _strict_int(artifact["chunk_size"], "chunk_size"),
        "window_frames": _strict_int(artifact["window_frames"], "window_frames"),
        "sink_frames": _strict_int(artifact["sink_frames"], "sink_frames", positive=False),
        "text_cache_max_len": _strict_int(artifact["text_cache_max_len"], "text_cache_max_len"),
        "latent_patch_size": _strict_int(artifact["latent_patch_size"], "latent_patch_size"),
        "vae_spatial_compression_factor": _strict_int(
            artifact["vae_spatial_compression_factor"], "vae_spatial_compression_factor"
        ),
        "temporal_compression_factor": _strict_int(
            artifact["temporal_compression_factor"], "temporal_compression_factor"
        ),
        "temporal_modality_margin": _strict_int(artifact["temporal_modality_margin"], "temporal_modality_margin"),
        "unified_3d_mrope_reset_spatial_ids": _strict_bool(
            artifact["unified_3d_mrope_reset_spatial_ids"], "unified_3d_mrope_reset_spatial_ids"
        ),
        "base_fps": base_fps,
        "enable_fps_modulation": _strict_bool(artifact["enable_fps_modulation"], "enable_fps_modulation"),
        "attention_mode": artifact["attention_mode"],
        "video_temporal_causal": _strict_bool(artifact["video_temporal_causal"], "video_temporal_causal"),
        "sample_type": fixed_step.get("sample_type"),
        "t_list": t_list,
        "num_train_timesteps": _strict_int(fixed_step.get("num_train_timesteps"), "num_train_timesteps"),
        "checkpoint_id": artifact["checkpoint_id"],
        "checkpoint_iteration": _strict_int(artifact["checkpoint_iteration"], "checkpoint_iteration"),
        "checkpoint_hash": artifact["checkpoint_hash"],
        "conditioning": parse_cosmos_dreams_conditioning(artifact["conditioning"]),
    }


@dataclass(frozen=True)
class CosmosDreamsManifest:
    """Immutable schema-v1 artifact fields required by Cosmos-Dreams runtime."""

    schema_version: int = COSMOS_DREAMS_SCHEMA_VERSION
    chunk_size: int = 4
    window_frames: int = 96
    sink_frames: int = 0
    text_cache_max_len: int = 512
    latent_patch_size: int = 2
    vae_spatial_compression_factor: int = 16
    temporal_compression_factor: int = 4
    temporal_modality_margin: int = 15_000
    unified_3d_mrope_reset_spatial_ids: bool = True
    base_fps: float = 24.0
    enable_fps_modulation: bool = True
    attention_mode: str = "three_way"
    video_temporal_causal: bool = True
    sample_type: str = "sde"
    t_list: tuple[float, ...] = (1.0, 15 / 16, 5 / 6, 5 / 8)
    num_train_timesteps: int = 1000
    checkpoint_id: str = "unknown"
    checkpoint_iteration: int = 0
    checkpoint_hash: str = "unknown"
    conditioning: CosmosDreamsConditioning | None = None

    def __post_init__(self) -> None:
        if self.schema_version != COSMOS_DREAMS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Cosmos-Dreams artifact schema_version={self.schema_version}; "
                f"expected {COSMOS_DREAMS_SCHEMA_VERSION}"
            )
        positive = {
            "chunk_size": self.chunk_size,
            "window_frames": self.window_frames,
            "text_cache_max_len": self.text_cache_max_len,
            "latent_patch_size": self.latent_patch_size,
            "vae_spatial_compression_factor": self.vae_spatial_compression_factor,
            "temporal_compression_factor": self.temporal_compression_factor,
            "temporal_modality_margin": self.temporal_modality_margin,
            "num_train_timesteps": self.num_train_timesteps,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Cosmos-Dreams manifest {name} must be a positive integer, got {value!r}")
        if isinstance(self.sink_frames, bool) or not isinstance(self.sink_frames, int) or self.sink_frames < 0:
            raise ValueError(f"Cosmos-Dreams manifest sink_frames must be non-negative, got {self.sink_frames!r}")
        if self.checkpoint_iteration < 0:
            raise ValueError(
                f"Cosmos-Dreams checkpoint_iteration must be non-negative, got {self.checkpoint_iteration}"
            )
        if self.attention_mode != "three_way":
            raise ValueError(f"Cosmos-Dreams requires attention_mode='three_way', got {self.attention_mode!r}")
        if self.video_temporal_causal is not True:
            raise ValueError("Cosmos-Dreams requires video_temporal_causal=True")
        if self.unified_3d_mrope_reset_spatial_ids is not True:
            raise ValueError("Cosmos-Dreams requires unified_3d_mrope_reset_spatial_ids=True")
        if self.enable_fps_modulation is not True:
            raise ValueError("Cosmos-Dreams AR inference requires FPS modulation")
        if not math.isfinite(self.base_fps) or self.base_fps <= 0:
            raise ValueError(f"Cosmos-Dreams base_fps must be positive, got {self.base_fps}")
        if self.sample_type != "sde":
            raise ValueError(f"Cosmos-Dreams sample_type must be 'sde', got {self.sample_type!r}")
        if not self.t_list:
            raise ValueError("Cosmos-Dreams t_list must not be empty")
        if abs(self.t_list[0] - 1.0) > 1e-6:
            raise ValueError(f"Cosmos-Dreams t_list must start at 1.0, got {self.t_list[0]}")
        if any(not math.isfinite(sigma) or sigma <= 0.0 or sigma > 1.0 for sigma in self.t_list):
            raise ValueError(f"Cosmos-Dreams t_list entries must be in (0, 1], got {self.t_list}")
        if any(left <= right for left, right in zip(self.t_list, self.t_list[1:])):
            raise ValueError(f"Cosmos-Dreams t_list must be strictly descending, got {self.t_list}")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("Cosmos-Dreams checkpoint_id must be a non-empty string")
        if not isinstance(self.checkpoint_hash, str):
            raise ValueError("Cosmos-Dreams checkpoint_hash must be a string")
        if self.checkpoint_hash != "unknown" and (
            len(self.checkpoint_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.checkpoint_hash)
        ):
            raise ValueError(
                f"Cosmos-Dreams checkpoint_hash must be a 64-character SHA-256 digest, got {self.checkpoint_hash!r}"
            )
        if self.checkpoint_hash != "unknown" and set(self.checkpoint_hash) == {"0"}:
            raise ValueError("Cosmos-Dreams checkpoint_hash cannot be the all-zero template value")
        if self.conditioning is not None and not isinstance(
            self.conditioning, CosmosDreamsActionConditioning | CosmosDreamsControlVideoConditioning
        ):
            raise ValueError("Cosmos-Dreams schema v1 requires a recognized conditioning payload")

    @property
    def action_schema(self) -> CosmosDreamsActionSchema | None:
        return self.conditioning if isinstance(self.conditioning, CosmosDreamsActionConditioning) else None

    def require_action_schema(self) -> CosmosDreamsActionSchema:
        schema = self.action_schema
        if schema is None:
            raise ValueError("Cosmos-Dreams action conditioning is unavailable.")
        return schema

    def require_control_video_conditioning(self) -> CosmosDreamsControlVideoConditioning:
        conditioning = self.conditioning
        if not isinstance(conditioning, CosmosDreamsControlVideoConditioning):
            raise ValueError("Cosmos-Dreams control_video conditioning is unavailable.")
        return conditioning

    @property
    def action_tokens_per_frame(self) -> int:
        return self.require_action_schema().action_tokens_per_frame

    @property
    def raw_action_dim(self) -> int:
        return self.require_action_schema().raw_action_dim

    def raw_action_dim_for(self, embodiment: str) -> int:
        return self.require_action_schema().raw_action_dim_for(embodiment)

    @property
    def max_action_dim(self) -> int:
        return self.require_action_schema().model_action_dim

    @property
    def num_embodiment_domains(self) -> int:
        return self.require_action_schema().num_embodiment_domains

    @property
    def embodiment_to_domain(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self.require_action_schema().embodiment_to_domain.items()))

    @property
    def action_contract_sha256(self) -> str:
        return self.require_action_schema().contract_sha256

    @property
    def conditioning_tokens_per_frame(self) -> int:
        if isinstance(self.conditioning, CosmosDreamsControlVideoConditioning):
            return 0
        return self.action_tokens_per_frame

    @property
    def conditioning_digest(self) -> str | None:
        if self.conditioning is None:
            return None
        digest = self.conditioning.digest
        if not isinstance(digest, str) or not digest:
            raise ValueError("Cosmos-Dreams conditioning payload must expose a non-empty digest")
        return digest

    @classmethod
    def from_od_config(cls, od_config: Any) -> CosmosDreamsManifest:
        """Load the sole schema-v1 artifact source; deployments never synthesize it."""

        _validate_deploy_layout(od_config)
        artifact = _exported_artifact_source(od_config)
        if not artifact:
            raise ValueError(
                "Cosmos-Dreams requires a schema-v1 artifact in tf_model_config['cosmos_dreams']; "
                "deployment defaults are not an artifact."
            )
        missing = sorted(COSMOS_DREAMS_ARTIFACT_FIELDS - set(artifact))
        unknown = sorted(set(artifact) - COSMOS_DREAMS_ARTIFACT_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"Cosmos-Dreams schema-v1 artifact fields are invalid: missing={missing}, unknown={unknown}."
            )
        return cls(**_parse_artifact(artifact))

    def require_exported_artifact(self) -> None:
        missing: list[str] = []
        if self.checkpoint_id == "unknown":
            missing.append("checkpoint_id")
        if self.checkpoint_iteration <= 0:
            missing.append("checkpoint_iteration")
        if self.checkpoint_hash == "unknown":
            missing.append("checkpoint_hash")
        if missing:
            raise ValueError(
                "Cosmos-Dreams requires a validated exported artifact; missing "
                f"{', '.join(missing)} from its causal manifest."
            )

    def resolve_domain_name(self, name: str) -> int:
        normalized = str(name).strip().lower()
        try:
            return dict(self.embodiment_to_domain)[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Cosmos-Dreams domain_name={name!r}; expected one of "
                f"{sorted(dict(self.embodiment_to_domain))}."
            ) from exc

    def resolve_embodiment(self, name: str | None, domain_id: int | None) -> str:
        return self.require_action_schema().resolve_embodiment(name, domain_id)

    @property
    def sampler_id(self) -> str:
        values = ",".join(f"{value:.12g}" for value in self.t_list)
        return f"{self.sample_type}:{values}:train={self.num_train_timesteps}"

    @property
    def digest(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "conditioning"}
        values["conditioning_digest"] = self.conditioning_digest
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
