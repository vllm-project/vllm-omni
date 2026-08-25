# SPDX-License-Identifier: Apache-2.0
"""Validated deployment manifest for Cosmos-Dreams checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import CosmosDreamsActionSchema
from vllm_omni.diffusion.models.cosmos_dreams.control_contract import (
    CosmosDreamsActionConditioning,
    CosmosDreamsControlVideoConditioning,
    parse_cosmos_dreams_v3_conditioning,
)

_V2_EXPORTED_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "checkpoint_iteration",
        "checkpoint_hash",
        "chunk_size",
        "window_frames",
        "sink_frames",
        "text_cache_max_len",
        "deploy_resolution",
        "attention_mode",
        "video_temporal_causal",
        "latent_patch_size",
        "vae_spatial_compression_factor",
        "temporal_compression_factor",
        "fixed_step_sampler_config",
        "action_schema",
        "temporal_modality_margin",
        "unified_3d_mrope_reset_spatial_ids",
        "base_fps",
        "enable_fps_modulation",
    }
)
_EXPORTED_ARTIFACT_REQUIRED_FIELDS_BY_VERSION = {2: _V2_EXPORTED_ARTIFACT_FIELDS}
_EXPORTED_ARTIFACT_ALLOWED_FIELDS_BY_VERSION = {2: _V2_EXPORTED_ARTIFACT_FIELDS}
_CONDITIONING_ARTIFACT_FIELDS = frozenset({"action_schema", "conditioning"})
_LEGACY_NORMALIZER_FIELDS = {"action_normalizer", "normalizer_id", "normalizer_source"}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    params = getattr(value, "params", None)
    return params if isinstance(params, dict) else {}


def _first(sources: list[dict[str, Any]], *keys: str, default: Any = None) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return default


def _manifest_sources(config: Any) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for attr in ("custom_pipeline_args", "model_config", "tf_model_config"):
        value = _mapping(getattr(config, attr, None))
        if value:
            roots.append(value)

    sources: list[dict[str, Any]] = []
    for root in roots:
        for key in (
            "cosmos_dreams",
            "causal_manifest",
            "interactive_config",
            "diffusion_expert_config",
        ):
            nested = _mapping(root.get(key))
            if nested:
                sources.append(nested)
        sources.append(root)
    return sources


def _exported_artifact_source(config: Any) -> dict[str, Any]:
    """Return the causal payload embedded in ``transformer/config.json``."""

    transformer_config = _mapping(getattr(config, "tf_model_config", None))
    return _mapping(transformer_config.get("cosmos_dreams"))


def _validate_deploy_overrides(config: Any, artifact: dict[str, Any]) -> None:
    """Reject deploy values that contradict the signed artifact contract."""

    for attr in ("custom_pipeline_args", "model_config"):
        root = _mapping(getattr(config, attr, None))
        legacy_root_fields = sorted(_LEGACY_NORMALIZER_FIELDS & set(root))
        if legacy_root_fields:
            raise ValueError(
                f"Cosmos-Dreams deployment contains legacy normalizer fields in {attr}: {legacy_root_fields}."
            )
        for conditioning_field in sorted(_CONDITIONING_ARTIFACT_FIELDS & set(root)):
            raise ValueError(
                f"Cosmos-Dreams {conditioning_field} may only come from transformer/config.json, "
                f"not {attr}.{conditioning_field}."
            )
        for key in ("cosmos_dreams", "causal_manifest", "interactive_config"):
            override = _mapping(root.get(key))
            legacy_override_fields = sorted(_LEGACY_NORMALIZER_FIELDS & set(override))
            if legacy_override_fields:
                raise ValueError(
                    "Cosmos-Dreams deployment contains legacy normalizer fields in "
                    f"{attr}.{key}: {legacy_override_fields}."
                )
            for conditioning_field in sorted(_CONDITIONING_ARTIFACT_FIELDS & set(override)):
                raise ValueError(
                    f"Cosmos-Dreams {conditioning_field} may only come from transformer/config.json, "
                    f"not {attr}.{key}.{conditioning_field}."
                )
            for field_name in artifact.keys() & override.keys():
                if artifact[field_name] != override[field_name]:
                    raise ValueError(
                        "Cosmos-Dreams deployment override contradicts the exported artifact: "
                        f"{attr}.{key}.{field_name}={override[field_name]!r}, "
                        f"artifact={artifact[field_name]!r}."
                    )


def _parse_v2_manifest(
    manifest_cls: type[CosmosDreamsManifest],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse every field in the immutable v2 action-conditioned contract."""

    fixed_step = _mapping(_first(sources, "fixed_step_sampler_config", default={}))
    resolution = _first(sources, "deploy_resolution", default=None)
    resolution_map = _mapping(resolution)
    resolution_list = resolution if isinstance(resolution, list | tuple) else None
    height = _first(sources, "height", "deploy_height", default=None)
    width = _first(sources, "width", "deploy_width", default=None)
    if height is None and resolution_map:
        height = resolution_map.get("height")
    if width is None and resolution_map:
        width = resolution_map.get("width")
    if resolution_list is not None and len(resolution_list) == 2:
        height = resolution_list[0] if height is None else height
        width = resolution_list[1] if width is None else width

    raw_t_list = fixed_step.get("t_list", _first(sources, "t_list", default=manifest_cls.t_list))
    raw_conditioning = _first(sources, "action_schema", default=None)
    conditioning = CosmosDreamsActionSchema.model_validate(raw_conditioning) if raw_conditioning is not None else None
    return {
        "schema_version": 2,
        "chunk_size": int(_first(sources, "chunk_size", "teacher_forcing_frames_per_chunk", default=4)),
        "window_frames": int(_first(sources, "window_frames", "kv_cache_inference_size", default=96)),
        "sink_frames": int(_first(sources, "sink_frames", "attention_sink_size", default=0)),
        "text_cache_max_len": int(_first(sources, "text_cache_max_len", "ar_static_und_cache_max_len", default=512)),
        "height": int(height if height is not None else 720),
        "width": int(width if width is not None else 1280),
        "latent_patch_size": int(_first(sources, "latent_patch_size", "patch_spatial", default=2)),
        "vae_spatial_compression_factor": int(
            _first(sources, "vae_spatial_compression_factor", "latent_downsample_factor", default=16)
        ),
        "temporal_compression_factor": int(_first(sources, "temporal_compression_factor", default=4)),
        "temporal_modality_margin": int(
            _first(
                sources,
                "temporal_modality_margin",
                "unified_3d_mrope_temporal_modality_margin",
                default=15_000,
            )
        ),
        "base_fps": float(_first(sources, "base_fps", default=24.0)),
        "enable_fps_modulation": bool(_first(sources, "enable_fps_modulation", default=True)),
        "attention_mode": str(_first(sources, "attention_mode", "joint_attn_implementation", default="three_way")),
        "video_temporal_causal": bool(_first(sources, "video_temporal_causal", default=True)),
        "sample_type": str(fixed_step.get("sample_type", _first(sources, "sample_type", default="sde"))),
        "t_list": tuple(float(value) for value in raw_t_list),
        "num_train_timesteps": int(
            fixed_step.get(
                "num_train_timesteps",
                _first(sources, "num_train_timesteps", default=1000),
            )
        ),
        "checkpoint_id": str(_first(sources, "checkpoint_id", default="unknown")),
        "checkpoint_iteration": int(_first(sources, "checkpoint_iteration", default=0)),
        "checkpoint_hash": str(_first(sources, "checkpoint_hash", default="unknown")),
        "conditioning": conditioning,
    }


_V3_EXPORTED_ARTIFACT_FIELDS = (_V2_EXPORTED_ARTIFACT_FIELDS - {"action_schema"}) | {"conditioning"}
_EXPORTED_ARTIFACT_REQUIRED_FIELDS_BY_VERSION[3] = _V3_EXPORTED_ARTIFACT_FIELDS
_EXPORTED_ARTIFACT_ALLOWED_FIELDS_BY_VERSION[3] = _V3_EXPORTED_ARTIFACT_FIELDS


def _parse_v3_manifest(
    manifest_cls: type[CosmosDreamsManifest],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse the additive v3 control-video conditioning contract."""

    values = _parse_v2_manifest(manifest_cls, sources)
    raw_conditioning = _first(sources, "conditioning", default=None)
    values.update(
        schema_version=3,
        conditioning=parse_cosmos_dreams_v3_conditioning(raw_conditioning) if raw_conditioning is not None else None,
    )
    return values


_MANIFEST_PARSERS_BY_VERSION: dict[
    int,
    Callable[[type[CosmosDreamsManifest], list[dict[str, Any]]], dict[str, Any]],
] = {2: _parse_v2_manifest, 3: _parse_v3_manifest}


@dataclass(frozen=True)
class CosmosDreamsManifest:
    """Causal settings stripped by the native Cosmos3 base-model export.

    Resolution is fixed at model load because ``tokens_per_frame`` is the AR
    paged-cache block size.  Pixel dimensions need not be multiples of 32; the
    latent patch grid uses ceiling division, matching Cosmos3 patchification.
    """

    schema_version: int = 2
    chunk_size: int = 4
    window_frames: int = 96
    sink_frames: int = 0
    text_cache_max_len: int = 512
    height: int = 720
    width: int = 1280
    latent_patch_size: int = 2
    vae_spatial_compression_factor: int = 16
    temporal_compression_factor: int = 4
    temporal_modality_margin: int = 15_000
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
    conditioning: Any | None = None
    # Compatibility-only constructor/attribute alias. On-disk v2
    # ``action_schema`` is parsed into ``conditioning``; new schema versions do
    # not need to add another action-named field to the manifest trunk.
    action_schema: CosmosDreamsActionSchema | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        conditioning = self.conditioning
        if self.action_schema is not None:
            if conditioning is not None and conditioning != self.action_schema:
                raise ValueError("Cosmos-Dreams conditioning and action_schema aliases disagree")
            conditioning = self.action_schema
            object.__setattr__(self, "conditioning", conditioning)
        if isinstance(conditioning, CosmosDreamsActionSchema) and self.action_schema is None:
            object.__setattr__(self, "action_schema", conditioning)
        if self.schema_version not in _MANIFEST_PARSERS_BY_VERSION:
            supported = sorted(_MANIFEST_PARSERS_BY_VERSION)
            raise ValueError(
                f"Unsupported Cosmos-Dreams manifest schema_version={self.schema_version}; expected one of {supported}"
            )
        positive = {
            "schema_version": self.schema_version,
            "chunk_size": self.chunk_size,
            "window_frames": self.window_frames,
            "text_cache_max_len": self.text_cache_max_len,
            "height": self.height,
            "width": self.width,
            "latent_patch_size": self.latent_patch_size,
            "vae_spatial_compression_factor": self.vae_spatial_compression_factor,
            "temporal_compression_factor": self.temporal_compression_factor,
            "num_train_timesteps": self.num_train_timesteps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"Cosmos-Dreams manifest {name} must be positive, got {value}")
        if self.sink_frames < 0:
            raise ValueError(f"Cosmos-Dreams manifest sink_frames must be non-negative, got {self.sink_frames}")
        if self.checkpoint_iteration < 0:
            raise ValueError(
                f"Cosmos-Dreams checkpoint_iteration must be non-negative, got {self.checkpoint_iteration}"
            )
        if self.attention_mode != "three_way":
            raise ValueError(f"Cosmos-Dreams requires attention_mode='three_way', got {self.attention_mode!r}")
        if not self.video_temporal_causal:
            raise ValueError("Cosmos-Dreams requires video_temporal_causal=True")
        if not self.enable_fps_modulation:
            raise ValueError("Cosmos-Dreams AR inference requires FPS modulation")
        if not math.isfinite(self.base_fps) or self.base_fps <= 0:
            raise ValueError(f"Cosmos-Dreams base_fps must be positive, got {self.base_fps}")
        if self.sample_type not in {"ode", "sde"}:
            raise ValueError(f"Cosmos-Dreams sample_type must be 'ode' or 'sde', got {self.sample_type!r}")
        if not self.t_list:
            raise ValueError("Cosmos-Dreams t_list must not be empty")
        if abs(self.t_list[0] - 1.0) > 1e-6:
            raise ValueError(f"Cosmos-Dreams t_list must start at 1.0, got {self.t_list[0]}")
        if any(not math.isfinite(sigma) or sigma <= 0.0 or sigma > 1.0 for sigma in self.t_list):
            raise ValueError(f"Cosmos-Dreams t_list entries must be in (0, 1], got {self.t_list}")
        if any(left <= right for left, right in zip(self.t_list, self.t_list[1:])):
            raise ValueError(f"Cosmos-Dreams t_list must be strictly descending, got {self.t_list}")
        if self.checkpoint_hash != "unknown" and (
            len(self.checkpoint_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.checkpoint_hash)
        ):
            raise ValueError(
                f"Cosmos-Dreams checkpoint_hash must be a 64-character SHA-256 digest, got {self.checkpoint_hash!r}"
            )
        if self.checkpoint_hash != "unknown" and set(self.checkpoint_hash) == {"0"}:
            raise ValueError("Cosmos-Dreams checkpoint_hash cannot be the all-zero template value")
        if self.schema_version == 3 and not isinstance(
            conditioning,
            CosmosDreamsActionConditioning | CosmosDreamsControlVideoConditioning,
        ):
            raise ValueError("Cosmos-Dreams schema v3 requires conditioning with a recognized mode discriminator.")

    def require_action_schema(self) -> CosmosDreamsActionSchema:
        schema = self.action_schema
        if schema is None:
            raise ValueError("Cosmos-Dreams action_schema is unavailable.")
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
        """Token contribution owned by the selected conditioning payload."""

        if isinstance(self.conditioning, CosmosDreamsControlVideoConditioning):
            return 0
        return self.action_tokens_per_frame

    @property
    def conditioning_digest(self) -> str | None:
        if self.conditioning is None:
            return None
        digest = getattr(self.conditioning, "digest", None)
        if callable(digest):
            digest = digest()
        if not isinstance(digest, str) or not digest:
            raise ValueError("Cosmos-Dreams conditioning payload must expose a non-empty digest")
        return digest

    @classmethod
    def from_od_config(
        cls,
        od_config: Any,
        *,
        require_explicit: bool = False,
    ) -> CosmosDreamsManifest:
        if require_explicit:
            artifact = _exported_artifact_source(od_config)
            if not artifact:
                raise ValueError(
                    "Cosmos-Dreams requires a causal manifest embedded in transformer/config.json; "
                    "deployment defaults are not a validated artifact."
                )
            raw_schema_version = artifact.get("schema_version", 2)
            if isinstance(raw_schema_version, bool):
                raise ValueError("Cosmos-Dreams transformer artifact schema_version must be an integer")
            try:
                schema_version = int(raw_schema_version)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Cosmos-Dreams transformer artifact schema_version must be an integer") from exc
            required_fields = _EXPORTED_ARTIFACT_REQUIRED_FIELDS_BY_VERSION.get(schema_version)
            allowed_fields = _EXPORTED_ARTIFACT_ALLOWED_FIELDS_BY_VERSION.get(schema_version)
            if required_fields is None or allowed_fields is None:
                raise ValueError(
                    f"Unsupported Cosmos-Dreams manifest schema_version={schema_version}; "
                    f"expected one of {sorted(_MANIFEST_PARSERS_BY_VERSION)}"
                )
            missing_artifact_fields = sorted(required_fields - set(artifact))
            unknown_artifact_fields = sorted(set(artifact) - allowed_fields)
            if missing_artifact_fields:
                raise ValueError(
                    f"Cosmos-Dreams transformer artifact is incomplete; missing {', '.join(missing_artifact_fields)}."
                )
            if unknown_artifact_fields:
                raise ValueError(
                    f"Cosmos-Dreams transformer artifact contains unknown fields: {unknown_artifact_fields}."
                )
            _validate_deploy_overrides(od_config, artifact)
        else:
            artifact = _exported_artifact_source(od_config)
        sources = [artifact] if artifact else _manifest_sources(od_config)
        raw_schema_version = _first(sources, "schema_version", default=2)
        if isinstance(raw_schema_version, bool):
            raise ValueError("Cosmos-Dreams manifest schema_version must be an integer")
        try:
            schema_version = int(raw_schema_version)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Cosmos-Dreams manifest schema_version must be an integer") from exc
        parser = _MANIFEST_PARSERS_BY_VERSION.get(schema_version)
        if parser is None:
            raise ValueError(
                f"Unsupported Cosmos-Dreams manifest schema_version={schema_version}; "
                f"expected one of {sorted(_MANIFEST_PARSERS_BY_VERSION)}"
            )
        if artifact and not require_explicit:
            _validate_deploy_overrides(od_config, artifact)
        return cls(**parser(cls, sources))

    def require_exported_artifact(self) -> None:
        """Reject deployment-only defaults without the Stage-2 artifact data."""

        missing: list[str] = []
        if not self.checkpoint_id or self.checkpoint_id == "unknown":
            missing.append("checkpoint_id")
        if self.checkpoint_iteration <= 0:
            missing.append("checkpoint_iteration")
        if not self.checkpoint_hash or self.checkpoint_hash == "unknown":
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
    def latent_height(self) -> int:
        return math.ceil(self.height / self.vae_spatial_compression_factor)

    @property
    def latent_width(self) -> int:
        return math.ceil(self.width / self.vae_spatial_compression_factor)

    @property
    def patch_grid(self) -> tuple[int, int]:
        return (
            math.ceil(self.latent_height / self.latent_patch_size),
            math.ceil(self.latent_width / self.latent_patch_size),
        )

    @property
    def vision_tokens_per_frame(self) -> int:
        grid_h, grid_w = self.patch_grid
        return grid_h * grid_w

    @property
    def tokens_per_frame(self) -> int:
        return self.conditioning_tokens_per_frame + self.vision_tokens_per_frame

    @property
    def sampler_id(self) -> str:
        values = ",".join(f"{value:.12g}" for value in self.t_list)
        return f"{self.sample_type}:{values}:train={self.num_train_timesteps}"

    @property
    def digest(self) -> str:
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"conditioning", "action_schema"}
        }
        values["conditioning_digest"] = self.conditioning_digest
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
