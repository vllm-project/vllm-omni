# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Cosmos3-Nano-Transfer-Auto deployment metadata, shared by pipeline and transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from .lidar import validate_lidar_config

COSMOS3_MULTIVIEW_BACKBONE_TYPE = "cosmos3_multiview"


def _tf_config_get(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Cosmos3 multiview {name} must be an object, got {type(value).__name__}.")
    return value


COSMOS3_MULTIVIEW_CONTRACT_FIELDS = frozenset(
    {
        "cameras",
        "cross_view_past_window_seconds",
        "rig_view_embedding",
        "lidar_latent_patch_size_hw",
        "lidar",
        "inference_defaults",
    }
)


def _required_deployment_field(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(f"Cosmos3 multiview transformer config requires field {name!r}.")
    return config[name]


def _positive_int(value: Any) -> TypeGuard[int]:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validated_lidar_patch(config: Mapping[str, Any]) -> list[int] | None:
    """Require the joint checkpoint's explicit transformer patch over LiDAR latents."""
    name = "lidar_latent_patch_size_hw"
    if "lidar" not in config:
        if name in config:
            raise ValueError(f"Cosmos3 multiview {name} requires a lidar block.")
        return None
    patch = _required_deployment_field(config, name)
    if not isinstance(patch, list | tuple) or len(patch) != 2 or not all(_positive_int(side) for side in patch):
        raise ValueError(f"Cosmos3 multiview {name} must be two positive integers, got {patch!r}.")
    return list(patch)


def _validated_rig_view_embedding(config: Mapping[str, Any], cameras: Sequence[str]) -> dict[str, Any]:
    """Validate the physical rig-ID table: one row per MADS camera ID plus a final LiDAR row."""
    raw = _required_deployment_field(config, "rig_view_embedding")
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    rig = _mapping(raw, "rig_view_embedding")
    if unknown := set(rig) - {"num_embeddings", "camera_ids", "lidar_id"}:
        raise ValueError(f"Unknown Cosmos3 multiview rig_view_embedding fields: {sorted(unknown)}.")
    num_embeddings = rig.get("num_embeddings")
    if not _positive_int(num_embeddings) or num_embeddings < 2:
        raise ValueError(
            f"Cosmos3 multiview rig_view_embedding.num_embeddings must be an integer >= 2, got {num_embeddings!r}."
        )
    camera_ids = _mapping(rig.get("camera_ids"), "rig_view_embedding.camera_ids")
    if set(camera_ids) != set(cameras):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must name exactly the exported cameras: "
            f"expected={sorted(cameras)}, got={sorted(camera_ids)}."
        )
    for camera, row in camera_ids.items():
        # Row N-1 is reserved for LiDAR.
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row <= num_embeddings - 2:
            raise ValueError(
                f"Cosmos3 multiview rig_view_embedding.camera_ids[{camera!r}] must be an integer in "
                f"[0, {num_embeddings - 2}], got {row!r}."
            )
    if len(set(camera_ids.values())) != len(camera_ids):
        raise ValueError("Cosmos3 multiview rig camera IDs must be unique.")
    lidar_id = rig.get("lidar_id")
    if isinstance(lidar_id, bool) or not isinstance(lidar_id, int) or lidar_id != num_embeddings - 1:
        raise ValueError(
            f"Cosmos3 multiview rig_view_embedding.lidar_id must be the final row {num_embeddings - 1}, "
            f"got {lidar_id!r}."
        )
    return {"num_embeddings": num_embeddings, "camera_ids": dict(camera_ids), "lidar_id": lidar_id}


def _validated_multiview_deployment_config(model_config: Any) -> dict[str, Any]:
    """Validate the sole Cosmos3-Nano-Transfer-Auto deployment contract before initialization."""
    backbone_type = _tf_config_get(model_config, "backbone_type", None)
    if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
        raise ValueError(
            "Cosmos3MultiviewPipeline requires transformer/config.json "
            f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
        )

    raw_config = _tf_config_get(model_config, "multiview", None)
    if raw_config is None:
        raise ValueError("Cosmos3 multiview transformer config must contain a 'multiview' object.")
    if hasattr(raw_config, "to_dict"):
        raw_config = raw_config.to_dict()
    config = _mapping(raw_config, "transformer config")

    if unknown := set(config) - COSMOS3_MULTIVIEW_CONTRACT_FIELDS:
        raise ValueError(f"Unknown Cosmos3 multiview contract fields: {sorted(unknown)}. Re-export the checkpoint.")
    temporal_window = _required_deployment_field(config, "cross_view_past_window_seconds")
    if isinstance(temporal_window, bool) or not isinstance(temporal_window, int | float):
        raise ValueError("Cosmos3 multiview cross_view_past_window_seconds must be a finite non-negative number.")
    if not math.isfinite(temporal_window) or temporal_window < 0:
        raise ValueError("Cosmos3 multiview cross_view_past_window_seconds must be finite and non-negative.")
    cameras = _required_deployment_field(config, "cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, str) and camera for camera in cameras)
    ):
        raise TypeError("Cosmos3 multiview cameras must be a non-empty list of strings.")
    if len(cameras) != len(set(cameras)):
        raise ValueError("Cosmos3 multiview cameras must be unique.")
    if "lidar" in config:
        validate_lidar_config(dict(_mapping(config["lidar"], "lidar")))
    lidar_patch = _validated_lidar_patch(config)
    rig = _validated_rig_view_embedding(config, cameras)
    defaults = _mapping(_required_deployment_field(config, "inference_defaults"), "inference_defaults")
    fields = {
        "resolution",
        "fps",
        "num_steps",
        "guidance",
        "shift",
        "control_guidance",
        "emphasize_control_in_prompt",
        "guidance_interval",
        "control_guidance_interval",
        "normalize_cfg",
    }
    if missing := fields - defaults.keys():
        raise ValueError(f"Incomplete inference_defaults metadata: {sorted(missing)}.")
    if unknown := defaults.keys() - fields:
        raise ValueError(f"Unknown inference_defaults fields: {sorted(unknown)}.")
    if defaults["resolution"] not in ("480", "720"):
        raise ValueError("inference_defaults.resolution must be 480 or 720.")
    for name in ("fps", "guidance", "shift", "control_guidance"):
        value = defaults[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"inference_defaults.{name} must be finite and non-negative.")
    if defaults["fps"] == 0 or defaults["shift"] == 0 or not _positive_int(defaults["num_steps"]):
        raise ValueError("inference_defaults requires positive FPS, integer step count and shift.")
    for name in ("emphasize_control_in_prompt", "normalize_cfg"):
        if not isinstance(defaults[name], bool):
            raise ValueError(f"inference_defaults.{name} must be boolean.")
    for name in ("guidance_interval", "control_guidance_interval"):
        interval = defaults[name]
        if interval is not None and (
            not isinstance(interval, list | tuple)
            or len(interval) != 2
            or any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) for v in interval)
            or interval[0] >= interval[1]
        ):
            raise ValueError(f"inference_defaults.{name} must be two finite increasing bounds or null.")
    validated = {**config, "cross_view_past_window_seconds": float(temporal_window), "rig_view_embedding": rig}
    if lidar_patch is not None:
        validated["lidar_latent_patch_size_hw"] = lidar_patch
    return validated
