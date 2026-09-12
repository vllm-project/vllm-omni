# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

COSMOS3_EXTRA_BODY_PARAMS = frozenset(
    {
        "flow_shift",
        "max_sequence_length",
        "use_resolution_template",
        "use_duration_template",
        "negative_metadata_mode",
        "use_system_prompt",
        "system_prompt",
        "negative_prompt",
        "guardrails",
        "condition_frame_indexes_vision",
        "condition_video_keep",
        "generate_sound",
        "sound_gen",
        "sound_duration",
        "audio_duration",
        "action_mode",
        "action",
        "domain_name",
        "domain_id",
        "raw_action_dim",
        "action_chunk_size",
        "action_space",
        "action_fps",
        "image_height",
        "image_width",
        "history_length",
        "conditioning_fps",
        "resolution",
        "image_size",
        "use_state",
        "format_prompt_as_json",
        "observation",
        "robot_obs",
        "deterministic_seed",
        "session_id",
    }
)
COSMOS3_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "action",
        "raw_action_dim",
        "domain_id",
        "action_mode",
    }
)

COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS = frozenset(
    {
        "multiview",
        "wsm",
        "flow_shift",
        "max_sequence_length",
        "negative_prompt",
        "resolution",
        "fps",
        "frame_rate",
        "resolved_frame_rate",
    }
)


COSMOS3_MADS_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)
MULTIVIEW_MAX_UPLOADS = 2 * len(COSMOS3_MADS_CAMERAS)
MULTIVIEW_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
COSMOS3_TRANSFER_HINT_KEYS = ("edge", "blur", "depth", "seg", "wsm")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Cosmos3 multiview {name} must be an object, got {type(value).__name__}.")
    return value


def path_media_kind(value: Any) -> str:
    if not isinstance(value, str | Path):
        raise TypeError("HTTP multiview media must be a file path or an uploaded reference index.")
    return "image" if Path(value).suffix.lower() in MULTIVIEW_IMAGE_EXTENSIONS else "video"


def validate_multiview_request(
    extra: Mapping[str, Any],
    cameras: Sequence[str] = COSMOS3_MADS_CAMERAS,
    *,
    media_kind: Callable[[Any], str] = path_media_kind,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    multiview = _mapping(extra.get("multiview"), "extra_args['multiview']")
    unknown = set(multiview) - {
        "views",
        "condition_video_as_image",
        "condition_frame_indexes_vision",
        "num_frames",
        "resolution",
    }
    if unknown:
        raise ValueError(f"Unsupported Cosmos3 multiview fields: {sorted(unknown)}.")
    raw_views = multiview.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, str | bytes) or not raw_views:
        raise ValueError("Cosmos3 multiview.views must contain at least one camera view.")
    views = [_mapping(view, f"view {index}") for index, view in enumerate(raw_views)]
    for index, view in enumerate(views):
        unknown = set(view) - {"camera_key", "vision_path", "control_path", "vision", "control"}
        if unknown:
            raise ValueError(f"Unsupported Cosmos3 multiview view {index} fields: {sorted(unknown)}.")
    camera_keys = [str(view.get("camera_key", "")) for view in views]
    if any(not key for key in camera_keys) or len(set(camera_keys)) != len(camera_keys):
        raise ValueError(f"Cosmos3 multiview camera_key values must be non-empty and unique: {camera_keys}.")
    if tuple(camera_keys) != tuple(cameras):
        raise ValueError(
            "Cosmos3 multiview camera order must exactly match the exported checkpoint order: "
            f"expected={list(cameras)}, got={camera_keys}."
        )
    if any("lidar" in key.lower() for key in camera_keys) or extra.get("lidar") is not None:
        raise ValueError("Cosmos3 multiview v1 does not support LiDAR items.")
    selected_hints = [key for key in COSMOS3_TRANSFER_HINT_KEYS if extra.get(key) is not None]
    if selected_hints != ["wsm"]:
        raise ValueError(
            f"Cosmos3 multiview requires exactly one top-level precomputed WSM hint; selected={selected_hints}."
        )
    wsm = extra.get("wsm")
    if wsm is not True and (not isinstance(wsm, Mapping) or len(wsm) != 0):
        raise ValueError(
            "Cosmos3 multiview WSM controls must be supplied per view; top-level wsm must be true or an empty object."
        )
    for field in ("vision", "control"):
        values = [view.get(f"{field}_path", view.get(field)) for view in views]
        present = [value is not None for value in values]
        if field == "vision" and any(present) and not all(present):
            raise ValueError("Cosmos3 multiview vision inputs must be supplied for every camera or none.")
        if field == "control" and not all(present):
            raise ValueError("Cosmos3 multiview requires a precomputed control input for every camera.")
        if any(present):
            kinds = {media_kind(value) for value in values}
            if len(kinds) != 1:
                raise ValueError(f"Cosmos3 multiview {field} inputs must be all images or all videos, got {kinds}.")
    return multiview, views


def has_multiview_upload_indexes(extra: Mapping[str, Any]) -> bool:
    multiview = extra.get("multiview")
    views = multiview.get("views") if isinstance(multiview, Mapping) else None
    return isinstance(views, list) and any(
        isinstance(view, Mapping) and any(f"{role}_reference_index" in view for role in ("control", "vision"))
        for view in views
    )


def resolve_multiview_uploads(extra: Mapping[str, Any], upload_paths: Sequence[str]) -> dict[str, Any]:
    """Copy the manifest and substitute paths; also usable with suffix-only paths before persistence."""
    if len(upload_paths) > MULTIVIEW_MAX_UPLOADS:
        raise ValueError(f"Cosmos3 multiview accepts at most {MULTIVIEW_MAX_UPLOADS} uploaded references.")
    multiview = dict(_mapping(extra.get("multiview"), "extra_params.multiview"))
    raw_views = multiview.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("Cosmos3 multiview.views must contain at least one camera view.")
    used: set[int] = set()
    views = []
    for raw_view in raw_views:
        view = dict(_mapping(raw_view, "view"))
        for role in ("control", "vision"):
            key = f"{role}_reference_index"
            if key not in view:
                continue
            index = view.pop(key)
            if type(index) is not int or not 0 <= index < len(upload_paths):
                raise ValueError(f"{key} must be an integer index into input_references, got {index!r}.")
            if index in used:
                raise ValueError(f"input_references index {index} is referenced more than once.")
            if any(view.get(field) is not None for field in (role, f"{role}_path")):
                raise ValueError(f"{key} cannot be combined with {role} or {role}_path for the same camera.")
            used.add(index)
            view[f"{role}_path"] = upload_paths[index]
        views.append(view)
    if used != set(range(len(upload_paths))):
        raise ValueError("Every input_references upload must be referenced exactly once in multiview.views.")
    multiview["views"] = views
    resolved = {**extra, "multiview": multiview}
    validate_multiview_request(resolved)
    return resolved
