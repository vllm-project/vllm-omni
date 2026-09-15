# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
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
        "lidar",
        "edge",
        "blur",
        "depth",
        "seg",
        "emphasize_control_in_prompt",
        "guidance_interval",
        "control_guidance",
        "control_guidance_interval",
        "sigma_max",
        "normalize_cfg",
        "flow_shift",
        "max_sequence_length",
        "negative_prompt",
        "resolution",
        "aspect_ratio",
        "fps",
        "frame_rate",
        "resolved_frame_rate",
    }
)

COSMOS3_MULTIVIEW_ASPECT_RATIOS = ("1,1", "4,3", "3,4", "16,9", "9,16")


def normalize_multiview_aspect_ratio(value: Any) -> str:
    """Normalize the explicit bucket label, or select automatic WSM sizing."""
    if value is None or value == "auto":
        return "auto"
    parts = str(value).strip().replace(":", ",").split(",")
    if len(parts) == 2:
        try:
            width, height = (int(part.strip()) for part in parts)
        except ValueError:
            pass
        else:
            if width > 0 and height > 0:
                divisor = math.gcd(width, height)
                ratio = f"{width // divisor},{height // divisor}"
                if ratio in COSMOS3_MULTIVIEW_ASPECT_RATIOS:
                    return ratio
    raise ValueError(
        f"Unsupported Cosmos3 multiview aspect_ratio={value!r}; expected auto, 1:1, 4:3, 3:4, 16:9, or 9:16."
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
MULTIVIEW_MAX_UPLOADS = 2 * len(COSMOS3_MADS_CAMERAS) + 1
MULTIVIEW_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
COSMOS3_TRANSFER_HINT_KEYS = ("edge", "blur", "depth", "seg", "wsm")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Cosmos3 multiview {name} must be an object, got {type(value).__name__}.")
    return value


def path_media_kind(value: Any) -> str:
    if not isinstance(value, str | Path):
        raise TypeError("HTTP multiview media must be a file path or an uploaded reference index.")
    if Path(value).suffix.lower() == ".safetensors":
        raise ValueError("Numeric LiDAR files cannot be used as camera image or video inputs.")
    return "image" if Path(value).suffix.lower() in MULTIVIEW_IMAGE_EXTENSIONS else "video"


def validate_multiview_request(
    extra: Mapping[str, Any],
    cameras: Sequence[str] = COSMOS3_MADS_CAMERAS,
    *,
    media_kind: Callable[[Any], str] = path_media_kind,
    separate_view_text_tokenization: bool = False,
    variable_view_count: bool = False,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    multiview = _mapping(extra.get("multiview"), "extra_args['multiview']")
    unknown = set(multiview) - {
        "views",
        "condition_video_as_image",
        "condition_frame_indexes_vision",
        "num_frames",
        "resolution",
        "aspect_ratio",
    }
    if unknown:
        raise ValueError(f"Unsupported Cosmos3 multiview fields: {sorted(unknown)}.")
    ratio = multiview.get("aspect_ratio")
    normalize_multiview_aspect_ratio(extra.get("aspect_ratio") if ratio is None else ratio)
    raw_views = multiview.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, str | bytes) or not raw_views:
        raise ValueError("Cosmos3 multiview.views must contain at least one camera view.")
    views = [_mapping(view, f"view {index}") for index, view in enumerate(raw_views)]
    for index, view in enumerate(views):
        unknown = set(view) - {"camera_key", "vision_path", "control_path", "vision", "control", "prompt"}
        if unknown:
            raise ValueError(f"Unsupported Cosmos3 multiview view {index} fields: {sorted(unknown)}.")
        caption = view.get("prompt")
        if separate_view_text_tokenization and (not isinstance(caption, str) or not caption.strip()):
            raise ValueError(f"Cosmos3 multiview requires one prompt per camera; missing views[{index}].prompt.")
        if caption is not None:
            validate_camera_caption(caption)
    camera_keys = [str(view.get("camera_key", "")) for view in views]
    if any(not key for key in camera_keys) or len(set(camera_keys)) != len(camera_keys):
        raise ValueError(f"Cosmos3 multiview camera_key values must be non-empty and unique: {camera_keys}.")
    if not set(camera_keys).issubset(cameras):
        raise ValueError(
            "Cosmos3 multiview cameras must be a subset of the exported checkpoint cameras: "
            f"expected={list(cameras)}, got={camera_keys}."
        )
    if not variable_view_count and tuple(camera_keys) != tuple(cameras):
        raise ValueError(
            "Cosmos3 multiview requires the full exported camera order unless the checkpoint enables "
            f"variable_view_count: expected={list(cameras)}, got={camera_keys}."
        )
    for field in ("guidance_interval", "control_guidance_interval"):
        interval = extra.get(field)
        if interval is not None and (
            not isinstance(interval, list | tuple)
            or len(interval) != 2
            or any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) for v in interval)
            or interval[0] >= interval[1]
        ):
            raise ValueError(f"{field} must be two finite increasing timestep bounds [lo, hi].")
    for field in ("control_guidance", "sigma_max"):
        value = extra.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{field} must be finite and non-negative.")
    joint = extra.get("lidar") is not None
    if joint:
        lidar = _mapping(extra["lidar"], "lidar")
        if set(lidar) - {"control_path", "return_output"} or not isinstance(lidar.get("control_path"), str | Path):
            raise ValueError("Cosmos3 lidar requires exactly one numeric control_path.")
        if "return_output" in lidar and type(lidar["return_output"]) is not bool:
            raise ValueError("Cosmos3 lidar.return_output must be boolean.")
        if Path(lidar["control_path"]).suffix.lower() != ".safetensors":
            raise ValueError("Cosmos3 lidar.control_path must be a .safetensors file.")
    selected_hints = [key for key in COSMOS3_TRANSFER_HINT_KEYS if extra.get(key) is not None]
    controls = [view.get("control_path", view.get("control")) is not None for view in views]
    vision = [view.get("vision_path", view.get("vision")) is not None for view in views]
    completion = any(vision) and not all(vision)
    if joint and selected_hints != ["wsm"]:
        raise ValueError("Joint camera+LiDAR requires exactly one WSM hint and no other transfer hints.")
    if joint and completion:
        raise ValueError("Joint RGB conditions must cover every camera or none.")
    if (any(controls) or completion or joint) and not all(controls):
        raise ValueError("Cosmos3 multiview requires a precomputed control input for every camera.")
    if len(selected_hints) != int(all(controls)):
        raise ValueError(
            "Cosmos3 multiview transfer requires controls and exactly one hint; ordinary generation uses neither."
        )
    for hint in selected_hints:
        value = extra[hint]
        if value is not True and (not isinstance(value, Mapping) or set(value) - {"weight"}):
            raise ValueError("Cosmos3 multiview controls must be supplied per view; hint must be true or {weight}.")
        weight = 1.0 if value is True else value.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, int | float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Cosmos3 multiview control weight must be finite and positive.")
    for field in ("vision", "control"):
        values = [view.get(f"{field}_path", view.get(field)) for view in views]
        present = [value is not None for value in values]
        if any(present):
            kinds = {media_kind(value) for value in values if value is not None}
            if len(kinds) != 1:
                raise ValueError(f"Cosmos3 multiview {field} inputs must be all images or all videos, got {kinds}.")
            if field == "vision" and completion and (kinds != {"video"} or multiview.get("condition_video_as_image")):
                raise ValueError("View completion requires complete RGB videos for known views, not partial images.")
    return multiview, views


def validate_camera_caption(caption: Any) -> None:
    """Inline captions are raw dataset text; the runtime owns camera labels and framing."""
    if not isinstance(caption, str):
        raise ValueError("Camera prompt must be a string.")
    reserved = (
        "camera_view",
        "the video is captured from",
        "follow the wsm",
        "follow the lidar",
        "control video precisely:",
        "this multiview driving sequence contains",
        "<|",
        "<camera",
        "the video has a resolution",
        "the video is of resolution",
        "the video has a duration",
        "this video is of",
        "seconds long and is of",
    )
    lowered = caption.lower()
    try:
        payload = json.loads(caption)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        raise ValueError(
            "JSON-object camera prompts are not supported; supply plain text without runtime camera labels or metadata."
        )
    if any(part in lowered for part in reserved) or any(key in caption for key in COSMOS3_MADS_CAMERAS):
        raise ValueError(
            "Camera prompt must not contain runtime camera labels, metadata templates, or control emphasis."
        )


def has_multiview_upload_indexes(extra: Mapping[str, Any]) -> bool:
    multiview = extra.get("multiview")
    views = multiview.get("views") if isinstance(multiview, Mapping) else None
    lidar = extra.get("lidar")
    return (isinstance(lidar, Mapping) and "control_reference_index" in lidar) or (
        isinstance(views, list)
        and any(
            isinstance(view, Mapping) and any(f"{role}_reference_index" in view for role in ("control", "vision"))
            for view in views
        )
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
    multiview["views"] = views
    resolved = {**extra, "multiview": multiview}
    if extra.get("lidar") is not None:
        lidar = dict(_mapping(extra["lidar"], "lidar"))
        if "control_reference_index" in lidar:
            index = lidar.pop("control_reference_index")
            if type(index) is not int or not 0 <= index < len(upload_paths):
                raise ValueError("lidar.control_reference_index must be an integer index into input_references.")
            if index in used:
                raise ValueError(f"input_references index {index} is referenced more than once.")
            if "control_path" in lidar:
                raise ValueError("lidar.control_reference_index cannot be combined with control_path.")
            used.add(index)
            lidar["control_path"] = upload_paths[index]
        resolved["lidar"] = lidar
    if used != set(range(len(upload_paths))):
        raise ValueError("Every input_references upload must be referenced exactly once in multiview.views or lidar.")
    # Upload admission has no deployment metadata. The pipeline enforces the
    # checkpoint's fixed/variable camera policy and per-camera caption requirements
    # after references are resolved. Missing required captions can therefore fail
    # an asynchronous job after admission; supplied captions are still validated.
    validate_multiview_request(resolved, variable_view_count=True)
    return resolved
