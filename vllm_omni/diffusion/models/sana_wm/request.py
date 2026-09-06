# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request normalization for Sana-WM image-to-video.

The released SANA-WM checkpoint is first-frame image-to-video. This module keeps
request-side normalization lightweight: it validates the image/camera/action
contract and stores a canonical payload under
``additional_information["sana_wm"]``. Model-internal raymap / Plucker
projection lives with the transformer.

This lives beside the model rather than under ``stage_input_processors``
because nothing loads it as one: the pipeline's ``pre_process_func`` is
registered from ``pipeline_sana_wm``, the same as every other diffusion model,
and no deploy config wires a ``custom_process_input_func`` here. Keeping it in
the stage package forced the pipeline to import *upwards* — the only such
import in ``diffusion/models/`` — which in turn made the VAE compression
constants impossible to share and so duplicated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_omni.diffusion.models.sana_wm.camera_control import action_rollout_num_frames
from vllm_omni.diffusion.models.sana_wm.config import (
    SANA_WM_VAE_SPATIAL_COMPRESSION,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)

SANA_WM_CANONICAL_KEY = "sana_wm"
# Request fallbacks used whenever the field is absent, mirroring the deploy YAML
# (161 frames) and the model's native resolution. Both are genuinely reachable:
# a served request that omits num_frames never gets one injected, and callers
# may ask for a different resolution.
SANA_WM_DEFAULT_NUM_FRAMES = 161
SANA_WM_DEFAULT_HEIGHT = 704
SANA_WM_DEFAULT_WIDTH = 1280
SANA_WM_DEFAULT_CAMERA_FORMAT = "c2w_4x4"
SANA_WM_DEFAULT_COORDINATE_SYSTEM = "official"


def _unwrap_single(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Sana-WM {name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_finite_float(value: Any, *, name: str) -> float:
    """Parse a float and reject NaN/inf.

    Non-finite geometry does not fail loudly downstream: it propagates through
    the raymap into the Plucker maps and comes back as an all-NaN video with a
    200 response.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sana-WM {name} must be a number, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Sana-WM {name} must be finite, got {parsed!r}.")
    return parsed


def _validate_numeric_matrix4x4(matrix: Any, *, name: str) -> None:
    if not _is_sequence(matrix) or len(matrix) != 4:
        raise ValueError(f"Sana-WM {name} must be a 4x4 matrix.")
    for row_idx, row in enumerate(matrix):
        if not _is_sequence(row) or len(row) != 4:
            raise ValueError(f"Sana-WM {name}[{row_idx}] must contain 4 values.")
        for col_idx, value in enumerate(row):
            _as_finite_float(value, name=f"{name}[{row_idx}][{col_idx}]")


def _validate_camera_poses(poses: Any, *, num_frames: int | None = None) -> None:
    if not _is_sequence(poses) or len(poses) == 0:
        raise ValueError("Sana-WM camera poses must be a non-empty sequence of 4x4 matrices.")
    if num_frames is not None and len(poses) != num_frames:
        raise ValueError(f"Sana-WM camera poses length {len(poses)} must equal num_frames {num_frames}.")
    for idx, pose in enumerate(poses):
        _validate_numeric_matrix4x4(pose, name=f"camera.poses[{idx}]")


def _validate_intrinsics(intrinsics: Any) -> None:
    # Only the ``{fx, fy, cx, cy}`` mapping is accepted (the one form the recipe
    # and examples use). Camera intrinsics are optional; when omitted the model
    # derives them from the output resolution.
    if intrinsics is None:
        return
    if not isinstance(intrinsics, Mapping):
        raise ValueError("Sana-WM intrinsics must be a {fx, fy, cx, cy} mapping.")
    required = {"fx", "fy", "cx", "cy"}
    missing = sorted(required - set(intrinsics))
    if missing:
        raise ValueError(f"Sana-WM intrinsics mapping is missing keys: {missing}.")
    for key in required:
        value = _as_finite_float(intrinsics[key], name=f"intrinsics[{key!r}]")
        # ``compute_raymap`` divides by fx/fy. cx/cy stay unconstrained beyond
        # finiteness — a principal point outside the frame is unusual, not wrong.
        if key in ("fx", "fy") and value <= 0:
            raise ValueError(f"Sana-WM intrinsics[{key!r}] must be positive, got {value}.")


def _as_positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sana-WM {name} must be an integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"Sana-WM {name} must be positive, got {parsed}.")
    return parsed


def _as_positive_float(value: Any, *, name: str) -> float:
    # Finiteness first: every comparison against NaN is False, so a bare
    # ``parsed <= 0`` would accept it.
    parsed = _as_finite_float(value, name=name)
    if parsed <= 0:
        raise ValueError(f"Sana-WM {name} must be positive, got {parsed}.")
    return parsed


def _extract_image(mm_data: Mapping[str, Any]) -> Any:
    return _unwrap_single(mm_data.get("image"))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _validate_latent_geometry(*, num_frames: int, height: int, width: int) -> None:
    """Reject geometry the VAE would silently floor.

    The SANA-WM VAE compresses 32x spatially and 8x temporally, so a request
    that is not aligned loses pixels/frames instead of failing.
    """
    if height % SANA_WM_VAE_SPATIAL_COMPRESSION or width % SANA_WM_VAE_SPATIAL_COMPRESSION:
        raise ValueError(
            f"Sana-WM height and width must be divisible by {SANA_WM_VAE_SPATIAL_COMPRESSION}, "
            f"got {height} and {width}."
        )
    if (num_frames - 1) % SANA_WM_VAE_TEMPORAL_COMPRESSION:
        raise ValueError(
            f"Sana-WM num_frames must satisfy (num_frames - 1) % {SANA_WM_VAE_TEMPORAL_COMPRESSION} == 0, "
            f"got {num_frames}."
        )


def normalize_sana_wm_payload(prompt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a prompt copy with canonical Sana-WM request metadata.

    The ``sana_wm`` block is read from the top-level ``sana_wm`` key, falling
    back to ``additional_information["sana_wm"]`` so the function is idempotent
    when called again on its own output. The first-frame image is read from
    ``multi_modal_data["image"]``.
    """

    if not isinstance(prompt, Mapping):
        raise ValueError(f"Sana-WM prompt must be a mapping, got {type(prompt).__name__}.")

    result = dict(prompt)
    mm_data = _as_dict(result.get("multi_modal_data"), name="multi_modal_data")
    image = _extract_image(mm_data)
    if image is None:
        raise ValueError("Sana-WM requires a first-frame image in `multi_modal_data['image']`.")

    additional = _as_dict(result.get("additional_information"), name="additional_information")
    raw = _first_present(
        result.get(SANA_WM_CANONICAL_KEY),
        additional.get(SANA_WM_CANONICAL_KEY),
        {},
    )
    raw = _as_dict(raw, name=SANA_WM_CANONICAL_KEY)

    num_frames = _as_positive_int(
        _first_present(raw.get("num_frames"), result.get("num_frames"), SANA_WM_DEFAULT_NUM_FRAMES),
        name="num_frames",
    )
    height = _as_positive_int(
        _first_present(raw.get("height"), result.get("height"), SANA_WM_DEFAULT_HEIGHT),
        name="height",
    )
    width = _as_positive_int(_first_present(raw.get("width"), result.get("width"), SANA_WM_DEFAULT_WIDTH), name="width")
    _validate_latent_geometry(num_frames=num_frames, height=height, width=width)

    camera = raw.get("camera")
    camera = _as_dict(camera, name="camera") if camera is not None else None
    action = raw.get("action")
    if action is not None:
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Sana-WM action must be a non-empty string when provided.")
        # Rejected here rather than in the model: a mismatch used to shorten the
        # camera tensor silently and then fail deep in the transformer on a
        # camera/latent frame mismatch, and an unbounded duration used to be
        # expanded one entry per frame before any of that.
        rollout_frames = action_rollout_num_frames(action)
        if rollout_frames != num_frames:
            raise ValueError(
                f"Sana-WM action rolls out {rollout_frames} frames but num_frames is {num_frames}. "
                "Action segment durations must sum to num_frames - 1."
            )
    if camera is not None and action is not None:
        raise ValueError("Sana-WM accepts exactly one of `camera` or `action`, not both.")
    if camera is None and action is None:
        raise ValueError("Sana-WM requires either explicit camera poses or an action DSL string.")

    canonical_camera = None
    if camera is not None:
        poses = camera.get("poses")
        if poses is None:
            raise ValueError("Sana-WM camera payload requires `poses`.")
        _validate_camera_poses(poses, num_frames=num_frames)
        # Only the defaults are accepted: nothing downstream reads these two
        # fields, so honouring another value would mean interpreting w2c poses
        # as c2w and silently generating the wrong trajectory.
        camera_format = camera.get("format", SANA_WM_DEFAULT_CAMERA_FORMAT)
        if camera_format != SANA_WM_DEFAULT_CAMERA_FORMAT:
            raise ValueError(
                f"Sana-WM only accepts camera format {SANA_WM_DEFAULT_CAMERA_FORMAT!r}; "
                f"convert the poses before sending, got {camera_format!r}."
            )
        coordinate_system = camera.get("coordinate_system", SANA_WM_DEFAULT_COORDINATE_SYSTEM)
        if coordinate_system != SANA_WM_DEFAULT_COORDINATE_SYSTEM:
            raise ValueError(
                f"Sana-WM only accepts coordinate system {SANA_WM_DEFAULT_COORDINATE_SYSTEM!r}, "
                f"got {coordinate_system!r}."
            )
        canonical_camera = {"poses": poses}

    intrinsics = raw.get("intrinsics")
    _validate_intrinsics(intrinsics)
    translation_speed = raw.get("translation_speed")
    rotation_speed_deg = raw.get("rotation_speed_deg")

    canonical = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "camera": canonical_camera,
        "action": action,
        "intrinsics": intrinsics,
    }
    if translation_speed is not None:
        canonical["translation_speed"] = _as_positive_float(translation_speed, name="translation_speed")
    if rotation_speed_deg is not None:
        canonical["rotation_speed_deg"] = _as_positive_float(rotation_speed_deg, name="rotation_speed_deg")

    mm_data["image"] = image
    result["multi_modal_data"] = mm_data
    additional[SANA_WM_CANONICAL_KEY] = canonical
    result["additional_information"] = additional
    result.pop(SANA_WM_CANONICAL_KEY, None)
    return result
