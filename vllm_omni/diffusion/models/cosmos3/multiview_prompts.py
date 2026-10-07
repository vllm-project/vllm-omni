# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact reference per-camera caption headers, system prompts and transfer emphasis."""

from collections.abc import Sequence

from .utils import COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE

# The MADS rig, verbatim from imaginaire4 ``datasets/multiview/camera_attributes.py``.
# Per-view caption headers quote these values, so a drifted copy still produces
# well-formed captions that describe the wrong rig.
MADS_CAMERA_ATTRIBUTES: dict[str, dict[str, str | int]] = {
    "camera_front_wide_120fov": {
        "camera_role": "front",
        "camera_type": "wide",
        "facing": "forward",
        "fov_degrees": 120,
    },
    "camera_cross_right_120fov": {
        "camera_role": "right_side",
        "camera_type": "wide",
        "facing": "right",
        "fov_degrees": 120,
    },
    "camera_rear_right_70fov": {
        "camera_role": "rear_right",
        "camera_type": "standard",
        "facing": "rear_right",
        "fov_degrees": 70,
    },
    "camera_rear_tele_30fov": {
        "camera_role": "rear",
        "camera_type": "telephoto",
        "facing": "backward",
        "fov_degrees": 30,
    },
    "camera_rear_left_70fov": {
        "camera_role": "rear_left",
        "camera_type": "standard",
        "facing": "rear_left",
        "fov_degrees": 70,
    },
    "camera_cross_left_120fov": {
        "camera_role": "left_side",
        "camera_type": "wide",
        "facing": "left",
        "fov_degrees": 120,
    },
    "camera_front_tele_30fov": {
        "camera_role": "front",
        "camera_type": "telephoto",
        "facing": "forward",
        "fov_degrees": 30,
    },
    "camera_front_fisheye_200fov": {
        "camera_role": "front",
        "camera_type": "fisheye",
        "facing": "forward",
        "fov_degrees": 200,
    },
    "camera_left_fisheye_200fov": {
        "camera_role": "left_side",
        "camera_type": "fisheye",
        "facing": "left",
        "fov_degrees": 200,
    },
    "camera_right_fisheye_200fov": {
        "camera_role": "right_side",
        "camera_type": "fisheye",
        "facing": "right",
        "fov_degrees": 200,
    },
    "camera_rear_fisheye_200fov": {
        "camera_role": "rear",
        "camera_type": "fisheye",
        "facing": "backward",
        "fov_degrees": 200,
    },
}

# imaginaire4 ``text_tokenizer.py`` task prompts for per-camera-caption AV checkpoints.
_AV_WSM_CONTROL_INSTRUCTION = (
    "Follow WSM controls for vehicles (including trucks), cyclists, pedestrians, traffic lights, traffic signs, "
    "road markings, lane boundaries, and road boundaries. "
    "Do not add objects or road features in these categories that are absent from WSM. "
    "Use captions for appearance and unconstrained background details; WSM takes precedence in any conflict."
)
COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates temporally synchronized, geometrically consistent autonomous-driving "
    "videos from per-camera scene descriptions and World Scenario Map (WSM) control videos depicting the controlled "
    "objects and road layout. Treat all camera views as simultaneous observations of the same driving scene, "
    "preserving each camera's viewpoint, shared ego motion, road layout, object identity and motion, weather, "
    f"lighting, and cross-view consistency.\n\n{_AV_WSM_CONTROL_INSTRUCTION}"
)
COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that jointly generates temporally synchronized, geometrically consistent "
    "autonomous-driving camera videos and LiDAR range-view sequences from per-camera scene descriptions and provided "
    "control signals: per-camera World Scenario Map (WSM) control videos depicting the controlled objects and road "
    "layout, and an HD-map control for LiDAR. Treat all camera views and LiDAR sweeps as synchronized observations "
    "of the same driving scene, preserving each camera's viewpoint, shared ego motion, road layout, object identity "
    "and motion, weather, lighting, cross-view consistency, and camera-LiDAR alignment."
    f"\n\n{_AV_WSM_CONTROL_INSTRUCTION}"
)


def _camera_identity(camera: str) -> str:
    attributes = MADS_CAMERA_ATTRIBUTES[camera]
    role = str(attributes["camera_role"]).replace("_", "-")
    camera_type = str(attributes["camera_type"]).replace("_", "-")
    camera_type = {"standard": "", "wide": "wide-angle"}.get(camera_type, camera_type)
    return f"{' '.join(part for part in (role, camera_type) if part)} camera"


def _facing(camera: str) -> str:
    return str(MADS_CAMERA_ATTRIBUTES[camera]["facing"]).replace("_", "-")


def format_rig_view_captions(captions: Sequence[str], cameras: Sequence[str]) -> list[str]:
    """Prefix each camera's caption with the sampled-rig and current-camera headers.

    Mirrors imaginaire4 ``format_separate_view_captions(add_camera_rig_prefix=True)``,
    which per-camera-caption training and reference inference both use. The rig
    sentence lists the request's cameras in request order.
    """
    if not captions or len(captions) != len(cameras):
        raise ValueError(f"Per-view captions must match the cameras: captions={len(captions)}, cameras={len(cameras)}.")
    descriptions = [
        f"{_camera_identity(camera)} ({_facing(camera)}-facing, {MADS_CAMERA_ATTRIBUTES[camera]['fov_degrees']}° FOV)"
        for camera in cameras
    ]
    rig_prefix = (
        f"This multiview driving sequence contains time-aligned recordings from {len(cameras)} vehicle-mounted "
        f"{'camera' if len(cameras) == 1 else 'cameras'}: {'; '.join(descriptions)}."
    )
    return [
        f"{rig_prefix}\n\nThe description below is for the {_camera_identity(camera)} mounted on the vehicle. "
        f"This camera is facing {_facing(camera)} and has a "
        f"{MADS_CAMERA_ATTRIBUTES[camera]['fov_degrees']}° field of view:\n\n{caption}"
        for camera, caption in zip(cameras, captions, strict=True)
    ]


def control_emphasis(hint: str, *, joint: bool) -> str:
    if joint:
        return (
            "Follow the wsm and lidar control videos precisely: every camera view must"
            " align with its world-scenario map, and the LiDAR rangemap must align with the"
            " HD-map rangemap, at every frame."
        )
    return COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE.format(hint_names=hint)
