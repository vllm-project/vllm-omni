# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact reference per-camera caption headers, system prompts and transfer emphasis."""

from collections.abc import Sequence

from vllm_omni.model_extras.cosmos3 import COSMOS3_MADS_CAMERA_ATTRIBUTES

from .utils import COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE

# Reference task prompts for per-camera-caption AV checkpoints.
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

COSMOS3_AV_JOINT_CONTROL_DIRECTIVE = (
    "Follow the wsm and lidar control videos precisely: every camera view must"
    " align with its world-scenario map, and the LiDAR rangemap must align with the"
    " HD-map rangemap, at every frame."
)


def _camera_identity(camera: str) -> str:
    attributes = COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]
    role = str(attributes["camera_role"]).replace("_", "-")
    camera_type = str(attributes["camera_type"]).replace("_", "-")
    camera_type = {"standard": "", "wide": "wide-angle"}.get(camera_type, camera_type)
    return f"{' '.join(part for part in (role, camera_type) if part)} camera"


def _facing(camera: str) -> str:
    return str(COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]["facing"]).replace("_", "-")


def _fov(camera: str) -> int:
    return int(COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]["fov_degrees"])


def format_rig_view_captions(captions: Sequence[str], cameras: Sequence[str]) -> list[str]:
    """Prefix each camera's caption with the sampled-rig and current-camera headers.

    Mirrors the reference ``format_separate_view_captions(add_camera_rig_prefix=True)``,
    which per-camera-caption training and reference inference both use. The rig
    sentence lists the request's cameras in request order.
    """
    if not captions or len(captions) != len(cameras):
        raise ValueError(f"Per-view captions must match the cameras: captions={len(captions)}, cameras={len(cameras)}.")
    descriptions = [f"{_camera_identity(camera)} ({_facing(camera)}-facing, {_fov(camera)}° FOV)" for camera in cameras]
    rig_prefix = (
        f"This multiview driving sequence contains time-aligned recordings from {len(cameras)} vehicle-mounted "
        f"{'camera' if len(cameras) == 1 else 'cameras'}: {'; '.join(descriptions)}."
    )
    return [
        f"{rig_prefix}\n\nThe description below is for the {_camera_identity(camera)} mounted on the vehicle. "
        f"This camera is facing {_facing(camera)} and has a "
        f"{_fov(camera)}° field of view:\n\n{caption}"
        for camera, caption in zip(cameras, captions, strict=True)
    ]


def control_emphasis(hint: str, *, joint: bool) -> str:
    if joint:
        return COSMOS3_AV_JOINT_CONTROL_DIRECTIVE
    return COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE.format(hint_names=hint)
