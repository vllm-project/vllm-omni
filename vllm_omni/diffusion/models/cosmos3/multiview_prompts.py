# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""Exact reference camera labels and transfer emphasis."""

from .utils import COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE

DEFAULT_CAPTION_PREFIXES = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": (
        "The video is captured from a camera mounted on a car. The camera is facing to the right."
    ),
    "camera_rear_right_70fov": (
        "The video is captured from a camera mounted on a car. The camera is facing the rear right side."
    ),
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": (
        "The video is captured from a camera mounted on a car. The camera is facing the rear left side."
    ),
    "camera_cross_left_120fov": (
        "The video is captured from a camera mounted on a car. The camera is facing to the left."
    ),
    "camera_front_tele_30fov": (
        "The video is captured from a telephoto camera mounted on a car. The camera is facing forward."
    ),
    "camera_front_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing forward."
    ),
    "camera_left_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing to the left."
    ),
    "camera_right_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing to the right."
    ),
    "camera_rear_fisheye_200fov": (
        "The video is captured from a fisheye camera mounted on a car. The camera is facing backwards."
    ),
}


def format_camera_caption(caption: str, camera: str) -> str:
    # Admission accepts plain text and rejects JSON objects. Reference
    # format_view_caption labels that string before metadata formatting.
    return f"{DEFAULT_CAPTION_PREFIXES[camera]} {caption}"


def control_emphasis(hint: str, *, joint: bool) -> str:
    if joint:
        return (
            "Follow the wsm and lidar control videos precisely: every camera view must"
            " align with its world-scenario map, and the LiDAR rangemap must align with the"
            " HD-map rangemap, at every frame."
        )
    return COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE.format(hint_names=hint)
