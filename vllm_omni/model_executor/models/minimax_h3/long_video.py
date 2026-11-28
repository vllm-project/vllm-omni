# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared long-video request limits for encoder and diffusion entry points."""

from collections.abc import Mapping, Sequence
from typing import Any

from vllm_omni.errors import OmniClientError

MINIMAX_H3_MAX_FULL_SECONDS = 30.0
MINIMAX_H3_MAX_CONTINUATION_SECONDS = 300.0
# Ref2VA carries the previous window's tail as a ``latent_guide`` reference; T2VA and FL2VA carry it
# as a frame-0 timeline guide, the layout ``continue_from`` uses across requests.
MINIMAX_H3_CONTINUATION_TASKS = ("ref2va", "t2va", "fl2va")


def resolve_long_video_mode(extra: Mapping[str, Any], task: str) -> str:
    long_video = extra.get("long_video", False)
    if not isinstance(long_video, bool):
        raise OmniClientError("MiniMax H3 long_video must be a boolean")
    mode = extra.get("long_video_mode", "continuation" if long_video and task == "ref2va" else "full")
    if mode not in ("full", "continuation"):
        raise OmniClientError("MiniMax H3 long_video_mode must be full or continuation")
    if mode == "continuation" and task not in MINIMAX_H3_CONTINUATION_TASKS:
        raise OmniClientError(
            f"MiniMax H3 continuation supports {', '.join(MINIMAX_H3_CONTINUATION_TASKS)} requests, got {task!r}"
        )
    return mode


def check_continuation_conditions(task: str, keyframe_frame_indices: Sequence[int]) -> None:
    """Reject request conditions the windowed loop cannot place.

    A T2VA or FL2VA window after the first carries only its predecessor's tail,
    so a keyframe past the first frame would have no window to land in.
    """
    if task != "ref2va" and any(keyframe_frame_indices):
        raise OmniClientError(
            "MiniMax H3 continuation conditions FL2VA on a first frame only; "
            "use long_video_mode=full for other keyframes"
        )


def max_output_seconds(extra: Mapping[str, Any], task: str) -> float:
    mode = resolve_long_video_mode(extra, task)
    if not extra.get("long_video", False):
        return 15.0
    return MINIMAX_H3_MAX_CONTINUATION_SECONDS if mode == "continuation" else MINIMAX_H3_MAX_FULL_SECONDS


def validate_encoded_frame_limit(extra: Mapping[str, Any], task: str, num_frames: int) -> None:
    """Apply the same cap to externally encoded requests, allowing grid rounding."""
    seconds = max_output_seconds(extra, task)
    requested_max = int(seconds * 24)
    aligned_max = ((requested_max - 5 + 16) // 17) * 17 + 5
    if num_frames > aligned_max:
        raise OmniClientError(
            f"MiniMax H3 output duration exceeds the {seconds:g}-second request limit "
            f"({aligned_max} frames after alignment); got {num_frames} frames"
        )
