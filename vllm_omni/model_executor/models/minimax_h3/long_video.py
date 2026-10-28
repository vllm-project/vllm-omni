# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared long-video request limits for encoder and diffusion entry points."""

from collections.abc import Mapping
from typing import Any

from vllm_omni.errors import OmniClientError

MINIMAX_H3_MAX_FULL_SECONDS = 30.0
MINIMAX_H3_MAX_CONTINUATION_SECONDS = 300.0


def resolve_long_video_mode(extra: Mapping[str, Any], task: str) -> str:
    long_video = extra.get("long_video", False)
    if not isinstance(long_video, bool):
        raise OmniClientError("MiniMax H3 long_video must be a boolean")
    mode = extra.get("long_video_mode", "continuation" if long_video and task == "ref2va" else "full")
    if mode not in ("full", "continuation"):
        raise OmniClientError("MiniMax H3 long_video_mode must be full or continuation")
    if mode == "continuation" and task != "ref2va":
        raise OmniClientError("MiniMax H3 continuation requires Ref2VA request execution (not step execution)")
    return mode


def max_output_seconds(extra: Mapping[str, Any], task: str) -> float:
    mode = resolve_long_video_mode(extra, task)
    if not extra.get("long_video", False):
        return 15.0
    return MINIMAX_H3_MAX_CONTINUATION_SECONDS if mode == "continuation" else MINIMAX_H3_MAX_FULL_SECONDS


def validate_encoded_frame_limit(
    extra: Mapping[str, Any],
    task: str,
    num_frames: int,
    fps: int = 24,
) -> None:
    """Apply the same cap to externally encoded requests, allowing grid rounding.

    ``fps`` is the request's output frame rate: the cap is a duration, so the
    frame budget it implies grows with the rate.
    """
    seconds = max_output_seconds(extra, task)
    requested_max = int(seconds * fps)
    aligned_max = ((requested_max - 5 + 16) // 17) * 17 + 5
    if num_frames > aligned_max:
        raise OmniClientError(
            f"MiniMax H3 output duration exceeds the {seconds:g}-second request limit "
            f"({aligned_max} frames after alignment); got {num_frames} frames"
        )
