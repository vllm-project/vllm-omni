# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero AR client frame schedule aligned with upstream ``test_client_AR.py``."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Frame schedule constants (matching upstream debug_inference.py / test_client_AR.py).
ACTION_HORIZON = 24
DEFAULT_ACTION_DIM = 8
RELATIVE_OFFSETS = [-23, -16, -8, 0]
DEFAULT_NUM_AR_CHUNKS = 15

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}


def build_frame_schedule(
    total_frames: int,
    num_chunks: int,
    *,
    logger: logging.Logger | None = None,
) -> list[list[int]]:
    """Build the frame index schedule for multi-frame chunks.

    Args:
        total_frames: Number of frames available in each camera video.
        num_chunks: Number of 4-frame chunks to schedule after the initial frame.

    Returns:
        A list of frame-index lists. Each inner list has four indices. The
        returned list may be shorter than ``num_chunks`` when the videos run
        out of frames (upstream stops early instead of erroring).
    """
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + offset, 0) for offset in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            if logger is not None:
                logger.info(
                    "Frame %s >= %s, stopping at %s chunks",
                    indices[-1],
                    total_frames,
                    len(chunks),
                )
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    *,
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    """Build an observation dict from real video frames.

    For one frame each image key is ``(H, W, 3)``. For four frames each key is
    ``(4, H, W, 3)``.
    """
    obs: dict[str, Any] = {}
    for camera_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[camera_key] = selected[0] if len(frame_indices) == 1 else selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def build_ar_observations(
    camera_frames: dict[str, np.ndarray],
    *,
    prompt: str,
    session_id: str,
    num_chunks: int = DEFAULT_NUM_AR_CHUNKS,
    repeat_chunk_observations: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Build the AR observation sequence used by upstream ``test_client_AR.py``.

    Step 0 sends frame ``[0]``. Each subsequent step sends one 4-frame chunk.

    ``num_chunks`` counts only the 4-frame chunks after the initial frame, not
    the total number of inferences.
    """
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")

    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    observations = [
        make_obs_from_video(
            camera_frames,
            [0],
            prompt=prompt,
            session_id=session_id,
        )
    ]

    chunk_schedule = build_frame_schedule(total_frames, num_chunks, logger=logger)
    if repeat_chunk_observations and chunk_schedule and len(chunk_schedule) < num_chunks:
        while len(chunk_schedule) < num_chunks:
            chunk_schedule.append(chunk_schedule[-1])

    for frame_indices in chunk_schedule:
        observations.append(
            make_obs_from_video(
                camera_frames,
                frame_indices,
                prompt=prompt,
                session_id=session_id,
            )
        )
    return observations
