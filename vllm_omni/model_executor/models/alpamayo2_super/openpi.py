# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reasoning-stage prompt for Alpamayo 2 Super OpenPI observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    CAMERA_DISPLAY_NAMES,
    build_alpamayo2_super_tokenizer,
    build_trajectory_prompt,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def _squeeze_leading(array: np.ndarray, ndim: int) -> np.ndarray:
    while array.ndim > ndim and array.shape[0] == 1:
        array = array[0]
    return array


class Alpamayo2SuperOpenPIPrompt:
    """Build the reasoning stage's prompt from an OpenPI observation.

    Observation keys: ``image_frames`` ``(cameras, frames, 3, H, W)`` uint8,
    strictly increasing ``camera_indices``, ``ego_history_xyz`` ``(16, 3)`` and
    ``ego_history_rot`` ``(16, 3, 3)``. Frames per camera and history length
    come from the deploy config's ``policy_server_config``. The expert stage
    reads the ego history from the same observation.
    """

    def __init__(self, model_config: ModelConfig, policy_server_config: Mapping[str, Any]) -> None:
        self.config = model_config.hf_config
        self.frames_per_camera = int(policy_server_config["frames_per_camera"])
        self.history_steps = int(policy_server_config["history_steps"])
        self.tokenizer = build_alpamayo2_super_tokenizer(
            model_config.tokenizer,
            self.config.history_vocab_size,
            self.config.future_vocab_size,
        )

    def build_prompt(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        frames_array = np.asarray(obs["image_frames"])
        if (
            frames_array.dtype != np.uint8
            or frames_array.ndim != 5
            or frames_array.shape[1] != self.frames_per_camera
            or frames_array.shape[2] != 3
        ):
            raise ValueError(
                f"image_frames must be uint8 (cameras, {self.frames_per_camera}, 3, H, W), "
                f"got {frames_array.dtype} {frames_array.shape}"
            )
        camera_indices = [int(index) for index in np.asarray(obs["camera_indices"]).reshape(-1)]
        if (
            len(camera_indices) != frames_array.shape[0]
            or any(index not in CAMERA_DISPLAY_NAMES for index in camera_indices)
            or any(a >= b for a, b in zip(camera_indices, camera_indices[1:]))
        ):
            raise ValueError(
                f"camera_indices must be one strictly increasing index per camera from "
                f"{sorted(CAMERA_DISPLAY_NAMES)}, got {camera_indices}"
            )
        history_xyz = _squeeze_leading(np.asarray(obs["ego_history_xyz"], dtype=np.float32), 2)
        history_rot = _squeeze_leading(np.asarray(obs["ego_history_rot"], dtype=np.float32), 3)
        if history_xyz.shape != (self.history_steps, 3) or history_rot.shape != (self.history_steps, 3, 3):
            raise ValueError(
                f"ego_history_xyz and ego_history_rot must be ({self.history_steps}, 3) and "
                f"({self.history_steps}, 3, 3), got {history_xyz.shape} and {history_rot.shape}"
            )
        frames = torch.from_numpy(frames_array)
        prompt_ids = build_trajectory_prompt(
            self.tokenizer,
            self.config,
            frames=frames,
            camera_indices=camera_indices,
            ego_history_xyz=torch.from_numpy(history_xyz),
        )
        return {"prompt_token_ids": prompt_ids, "multi_modal_data": {"image": list(frames.flatten(0, 1))}}
