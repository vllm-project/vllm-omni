# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observation and action processing for FastWAM-family policies.

Mirrors the reference LIBERO evaluator (``eval_libero_single.py``) around the model:
each camera is resized (PIL bilinear, scale to cover, centre crop) and the cameras
are concatenated left to right; the 8-D proprioception is min/max normalised to
[-1, 1] and clamped to [-5, 5]; actions are de-normalised with the same statistics
and, for LIBERO, the gripper is mapped from the dataset convention (0 close, 1
open) to the simulator's (+1 close, -1 open) and binarised.

Observation keys follow the OpenPI LIBERO client: ``observation/image`` (third
person), ``observation/wrist_image``, ``observation/state`` and ``prompt``, plus
``task_suite``, which selects the per-suite statistics and action-codec conditioning
of CAE-FastWAM. CAE-FastWAM needs it on every request unless the deploy config pins
one suite (the stock OpenPI LIBERO client does not send it); checkpoints with pooled
statistics ignore it. Images are HWC uint8 and must already be upright
(the OpenPI LIBERO client rotates LIBERO's frames by 180 degrees before sending).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

IMAGE_KEYS = ("observation/image", "observation/wrist_image")
STATE_KEY = "observation/state"
PROMPT_KEY = "prompt"
SUITE_KEY = "task_suite"
ACTION_POSTPROCESS_MODES = ("libero", "none")


@dataclass(frozen=True)
class FastWAMObservation:
    images: tuple[np.ndarray, ...]
    state: np.ndarray
    task: str
    # None when neither the request nor the deploy config names a suite.
    suite: str | None


def parse_observation(
    robot_obs: Mapping[str, Any], *, proprio_dim: int, num_cameras: int, default_suite: str | None
) -> FastWAMObservation:
    """Validate an OpenPI observation dict and pull out the FastWAM inputs."""
    image_keys = IMAGE_KEYS[:num_cameras]
    missing = [key for key in (*image_keys, STATE_KEY, PROMPT_KEY) if key not in robot_obs]
    if missing:
        raise ValueError(
            f"robot_obs is missing {missing}; FastWAM expects keys {[*image_keys, STATE_KEY, PROMPT_KEY]}."
        )
    images = []
    for key in image_keys:
        image = np.asarray(robot_obs[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"robot_obs[{key!r}] must be an HxWx3 uint8 image, got {image.dtype} {image.shape}.")
        images.append(image)
    state = np.asarray(robot_obs[STATE_KEY], dtype=np.float32).reshape(-1)
    if state.shape != (proprio_dim,):
        raise ValueError(f"robot_obs[{STATE_KEY!r}] must have {proprio_dim} values, got {state.shape[0]}.")
    task = robot_obs[PROMPT_KEY]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("robot_obs['prompt'] must be a non-empty task instruction string.")
    suite = robot_obs.get(SUITE_KEY) or default_suite
    if suite is not None and not isinstance(suite, str):
        raise ValueError(f"robot_obs[{SUITE_KEY!r}] must be a string, got {type(suite).__name__}.")
    return FastWAMObservation(images=tuple(images), state=state, task=task, suite=suite)


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover ``width`` x ``height`` (PIL bilinear) and centre-crop, as the reference does."""
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.Resampling.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def build_image(images: Sequence[np.ndarray], camera_size: tuple[int, int]) -> torch.Tensor:
    """Resize each camera and concatenate horizontally -> uint8 tensor [1, 3, H, N*W]."""
    height, width = camera_size
    rgb = np.concatenate([center_crop_resize(image, width=width, height=height) for image in images], axis=1)
    return torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)


class MinMaxNormalizer:
    """FastWAM ``SingleFieldLinearNormalizer`` in ``min/max`` mode (float32, CPU)."""

    range_tol = 1e-4

    def __init__(self, global_min: Sequence[float], global_max: Sequence[float]) -> None:
        input_min = torch.tensor(global_min, dtype=torch.float32)
        input_max = torch.tensor(global_max, dtype=torch.float32)
        if input_min.shape != input_max.shape or input_min.dim() != 1:
            raise ValueError("global_min/global_max must be 1-D vectors of equal length.")
        input_range = input_max - input_min
        ignore_dim = input_range < self.range_tol
        input_range[ignore_dim] = 2.0
        self.scale = 2.0 / input_range
        self.offset = -1.0 - self.scale * input_min
        # Constant dimensions are only centred.
        self.offset[ignore_dim] = 0.0 - input_min[ignore_dim]

    @property
    def dim(self) -> int:
        return int(self.scale.shape[0])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.scale + self.offset, -5.0, 5.0)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale


@dataclass(frozen=True)
class FastWAMStats:
    state: MinMaxNormalizer
    action: MinMaxNormalizer


def _single_field(stats: Mapping[str, Any], field: str, path: str) -> Mapping[str, Any]:
    entries = stats.get(field)
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError(f"{path}: missing {field!r} statistics.")
    if len(entries) != 1:
        raise ValueError(f"{path}: expected one merged {field!r} key, got {sorted(entries)}.")
    return next(iter(entries.values()))


def load_stats(path: str, *, action_dim: int, proprio_dim: int) -> FastWAMStats:
    """Read a FastWAM ``*_dataset_stats.json`` / ``stats/<suite>.json`` file (global min/max)."""
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    state = _single_field(stats, "state", path)
    action = _single_field(stats, "action", path)
    result = FastWAMStats(
        state=MinMaxNormalizer(state["global_min"], state["global_max"]),
        action=MinMaxNormalizer(action["global_min"], action["global_max"]),
    )
    if result.state.dim != proprio_dim or result.action.dim != action_dim:
        raise ValueError(
            f"{path}: state/action stats have {result.state.dim}/{result.action.dim} dims, "
            f"expected {proprio_dim}/{action_dim}."
        )
    return result


def postprocess_actions(actions: np.ndarray, mode: str) -> np.ndarray:
    """Apply the benchmark's action convention to de-normalised actions [T, D]."""
    actions = np.array(actions, dtype=np.float32, copy=True)
    if mode == "libero":
        # Dataset gripper is 0 (close) .. 1 (open); LIBERO wants -1 open, +1 close.
        actions[..., -1] = actions[..., -1] * 2 - 1
        actions[..., -1] = actions[..., -1] * -1.0
        actions[..., -1] = np.sign(actions[..., -1])
    elif mode != "none":
        raise ValueError(f"Unknown action_postprocess {mode!r}; choose one of {ACTION_POSTPROCESS_MODES}.")
    return actions
