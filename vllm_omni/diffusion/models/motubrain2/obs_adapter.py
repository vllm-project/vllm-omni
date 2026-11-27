# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert robot observations into native Motubrain policy inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def camera_source(view: dict) -> str:
    """Return the observation key that supplies one profile camera."""
    return str(view.get("source", view["name"]))


def camera_sources(views: list[dict]) -> list[str]:
    """Return camera observation keys in profile order, including duplicates."""
    return [camera_source(view) for view in views]


def expected_rgb_frames(observed_frames: int, temporal_stride: int) -> int:
    """Return the RGB frame count for one observed Wan latent window."""
    if type(observed_frames) is not int or observed_frames < 1:
        raise ValueError("observed_frames must be a positive integer")
    if type(temporal_stride) is not int or temporal_stride < 1:
        raise ValueError("temporal_stride must be a positive integer")
    return 1 + temporal_stride * (observed_frames - 1)


def _frame_to_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "convert"):
        value = np.asarray(value.convert("RGB"))
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("camera frames must be HWC or CHW arrays")
    if array.shape[-1] == 3:
        hwc = array
    elif array.shape[0] == 3:
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError("camera frames need three RGB channels")
    if min(hwc.shape) < 1:
        raise ValueError("camera frames must contain pixels")
    if hwc.dtype == np.uint8:
        return np.ascontiguousarray(hwc)
    if np.issubdtype(hwc.dtype, np.floating):
        if not np.isfinite(hwc).all():
            raise ValueError("camera frames must be finite")
        if hwc.size and float(hwc.max()) <= 1.0:
            hwc = hwc * 255.0
    hwc = np.clip(np.rint(hwc.astype(np.float64)), 0, 255)
    return np.ascontiguousarray(hwc.astype(np.uint8))


def camera_clip(value: Any, key: str) -> torch.Tensor:
    """Return one camera observation as a uint8 ``[T, H, W, 3]`` tensor."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    if isinstance(value, np.ndarray) and value.ndim == 4:
        value = [value[index] for index in range(value.shape[0])]
    elif isinstance(value, torch.Tensor) and value.ndim == 4:
        value = [value[index] for index in range(value.shape[0])]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"camera {key!r} has no frames")
        frames = [_frame_to_hwc_uint8(frame) for frame in value]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f"camera {key!r} frames must share one shape")
        clip = np.stack(frames)
    else:
        clip = _frame_to_hwc_uint8(value)[None]
    return torch.from_numpy(clip)


def _lookup(robot_obs: dict, view: dict, kind: str) -> Any:
    keys = (camera_source(view), str(view["name"]))
    for key in keys:
        value = robot_obs.get(key)
        if value is not None:
            return value
    raise ValueError(f"missing {kind} camera {keys[0]!r}; accepted keys: {', '.join(dict.fromkeys(keys))}")


def _camera_dict(
    robot_obs: dict,
    views: list[dict],
    *,
    expected_frames: int,
    kind: str,
) -> dict[str, torch.Tensor]:
    clips = {}
    for view in views:
        source = camera_source(view)
        clip = camera_clip(_lookup(robot_obs, view, kind), source)
        if clip.shape[0] != expected_frames:
            raise ValueError(f"{kind} camera {source!r} must carry {expected_frames} RGB frames, got {clip.shape[0]}")
        clips[source] = clip
    return clips


def _float_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def build_observation(
    robot_obs: dict,
    views: list[dict],
    *,
    wire_dim: int | None = None,
    observed_frames: int = 1,
    temporal_stride: int = 1,
    history_frames: int | None = None,
) -> dict:
    """Build the observation expected by ``RawPolicy.predict``.

    Robot clients send cameras as HWC uint8 arrays (or lists of frames). The native
    policy expects ``[T, H, W, 3]`` uint8 clips keyed by the profile camera
    source. Optional ``state`` and ``camera_extrinsic`` values pass through.
    """
    if not isinstance(robot_obs, dict):
        raise TypeError("robot observation must be a mapping")

    instruction = robot_obs.get("prompt")
    if instruction is None:
        instruction = robot_obs.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("robot observation needs a nonempty 'prompt' or 'instruction'")

    cameras = robot_obs.get("cameras", robot_obs)
    if not isinstance(cameras, dict):
        raise ValueError("cameras must map camera names to RGB frames")
    observation: dict[str, Any] = {
        "instruction": instruction,
        "cameras": _camera_dict(
            cameras,
            views,
            expected_frames=expected_rgb_frames(observed_frames, temporal_stride),
            kind="current",
        ),
    }

    state = robot_obs.get("state")
    if state is not None:
        state = _float_tensor(state, "state").reshape(-1)
        if wire_dim is not None and state.numel() != wire_dim:
            raise ValueError(f"state must have {wire_dim} controller values, got {state.numel()}")
        observation["state"] = state

    extrinsic = robot_obs.get("camera_extrinsic")
    if extrinsic is not None:
        extrinsic = _float_tensor(extrinsic, "camera_extrinsic")
        if extrinsic.shape != (4, 4):
            raise ValueError("camera_extrinsic must be a 4x4 base-from-camera matrix")
        observation["camera_extrinsic"] = extrinsic

    if history_frames is not None:
        past = robot_obs.get("history_cameras")
        if not isinstance(past, dict):
            raise ValueError(
                "this policy uses sparse history; provide 'history_cameras' "
                "with the same camera keys as the current observation"
            )
        observation["history_cameras"] = _camera_dict(past, views, expected_frames=history_frames, kind="history")
    elif robot_obs.get("history_cameras") is not None:
        raise ValueError("history_cameras was supplied but the profile does not use sparse history")

    return observation
