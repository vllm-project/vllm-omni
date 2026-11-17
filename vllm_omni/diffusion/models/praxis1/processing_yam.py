# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YAM geometry on the shared Praxis observation-history state machine."""

import numpy as np
import torch

from . import processing as common

SIDES = ("left", "right")
VIEW_KEYS = {
    "top_left": ("image.exterior", "image.top", "image.top_left"),
    "wrist_left": ("image.wrist_left", "image.left_wrist"),
    "wrist_right": ("image.wrist_right", "image.right_wrist"),
}
RIG_FRAME = common.RigFrame(0.0, common._rows(common.NO_TURN))


class StackMismatchError(ValueError):
    """An arm's pose and grip histories have different lengths."""


def pose_rows(obs: dict, side: str, broadcast: bool = True) -> np.ndarray:
    ee = np.asarray(common._pick(obs, (f"robot_state.{side}.ee_pose", f"ee_pose.{side}")), np.float32)
    grip = np.asarray(common._pick(obs, (f"grip.{side}", f"robot_state.{side}.grip", f"gripper.{side}")), np.float32)
    ee = ee.reshape(-1, ee.shape[-1]) if ee.ndim >= 2 else ee.reshape(1, -1)
    if ee.shape[1] < 7:
        raise ValueError(f"{side} ee_pose must have at least 7 values")
    grip = grip.reshape(grip.shape[0], -1) if grip.ndim >= 2 else grip.reshape(-1, 1)
    if len(ee) != len(grip):
        if not broadcast or (len(ee) != 1 and len(grip) != 1):
            raise StackMismatchError(f"{side} pose and grip stack lengths differ")
        n = max(len(ee), len(grip))
        ee, grip = np.broadcast_to(ee, (n, ee.shape[1])), np.broadcast_to(grip, (n, grip.shape[1]))
    rows = np.concatenate([ee[:, :7], grip[:, :1]], axis=1).astype(np.float32)
    if not np.isfinite(rows).all() or np.any(np.abs(np.linalg.norm(rows[:, 3:7], axis=1) - 1) > 1e-2):
        raise ValueError(f"{side} pose/grip must be finite with unit wxyz quaternions")
    return rows


def view_stacks(obs: dict) -> dict[str, np.ndarray]:
    views = {}
    for view, keys in VIEW_KEYS.items():
        image = np.asarray(common._pick(obs, keys))
        if image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{view} must be a uint8 (T,H,W,3) image stack")
        views[view] = image
    return views


def canvas_clip(views: dict[str, np.ndarray], device: torch.device | str = "cpu") -> torch.Tensor:
    stacks = [views[v] for v in VIEW_KEYS]
    if len({len(s) for s in stacks}) != 1:
        raise ValueError("YAM cameras must have equal frame counts")
    frames = [torch.from_numpy(np.ascontiguousarray(s)).permute(0, 3, 1, 2) for s in stacks]
    native = [tuple(f.shape[-2:]) for f in frames]
    fitted = [(min(h, 416, h * 416 // w), min(w, 416, w * 416 // h)) for h, w in native]
    common_h = min(h for h, _ in fitted)
    sizes = [(common_h, w * common_h // h) if rh > common_h else (rh, rw) for (h, w), (rh, rw) in zip(native, fitted)]
    if any((h // 32 * 32, w // 32 * 32) != (256, 416) for h, w in sizes):
        raise ValueError("YAM requires native-aspect 16:10 views fitting 256x416 tiles")
    tiles = []
    for frame, (h, w) in zip(frames, sizes):
        frame = common._resize_view(frame, h, w)
        y, x = (h - 256) // 2, (w - 416) // 2
        tiles.append(frame[..., y : y + 256, x : x + 416])
    return torch.cat(tiles, dim=-1).float().permute(1, 0, 2, 3).unsqueeze(0).to(device)


class YamServeSession(common.ServeSession):
    """Two-arm rig poses; context, normalization and decoding stay shared."""

    def _current(self, obs: dict) -> tuple[np.ndarray, torch.Tensor]:
        if self.control_mode != "none":
            raise ValueError("YAM control_mode must be none")
        pose = np.concatenate([pose_rows(obs, side)[-1] for side in SIDES]).astype(np.float32)
        views = {v: s[-1:] for v, s in view_stacks(obs).items()}
        return pose, canvas_clip(views, self.device)

    def _executed(self, obs: dict, need: int) -> tuple[torch.Tensor, np.ndarray | None] | None:
        views = view_stacks(obs)
        if any(len(s) < need + 1 for s in views.values()):
            return None
        clip = canvas_clip({v: s[-(need + 1) : -1] for v, s in views.items()}, self.device)
        try:
            rows = [pose_rows(obs, side, broadcast=False) for side in SIDES]
        except (KeyError, StackMismatchError):
            return clip, None
        history = (
            None
            if any(len(r) < need + 1 for r in rows)
            else np.concatenate([r[-(need + 1) : -1] for r in rows], axis=1)
        )
        return clip, history

    def rel_to_wire(self, rel: np.ndarray, pose: np.ndarray) -> list[dict]:
        commands = [
            common.rel10_to_wire(
                rel[:, 10 * i : 10 * (i + 1)], pose[8 * i : 8 * (i + 1)], 1 / max(self.fps, 1e-6), frame=RIG_FRAME
            )
            for i in range(2)
        ]
        return [
            {
                **{
                    f"{key}.{side}": commands[i][t][key]
                    for i, side in enumerate(SIDES)
                    for key in ("robot_command", "target_grip")
                },
                "timestamp": row["timestamp"],
            }
            for t, row in enumerate(commands[0])
        ]
