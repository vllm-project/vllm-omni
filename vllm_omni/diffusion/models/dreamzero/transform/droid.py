# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DROID dataset transform.

DROID uses 1-indexed exterior cameras, 3 views total (OXE_DROID embodiment).
Stitching layout (same as RoboArena — both are OXE_DROID):
    ┌─────────────────────────┐
    │   wrist (2x width)      │  ← pixel-repeat along width
    ├────────────┬────────────┤
    │  left ext  │ right ext  │
    └────────────┴────────────┘
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2 as T

from vllm_omni.diffusion.models.dreamzero.transform.base import (
    RobotPolicyTransform,
    register_transform,
)


class DroidTransform(RobotPolicyTransform):
    """Transform for DROID dataset (OXE_DROID embodiment).

    DROID observation keys (1-indexed exterior cameras):
        observation/exterior_image_1_left  → left exterior
        observation/exterior_image_2_left  → right exterior
        observation/wrist_image_left       → wrist
    """

    IMAGE_KEY_MAP = {
        "observation/exterior_image_1_left": "images/exterior_0",
        "observation/exterior_image_2_left": "images/exterior_1",
        "observation/wrist_image_left": "images/wrist",
    }
    EMBODIMENT_NAME = "oxe_droid"
    ACTION_DIM = 8  # 7 joint + 1 gripper
    _VIDEO_CROP_SCALE = 0.95
    _VIDEO_RESIZE_HW = (176, 320)

    @classmethod
    def _preprocess_view(cls, arr: np.ndarray) -> np.ndarray:
        """Apply per-view crop and resize before stitching."""
        frames = torch.from_numpy(arr).to(torch.float32).permute(0, 3, 1, 2) / 255.0
        crop_h = int(arr.shape[1] * cls._VIDEO_CROP_SCALE)
        crop_w = int(arr.shape[2] * cls._VIDEO_CROP_SCALE)
        frames = T.CenterCrop((crop_h, crop_w))(frames)
        frames = T.Resize(
            cls._VIDEO_RESIZE_HW,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )(frames)
        return (frames.permute(0, 2, 3, 1) * 255.0).to(torch.uint8).cpu().numpy()

    def _stitch_views(self, images: dict[str, np.ndarray]) -> np.ndarray:
        """OXE_DROID 2x2 stitching: wrist top (2x wide), exteriors bottom."""
        left_ext = images.get("images/exterior_0")
        right_ext = images.get("images/exterior_1")
        wrist = images.get("images/wrist")

        # Ensure 4D: (T, H, W, C)
        def ensure_4d(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return arr if arr.ndim == 4 else arr[np.newaxis]

        left_ext = ensure_4d(left_ext)
        right_ext = ensure_4d(right_ext)
        wrist = ensure_4d(wrist)

        # Determine shape from first available view.
        ref = next((v for v in [wrist, left_ext, right_ext] if v is not None), None)
        if ref is None:
            return np.zeros((1, 352, 640, 3), dtype=np.uint8)

        # Apply per-view crop + resize before stitching.
        def maybe_preprocess(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return self._preprocess_view(arr)

        left_ext = maybe_preprocess(left_ext)
        right_ext = maybe_preprocess(right_ext)
        wrist = maybe_preprocess(wrist)
        ref = next((v for v in [wrist, left_ext, right_ext] if v is not None), None)
        if ref is None:
            raise RuntimeError("Expected at least one DROID camera view after preprocessing.")
        t, h, w, c = ref.shape

        out = np.zeros((t, 2 * h, 2 * w, c), dtype=ref.dtype)  # (T, 2H, 2W, C)

        # Top row: wrist repeated 2x along width.
        if wrist is not None:
            wrist_wide = np.repeat(wrist, 2, axis=2)  # (T, H, 2W, C)
            out[:, :h, :] = wrist_wide

        # Bottom row: left exterior | right exterior.
        if left_ext is not None:
            out[:, h:, :w] = left_ext
        if right_ext is not None:
            out[:, h:, w:] = right_ext

        return out

    def _language_template(self, prompt: str) -> str:
        """Expand the language prompt for the OXE_DROID multi-view format."""
        prompt = (prompt or "Perform the default behavior.").strip()
        prompt_lower = prompt.lower()
        return (
            "A multi-view video shows that a robot "
            + prompt_lower
            + " The video is split into three views: The top view shows the "
            + "camera view from the robot's wrist, the bottom-left view shows "
            + "the camera view from the left exterior camera, and the "
            + "bottom-right view shows the camera view from the right exterior "
            + "camera. During training, one of the two bottom exterior views "
            + "may be a black screen (dropped view). The robot "
            + prompt_lower
        )

    def _extract_raw_state(self, obs: dict) -> np.ndarray:
        """OXE_DROID state: 7 joint + 1 gripper = 8 dims."""
        parts = []
        if "observation/joint_position" in obs:
            parts.append(np.asarray(obs["observation/joint_position"], dtype=np.float64).flatten())
        if "observation/gripper_position" in obs:
            parts.append(np.asarray(obs["observation/gripper_position"], dtype=np.float64).flatten())
        if parts:
            return np.concatenate(parts)
        return np.zeros(8, dtype=np.float64)


register_transform("droid", DroidTransform())
