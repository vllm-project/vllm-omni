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

Direct stitching source:
    `third_party/dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:337`
    to
    `third_party/dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:355`

Size assumptions for the current DreamZero path:
    `third_party/dreamzero/groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml:17`
    `third_party/dreamzero/scripts/train/droid_training_full_finetune.sh:82`
    `third_party/dreamzero/scripts/train/droid_training_full_finetune.sh:83`
    `third_party/dreamzero/scripts/train/droid_training_full_finetune.sh:86`
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2 as T

from vllm_omni.entrypoints.openai.realtime.robot.transform.base import (
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
        """Match source eval transform for OXE_DROID camera views.

        Source transform chain from `experiment_cfg/conf.yaml`:
        `VideoToTensor -> VideoCrop(scale=0.95, eval=center crop) ->
         VideoResize(height=176, width=320, interpolation=linear, antialias=True) ->
         VideoToNumpy`
        """
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
        """OXE_DROID 2x2 stitching: wrist top (2x wide), exteriors bottom.
        Direct layout correspondence:
            - output canvas `(t, 2H, 2W)` ↔ `dreamzero_cotrain.py:337`
            - wrist repeat-along-width ↔ `dreamzero_cotrain.py:339`-`342`
            - bottom left/right placement ↔ `dreamzero_cotrain.py:344`-`353`

        The resize-to-176x320 step is not done inside upstream
        `_prepare_video()`. Upstream expects the video path to already satisfy
        the model's spatial assumptions; for the current DreamZero config that
        assumption comes from:
            - `wan_flow_matching_action_tf.yaml:17` (`frame_seqlen: 880`)
            - `droid_training_full_finetune.sh:82`-`86`
        so we materialize that precondition here for online serving.
        """
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
        # Upstream `_prepare_video()` assumes views already share the same H/W
        # before it allocates `concat_images`; see `dreamzero_cotrain.py:337`.
        ref = next((v for v in [wrist, left_ext, right_ext] if v is not None), None)
        if ref is None:
            # No direct upstream line: this is a serving-side empty placeholder.
            # We choose 352x640 so the empty sample matches the active DreamZero
            # DROID path (per-view 176x320 -> stitched 352x640), consistent with
            # `droid_training_full_finetune.sh:82`-`86` and
            # `wan_flow_matching_action_tf.yaml:17`.
            return np.zeros((1, 352, 640, 3), dtype=np.uint8)

        # Match the source eval transform chain before `ConcatTransform` /
        # `DreamTransform`: center crop by 0.95, then resize each view to
        # 176x320. This is the actual preprocessing path used by
        # `GrootSimPolicy.eval_transform`, not just a serving-side heuristic.
        def maybe_preprocess(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return self._preprocess_view(arr)

        left_ext = maybe_preprocess(left_ext)
        right_ext = maybe_preprocess(right_ext)
        wrist = maybe_preprocess(wrist)
        ref = next((v for v in [wrist, left_ext, right_ext] if v is not None), None)
        assert ref is not None
        t, h, w, c = ref.shape

        # Match upstream canvas dtype exactly:
        # `concat_images = np.zeros(..., dtype=images.dtype)` at
        # `dreamzero_cotrain.py:337`.
        out = np.zeros((t, 2 * h, 2 * w, c), dtype=ref.dtype)  # (T, 2H, 2W, C)

        # Top row: wrist repeated 2x along width.
        # Corresponds to `dreamzero_cotrain.py:339`-`342`.
        if wrist is not None:
            wrist_wide = np.repeat(wrist, 2, axis=2)  # (T, H, 2W, C)
            out[:, :h, :] = wrist_wide

        # Bottom row: left exterior | right exterior.
        # Corresponds to `dreamzero_cotrain.py:344`-`353`.
        if left_ext is not None:
            out[:, h:, :w] = left_ext
        if right_ext is not None:
            out[:, h:, w:] = right_ext

        return out

    def _language_template(self, prompt: str) -> str:
        """Match the source OXE_DROID language prompt expansion exactly.

        Source correspondence:
        - `dreamzero_cotrain.py:collate()` OXE_DROID branch
        - `dreamzero_cotrain.py:HuggingfaceTokenizer(clean='whitespace')`

        Upstream online eval does *not* tokenize the raw instruction directly.
        After `DreamTransform.apply_single()` emits the raw language string,
        `collate()` rewrites it into the multi-view description below and only
        then tokenizes it. Using the raw prompt here changes the token ids and
        measurably changes the denoising trajectory.
        """
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
        """OXE_DROID state: 7 joint + 1 gripper = 8 dims.
        Source: dreamzero_cotrain.py _prepare_state() L436-467
        """
        parts = []
        if "observation/joint_position" in obs:
            parts.append(np.asarray(obs["observation/joint_position"], dtype=np.float64).flatten())
        if "observation/gripper_position" in obs:
            parts.append(np.asarray(obs["observation/gripper_position"], dtype=np.float64).flatten())
        if parts:
            return np.concatenate(parts)
        return np.zeros(8, dtype=np.float64)


register_transform("droid", DroidTransform())
