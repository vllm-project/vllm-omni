# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base transform interface for DreamZero robot policy serving.

Transforms handle dataset-specific concerns ONLY:
  - Observation key mapping
  - Multi-view stitching (embodiment-specific layout)
  - Language template wrapping (embodiment-specific)
  - Raw state extraction (dataset-specific keys)
  - Output action slicing (to actual action_dim)

Model-specific concerns belong in the pipeline:
  - Tokenization (pipeline owns tokenizer)
  - State padding (pipeline knows MAX_STATE_DIM)
  - Negative prompt (pipeline owns the string)
  - Noise generation, encoding, decoding

Flow:
  raw obs (dataset format)
    → DreamZeroPipeline selects transform by embodiment
      → unified dict (stitched video, templated prompt str, raw state)
        → tokenize, pad, encode, denoise
          → transform_action_output()
            → ndarray (N, action_dim)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RobotPolicyTransform:
    """Base class for dataset-specific observation transforms.

    Subclasses MUST define:
        IMAGE_KEY_MAP: dict — dataset obs keys → unified keys
        EMBODIMENT_NAME: str — embodiment identity (pipeline maps to numeric ID)
        ACTION_DIM: int — actual action dimensions (for output slicing)

    Subclasses MUST override:
        _stitch_views()      — multi-view → single stitched image
        _language_template() — prompt → embodiment-aware template
        _extract_raw_state() — obs → raw state ndarray
    """

    IMAGE_KEY_MAP: dict[str, str]
    EMBODIMENT_NAME: str
    ACTION_DIM: int

    def transform_input(self, obs: dict) -> dict:
        """Dataset-specific transform: key map → stitch → template → state."""
        # 1. Map image keys → unified keys
        images: dict[str, np.ndarray] = {}
        for src_key, dst_key in self.IMAGE_KEY_MAP.items():
            if src_key in obs:
                images[dst_key] = np.asarray(obs[src_key])

        # 2. Multi-view stitching
        stitched = self._stitch_views(images)

        # 3. Language template (string only, pipeline tokenizes)
        prompt = obs.get("prompt", "")
        templated_prompt = self._language_template(prompt)

        # 4. Raw state extraction (pipeline pads)
        raw_state = self._extract_raw_state(obs)

        # 5. Build unified output
        unified: dict[str, Any] = {
            "images": stitched,  # ndarray (T, H_out, W_out, 3)
            "prompt": templated_prompt,  # str (templated, not tokenized)
            "state": raw_state,  # ndarray (state_dim,) — pipeline pads
            "embodiment_name": self.EMBODIMENT_NAME,
        }
        if "session_id" in obs:
            unified["session_id"] = obs["session_id"]
        return unified

    def transform_action_output(self, actions: Any) -> np.ndarray:
        """Adapt model action output to this transform's action dimensions."""
        actions = np.asarray(actions, dtype=np.float32)
        # Handle any remaining batch dims: squeeze to 2D (horizon, dim)
        while actions.ndim > 2:
            actions = actions[0]
        # Slice padded dim to actual ACTION_DIM
        if actions.ndim == 2 and actions.shape[-1] > self.ACTION_DIM:
            actions = actions[:, : self.ACTION_DIM]
        return actions

    # ------------------------------------------------------------------
    # Subclass MUST override
    # ------------------------------------------------------------------

    def _stitch_views(self, images: dict[str, np.ndarray]) -> np.ndarray:
        """Stitch camera views into single image.
        Input: unified key → ndarray (H,W,3) or (T,H,W,3).
        Output: ndarray (T, H_out, W_out, 3).
        """
        raise NotImplementedError

    def _language_template(self, prompt: str) -> str:
        """Wrap prompt in embodiment-specific template string."""
        raise NotImplementedError

    def _extract_raw_state(self, obs: dict) -> np.ndarray:
        """Extract raw state vector from obs.
        Returns: ndarray (state_dim,) float64. Pipeline handles padding.
        """
        raise NotImplementedError


# Transform registry — keyed by embodiment/dataset name
TRANSFORMS: dict[str, RobotPolicyTransform] = {}


def register_transform(name: str, transform: RobotPolicyTransform) -> None:
    TRANSFORMS[name] = transform


def get_transform(name: str) -> RobotPolicyTransform:
    if name not in TRANSFORMS:
        raise KeyError(f"Unknown transform '{name}'. Available: {list(TRANSFORMS.keys())}")
    return TRANSFORMS[name]
