# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Flatten the released Alpamayo 2 Super wrapper config for native Qwen3-VL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import Qwen3VLConfig


class Alpamayo2SuperConfig(Qwen3VLConfig):
    """Expose Super's nested ``vlm_config`` to vLLM's Qwen3-VL runner.

    The released checkpoint wraps Qwen3-VL and stores the action expert beside
    it. vLLM needs the Qwen fields at the top level for multimodal sizing and KV
    allocation, while the original wrapper fields remain available to the
    model-owned expert hook.
    """

    model_type = "alpamayo2_super"

    def __init__(
        self,
        *,
        vlm_config: Mapping[str, Any] | None = None,
        expert_config: Mapping[str, Any] | None = None,
        traj_ids: Mapping[str, int] | None = None,
        history_vocab_size: int = 1000,
        future_vocab_size: int = 3000,
        tokens_per_history_traj: int = 45,
        include_camera_ids: bool = True,
        frame_label: str = "frame_num",
        min_pixels: int | None = 163840,
        max_pixels: int | None = 196608,
        **kwargs: Any,
    ) -> None:
        nested = dict(vlm_config or {})
        text_config = nested.pop("text_config", kwargs.pop("text_config", None))
        vision_config = nested.pop("vision_config", kwargs.pop("vision_config", None))
        for key in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            if key in nested:
                kwargs.setdefault(key, nested[key])
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs,
        )
        self.vlm_config = dict(vlm_config or {})
        self.expert_config = dict(expert_config or {})
        self.traj_ids = dict(traj_ids or {})
        self.history_vocab_size = history_vocab_size
        self.future_vocab_size = future_vocab_size
        self.traj_vocab_size = history_vocab_size + future_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.include_camera_ids = include_camera_ids
        self.frame_label = frame_label
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
