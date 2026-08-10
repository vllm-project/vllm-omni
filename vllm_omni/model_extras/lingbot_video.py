# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

LINGBOT_VIDEO_EXTRA_BODY_PARAMS = frozenset(
    {
        "batch_cfg",
        "duration",
        "flow_shift",
        "negative_prompt",
        "null_cond_clone_zero",
        "offload_vae_during_denoise",
        "output_type",
        "refiner_sigma_tail_steps",
        "resolution",
        "ratio",
        "shift",
        "t_thresh",
    }
)


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    del height, width
    result: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
    }
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result


def build_image_to_video_prompt(
    prompt: str,
    negative_prompt: str | None,
    media_inputs: Mapping[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    del height, width
    if set(media_inputs) != {"image"} or not isinstance(media_inputs["image"], Image.Image):
        raise ValueError("LingBot text-image-to-video requires exactly one PIL image input.")

    result: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["video"],
        "multi_modal_data": {"image": media_inputs["image"]},
    }
    if num_frames is not None:
        result["num_frames"] = num_frames
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result
