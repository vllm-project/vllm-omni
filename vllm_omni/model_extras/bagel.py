# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from PIL import Image


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    text_prompt: dict[str, Any] = {
        "prompt": f"<|im_start|>{prompt}<|im_end|>",
        "modalities": ["image"],
        "mm_processor_kwargs": {
            "target_h": height,
            "target_w": width,
            "modalities": ["image"],
        },
    }
    if negative_prompt is not None:
        text_prompt["negative_prompt"] = negative_prompt
    return text_prompt


def build_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    img_prompt: dict[str, Any] = {
        "prompt": f"<|fim_middle|><|im_start|>{prompt}<|im_end|>",
        "modalities": ["img2img"],
        "multi_modal_data": {"img2img": input_image},
        "mm_processor_kwargs": {"modalities": ["img2img"]},
    }
    if height is not None:
        img_prompt["mm_processor_kwargs"]["target_h"] = height
    if width is not None:
        img_prompt["mm_processor_kwargs"]["target_w"] = width
    if negative_prompt is not None:
        img_prompt["negative_prompt"] = negative_prompt
    return img_prompt


BAGEL_EXTRA_BODY_PARAMS = frozenset(
    {
        "cfg_text_scale",
        "cfg_img_scale",
        "cfg_interval",
        "cfg_renorm_type",
        "cfg_renorm_min",
        "negative_prompt",
        "think",
        "max_think_tokens",
        "do_sample",
        "text_temperature",
        "timestep_shift",
    }
)
BAGEL_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "text_output",
        "think_text",
    }
)
BAGEL_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES = True
