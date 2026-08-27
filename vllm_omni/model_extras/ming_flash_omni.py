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
        "prompt": prompt,
        "modalities": ["image"],
        "mm_processor_kwargs": {"modalities": ["image"]},
    }
    if height is not None:
        text_prompt["mm_processor_kwargs"]["target_h"] = height
    if width is not None:
        text_prompt["mm_processor_kwargs"]["target_w"] = width
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
        "prompt": prompt,
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


# For Image-generation (diffusion-stage)
MING_FLASH_OMNI_EXTRA_BODY_PARAMS = frozenset(
    {
        "height",
        "width",
        "steps",
        "cfg",
        "seed",
        "byte5_text",
        "negative_prompt",
    }
)
MING_FLASH_OMNI_EXTRA_OUTPUT_PARAMS = frozenset()
MING_FLASH_OMNI_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES = True
