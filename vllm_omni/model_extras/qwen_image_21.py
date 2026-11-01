# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

from PIL import Image


def build_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    img_prompt: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": {"image": input_image},
    }
    if height is not None:
        img_prompt["height"] = height
    if width is not None:
        img_prompt["width"] = width
    if negative_prompt is not None:
        img_prompt["negative_prompt"] = negative_prompt
    return img_prompt
