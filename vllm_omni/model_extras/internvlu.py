# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from PIL import Image


def _reject_negative_prompt(negative_prompt: str | None) -> None:
    if negative_prompt is not None and negative_prompt.strip():
        raise ValueError(
            "InternVL-U uses fixed partial and unconditional CFG prompts; negative prompts are not supported"
        )


def _processor_kwargs(
    modality: str,
    height: int | None,
    width: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"modalities": [modality]}
    if height is not None:
        kwargs["target_h"] = height
    if width is not None:
        kwargs["target_w"] = width
    return kwargs


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    _reject_negative_prompt(negative_prompt)
    result: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
        "mm_processor_kwargs": _processor_kwargs("image", height, width),
    }
    return result


def build_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    _reject_negative_prompt(negative_prompt)
    result: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
        "multi_modal_data": {"image": input_image},
        "mm_processor_kwargs": _processor_kwargs("img2img", height, width),
    }
    return result


INTERNVLU_EXTRA_BODY_PARAMS = frozenset(
    {
        "all_cfg_scale",
        "flow_shift",
        "part_cfg_scale",
        "timestep_trunc",
    }
)
INTERNVLU_EXTRA_OUTPUT_PARAMS = frozenset()
