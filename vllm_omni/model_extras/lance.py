# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VIDEO_PAD = "<|video_pad|>"

_VISION_BLOCK = f"{VISION_START}{VIDEO_PAD}{VISION_END}"

_SYSTEM_PROMPTS: dict[tuple[str, str], str] = {
    ("t2i", "image"): (
        "Describe the image by detailing the color, quantity, text, shape, "
        "size, texture, spatial relationships of the objects and background:"
    ),
    ("i2v", "video"): (
        "Describe the video by detailing the color, quantity, visible text, "
        "shape, size, texture, spatial relationships and motion/camera "
        "movements of the objects and background:"
    ),
    ("image_edit", "image"): (
        "Describe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text "
        "instruction should alter or modify the image. Generate a new image "
        "that meets the user's requirements while maintaining consistency "
        "with the original input where appropriate."
    ),
    ("x2t_image", "image"): (
        "Generate a detailed and accurate description of the image, including all the key moments and visual details."
    ),
}

_VIDEO_TASKS = frozenset({"t2v", "i2v", "x2t_video", "video_edit"})


def _render_lance_prompt(
    task: str,
    user_text: str,
    *,
    vision_token: str | None = None,
    system_prompt: str | None = None,
) -> str:
    vision_type = "video" if task in _VIDEO_TASKS else "image"
    sys_prompt = system_prompt or _SYSTEM_PROMPTS.get((task, vision_type), "You are a helpful assistant.")
    user_msg = user_text if vision_token is None else f"{vision_token}{user_text}"
    return (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    )


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    del height, width
    text_prompt: dict[str, Any] = {
        "prompt": _render_lance_prompt("t2i", prompt),
        "modalities": ["image"],
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
    del height, width
    img_prompt: dict[str, Any] = {
        "prompt": _render_lance_prompt("image_edit", prompt, vision_token=_VISION_BLOCK),
        "modalities": ["image"],
        "multi_modal_data": {"img2img": input_image},
    }
    if negative_prompt is not None:
        img_prompt["negative_prompt"] = negative_prompt
    return img_prompt


def build_image_to_video_prompt(
    prompt: str,
    negative_prompt: str | None,
    media_inputs: Mapping[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    if set(media_inputs) != {"image"} or not isinstance(media_inputs.get("image"), Image.Image):
        raise ValueError("Lance image-to-video expects a single --image input (multi_modal_data == {'image': ...}).")
    video_prompt: dict[str, Any] = {
        "prompt": _render_lance_prompt("i2v", prompt, vision_token=_VISION_BLOCK),
        "modalities": ["video"],
        "multi_modal_data": {"first_frame": media_inputs["image"]},
    }
    extra_args = {
        key: value
        for key, value in (
            ("video_height", height),
            ("video_width", width),
            ("num_frames", num_frames),
        )
        if value is not None
    }
    if extra_args:
        video_prompt["extra_args"] = extra_args
    if negative_prompt:
        video_prompt["negative_prompt"] = negative_prompt
    return video_prompt


def build_x_to_text_prompt(
    model: str,
    prompt: str,
    has_image: bool,
) -> tuple[dict[str, Any], list[int] | None]:
    del model
    vision_token = _VISION_BLOCK if has_image else None
    return (
        {
            "prompt": _render_lance_prompt("x2t_image", prompt, vision_token=vision_token),
            "modalities": ["text"],
        },
        None,
    )


LANCE_EXTRA_BODY_PARAMS = frozenset(
    {
        "cfg_text_scale",
        "cfg_img_scale",
        "cfg_interval",
        "cfg_renorm_type",
        "cfg_renorm_min",
        "negative_prompt",
        "timestep_shift",
        "num_frames",
        "video_height",
        "video_width",
        "origin_fps",
        "max_think_tokens",
        "do_sample",
        "text_temperature",
        "system_prompt",
        "user_text",
    }
)
LANCE_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "text_output",
    }
)
