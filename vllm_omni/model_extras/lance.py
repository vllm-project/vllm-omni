# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

from vllm_omni.diffusion.models.lance.prompts import (
    VIDEO_PAD,
    VISION_END,
    VISION_START,
    render_lance_prompt,
)

# Lance frames visual content as <|vision_start|><|video_pad|><|vision_end|>.
# Upstream's renderer uses <|video_pad|> for image inputs too, so image and
# video tasks share the same block.
_VISION_BLOCK = f"{VISION_START}{VIDEO_PAD}{VISION_END}"

# Lance's video defaults, mirroring the bespoke example this replaces.
_DEFAULT_NUM_FRAMES = 25
_DEFAULT_VIDEO_HEIGHT = 480
_DEFAULT_VIDEO_WIDTH = 768


def _single_image(input_image: Image.Image | list[Image.Image], task: str) -> Image.Image:
    if isinstance(input_image, list):
        if len(input_image) != 1:
            raise ValueError(f"Lance {task} accepts exactly one input image, got {len(input_image)}.")
        input_image = input_image[0]
    if not isinstance(input_image, Image.Image):
        raise ValueError(f"Lance {task} requires a PIL image input.")
    return input_image


def _video_extra_args(
    height: int | None,
    width: int | None,
    num_frames: int | None,
) -> dict[str, Any]:
    """Lance's video shape is read from ``extra_args``, not sampling_params."""
    return {
        "num_frames": num_frames if num_frames is not None else _DEFAULT_NUM_FRAMES,
        "video_height": height if height is not None else _DEFAULT_VIDEO_HEIGHT,
        "video_width": width if width is not None else _DEFAULT_VIDEO_WIDTH,
    }


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    # Lance resolves image height/width from sampling_params, so they are not
    # echoed into the prompt here.
    del height, width
    text_prompt: dict[str, Any] = {
        "prompt": render_lance_prompt("t2i", prompt),
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
        "prompt": render_lance_prompt("image_edit", prompt, vision_token=_VISION_BLOCK),
        "modalities": ["image"],
        "multi_modal_data": {"img2img": _single_image(input_image, "image editing")},
    }
    if negative_prompt is not None:
        img_prompt["negative_prompt"] = negative_prompt
    return img_prompt


def build_text_to_video_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    video_prompt: dict[str, Any] = {
        "prompt": render_lance_prompt("t2v", prompt),
        "modalities": ["video"],
        "extra_args": _video_extra_args(height, width, num_frames),
    }
    if negative_prompt is not None:
        video_prompt["negative_prompt"] = negative_prompt
    return video_prompt


def build_image_to_video_prompt(
    prompt: str,
    negative_prompt: str | None,
    media_inputs: Mapping[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    if set(media_inputs) != {"image"}:
        raise ValueError("Lance image-to-video accepts only a single --image input.")

    # The image is a reference frame for VAE+ViT prefill, not a pinned first
    # frame; the pipeline selects _forward_i2v off the ``first_frame`` key.
    video_prompt: dict[str, Any] = {
        "prompt": render_lance_prompt("i2v", prompt, vision_token=_VISION_BLOCK),
        "modalities": ["video"],
        "multi_modal_data": {"first_frame": _single_image(media_inputs["image"], "image-to-video")},
        "extra_args": _video_extra_args(height, width, num_frames),
    }
    if negative_prompt is not None:
        video_prompt["negative_prompt"] = negative_prompt
    return video_prompt


# Knobs LancePipeline reads from sampling_params.extra_args.
#
# LancePipeline.forward dispatches the video / edit / x2t paths itself and then
# falls through to ``super().forward(req)`` (BagelPipeline) for t2i, so the set
# below is the union of both classes' ``extra_args`` reads -- notably ``think``,
# which is consumed by the inherited Bagel path and would otherwise be filtered
# out of extra_args before reaching the pipeline.
LANCE_EXTRA_BODY_PARAMS = frozenset(
    {
        "cfg_text_scale",
        "cfg_img_scale",
        "cfg_interval",
        "cfg_renorm_type",
        "cfg_renorm_min",
        "negative_prompt",
        "timestep_shift",
        # think / text decoding: ``think`` is read by the inherited Bagel t2i
        # path; the rest are shared with Lance's own x2t (img2text/video2text).
        "think",
        "max_think_tokens",
        "do_sample",
        "text_temperature",
        "system_prompt",
        "user_text",
        # video shape / sampling
        "num_frames",
        "video_height",
        "video_width",
        "origin_fps",
    }
)

# Lance's x2t paths emit metadata["text"]["text_output"]; the inherited Bagel
# t2i path emits metadata["text"]["think_text"] when think mode is enabled.
LANCE_EXTRA_OUTPUT_PARAMS = frozenset({"text_output", "think_text"})
