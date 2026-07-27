# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance chat / system prompts.

Matches the upstream Lance training distribution (see
``bytedance/Lance/data/system_prompt_render.py`` and
``data/common.py::generate_system_prompt``):

  * each task has a specific system prompt describing what to attend to;
  * the prompt is wrapped in the Qwen chat template
    ``<|im_start|>system\\n…<|im_end|>\\n<|im_start|>user\\n…<|im_end|>\\n``
    ``<|im_start|>assistant\\n``;
  * vision tokens are framed by ``<|vision_start|><|video_pad|><|vision_end|>``
    (upstream uses ``<|video_pad|>`` even for image type by default).

These are pure string formatting helpers — the runtime pipeline still does
the actual VAE / ViT prefill separately.
"""

from __future__ import annotations

# Task-specific system prompts (verbatim from upstream's
# ``generate_system_prompt`` — first variant of each pool).
SYSTEM_PROMPTS: dict[tuple[str, str], str] = {
    # (task, vision_type) -> system prompt
    ("t2i", "image"): (
        "Describe the image by detailing the color, quantity, text, shape, "
        "size, texture, spatial relationships of the objects and background:"
    ),
    ("t2v", "video"): (
        "Describe the video by detailing the color, quantity, visible text, "
        "shape, size, texture, spatial relationships and motion/camera "
        "movements of the objects and background:"
    ),
    # i2v reuses the t2v system prompt (matches upstream Lance PR #33's
    # ``data/common.py`` where ``system_prompt_type in ("t2v", "i2v")``
    # selects the same prompt pool).
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
    ("video_edit", "video"): (
        "Describe the key features of the input video (color, shape, size, "
        "texture, objects, background), then explain how the user's text "
        "instruction should alter or modify the video. Generate a new video "
        "that meets the user's requirements while maintaining consistency "
        "with the original input where appropriate."
    ),
    ("x2t_image", "image"): (
        "Generate a detailed and accurate description of the image, including all the key moments and visual details."
    ),
    ("x2t_video", "video"): (
        "Generate a detailed and accurate description of the video, including all the key moments and visual details."
    ),
}

# Vision-content placeholders (Qwen2.5-VL convention).  Upstream uses
# ``<|video_pad|>`` for image inputs too unless ``force_video_pad=False``
# (counter-intuitively named; see ``render_qwenvl_prompt`` upstream).
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VIDEO_PAD = "<|video_pad|>"
IMAGE_PAD = "<|image_pad|>"


def render_lance_prompt(
    task: str,
    user_text: str,
    *,
    vision_token: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Render a Lance-compatible user-side prompt.

    ``task`` is one of ``t2i``, ``t2v``, ``image_edit``, ``video_edit``,
    ``x2t_image``, ``x2t_video``.  ``vision_token`` should be the visual
    placeholder to embed inside the user message (e.g. ``VIDEO_PAD`` or a
    ``VISION_START + VIDEO_PAD + VISION_END`` block); pass ``None`` for
    text-only inputs.  ``system_prompt`` overrides the default task system
    prompt — required for x2t QA examples whose upstream JSON carries a
    per-example instruction (e.g. ``"Look at the image carefully and answer
    the question."``); without it x2t falls back to the caption-style
    default and the model describes instead of answering.

    Returns a single string ready to be tokenized; no further wrapping is
    needed by the caller.
    """
    # ``t2v`` / ``x2t_video`` / ``video_edit`` use video system prompts;
    # the rest are image.
    vision_type = "video" if task in {"t2v", "i2v", "x2t_video", "video_edit"} else "image"
    sys_prompt = (
        system_prompt if system_prompt else SYSTEM_PROMPTS.get((task, vision_type), "You are a helpful assistant.")
    )

    if vision_token is None:
        user_msg = user_text
    else:
        # Upstream concatenates the vision block immediately before the text.
        user_msg = f"{vision_token}{user_text}"

    return (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    )


__all__ = [
    "IMAGE_PAD",
    "SYSTEM_PROMPTS",
    "VIDEO_PAD",
    "VISION_END",
    "VISION_START",
    "render_lance_prompt",
]
