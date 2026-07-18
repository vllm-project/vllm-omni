# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

# AR image-grid patch size (pixels per visual token). Mirrors the constant the
# bespoke MammothModa2 example used to convert image dimensions -> AR grid dims.
_PATCH_SIZE = 16

_AR_SYSTEM_PROMPT = "You are a helpful image generator."


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Build the MammothModa2 AR-stage prompt for text-to-image generation.

    Reproduces the prompt string and the structural ``additional_information``
    that the former bespoke example (``run_mammothmoda2_t2i.py``) constructed,
    deriving the AR image grid from ``height`` / ``width``.

    Model-specific sampling knobs (``text_guidance_scale``, ``cfg_range``,
    ``num_inference_steps``) flow separately via ``extra_body`` -> ``extra_args``.
    Config-derived token ids (``eol_token_id``, ``visual_token_start_id``,
    ``visual_token_end_id``, ``visual_ids``) are sourced inside the AR stage
    rather than passed by the caller -- see the ar2dit stage input processor.

    MammothModa2 t2i uses classifier-free guidance via ``text_guidance_scale``
    and has no explicit negative-prompt path, so ``negative_prompt`` is accepted
    for signature compatibility but not injected.
    """
    h = height or 1024
    w = width or 1024
    ar_height = h // _PATCH_SIZE
    ar_width = w // _PATCH_SIZE

    return {
        "prompt": (
            f"<|im_start|>system\n{_AR_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"<|image start|>{ar_width}*{ar_height}<|image token|>"
        ),
        "additional_information": {
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "ar_height": [ar_height],
            "image_height": [h],
            "image_width": [w],
        },
    }


MAMMOTHMODA2_PREVIEW_EXTRA_BODY_PARAMS = frozenset(
    {
        "text_guidance_scale",
        "cfg_range",
        # MammothModa2's DiT stage consumes inputs via the kwargs interface rather
        # than OmniDiffusionRequest, so the standard --num-inference-steps flag does
        # not reach it; it is routed through extra_body like the CFG knobs.
        "num_inference_steps",
    }
)
MAMMOTHMODA2_PREVIEW_EXTRA_OUTPUT_PARAMS = frozenset()
MAMMOTHMODA2_PREVIEW_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES = True
