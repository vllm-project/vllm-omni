# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from Ming repo's usage cookbook:
# https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/cookbook.ipynb
"""Ming-flash-omni-2.0 prompt utilities.

Two unrelated families of helpers live in the same file because they
both belong to the Ming module and are tightly coupled to Ming-specific
prompt conventions:

1. **Image-gen query-token expansion** — string-level helpers used by
   the thinker stage to mark the ``<image><imagePatch>*N</image>`` block
   that gets substituted with learnable ``query_tokens_dict`` embeddings
   during forward.

2. **TTS / talker caption builder** — JSON caption template for the
   Ming-flash-omni standalone talker (TTS) stage.
"""

from __future__ import annotations

import copy
import json
from typing import Any

# ============================================================
# Image-gen query-token block (thinker stage)
# ============================================================

# Ming's thinker uses these tokens to mark a learnable image-generation
# query block inside the text prompt. The thinker substitutes its
# ``query_tokens_dict`` embeddings at each ``<imagePatch>`` position
# during forward; see
# ``MingFlashOmniThinker._maybe_inject_image_gen_query_embeds``.
_IMAGE_OPEN_TOKEN = "<image>"
_IMAGE_CLOSE_TOKEN = "</image>"
IMAGE_PATCH_TOKEN = "<imagePatch>"

# Default query-token count matches ``MingImageGenConfig(img_gen_scales=[16])``
# (16 * 16 = 256), which is what the released inclusionAI/Ming-flash-omni-2.0
# checkpoint ships.
DEFAULT_NUM_QUERY_TOKENS = 256


def maybe_expand_image_gen_prompt(
    prompt: str,
    num_query_tokens: int = DEFAULT_NUM_QUERY_TOKENS,
) -> str:
    """Append the ``<image><imagePatch>*N</image>`` suffix for text-to-image.

    The thinker expects image-generation requests to end with an N-wide
    block of ``<imagePatch>`` tokens (wrapped in ``<image>``/``</image>``)
    — those positions get substituted with learnable
    ``query_tokens_dict`` embeddings during forward.

    This helper is a no-op (returns the input unchanged) when:

      * ``prompt`` is not a non-empty string, or
      * the prompt already contains an ``<imagePatch>`` block (avoids
        double expansion for tests / manual calls that pre-format the
        prompt).

    Args:
        prompt: Raw user prompt text.
        num_query_tokens: Total number of query tokens to emit. Defaults
            to 256 (the released checkpoint's ``img_gen_scales=[16]``).

    Returns:
        The (possibly expanded) prompt text.
    """
    if not isinstance(prompt, str) or not prompt:
        return prompt
    if IMAGE_PATCH_TOKEN in prompt:
        return prompt

    # TODO(multi-scale): single-block emission assumes img_gen_scales=[16].
    suffix = _IMAGE_OPEN_TOKEN + (IMAGE_PATCH_TOKEN * num_query_tokens) + _IMAGE_CLOSE_TOKEN
    return prompt + suffix


# ============================================================
# TTS / talker caption builder
# ============================================================

DEFAULT_PROMPT = "Please generate speech based on the following description.\n"

BASE_CAPTION_TEMPLATE: dict[str, Any] = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}


def create_instruction(user_input: dict[str, Any]) -> str:
    """Return a JSON caption string for ``audio_sequence[0]``.

    Only keys already present on the base template are merged in; unknown
    keys are silently ignored to keep the output schema stable.
    """
    caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
    item = caption["audio_sequence"][0]
    for key, value in user_input.items():
        if key in item:
            item[key] = value
    return json.dumps(caption, ensure_ascii=False)


__all__ = [
    "IMAGE_PATCH_TOKEN",
    "DEFAULT_NUM_QUERY_TOKENS",
    "maybe_expand_image_gen_prompt",
    "DEFAULT_PROMPT",
    "BASE_CAPTION_TEMPLATE",
    "create_instruction",
]
