# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Build a Cosmos3 T2I prompt by selecting image output modality.

    Cosmos3 uses the same pipeline class for T2I and video modes. The OpenAI
    image endpoint selects T2I by adding ``modalities=["image"]``. ``height``
    and ``width`` are accepted for registry compatibility; sizing travels
    through sampling params instead of the prompt payload.
    """
    del height, width
    text_prompt: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
    }
    if negative_prompt is not None:
        text_prompt["negative_prompt"] = negative_prompt
    return text_prompt


COSMOS3_EXTRA_BODY_PARAMS = frozenset(
    {
        "flow_shift",
        "max_sequence_length",
        "use_resolution_template",
        "use_duration_template",
        "use_system_prompt",
        "system_prompt",
        "negative_prompt",
        "guardrails",
        "condition_frame_indexes_vision",
        "condition_video_keep",
        "generate_sound",
        "sound_gen",
        "sound_duration",
        "audio_duration",
        "action_mode",
        "action",
        "domain_name",
        "domain_id",
        "raw_action_dim",
        "action_chunk_size",
        "action_space",
        "action_fps",
        "image_height",
        "image_width",
        "history_length",
        "conditioning_fps",
        "resolution",
        "image_size",
        "use_state",
        "observation",
        "robot_obs",
        "deterministic_seed",
        "session_id",
    }
)
COSMOS3_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "action",
        "raw_action_dim",
        "domain_id",
        "action_mode",
    }
)
