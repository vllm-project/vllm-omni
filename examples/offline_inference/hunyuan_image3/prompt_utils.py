# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prompt construction utilities for HunyuanImage-3.0-Instruct examples.

Wraps system_prompt.get_system_prompt() with task-aware presets so that
examples and tests don't need to manually concatenate system prompts,
<img>, <think>, and <recaption> tags.

Usage:
    from prompt_utils import build_prompt

    # IT2I (image editing, think+recaption mode)
    prompt = build_prompt("Make the petals neon pink", task="it2i_think")

    # I2T (image understanding)
    prompt = build_prompt("Describe the content of the picture.", task="i2t")
"""

from __future__ import annotations

from vllm_omni.diffusion.models.hunyuan_image3.system_prompt import (
    get_system_prompt,
)

# task → (sys_type, bot_task, trigger_tag)
# trigger_tag: "<think>", "<recaption>", or None
_TASK_PRESETS: dict[str, tuple[str, str | None, str | None]] = {
    # Pure text generation (text → text, no image)
    "t2t": ("en_unified", None, None),
    # Image understanding (image → text)
    "i2t": ("en_unified", None, None),
    # Image editing (image+text → image), think+recaption mode
    "it2i_think": ("en_unified", "think", "<think>"),
    # Image editing, recaption-only mode
    "it2i_recaption": ("en_unified", "recaption", "<recaption>"),
    # Text-to-image, think mode
    "t2i_think": ("en_unified", "think", "<think>"),
    # Text-to-image, recaption mode
    "t2i_recaption": ("en_unified", "recaption", "<recaption>"),
    # Text-to-image, vanilla (no CoT)
    "t2i_vanilla": ("en_vanilla", "image", None),
}


def build_prompt(
    user_prompt: str,
    task: str = "it2i_think",
    sys_type: str | None = None,
    custom_system_prompt: str | None = None,
) -> str:
    """Build a complete HunyuanImage-3.0 prompt with auto-selected system
    prompt and mode trigger tags.

    Args:
        user_prompt: The user's raw instruction or question.
        task: One of the preset task keys (see _TASK_PRESETS).
        sys_type: Override the preset's sys_type for get_system_prompt().
        custom_system_prompt: Custom system prompt text (used when
            sys_type="custom").

    Returns:
        Fully formatted prompt string ready for Omni.generate().
    """
    if task not in _TASK_PRESETS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {sorted(_TASK_PRESETS)}")

    preset_sys_type, preset_bot_task, trigger_tag = _TASK_PRESETS[task]
    effective_sys_type = sys_type or preset_sys_type

    system_prompt = get_system_prompt(effective_sys_type, preset_bot_task, custom_system_prompt)
    sys_text = system_prompt.strip() if system_prompt else ""

    has_image_input = task.startswith("i2t") or task.startswith("it2i")

    parts = ["<|startoftext|>"]
    if sys_text:
        parts.append(sys_text)
    # Instruct conversation template: \n\nUser: ... \n\nAssistant:
    parts.append("\n\nUser: ")
    if has_image_input:
        parts.append("<img>")
    parts.append(user_prompt)
    parts.append("\n\nAssistant: ")
    if trigger_tag:
        parts.append(trigger_tag)

    return "".join(parts)
