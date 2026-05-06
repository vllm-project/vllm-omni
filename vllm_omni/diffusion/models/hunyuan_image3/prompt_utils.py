# SPDX-License-Identifier: Apache-2.0
"""Shared prompt-template construction for HunyuanImage-3.0-Instruct.

Single source of truth for the AR-prefill prompt format used by the
example scripts and any downstream caller that needs to build
HunyuanImage3 chat-template token sequences without invoking the full
diffusion pipeline tokenizer wrapper.

The DiT pipeline (`pipeline_hunyuan_image3.py`) builds prompts through
`TokenizerWrapper.apply_chat_template`, which eagerly consumes
`JointImageInfo` objects produced by image preprocessing. The example
flow uses an `<img>` placeholder + `multi_modal_data` instead, so it
needs a lighter-weight builder that only requires a HF tokenizer. This
module provides that builder; the task -> template mapping below is the
canonical mapping for both flows.
"""

from __future__ import annotations

from .system_prompt import get_system_prompt

# task -> (sys_type, bot_task, trigger_tag)
_TASK_PRESETS: dict[str, tuple[str, str | None, str | None]] = {
    "t2t": ("en_unified", None, None),
    "i2t": ("en_unified", None, None),
    "it2i_think": ("en_unified", "think", "<think>"),
    "it2i_recaption": ("en_unified", "recaption", "<recaption>"),
    "t2i_think": ("en_unified", "think", "<think>"),
    "t2i_recaption": ("en_unified", "recaption", "<recaption>"),
    "t2i_vanilla": ("en_vanilla", "image", None),
}


def available_tasks() -> list[str]:
    """Sorted list of task keys accepted by `build_prompt` / `build_prompt_tokens`."""
    return sorted(_TASK_PRESETS)


def build_prompt(
    user_prompt: str,
    task: str = "it2i_think",
    sys_type: str | None = None,
    custom_system_prompt: str | None = None,
) -> str:
    """Build a HunyuanImage-3.0 prompt as a string (legacy/compat path).

    NOTE: when this string is passed to the engine, the engine's tokenizer
    will run a single BPE pass over the whole string, which can merge
    tokens across segment boundaries (e.g. `。\\n\\n` -> id 3490). For
    inputs that need to match HF baseline byte-for-byte, use
    `build_prompt_tokens` instead and feed the result via prompt_token_ids.
    """
    if task not in _TASK_PRESETS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {available_tasks()}")

    preset_sys_type, preset_bot_task, trigger_tag = _TASK_PRESETS[task]
    effective_sys_type = sys_type or preset_sys_type

    system_prompt = get_system_prompt(effective_sys_type, preset_bot_task, custom_system_prompt)
    sys_text = system_prompt.strip() if system_prompt else ""

    has_image_input = task.startswith("i2t") or task.startswith("it2i")

    # t2i_vanilla: pretrain mode for direct text->image generation. The
    # vanilla system prompt drives the model with no chat structure.
    if task == "t2i_vanilla":
        parts = ["<|startoftext|>"]
        if sys_text:
            parts.append(sys_text)
        parts.append(user_prompt)
        return "".join(parts)

    # All other tasks (t2t / i2t / t2i_think / t2i_recaption /
    # it2i_think / it2i_recaption) use HunyuanImage3 Instruct chat template:
    #   <|startoftext|>{system?}\n\nUser: {<img>?}{user_prompt}\n\nAssistant: {trigger?}
    # generation_config.json declares sequence_template="instruct", so the
    # AR prefill MUST use this template -- verified to match HF's
    # apply_chat_template output token-for-token (modulo BPE boundary merges).
    # The trigger_tag (e.g. <think>) MUST come AFTER the `Assistant: ` prefix:
    # if it goes BEFORE user_prompt (the old pretrain layout) the model puts
    # the user's instructions inside the "thinking section" and collapses
    # into repetition garbage under greedy decoding.
    parts = ["<|startoftext|>"]
    if sys_text:
        parts.append(f"{sys_text}\n\n")
    parts.append("User: ")
    if has_image_input:
        parts.append("<img>")
    parts.append(user_prompt)
    parts.append("\n\nAssistant: ")
    if trigger_tag:
        parts.append(trigger_tag)

    return "".join(parts)


def build_prompt_tokens(
    user_prompt: str,
    tokenizer,
    task: str = "it2i_think",
    sys_type: str | None = None,
    custom_system_prompt: str | None = None,
) -> list[int]:
    """Segment-by-segment tokenization that matches HF apply_chat_template.

    Calling tokenizer.encode(build_prompt(...)) on the full string lets BPE
    merge tokens across segment boundaries (e.g. user_prompt ends with `。`
    and the next segment is `\\n\\n` -> they merge into a single token id
    3490 instead of HF's [1811, 271]). HF's apply_chat_template tokenizes
    each segment independently and concatenates token_ids, so no cross-
    boundary merge happens. We replicate that here and feed the result to
    Omni via OmniTokensPrompt (prompt_token_ids).
    """
    if task not in _TASK_PRESETS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {available_tasks()}")

    preset_sys_type, preset_bot_task, trigger_tag = _TASK_PRESETS[task]
    effective_sys_type = sys_type or preset_sys_type

    bos_id = tokenizer.convert_tokens_to_ids("<|startoftext|>")
    img_id = tokenizer.convert_tokens_to_ids("<img>")
    trig_id = tokenizer.convert_tokens_to_ids(trigger_tag) if trigger_tag else None

    has_image_input = task.startswith("i2t") or task.startswith("it2i")

    # t2i_vanilla uses pretrain template with no chat structure; the vanilla
    # system prompt drives the model directly. No segment boundaries to
    # protect, fall back to whole-string encode.
    if task == "t2i_vanilla":
        s = build_prompt(user_prompt, task, sys_type, custom_system_prompt)
        return tokenizer.encode(s, add_special_tokens=False)

    system_prompt = get_system_prompt(effective_sys_type, preset_bot_task, custom_system_prompt)
    # Do NOT strip -- HF apply_chat_template keeps the system prompt's
    # natural trailing newline; stripping it would shift one token id.
    sys_text = system_prompt or ""

    ids: list[int] = [bos_id]
    if sys_text:
        ids += tokenizer.encode(sys_text, add_special_tokens=False)
        ids += tokenizer.encode("\n\n", add_special_tokens=False)
    ids += tokenizer.encode("User: ", add_special_tokens=False)
    if has_image_input:
        ids += [img_id]
    ids += tokenizer.encode(user_prompt, add_special_tokens=False)
    ids += tokenizer.encode("\n\nAssistant: ", add_special_tokens=False)
    if trig_id is not None:
        ids += [trig_id]
    return ids


__all__ = ["build_prompt", "build_prompt_tokens", "available_tasks"]
