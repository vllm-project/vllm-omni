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

from dataclasses import dataclass
from typing import Any

from .system_prompt import get_system_prompt

BOT_TASKS = ("auto", "image", "recaption", "think_recaption")
PROMPT_BOT_TASKS = ("auto", "none", "think", "recaption", "vanilla")

# HunyuanImage-3.0-Instruct special token ids from tokenizer.json.
# Keep offline AR prompt/stop-token behavior independent of runtime
# tokenizer lookup for these fixed control tokens.
HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|endoftext|>": 127957,
    "<|startoftext|>": 127958,
    "<boi>": 128000,
    "<eoi>": 128001,
    "<img>": 128006,
    "<cfg>": 128010,
    "<recaption>": 128018,
    "</recaption>": 128019,
    "<think>": 128023,
    "</think>": 128024,
    "<answer>": 128025,
    "</answer>": 128026,
    "<img_size_1024>": 128037,
    "<img_ratio_0>": 128044,
    "<img_ratio_32>": 128076,
    "<img_ratio_33>": 130103,
    "<img_ratio_36>": 130106,
}
_BOT_TASK_TO_TOKENIZER_TASK = {
    "auto": "auto",
    "image": "image",
    "recaption": "recaption",
    "think_recaption": "think",
}

# task -> (sys_type, bot_task, trigger_tag)
_TASK_PRESETS: dict[str, tuple[str, str | None, str | None]] = {
    "t2t": ("en_unified", None, None),
    "t2t_think": ("en_unified", "think", "<think>"),
    "i2t": ("en_unified", None, None),
    "i2t_think": ("en_unified", "think", "<think>"),
    "it2i_think": ("en_unified", "think", "<think>"),
    "it2i_recaption": ("en_unified", "recaption", "<recaption>"),
    "t2i_think": ("en_unified", "think", "<think>"),
    "t2i_recaption": ("en_unified", "recaption", "<recaption>"),
    "t2i_vanilla": ("en_vanilla", "image", None),
}

_MODALITY_TO_TASK_PREFIX = {
    "text2text": "t2t",
    "t2t": "t2t",
    "img2text": "i2t",
    "image2text": "i2t",
    "i2t": "i2t",
    "text2img": "t2i",
    "text2image": "t2i",
    "t2i": "t2i",
    "img2img": "it2i",
    "image2image": "it2i",
    "it2i": "it2i",
    "ti2i": "it2i",
}

_DEFAULT_BOT_TASK_BY_PREFIX: dict[str, str | None] = {
    "t2t": None,
    "i2t": None,
    "t2i": "think",
    "it2i": "think",
}

_TASK_BY_PREFIX_AND_BOT_TASK: dict[tuple[str, str | None], str] = {
    ("t2t", None): "t2t",
    ("t2t", "think"): "t2t_think",
    ("i2t", None): "i2t",
    ("i2t", "think"): "i2t_think",
    ("t2i", "think"): "t2i_think",
    ("t2i", "recaption"): "t2i_recaption",
    ("t2i", "vanilla"): "t2i_vanilla",
    ("it2i", "think"): "it2i_think",
    ("it2i", "recaption"): "it2i_recaption",
}

_PROMPT_BOT_TASK_ALIASES: dict[str, str | None] = {
    "auto": "auto",
    "default": "auto",
    "none": None,
    "no": None,
    "false": None,
    "think": "think",
    "think_recaption": "think",
    "recaption": "recaption",
    "image": "vanilla",
    "vanilla": "vanilla",
}


@dataclass(frozen=True)
class BotTaskResolution:
    """Resolved HunyuanImage3 prompt/bot-task settings."""

    task: str | None
    sys_type: str | None
    prompt_bot_task: str | None
    bot_task: str
    tokenizer_bot_task: str
    trigger_tag: str | None
    stop_token_ids: list[int] | None = None


def available_tasks() -> list[str]:
    """Sorted list of task keys accepted by `build_prompt` / `build_prompt_tokens`."""
    return sorted(_TASK_PRESETS)


def available_prompt_bot_tasks() -> list[str]:
    """Sorted public bot_task values accepted by `resolve_bot_task` with modality."""
    return sorted(PROMPT_BOT_TASKS)


def _task_preset(task: str) -> tuple[str, str | None, str | None]:
    if task not in _TASK_PRESETS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {available_tasks()}")
    return _TASK_PRESETS[task]


def _task_has_image_input(task: str) -> bool:
    return task.startswith(("i2t", "it2i"))


def _normalize_prompt_bot_task(bot_task: str | None) -> str | None:
    if bot_task is None:
        return "auto"

    normalized = bot_task.strip().lower()
    if normalized not in _PROMPT_BOT_TASK_ALIASES:
        raise ValueError(f"Unknown bot_task {bot_task!r}. Choose from: {available_prompt_bot_tasks()}")
    return _PROMPT_BOT_TASK_ALIASES[normalized]


def _task_for_modality_and_prompt_bot_task(modality: str, bot_task: str | None = "auto") -> str:
    modality_key = modality.strip().lower()
    if modality_key not in _MODALITY_TO_TASK_PREFIX:
        raise ValueError(f"Unknown modality {modality!r}. Choose from: {sorted(_MODALITY_TO_TASK_PREFIX)}")

    task_prefix = _MODALITY_TO_TASK_PREFIX[modality_key]
    normalized_bot_task = _normalize_prompt_bot_task(bot_task)
    if normalized_bot_task == "auto":
        normalized_bot_task = _DEFAULT_BOT_TASK_BY_PREFIX[task_prefix]

    task_key = (task_prefix, normalized_bot_task)
    if task_key not in _TASK_BY_PREFIX_AND_BOT_TASK:
        valid_bot_tasks = sorted(
            "none" if candidate is None else candidate
            for prefix, candidate in _TASK_BY_PREFIX_AND_BOT_TASK
            if prefix == task_prefix
        )
        raise ValueError(
            f"bot_task {bot_task!r} is not supported for modality {modality!r}. Choose from: {valid_bot_tasks}"
        )

    return _TASK_BY_PREFIX_AND_BOT_TASK[task_key]


def _stop_token_ids_for_tokenizer_bot_task(
    tokenizer,
    tokenizer_bot_task: str,
    image_size: int | str | None = None,
) -> list[int]:
    eos_id = _eos_token_id(tokenizer)

    if image_size == "auto":
        start_ratio_id = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
        end_ratio_id = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"]
        extra_auto_stops = list(range(start_ratio_id, end_ratio_id + 1))
    else:
        extra_auto_stops = [_token_id(tokenizer, "<boi>")]

    stop_token_id = {
        "auto": [eos_id] + extra_auto_stops,
        "image": [eos_id],
        "recaption": [
            _token_id(tokenizer, "</recaption>"),
            _token_id(tokenizer, "</answer>"),
            eos_id,
        ],
        "think": [
            _token_id(tokenizer, "</recaption>"),
            _token_id(tokenizer, "</answer>"),
            eos_id,
        ],
    }
    return stop_token_id[tokenizer_bot_task]


def resolve_bot_task(
    bot_task: str | None = "auto",
    *,
    modality: str | None = None,
    task: str | None = None,
    tokenizer: Any | None = None,
    image_size: int | str | None = None,
) -> BotTaskResolution:
    """Resolve HunyuanImage3 bot-task related prompt settings.

    Pass `modality + bot_task` for CLI/request-level behavior, `task` for a
    canonical prompt task, or only `bot_task` to validate/map a pipeline
    HunyuanImage3 bot_task.
    """
    if task is not None and modality is not None:
        raise ValueError("Pass either task or modality, not both.")

    if task is None and modality is not None:
        task = _task_for_modality_and_prompt_bot_task(modality, bot_task)

    if task is not None:
        sys_type, preset_bot_task, trigger_tag = _task_preset(task)
        resolved_bot_task = "think_recaption" if preset_bot_task == "think" else preset_bot_task or "auto"
        prompt_bot_task = preset_bot_task
    else:
        sys_type = None
        trigger_tag = None
        prompt_bot_task = None
        resolved_bot_task = "auto" if bot_task is None else bot_task.strip().lower()

    if resolved_bot_task not in _BOT_TASK_TO_TOKENIZER_TASK:
        raise ValueError(f"Unknown bot_task {resolved_bot_task!r}. Choose from: {list(BOT_TASKS)}")
    tokenizer_bot_task = _BOT_TASK_TO_TOKENIZER_TASK[resolved_bot_task]
    stop_token_ids = (
        _stop_token_ids_for_tokenizer_bot_task(tokenizer, tokenizer_bot_task, image_size=image_size)
        if tokenizer is not None
        else None
    )

    return BotTaskResolution(
        task=task,
        sys_type=sys_type,
        prompt_bot_task=prompt_bot_task,
        bot_task=resolved_bot_task,
        tokenizer_bot_task=tokenizer_bot_task,
        trigger_tag=trigger_tag,
        stop_token_ids=stop_token_ids,
    )


def sys_type_for_task(task: str) -> str:
    """Return the default system prompt type for a canonical prompt task."""
    sys_type = resolve_bot_task(task=task).sys_type
    assert sys_type is not None
    return sys_type


def _token_id(tokenizer, token: str) -> int:
    if token in HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS:
        return HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS[token]

    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not know special token {token!r}")
    return int(token_id)


def _eos_token_id(tokenizer) -> int:
    return HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<|endoftext|>"]


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
    preset_sys_type, preset_bot_task, trigger_tag = _task_preset(task)
    effective_sys_type = sys_type or preset_sys_type

    system_prompt = get_system_prompt(effective_sys_type, preset_bot_task, custom_system_prompt)
    sys_text = system_prompt.strip() if system_prompt else ""

    has_image_input = _task_has_image_input(task)

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
    preset_sys_type, preset_bot_task, trigger_tag = _task_preset(task)
    effective_sys_type = sys_type or preset_sys_type

    bos_id = _token_id(tokenizer, "<|startoftext|>")
    img_id = _token_id(tokenizer, "<img>")
    trig_id = _token_id(tokenizer, trigger_tag) if trigger_tag else None

    has_image_input = _task_has_image_input(task)

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


__all__ = [
    "BotTaskResolution",
    "available_tasks",
    "available_prompt_bot_tasks",
    "BOT_TASKS",
    "build_prompt",
    "build_prompt_tokens",
    "HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS",
    "PROMPT_BOT_TASKS",
    "resolve_bot_task",
    "sys_type_for_task",
]
