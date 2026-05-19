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
module provides that builder; the (task, bot_task) -> template mapping
below is the canonical mapping for both flows.

Two orthogonal axes:

  * `task` selects the I/O modality combination, which only controls
    whether `<img>` placeholders are emitted between `User: ` and the
    user prompt: ``i2t`` / ``it2i`` produce them, ``t2t`` / ``t2i`` do
    not.

  * `bot_task` selects the prompting mode and drives both the system
    prompt and the trigger tag appended after ``Assistant: ``. ``None``
    (default) gives a plain Assistant turn under the unified prompt;
    ``think`` / ``recaption`` switch the trigger tag to ``<think>`` /
    ``<recaption>``; ``think_recaption`` swaps the system prompt for
    the dedicated combined-mode template; ``vanilla`` drops the chat
    structure entirely (pretrain template, ``t2i`` only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .system_prompt import get_system_prompt

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

# bot_task -> (sys_type, trigger_tag).
# ``vanilla`` is special-cased downstream: it bypasses the chat template
# (no ``User:`` / ``Assistant:`` framing) and is only valid with
# ``task='t2i'``.
_BOT_TASK_PRESETS: dict[str | None, tuple[str, str | None]] = {
    None: ("en_unified", None),
    "think": ("en_unified", "<think>"),
    "recaption": ("en_unified", "<recaption>"),
    "think_recaption": ("en_think_recaption", "<think>"),
    "vanilla": ("en_vanilla", None),
}

_TASKS: frozenset[str] = frozenset({"t2t", "i2t", "it2i", "t2i"})


class _DefaultBotTask:
    pass


_DEFAULT_BOT_TASK = _DefaultBotTask()

# Legacy composite task alias -> (task, bot_task). Keep this during rebase so
# older callers and intermediate commits still resolve cleanly.
_TASK_PRESETS: dict[str, tuple[str, str | None, str | None]] = {
    "t2t": ("en_unified", None, None),
    "i2t": ("en_unified", None, None),
    "it2i_think": ("en_unified", "think", "<think>"),
    "it2i_recaption": ("en_unified", "recaption", "<recaption>"),
    "it2i_think_recaption": ("en_unified", "think_recaption", "<think>"),
    "t2i": ("en_unified", None, None),
    "t2i_vanilla": ("en_vanilla", "vanilla", None),
    "t2i_think": ("en_unified", "think", "<think>"),
    "t2i_recaption": ("en_unified", "recaption", "<recaption>"),
}

_LEGACY_COMPOSITE_TASKS: frozenset[str] = frozenset(_TASK_PRESETS) - {"t2t", "i2t", "t2i"}


def _normalize_task_and_bot_task(
    task: str,
    bot_task: str | None | _DefaultBotTask,
) -> tuple[str, str | None]:
    bot_task_was_omitted = bot_task is _DEFAULT_BOT_TASK
    if task in _TASK_PRESETS:
        _, legacy_bot_task, _ = _TASK_PRESETS[task]
        base_task = task.split("_", 1)[0]
        if base_task == "t2i" and task == "t2i":
            base_task = "t2i"
        if task in ("t2t", "i2t", "t2i"):
            base_task = task
        if bot_task_was_omitted:
            bot_task = legacy_bot_task
        elif task in _LEGACY_COMPOSITE_TASKS and bot_task is None:
            # Composite task names already encode the legacy bot_task. Keep
            # calls like build_prompt_tokens(task="it2i_think", bot_task=None)
            # on their historical meaning; explicit None is the plain-mode
            # escape hatch only for the new two-axis base tasks.
            bot_task = legacy_bot_task
        task = base_task
    elif bot_task_was_omitted:
        bot_task = "think"
    return task, bot_task


def available_tasks() -> list[str]:
    """Sorted list of `task` values accepted by the prompt builders."""
    return sorted(_TASKS)


def available_bot_tasks() -> list[str | None]:
    """Sorted list of `bot_task` values (with ``None`` first)."""
    rest = sorted(k for k in _BOT_TASK_PRESETS if k is not None)
    return [None, *rest]


def resolve_sys_type(bot_task: str | None) -> str:
    """Default system-prompt type for a given ``bot_task``."""
    if bot_task not in _BOT_TASK_PRESETS:
        raise ValueError(f"Unknown bot_task {bot_task!r}. Choose from: {available_bot_tasks()}")
    return _BOT_TASK_PRESETS[bot_task][0]


def resolve_stop_token_ids(
    task: str = "it2i",
    bot_task: str | None | _DefaultBotTask = _DEFAULT_BOT_TASK,
    tokenizer: Any | None = None,
) -> list[int]:
    """AR stop-token ids for a given (task, bot_task) generation request.

    Image-output tasks (``it2i`` / ``t2i``) stop on any ``<img_ratio_*>``
    token. Upstream ``modeling_hunyuan_image_3.py::generate_image``
    (line 3289-3303) sets ``final_stop_tokens`` to the full ratio token
    range when ``need_ratio`` is true, then strips the trailing ratio
    token before passing the cot to the image stage. AR's natural
    trajectory under ``_stage_transitions`` is
    ``</recaption><answer><boi><img_size_base><img_ratio_X>``; stopping
    AT the ratio token means KV ends exactly at the prefix DiT reuses,
    and ``ar2diffusion`` can read the ratio off the last sampled token
    without AR wasting decode steps on ``<|endoftext|>``.

    Text-output tasks (``i2t`` / ``t2t``) stop on ``<answer>`` -- the AR
    is the final stage, and the comprehension response sits inside the
    ``<answer>`` body so the answer-open is the natural cot/recaption
    terminator.
    """
    task, bot_task = _normalize_task_and_bot_task(task, bot_task)
    if task not in _TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {available_tasks()}")
    if bot_task not in _BOT_TASK_PRESETS:
        raise ValueError(f"Unknown bot_task {bot_task!r}. Choose from: {available_bot_tasks()}")
    if task in ("it2i", "t2i"):
        # Main ratio range: <img_ratio_0> .. <img_ratio_32>.
        start = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
        end = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"]
        stops = list(range(start, end + 1))
        # Other slices (upstream tokenizer ``ratio_token_other_slices``):
        # <img_ratio_33> .. <img_ratio_36>.
        other_start = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]
        other_end = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]
        stops.extend(range(other_start, other_end + 1))
        return stops
    return [HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"]]


# Upstream "Multi-Image Fusion" caps reference images at 3 per request.
MAX_IMAGES_PER_REQUEST = 3


def _validate_num_images(num_images: int) -> None:
    if not (1 <= num_images <= MAX_IMAGES_PER_REQUEST):
        raise ValueError(f"num_images must be in [1, {MAX_IMAGES_PER_REQUEST}], got {num_images}")


def _resolve_preset(task: str, bot_task: str | None) -> tuple[str, str | None]:
    """Validate (task, bot_task) and return ``(sys_type, trigger_tag)``."""
    task, bot_task = _normalize_task_and_bot_task(task, bot_task)
    if task not in _TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from: {available_tasks()}")
    if bot_task not in _BOT_TASK_PRESETS:
        raise ValueError(f"Unknown bot_task {bot_task!r}. Choose from: {available_bot_tasks()}")
    if bot_task == "vanilla" and task != "t2i":
        raise ValueError(f"bot_task='vanilla' is only valid with task='t2i' (pretrain template); got task={task!r}")
    return _BOT_TASK_PRESETS[bot_task]


def build_prompt(
    user_prompt: str,
    task: str = "it2i",
    bot_task: str | None | _DefaultBotTask = _DEFAULT_BOT_TASK,
    sys_type: str | None = None,
    custom_system_prompt: str | None = None,
    num_images: int = 1,
) -> str:
    """Build a HunyuanImage-3.0 prompt as a string (legacy/compat path)."""
    task, bot_task = _normalize_task_and_bot_task(task, bot_task)
    preset_sys_type, trigger_tag = _resolve_preset(task, bot_task)
    effective_sys_type = sys_type or preset_sys_type

    system_prompt = get_system_prompt(effective_sys_type, bot_task, custom_system_prompt)
    sys_text = system_prompt or ""

    has_image_input = task in ("i2t", "it2i")
    if has_image_input:
        _validate_num_images(num_images)

    if bot_task == "vanilla":
        parts = ["<|startoftext|>"]
        if sys_text:
            parts.append(sys_text)
        parts.append(user_prompt)
        return "".join(parts)

    parts = ["<|startoftext|>"]
    if sys_text:
        parts.append(f"{sys_text}\n\n")
    parts.append("User: ")
    if has_image_input:
        parts.extend(["<img>"] * num_images)
    parts.append(user_prompt)
    parts.append("\n\nAssistant: ")
    if trigger_tag:
        parts.append(trigger_tag)
    return "".join(parts)


@dataclass
class PromptTokensResult:
    token_ids: list[int]
    system_prompt_type: str


def build_prompt_tokens(
    user_prompt: str,
    tokenizer,
    task: str = "it2i",
    bot_task: str | None | _DefaultBotTask = _DEFAULT_BOT_TASK,
    sys_type: str | None = None,
    custom_system_prompt: str | None = None,
    num_images: int = 1,
) -> PromptTokensResult:
    """Segment-by-segment tokenization that matches HF apply_chat_template."""
    task, bot_task = _normalize_task_and_bot_task(task, bot_task)
    preset_sys_type, trigger_tag = _resolve_preset(task, bot_task)
    effective_sys_type = sys_type or preset_sys_type

    bos_id = tokenizer.convert_tokens_to_ids("<|startoftext|>")
    img_id = tokenizer.convert_tokens_to_ids("<img>")
    trig_id = tokenizer.convert_tokens_to_ids(trigger_tag) if trigger_tag else None

    has_image_input = task in ("i2t", "it2i")
    if has_image_input:
        _validate_num_images(num_images)

    if bot_task == "vanilla":
        s = build_prompt(
            user_prompt,
            task=task,
            bot_task=bot_task,
            sys_type=sys_type,
            custom_system_prompt=custom_system_prompt,
        )
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        return PromptTokensResult(
            token_ids=token_ids,
            system_prompt_type=effective_sys_type,
        )

    system_prompt = get_system_prompt(effective_sys_type, bot_task, custom_system_prompt)
    sys_text = system_prompt or ""

    ids: list[int] = [bos_id]
    if sys_text:
        ids += tokenizer.encode(sys_text, add_special_tokens=False)
        ids += tokenizer.encode("\n\n", add_special_tokens=False)
    ids += tokenizer.encode("User: ", add_special_tokens=False)
    if has_image_input:
        ids += [img_id] * num_images
    ids += tokenizer.encode(user_prompt, add_special_tokens=False)
    ids += tokenizer.encode("\n\nAssistant: ", add_special_tokens=False)
    if trig_id is not None:
        ids += [trig_id]

    return PromptTokensResult(
        token_ids=ids,
        system_prompt_type=effective_sys_type,
    )


__all__ = [
    "HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS",
    "MAX_IMAGES_PER_REQUEST",
    "_TASK_PRESETS",
    "available_bot_tasks",
    "available_tasks",
    "build_prompt",
    "build_prompt_tokens",
    "resolve_stop_token_ids",
    "resolve_sys_type",
]
