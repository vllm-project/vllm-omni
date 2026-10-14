"""Shared Covo-Audio prompt constants and construction helpers."""

from __future__ import annotations

from typing import Any

COVO_AUDIO_INPUT_PREFIX = "<|begofcAUDIO|><|cAUDIO|><|endofcAUDIO|>"

COVO_AUDIO_SYSTEM_PROMPT = (
    '你是"小腾"，英文名是"Covo"，由腾讯开发的AI助手。\n'
    "1、请使用简洁、口语化的语言和用户聊天，"
    "你的态度积极、耐心，像一位值得信赖的朋友。\n"
    "2、不要使用列表或编号，避免输出网址、表情符号和复杂的公式。\n"
    "3、不评价竞争对手，不发表主观政治观点，"
    "针对色情类、政治类、恐怖类、歧视类、暴力类的用户问题，"
    "你要妥善应对潜在的安全风险，并给出幽默，情绪安抚以及安全的劝导。\n"
    "请用文本和音频进行对话，交替生成5个文本token和15个音频token，"
    "音频部分使用发音人：default_female"
)


def build_covo_audio_chat_prompt(
    user_content: str,
    system_prompt: str = COVO_AUDIO_SYSTEM_PROMPT,
) -> str:
    """Build a raw chat-template prompt string for Covo-Audio-Chat.

    Used by offline inference and tests where the prompt is passed as a
    plain string (tokenization happens downstream).
    """
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_covo_audio_prompt_token_ids(
    tokenizer: Any,
    user_text: str,
    system_prompt: str = COVO_AUDIO_SYSTEM_PROMPT,
) -> list[int]:
    """Build tokenized prompt IDs for Covo-Audio-Chat.

    Used by the OpenAI-compatible serving layer where prompt_token_ids
    are passed directly to the engine.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
