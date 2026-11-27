# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 NetEase Youdao

from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

LANGUAGE_TOKEN_MAP = {
    "zh": "请用中文朗读接下来的文字",
    "en": "请用英文朗读接下来的文字",
    "ja": "请用日语朗读接下来的文字",
    "ko": "请用韩语朗读接下来的文字",
    "de": "请用德语朗读接下来的文字",
    "fr": "请用法语朗读接下来的文字",
    "es": "请用西班牙语朗读接下来的文字",
    "id": "请用印尼语朗读接下来的文字",
    "it": "请用意大利语朗读接下来的文字",
    "th": "请用泰语朗读接下来的文字",
    "pt": "请用葡萄牙语朗读接下来的文字",
    "ru": "请用俄语朗读接下来的文字",
    "ms": "请用马来语朗读接下来的文字",
    "vi": "请用越南语朗读接下来的文字",
}


def format_confucius4_prompt(text: str, lang: str) -> str:
    try:
        instruction = LANGUAGE_TOKEN_MAP[lang]
    except KeyError as exc:
        raise ValueError(f"Unsupported Confucius4-TTS language: {lang}") from exc
    return f"You are a helpful assistant. {instruction}:{text}"


@lru_cache(maxsize=4)
def _tokenizer(model_path: str):
    path = Path(model_path)
    if path.is_dir():
        root_config = path / "tokenizer_config.json"
        checkpoint_dir = path / "checkpoints"
        checkpoint_config = checkpoint_dir / "tokenizer_config.json"
        if root_config.is_file():
            tokenizer_path = str(path)
        elif checkpoint_config.is_file():
            tokenizer_path = str(checkpoint_dir)
        else:
            raise FileNotFoundError(f"Confucius4-TTS tokenizer_config.json was not found at {path} or {checkpoint_dir}")
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
        )

    try:
        return AutoTokenizer.from_pretrained(
            model_path,
        )
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained(
            model_path,
            subfolder="checkpoints",
        )


def tokenize_confucius4_prompt(model_path: str, text: str, lang: str) -> list[int]:
    """Tokenize the formatted prompt the way the canonical front-end does.

    The checkpoint was trained on ``<s> ... </s>``, and the tokenizer config declares
    ``add_bos_token``/``add_eos_token``. Their effect is not stable across transformers
    releases: 5.x no longer appends ``</s>`` for this config, which silently shortens
    the prompt and shifts every semantic position by one. Add both explicitly so the
    prompt is identical whatever transformers version serves the model.
    """
    formatted = format_confucius4_prompt(text, lang)
    tokenizer = _tokenizer(model_path)
    token_ids = list(tokenizer.encode(formatted, add_special_tokens=False))
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is not None and (not token_ids or token_ids[0] != bos):
        token_ids.insert(0, bos)
    if eos is not None and (not token_ids or token_ids[-1] != eos):
        token_ids.append(eos)
    return token_ids
