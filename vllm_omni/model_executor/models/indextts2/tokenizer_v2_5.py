# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS 2.5 multilingual tiktoken vocabulary.

The vocabulary layout mirrors the official IndexTTS 2.5 tokenizer: 58,836
mergeable ranks followed by 1,673 registered special tokens.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any

INDEXTTS25_TOKENIZER_FILE = "multilingual_zh_ja_yue_char_del.tiktoken"
INDEXTTS25_MERGEABLE_RANKS = 58836
INDEXTTS25_VOCAB_SIZE = 60509
INDEXTTS25_NUM_SPECIAL_LANGUAGES = 99

# Order is checkpoint ABI: lang_embedding rows were trained against these IDs.
LANGUAGES = (
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
    "yue",
    "minnan",
    "wuyu",
    "dialect",
    "zh/en",
    "en/zh",
    "common",
)
LANGUAGE_DICT = {lang: index for index, lang in enumerate(LANGUAGES)}

# ``zhen`` is an official mixed Chinese/English frontend mode, but it has no
# dedicated checkpoint language-embedding row or registered special token.
_FRONTEND_LANGUAGE_TO_EMBEDDING = {"zhen": "common"}

TO_LANGUAGE_CODE = {
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "korean": "ko",
    "french": "fr",
    "japanese": "ja",
    "portuguese": "pt",
    "turkish": "tr",
    "cantonese": "yue",
    # Intentional vLLM-Omni convenience alias; upstream would route the name
    # ``mandarin`` to the ``common`` language-embedding row.
    "mandarin": "zh",
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

_AUDIO_EVENTS = (
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
)
_EMOTIONS = ("HAPPY", "SAD", "ANGRY", "NEUTRAL")
_TTS_VOCAL_TOKENS = (
    "TTS/B",
    "TTS/O",
    "TTS/Q",
    "TTS/A",
    "TTS/CO",
    "TTS/CL",
    "TTS/H",
    *(f"TTS/SP{i:02d}" for i in range(1, 14)),
)

INDEXTTS25_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|startoftranscript|>",
    *(f"<|{lang}|>" for lang in LANGUAGES[:INDEXTTS25_NUM_SPECIAL_LANGUAGES]),
    *(f"<|{event}|>" for event in _AUDIO_EVENTS),
    *(f"<|{emotion}|>" for emotion in _EMOTIONS),
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    *(f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)),
    *(f"<|{token}|>" for token in _TTS_VOCAL_TOKENS),
    *(f"<|{i * 0.02:.2f}|>" for i in range(1501)),
)

_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def normalize_language_code(lang: str) -> str:
    normalized = lang.strip().lower()
    normalized = TO_LANGUAGE_CODE.get(normalized, normalized)
    if normalized not in LANGUAGE_DICT and normalized not in _FRONTEND_LANGUAGE_TO_EMBEDDING:
        supported = (*LANGUAGES, *sorted(_FRONTEND_LANGUAGE_TO_EMBEDDING))
        raise ValueError(f"Unsupported IndexTTS 2.5 language {lang!r}; expected one of {', '.join(supported)}")
    return normalized


def lang_to_token(lang: str) -> int:
    normalized = normalize_language_code(lang)
    embedding_language = _FRONTEND_LANGUAGE_TO_EMBEDDING.get(normalized, normalized)
    return LANGUAGE_DICT[embedding_language]


def resolve_indextts25_tokenizer_file(
    model_dir: str,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
) -> str:
    model_path = Path(model_dir).expanduser()
    for root in (model_path, model_path / "checkpoints"):
        direct = root / tokenizer_file
        if direct.is_file():
            return str(direct)
    if model_path.is_file() and model_path.name == tokenizer_file:
        return str(model_path)

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_dir, filename=tokenizer_file)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve IndexTTS 2.5 tokenizer {tokenizer_file!r} from {model_dir!r}"
        ) from exc


@lru_cache(maxsize=16)
def load_indextts25_encoding(
    model_dir: str,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
) -> Any:
    import tiktoken

    vocab_path = resolve_indextts25_tokenizer_file(model_dir, tokenizer_file)
    with open(vocab_path, encoding="utf-8") as vocab:
        ranks = {
            base64.b64decode(token): int(rank) for line in vocab if line.strip() for token, rank in (line.split(),)
        }

    if len(ranks) != INDEXTTS25_MERGEABLE_RANKS:
        raise ValueError(
            f"IndexTTS 2.5 tokenizer {vocab_path!r} has {len(ranks)} "
            f"mergeable ranks; expected {INDEXTTS25_MERGEABLE_RANKS}"
        )
    special_tokens = {token: len(ranks) + index for index, token in enumerate(INDEXTTS25_SPECIAL_TOKENS)}
    explicit_n_vocab = len(ranks) + len(special_tokens)
    if explicit_n_vocab != INDEXTTS25_VOCAB_SIZE:
        raise ValueError(
            f"IndexTTS 2.5 tokenizer builds vocab size {explicit_n_vocab}; expected {INDEXTTS25_VOCAB_SIZE}"
        )
    return tiktoken.Encoding(
        name=Path(vocab_path).name,
        explicit_n_vocab=explicit_n_vocab,
        pat_str=_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


def encode_indextts25_text(
    text: str,
    *,
    model_dir: str,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
) -> list[int]:
    encoding = load_indextts25_encoding(model_dir, tokenizer_file)
    return encoding.encode(text, allowed_special="all")
