# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt helpers for IndexTTS2 talker prefill."""

from __future__ import annotations

from functools import lru_cache

from vllm_omni.model_executor.models.indextts2.preprocess_utils import resolve_model_file
from vllm_omni.model_executor.models.indextts2.text_processing_v2_5 import (
    prepare_indextts25_text,
)
from vllm_omni.model_executor.models.indextts2.tokenizer import IndexTTS2Tokenizer
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
)

_V2_CONDITIONING_PREFIX_TOKENS = 34
_V25_CONDITIONING_PREFIX_TOKENS = 3
_V2_TEXT_WRAPPER_TOKENS = 2
_V25_TEXT_WRAPPER_TOKENS = 2
_V25_TEXT_WRAPPER_TOKEN_IDS = frozenset({0, 1})
_START_MEL_TOKENS = 1


def _resolve_bpe_model_path(model_id_or_path: str) -> str:
    path = resolve_model_file(model_id_or_path, "bpe.model")
    if path is None:
        raise FileNotFoundError(f"Could not resolve bpe.model for {model_id_or_path!r}")
    return path


@lru_cache(maxsize=16)
def _get_text_tokenizer(model_id_or_path: str) -> IndexTTS2Tokenizer:
    return IndexTTS2Tokenizer(_resolve_bpe_model_path(model_id_or_path), model_dir=model_id_or_path)


def estimate_indextts2_prefill_prompt_len(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
) -> int:
    """Return the placeholder prompt length expected by the IndexTTS2 talker.

    IndexTTS 2 uses 34 conditioning tokens. IndexTTS 2.5 uses one projected
    CAMPPlus speaker token followed by two zero tokens.
    """
    if model_type == "indextts2_5":
        text_ids, _ = prepare_indextts25_text(
            text,
            lang=lang,
            model_dir=model_id_or_path,
            text_normalization=text_normalization,
            tokenizer_file=tokenizer_file,
        )
        conditioning_prefix_tokens = _V25_CONDITIONING_PREFIX_TOKENS
        text_token_count = (
            sum(token_id not in _V25_TEXT_WRAPPER_TOKEN_IDS for token_id in text_ids) + _V25_TEXT_WRAPPER_TOKENS
        )
    else:
        tokenizer = _get_text_tokenizer(model_id_or_path)
        text_token_count = len(tokenizer.encode(text, add_special_tokens=False)) + _V2_TEXT_WRAPPER_TOKENS
        conditioning_prefix_tokens = _V2_CONDITIONING_PREFIX_TOKENS
    return conditioning_prefix_tokens + text_token_count + _START_MEL_TOKENS


def build_indextts2_prefill_prompt_ids(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
    placeholder_token_id: int = 1,
) -> list[int]:
    prompt_len = estimate_indextts2_prefill_prompt_len(
        model_id_or_path,
        text,
        model_type=model_type,
        lang=lang,
        text_normalization=text_normalization,
        tokenizer_file=tokenizer_file,
    )
    return [placeholder_token_id] * prompt_len
