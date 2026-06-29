# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt helpers for IndexTTS2 talker prefill."""

from __future__ import annotations

from functools import lru_cache

from vllm_omni.model_executor.models.indextts2.preprocess_utils import resolve_model_file
from vllm_omni.model_executor.models.indextts2.tokenizer import IndexTTS2Tokenizer

_CONDITIONING_PREFIX_TOKENS = 34
_TEXT_WRAPPER_TOKENS = 2
_START_MEL_TOKENS = 1


def _resolve_bpe_model_path(model_id_or_path: str) -> str:
    path = resolve_model_file(model_id_or_path, "bpe.model")
    if path is None:
        raise FileNotFoundError(f"Could not resolve bpe.model for {model_id_or_path!r}")
    return path


@lru_cache(maxsize=16)
def _get_text_tokenizer(model_id_or_path: str) -> IndexTTS2Tokenizer:
    return IndexTTS2Tokenizer(_resolve_bpe_model_path(model_id_or_path), model_dir=model_id_or_path)


def estimate_indextts2_prefill_prompt_len(model_id_or_path: str, text: str) -> int:
    """Return the placeholder prompt length expected by the IndexTTS2 talker.

    The model-side prefill replaces placeholder embeddings with:
    34 conditioning tokens, text tokens wrapped with start/stop markers, and
    one start-mel token.
    """
    tokenizer = _get_text_tokenizer(model_id_or_path)
    text_token_count = len(tokenizer.encode(text, add_special_tokens=False)) + _TEXT_WRAPPER_TOKENS
    return _CONDITIONING_PREFIX_TOKENS + text_token_count + _START_MEL_TOKENS


def build_indextts2_prefill_prompt_ids(
    model_id_or_path: str,
    text: str,
    *,
    placeholder_token_id: int = 1,
) -> list[int]:
    return [placeholder_token_id] * estimate_indextts2_prefill_prompt_len(model_id_or_path, text)
