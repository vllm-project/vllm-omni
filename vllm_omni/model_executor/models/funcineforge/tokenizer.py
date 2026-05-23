# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge tokenizer wrapper.

Uses the Qwen2 tokenizer from the LLM's pretrain directory, consistent
with FunCineForge's original usage of ``AutoTokenizer.from_pretrained``.
Adds FunCineForge-specific special tokens (startofclue, endofclue).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# FunCineForge special tokens added on top of Qwen's vocabulary
FUNCINEFORGE_SPECIAL_TOKENS = {
    "startofclue": 151646,
    "endofclue": 151647,
}


@lru_cache(maxsize=4)
def get_funcineforge_tokenizer(token_path: str) -> Any:
    """Get or create a cached FunCineForge tokenizer.

    Args:
        token_path: Path to the Qwen2 tokenizer directory
                    (e.g. ``model_dir/llm``).

    Returns:
        HuggingFace ``AutoTokenizer`` instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

    # Add FunCineForge-specific special tokens if not already present
    special_tokens_dict = {}
    for token_name, _token_id in FUNCINEFORGE_SPECIAL_TOKENS.items():
        token_str = f"<|{token_name}|>"
        if token_str not in tokenizer.get_vocab():
            special_tokens_dict[token_name] = token_str

    if special_tokens_dict:
        tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens_dict.values())})

    return tokenizer
