# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tokenizer invocation shared by Pi-family prompt processors."""

from __future__ import annotations


def tokenize_fixed_length(tokenizer, text: str, max_token_len: int):
    """Tokenize and right-pad ``text`` to exactly ``max_token_len`` tokens."""
    enc = tokenizer(
        text,
        padding="max_length",
        max_length=max_token_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    return list(enc["input_ids"]), list(enc["attention_mask"])
