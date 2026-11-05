# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the token layout used by the Spark speech-code parser."""

from collections.abc import Mapping

# Spark-TTS-0.5B tokenizer revision 642071559bfc6346c2359d19dcb6be3f9dd8a05d.
GLOBAL_OFFSET = 151665
GLOBAL_SIZE = 4096
SEMANTIC_OFFSET = 155761
SEMANTIC_SIZE = 8192
START_GLOBAL = 165150
END_GLOBAL = 165156
STOP_TOKEN_ID = 151645


def validate_tokenizer_vocab(vocab: Mapping[str, int]) -> None:
    expected = {
        "<|start_global_token|>": START_GLOBAL,
        "<|end_global_token|>": END_GLOBAL,
        "<|im_end|>": STOP_TOKEN_ID,
    }
    for prefix, offset, size in (
        ("bicodec_global", GLOBAL_OFFSET, GLOBAL_SIZE),
        ("bicodec_semantic", SEMANTIC_OFFSET, SEMANTIC_SIZE),
    ):
        expected.update((f"<|{prefix}_{i}|>", offset + i) for i in range(size))
    for token, token_id in expected.items():
        actual = vocab.get(token)
        if actual != token_id:
            raise ValueError(
                f"Incompatible Spark-TTS tokenizer: {token} must have id {token_id}, got {actual}. "
                "Use the matching Spark-TTS-0.5B LLM tokenizer."
            )
