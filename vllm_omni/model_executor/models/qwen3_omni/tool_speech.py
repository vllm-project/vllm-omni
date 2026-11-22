# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stop the Talker at a Thinker tool call.

Qwen3-Omni encodes the Hermes ``<tool_call>`` marker as one token. Codec
packets are not aligned to that token, so the cut has to happen on the token
ids before they become Talker text.
"""

from collections.abc import Sequence

# tokenizer_config.json of Qwen3-Omni-30B-A3B-Instruct: 151657 = "<tool_call>".
TOOL_CALL_START_TOKEN_ID = 151657


def tool_call_start_offset(token_ids: Sequence[int]) -> int | None:
    """Index of the first ``<tool_call>`` token, or None."""
    for offset, token_id in enumerate(token_ids):
        try:
            if int(token_id) == TOOL_CALL_START_TOKEN_ID:
                return offset
        except (TypeError, ValueError):
            continue
    return None


def spoken_span_end(token_ids: Sequence[int], start: int, end: int) -> int:
    """Exclusive end of ``token_ids[start:end]`` with the tool call removed.

    The token at the marker and everything after it are not speech. The
    preface before the marker stays.
    """
    offset = tool_call_start_offset(token_ids[start:end])
    if offset is None:
        return end
    return start + offset
