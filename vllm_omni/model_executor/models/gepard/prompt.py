# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gepard-1.0 prompt layout.

``preprocess`` only consumes this layout — it injects the speaker prefix at the
placeholder slots and samples the first audio frame from the SOS position. The
layout itself is built here, by whichever entry point is assembling a request.
"""

from __future__ import annotations

import math


def build_gepard_prompt_ids(
    text_token_ids: list[int],
    *,
    start_of_text: int,
    end_of_text: int,
    start_of_speech: int,
    speaker_token_base: int,
    num_speaker_prefix: int,
    target_text_tokens: int = 16,
    apply_below: int = 13,
    max_repeats: int = 8,
    repetition_enabled: bool = True,
) -> list[int]:
    """Assemble ``[speaker_slots] + [SOT, *text, EOT]*(R-1) + [SOT, *text, EOT, SOS]``.

    Only the final copy carries SOS, the learned "audio starts now" gate, so
    the earlier copies are read as context and never voiced. Short texts repeat
    because a couple of text tokens carry too little mass against the speaker
    prefix. This must match the training layout or WER collapses.
    """
    n = len(text_token_ids)
    if not repetition_enabled or n <= 0 or n >= apply_below or n >= target_text_tokens:
        R = 1
    else:
        R = max(1, min(math.ceil(target_text_tokens / n), max_repeats))

    block = [start_of_text, *text_token_ids, end_of_text]
    text_region = block * (R - 1) + [start_of_text, *text_token_ids, end_of_text, start_of_speech]
    speaker_slots = [speaker_token_base + i for i in range(num_speaker_prefix)]
    return speaker_slots + text_region
