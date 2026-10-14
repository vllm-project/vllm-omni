# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gepard-1.0 prompt layout.

``preprocess`` only consumes this layout — it injects the speaker prefix at the
placeholder slots and samples the first audio frame from the SOS position. The
layout itself is built here, by whichever entry point is assembling a request.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig


def build_gepard_prompt_ids(text_token_ids: list[int], *, config: GepardConfig) -> list[int]:
    """Assemble ``[speaker_slots] + [SOT, *text, EOT]*(R-1) + [SOT, *text, EOT, SOS]``.

    Only the final copy carries SOS, the learned "audio starts now" gate, so
    the earlier copies are read as context and never voiced. Short texts repeat
    because a couple of text tokens carry too little mass against the speaker
    prefix. This must match the training layout or WER collapses, which is why
    the thresholds come from the checkpoint's ``text_repetition`` block rather
    than from literals here.

    Raises ``ValueError`` on empty text: the layout would still be structurally
    valid and the model would voice it as arbitrary audio, so an empty request
    has to fail loudly. This is the only entry point in the offline path — there
    is no adapter above it to check first.
    """
    n = len(text_token_ids)
    if n == 0:
        raise ValueError("text_token_ids is empty; Gepard needs at least one text token to voice")
    if (
        not config.text_repetition_enabled
        or n >= config.text_repetition_apply_below
        or n >= config.text_repetition_target_tokens
    ):
        repeats = 1
    else:
        repeats = max(
            1,
            min(
                math.ceil(config.text_repetition_target_tokens / n),
                config.text_repetition_max_repeats,
            ),
        )

    block = [config.start_of_text, *text_token_ids, config.end_of_text]
    text_region = block * (repeats - 1) + [*block, config.start_of_speech]
    speaker_slots = [config.speaker_token_base + i for i in range(config.num_speaker_prefix)]
    return speaker_slots + text_region
