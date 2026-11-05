# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Thinker rows the Talker reads as hidden states.

The Talker projects the accept-layer hidden state at audio, image and video
placeholders inside user turns and the word embedding everywhere else: system
turns are dropped and the final assistant turn is text-projected as a whole.
The Thinker -> Talker chunk 0 relies on this to send one merged prefill tensor.
"""

from collections.abc import Sequence

import torch
from transformers import PretrainedConfig


def talker_placeholder_token_ids(thinker_config: PretrainedConfig) -> tuple[int, int, int]:
    """Audio, image and video placeholder ids, in that order."""
    return thinker_config.audio_token_id, thinker_config.image_token_id, thinker_config.video_token_id


def talker_hidden_state_rows(prompt_token_ids: Sequence[int], config: PretrainedConfig) -> torch.Tensor:
    """Bool [P] mask of the prompt rows the Talker takes from the hidden state.

    Follows the segment walk of ``_thinker_to_talker_prefill``: a segment runs
    from one ``<|im_start|>`` to the next and its role is the token after it.
    """
    ids = torch.tensor(list(prompt_token_ids), dtype=torch.long)
    placeholder = torch.isin(ids, torch.tensor(talker_placeholder_token_ids(config.thinker_config)))
    rows = torch.zeros_like(placeholder)
    starts = torch.nonzero(ids == config.im_start_token_id).flatten().tolist()
    for start, end in zip(starts, [*starts[1:], len(ids)]):
        if start + 1 < len(ids) and int(ids[start + 1]) == config.user_token_id:
            rows[start:end] = placeholder[start:end]
    return rows
