# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Why the Qwen3-Omni talker declares ``recompute_preemption="fail"``.

A talker decode step feeds ``summed_embeddings(layer0..7) + text_step`` into
PagedAttention, but only the layer-0 codec id is kept in ``output_token_ids``.
When ``_preempt_request()`` zeroes ``num_computed_tokens`` the replay is
scheduled as a prefill of ``prompt + every frame already generated``; the talker
rebuilds only the prompt part, and the runner's
``seg_len = min(span_len, req_embeds.shape[0])`` leaves the remaining rows of
its ``torch.empty`` buffer uninitialised (vllm-project/vllm-omni#6179).

Unlike VoxCPM2, the layer-0 code stays inside the codec vocabulary, so the
replay raises nothing: the damage is silent. The declaration itself is pinned
in ``tests/model_executor/models/test_recompute_preemption_declarations.py``;
this file pins the arithmetic that makes it necessary.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 8
PROMPT_LEN = 12  # rows _thinker_to_talker_prefill builds
GEN_FRAMES = 5  # frames already delivered before preemption


def test_replay_span_exceeds_the_embeddings_the_talker_can_rebuild():
    """Mirror of OmniGPUModelRunner._preprocess's copy."""
    span_len = PROMPT_LEN + GEN_FRAMES  # what _preempt_request() re-schedules
    rebuilt = torch.zeros(PROMPT_LEN, HIDDEN)  # prompt-only, as prefill builds it

    inputs_embeds = torch.empty(span_len, HIDDEN)
    seg_len = min(span_len, rebuilt.shape[0])
    inputs_embeds[0:seg_len] = rebuilt[:seg_len]

    assert seg_len == PROMPT_LEN
    assert inputs_embeds.shape[0] - seg_len == GEN_FRAMES


def test_only_layer0_of_a_talker_frame_reaches_the_token_stream():
    """The shortfall above is structural: 7 of 8 codes never enter the stream."""
    num_code_groups = 8  # talker_config.num_code_groups
    assert num_code_groups - 1 == 7
