# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
    HiggsAudioV3TalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fallback", [None, "prefill", "logprobs", "allowed", "bad_words", "penalty", "inactive"])
def test_text_head_only_skipped_for_direct_audio_decode(fallback):
    hidden = torch.randn(3, 4)
    calls = []

    def logits_processor(head, h):
        calls.append(h)
        return torch.ones(3, 7)

    model = SimpleNamespace(
        _audio_continuation_id=5,
        _last_step_input_ids=torch.ones(4 if fallback == "prefill" else 3, dtype=torch.long),
        _fast_audio_direct_rows=0 if fallback == "inactive" else 3,
        _text_vocab_size=7,
        lm_head=None,
        logits_processor=logits_processor,
    )
    metadata = SimpleNamespace(
        no_penalties=fallback != "penalty",
        max_num_logprobs=1 if fallback == "logprobs" else None,
        allowed_token_ids_mask=torch.ones(3, 7) if fallback == "allowed" else None,
        bad_words_token_ids={0: [1]} if fallback == "bad_words" else {},
    )
    logits = HiggsAudioV3TalkerForConditionalGeneration.compute_logits(model, hidden, metadata)
    assert logits.shape == (3, 7)
    assert model._last_logits_hidden is hidden
    assert len(calls) == (fallback is not None)
