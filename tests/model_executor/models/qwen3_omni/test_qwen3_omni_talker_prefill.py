# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test for Qwen3-Omni talker prefill projection on empty segments.

``_thinker_to_talker_prefill`` walks the chatml segments and collects per-role
talker inputs. When a projected chatml carries no user/assistant segment that
the talker consumes (e.g. a system-only prompt, or an assistant turn that is
not the final segment), both collected lists stay empty and the previous
unconditional ``torch.cat([])`` raised ``ValueError: expected a non-empty list
of Tensors`` inside the speech pipeline.

The fix returns empty talker inputs shaped consistently with
``_get_talker_user_parts`` instead of crashing. Pure shape test: no weights.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 8
_IM_START = 1
_SYSTEM = 2
_USER = 3
_ASSISTANT = 4
_AUDIO = 5
_IMAGE = 6
_VIDEO = 7


def _make_model() -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.config = SimpleNamespace(
        im_start_token_id=_IM_START,
        system_token_id=_SYSTEM,
        user_token_id=_USER,
        assistant_token_id=_ASSISTANT,
        talker_config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=_HIDDEN)),
    )
    model.thinker_config = SimpleNamespace(
        audio_token_id=_AUDIO,
        image_token_id=_IMAGE,
        video_token_id=_VIDEO,
    )
    filler = torch.zeros(1, _HIDDEN)
    model._get_tts_embed = lambda *_a, **_k: (filler, filler.clone(), filler.clone())
    return model


def _prefill(model, chatml_row):
    input_ids = torch.tensor([chatml_row], dtype=torch.long)
    thinker_result_ids = torch.tensor([chatml_row], dtype=torch.long)
    embed = torch.zeros(len(chatml_row), _HIDDEN)
    return model._thinker_to_talker_prefill(
        thinker_embed=embed,
        thinker_hidden=embed,
        multimodal_mask=None,
        input_ids=input_ids,
        thinker_result_ids=thinker_result_ids,
        speaker_id=0,
    )


def test_system_only_chatml_returns_empty_talker_inputs():
    """No consumable segment -> empty tensors, not a torch.cat crash."""
    model = _make_model()
    talker_ids, talker_embeds, trailing = _prefill(
        model,
        [_IM_START, _SYSTEM, 10, 11, _IM_START, _SYSTEM],
    )
    assert talker_embeds.shape == (0, _HIDDEN)
    assert talker_embeds.dtype == torch.bfloat16
    assert talker_ids.shape == (0,)
    assert talker_ids.dtype == torch.long
    assert trailing is None


def test_empty_projection_result_survives_downstream_slicing():
    """Downstream slices [start:end]; empty result must slice without error."""
    model = _make_model()
    _talker_ids, talker_embeds, _trailing = _prefill(
        model,
        [_IM_START, _SYSTEM, 10, 11, _IM_START, _SYSTEM],
    )
    assert talker_embeds[0:4].shape[0] == 0
