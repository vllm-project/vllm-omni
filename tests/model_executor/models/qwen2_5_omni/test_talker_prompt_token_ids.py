# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the talker pad-token remap in Qwen2.5-Omni ``forward``.

``SamplingMetadata.prompt_token_ids`` is populated lazily and is ``None`` unless
a sampler needs it, so the remap must tolerate a missing tensor.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEXT_PAD_ID = 152064
TALKER_PAD_ID = 8448


def _run_talker_forward(sampling_metadata):
    talker_hidden = torch.zeros(3, 4)
    fake_self = SimpleNamespace(
        model_stage="talker",
        talker=lambda **_: talker_hidden,
        voice_type="Chelsie",
    )
    out = Qwen2_5OmniForConditionalGeneration.forward(
        fake_self,
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3).unsqueeze(0),
        inputs_embeds=torch.zeros(3, 4),
        sampling_metadata=sampling_metadata,
    )
    assert out.text_hidden_states is talker_hidden
    return out


def test_talker_forward_without_sampling_metadata():
    _run_talker_forward(None)


def test_talker_forward_with_missing_prompt_token_ids():
    sampling_metadata = SimpleNamespace(prompt_token_ids=None)

    _run_talker_forward(sampling_metadata)

    assert sampling_metadata.prompt_token_ids is None


def test_talker_forward_remaps_pad_token_in_place():
    prompt_token_ids = torch.tensor([[5, TEXT_PAD_ID, 7, TEXT_PAD_ID], [TEXT_PAD_ID, 9, 11, 13]])
    sampling_metadata = SimpleNamespace(prompt_token_ids=prompt_token_ids)

    _run_talker_forward(sampling_metadata)

    expected = torch.tensor([[5, TALKER_PAD_ID, 7, TALKER_PAD_ID], [TALKER_PAD_ID, 9, 11, 13]])
    assert torch.equal(sampling_metadata.prompt_token_ids, expected)
