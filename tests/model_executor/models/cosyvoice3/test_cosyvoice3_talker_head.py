# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The talker head is padded to an 8-aligned output only under batch invariance."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("batch_invariant", "out_features", "padded_rows"),
    [
        (False, 6761, None),  # default mode keeps the unpadded head
        (True, 6761, 6768),
        (True, 6768, None),  # already aligned
    ],
)
def test_speech_head_padding(monkeypatch, batch_invariant, out_features, padded_rows):
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.cosyvoice3.cosyvoice3.envs.VLLM_BATCH_INVARIANT", batch_invariant
    )
    torch.manual_seed(0)
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.llm_decoder = nn.Linear(16, out_features, bias=False)
    model._pad_speech_head_for_batch_invariance()

    x = torch.randn(3, 16)
    out = model._decode_speech_logits(x)
    assert out.shape == (3, out_features)
    torch.testing.assert_close(out, model.model.llm_decoder(x))
    if padded_rows is None:
        assert model._padded_head_weight is None
    else:
        assert model._padded_head_weight.shape == (padded_rows, 16)
        assert torch.equal(model._padded_head_weight[out_features:], torch.zeros(padded_rows - out_features, 16))
        assert model._padded_head_bias is None
