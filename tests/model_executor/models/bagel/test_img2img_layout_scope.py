# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The img2img context layout must apply only to img2img requests.

An img2img prompt packs VAE tokens, a separator and ViT tokens behind
``<|fim_middle|>`` placeholders; the AR model rewrites their positions and
routes the VAE tokens through the generation expert. The layout of the last
img2img image is reused when its embeddings come from the encoder cache (the
CFG companion of the same image), so a later request must carry the img2img
placeholder to get it; a long understanding or text request must not.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.models.bagel import BagelForConditionalGeneration

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMG2IMG = 7
IMAGE = 9
LAYOUT = (4, 5, 64, 64)


def _model() -> OmniBagelForConditionalGeneration:
    model = object.__new__(OmniBagelForConditionalGeneration)
    nn.Module.__init__(model)
    model._img2img_token_id = IMG2IMG
    model._pending_img2img_info = []
    model._last_img2img_info = LAYOUT
    model._ropes_pending = []
    model._vae_token_mask = None
    model._has_vae_tokens = False
    model._has_non_vae_tokens = True
    model._mot_forward = Mock(return_value="img2img")
    return model


@pytest.fixture
def plain_forward(monkeypatch: pytest.MonkeyPatch) -> Mock:
    forward = Mock(return_value="plain")
    monkeypatch.setattr(BagelForConditionalGeneration, "forward", forward)
    return forward


def test_long_request_after_img2img_keeps_the_plain_layout(plain_forward: Mock):
    model = _model()
    positions = torch.arange(12)

    out = model.forward(torch.tensor([IMAGE] * 10 + [1, 2]), positions)

    assert out == "plain"
    model._mot_forward.assert_not_called()
    assert torch.equal(plain_forward.call_args.args[1], torch.arange(12))
    assert model._ropes_pending[-1]["ropes"][0] == 12


def test_cached_companion_reuses_the_last_img2img_layout(plain_forward: Mock):
    model = _model()

    out = model.forward(torch.tensor([IMG2IMG] * 10 + [1, 2]), torch.arange(12))

    assert out == "img2img"
    plain_forward.assert_not_called()
    positions = model._mot_forward.call_args.args[1]
    assert positions.tolist() == [0] * 4 + [0] + [1] * 5 + [2, 3]


def test_mixed_batch_rewrites_only_the_img2img_request():
    model = _model()
    model._pending_img2img_info = [LAYOUT]
    input_ids = torch.tensor([IMG2IMG] * 10 + [1, 2] + [IMAGE] * 10 + [3, 4])
    positions = torch.cat([torch.arange(12), torch.arange(12)])

    out = model._adjust_positions_for_img2img(positions, input_ids)

    assert out[:12].tolist() == [0] * 4 + [0] + [1] * 5 + [2, 3]
    assert out[12:].tolist() == list(range(12))
    assert model._vae_token_mask[1:3].all() and not model._vae_token_mask[12:].any()
