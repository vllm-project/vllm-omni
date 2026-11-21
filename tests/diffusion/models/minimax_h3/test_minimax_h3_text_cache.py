# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniMax-H3 text branch runs once per request, not once per denoising step."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _model():
    calls = []

    def condition_proj(rows):
        calls.append("proj")
        return rows * 2, None

    def token_refiner(x, *, cu_seqlens, max_seqlen, num_requests):
        calls.append("refiner")
        return x + 1

    return SimpleNamespace(condition_proj=condition_proj, token_refiner=token_refiner, _text_cache=None), calls


def _refined(model, text, cu, max_seqlen=4, num_requests=1):
    return MiniMaxH3DiTModel._refined_text(model, text, cu, max_seqlen, num_requests, torch.device("cpu"))


def test_same_inputs_reuse_the_refined_text():
    model, calls = _model()
    text, cu = torch.randn(4, 8), torch.tensor([0, 4, 4], dtype=torch.int32)
    first = _refined(model, text, cu)
    assert _refined(model, text, cu) is first
    assert calls == ["proj", "refiner"]
    assert torch.equal(first, (text.to(torch.bfloat16) * 2) + 1)


def test_new_prompt_tensor_or_layout_recomputes():
    model, calls = _model()
    text, cu = torch.randn(4, 8), torch.tensor([0, 4, 4], dtype=torch.int32)
    first = _refined(model, text, cu)
    assert _refined(model, text.clone(), cu) is not first
    assert _refined(model, text, cu.clone()) is not first
    assert _refined(model, text, cu, max_seqlen=5) is not first
    assert calls.count("refiner") == 4
