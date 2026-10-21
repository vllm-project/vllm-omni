# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``residual_layer_norm`` matches the eager gated residual + LayerNorm chain."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.ops import residual_layer_norm

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_gated_residual_modulated_layer_norm(device: str) -> None:
    torch.manual_seed(0)
    channels = 64
    x = torch.randn(3, 7, channels, device=device)
    y = torch.randn(3, 7, channels, device=device)
    gate, scale, shift = (torch.randn(channels, device=device) for _ in range(3))
    expected_residual = x + gate * y
    expected = F.layer_norm(expected_residual, (channels,), eps=1e-6) * (1 + scale) + shift
    residual = x.clone()
    out = residual_layer_norm(residual, y, gate=gate, weight=1 + scale, bias=shift, eps=1e-6, residual_out=residual)
    torch.testing.assert_close(residual, expected_residual, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
