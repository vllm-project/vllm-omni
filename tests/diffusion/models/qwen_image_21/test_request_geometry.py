# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21SequencePrepare

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@torch.no_grad()
def test_geometry_reuses_exact_positions_but_recomputes_latents():
    frequency = torch.arange(6 * 64, dtype=torch.float32).reshape(6, 64).to(torch.complex64)
    position = Mock(return_value=frequency)
    module = QwenImage21SequencePrepare(nn.Identity(), position)
    module._request_geometry = {}
    mask = torch.tensor([[False, False, True]])
    first = torch.randn(1, 4, 8)
    second = torch.randn(1, 4, 8)
    module(first, torch.randn(1, 2, 8), [(1, 2, 2)], mask, False)
    result = module(second, None, [(1, 2, 2)], mask, True)
    assert position.call_count == 1
    torch.testing.assert_close(result[1], second, rtol=0, atol=0)
    torch.testing.assert_close(result[3], frequency[2:], rtol=0, atol=0)
    module._request_geometry = {}
    module(first, torch.randn(1, 2, 8), [(1, 2, 2)], mask, False)
    assert position.call_count == 2
