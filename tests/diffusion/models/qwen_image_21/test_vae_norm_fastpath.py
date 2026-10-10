# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import QwenImage21RMS_norm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_training_preserves_autograd():
    norm = QwenImage21RMS_norm(8, images=False)
    x = torch.randn(1, 8, 1, 5, 7, requires_grad=True)
    expected = F.normalize(x, dim=1) * norm.scale * norm.gamma + norm.bias
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
    norm(x).sum().backward()
    assert x.grad is not None and norm.gamma.grad is not None


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    "channels,height,width", [(1152, 44, 80), (1152, 88, 160), (576, 176, 320), (288, 352, 640), (144, 704, 1280)]
)
@torch.no_grad()
def test_decoder_shapes_match_original_normalization(channels, height, width):
    torch.manual_seed(channels)
    norm = QwenImage21RMS_norm(channels, images=False).to("cuda", torch.bfloat16)
    norm.gamma.copy_(torch.randn_like(norm.gamma) * 0.05 + 1)
    x = torch.randn(1, channels, 1, height, width, device="cuda", dtype=torch.bfloat16)
    expected = F.normalize(x.float(), dim=1).to(x.dtype) * norm.scale * norm.gamma + norm.bias
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
