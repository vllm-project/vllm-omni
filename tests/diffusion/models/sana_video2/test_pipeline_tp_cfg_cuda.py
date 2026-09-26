# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from tests.diffusion.distributed.test_sana_video2_tp_cfg_pipeline import _VAE, _model
from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline

pytestmark = [pytest.mark.cuda, pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bfloat16_branch_preserves_flow_and_noise_space_on_cuda():
    torch.manual_seed(8104)
    device = torch.device("cuda:0")
    model = _model().to(device=device, dtype=torch.bfloat16)
    pipeline = SanaVideo2Pipeline(tokenizer=object(), text_encoder=nn.Identity(), vae=_VAE(), transformer=model)
    x = torch.randn(1, 128, 3, 1, 3, device=device)
    embeddings = torch.randn(1, 4, 16, device=device)
    mask = torch.tensor([[True, True, False, False]], device=device)

    frame_time = torch.tensor([0.0, 500.0, 750.0], device=device).reshape(1, 1, 3, 1, 1)
    flow = pipeline.predict_noise(x=x, time=frame_time, embeddings=embeddings, mask=mask)
    assert flow.dtype == torch.bfloat16

    sigma = torch.tensor(0.25, device=device)
    noise = pipeline.predict_noise(x=x, time=sigma, embeddings=embeddings, mask=mask, noise_space=True)
    raw = pipeline.predict_noise(x=x, time=sigma, embeddings=embeddings, mask=mask)
    expected = (1 - sigma.reshape((1,) * x.ndim).to(x)) * raw + x
    assert raw.dtype == torch.bfloat16
    assert noise.dtype == torch.float32
    torch.testing.assert_close(noise, expected, rtol=0, atol=0)
