# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.first_frame_decoder import Qwen3OmniFirstFrameDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [16, 19])
@torch.inference_mode()
def test_first_frame_graph_outputs_own_storage(dtype, rows):
    class Decoder(torch.nn.Linear):
        config = SimpleNamespace(num_quantizers=1)

        def forward(self, codes):
            return codes.to(dtype) * self.weight

    model = Decoder(1, 1, bias=False, device="cuda", dtype=dtype)
    model.weight.fill_(1)
    decoder = Qwen3OmniFirstFrameDecoder(model, sample_rate=24000)
    decoder.capture()
    codes = torch.arange(rows, device="cuda").reshape(-1, 1)
    first = decoder.decode(codes)
    second = decoder.decode(codes + 32)
    assert first.dtype == second.dtype == torch.float32
    torch.testing.assert_close(first, codes.float(), rtol=0, atol=0)
    torch.testing.assert_close(second, (codes + 32).float(), rtol=0, atol=0)
