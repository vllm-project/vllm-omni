# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav must preserve its model dtype across overridden reductions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_codec_mean_preserves_embedding_dtype_with_fp32_reduction(dtype, monkeypatch):
    model = object.__new__(Qwen3OmniMoeCode2Wav)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_quantizers=3)
    model.code_embedding = nn.Embedding(24, 4, dtype=dtype)
    model.register_buffer("code_offset", torch.arange(3).reshape(1, 3, 1) * 8)
    model.upsample = nn.ModuleList()
    model.decoder = nn.ModuleList()

    def check_transformer_input(*, inputs_embeds):
        assert inputs_embeds.dtype == dtype
        return SimpleNamespace(last_hidden_state=inputs_embeds)

    model.pre_transformer = check_transformer_input
    codes = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
    expected = model(codes)
    native_mean = torch.Tensor.mean

    def fp32_mean(tensor, *args, **kwargs):
        # vLLM 0.30 batch-invariant aten::mean.dim returns FP32 even for
        # BF16/FP16 input. Model dtype must not depend on that override.
        return native_mean(tensor.float(), *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "mean", fp32_mean)
    torch.testing.assert_close(model(codes), expected, rtol=0, atol=0)
