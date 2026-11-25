# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.phonon.mimi_decoder import causal_conv_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_convolution_chunking_and_history_aliasing():
    generator = torch.Generator().manual_seed(29)
    inputs = torch.randn(2, 3, 17, generator=generator)
    weight = torch.randn(4, 3, 3, generator=generator)
    bias = torch.randn(4, generator=generator)
    history = torch.zeros(4, 3, 4)
    slots = torch.tensor([2, 0])
    outputs = [
        causal_conv_step(chunk, weight, bias, history, slots, dilation=2)
        for chunk in inputs.split([1, 3, 2, 11], dim=-1)
    ]
    scratch_history = torch.zeros_like(history)
    scratch_outputs = []
    for chunk in inputs.split([1, 3, 2, 11], dim=-1):
        scratch = torch.empty(2, 3, 4 + chunk.shape[-1])
        scratch_outputs.append(
            causal_conv_step(
                chunk,
                weight,
                bias,
                scratch_history,
                slots,
                dilation=2,
                scratch=scratch,
            )
        )
    torch.testing.assert_close(torch.cat(scratch_outputs, -1), torch.cat(outputs, -1))
    torch.testing.assert_close(scratch_history, history)
    expected = F.conv1d(F.pad(inputs, (4, 0)), weight, bias, dilation=2)
    torch.testing.assert_close(torch.cat(outputs, -1), expected)
    torch.testing.assert_close(history[slots], inputs[..., -4:])
    assert history[1].count_nonzero() == 0
