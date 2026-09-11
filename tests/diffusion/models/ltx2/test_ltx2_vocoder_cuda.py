# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU regression tests for LTX-2 BWE vocoder precision."""

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class _BWEConvVocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Match the production LTX vocoder's conv_pre geometry. The previous
        # single-channel 1x1 probe intermittently had no MIOpen algorithm.
        self.conv = torch.nn.Conv1d(
            128,
            1024,
            kernel_size=7,
            padding=3,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.bwe_generator = torch.nn.Identity()
        self.input_dtype = None
        self.conv_output_dtype = None

    def forward(self, value):
        self.input_dtype = value.dtype
        output = self.conv(value)
        self.conv_output_dtype = output.dtype
        return output


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_ltx_bwe_vocoder_runs_real_cuda_autocast_in_fp32():
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _run_ltx_vocoder

    vocoder = _BWEConvVocoder()
    generated_mel = torch.ones((1, 128, 64), device="cuda", dtype=torch.bfloat16)

    output = _run_ltx_vocoder(vocoder, generated_mel)

    assert next(vocoder.parameters()).dtype == torch.bfloat16
    assert vocoder.input_dtype == torch.float32
    assert vocoder.conv_output_dtype == torch.float32
    assert output.dtype == torch.bfloat16
