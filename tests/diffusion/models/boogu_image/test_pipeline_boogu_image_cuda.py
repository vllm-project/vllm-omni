# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Focused CUDA correctness coverage for Boogu-Image Turbo DMD helpers."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.boogu_image import BooguImageTurboPipeline

pytestmark = [pytest.mark.advanced_model, pytest.mark.cuda, pytest.mark.diffusion]


class _CudaDMDPipeline(BooguImageTurboPipeline):
    def predict(
        self,
        t,
        latents,
        instruction_embeds,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states=None,
    ):
        return torch.full_like(latents, 0.125)


def test_boogu_turbo_dmd_helpers_cuda_bf16():
    if not torch.cuda.is_available():
        pytest.skip("Boogu-Image Turbo DMD CUDA correctness requires a CUDA device")

    pipeline = object.__new__(_CudaDMDPipeline)
    nn.Module.__init__(pipeline)
    device = torch.device("cuda")
    latents = torch.full((2, 4, 8, 8), 2.0, device=device, dtype=torch.bfloat16)
    embeds = torch.zeros(2, 4, 8, device=device, dtype=torch.bfloat16)
    mask = torch.ones(2, 4, device=device, dtype=torch.long)

    sigmas = pipeline._build_dmd_student_sigmas(4, device, latents.dtype, 0.001)
    assert sigmas.device.type == "cuda"
    assert sigmas.dtype == torch.bfloat16

    x0 = pipeline._predict_dmd_student_step(latents, sigmas[0], embeds, None, mask)
    expected_x0 = latents + (1.0 - sigmas[0]) * torch.full_like(latents, 0.125)
    torch.testing.assert_close(x0, expected_x0)

    first_generator = torch.Generator(device=device).manual_seed(13)
    second_generator = torch.Generator(device=device).manual_seed(13)
    first = pipeline._renoise_dmd_latents(x0, sigmas[1], first_generator)
    second = pipeline._renoise_dmd_latents(x0, sigmas[1], second_generator)
    assert first.device.type == "cuda"
    assert torch.equal(first, second)
