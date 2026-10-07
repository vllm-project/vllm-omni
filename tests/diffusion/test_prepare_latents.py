# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch

from vllm_omni.diffusion.models.utils import prepare_latents


class TestPrepareLatents:
    def test_random_4d_image(self):
        latents = prepare_latents((2, 4, 64, 64), dtype=torch.float32, device="cpu")
        assert latents.shape == (2, 4, 64, 64)
        assert latents.dtype == torch.float32

    def test_random_5d_video(self):
        latents = prepare_latents((1, 16, 5, 30, 52), dtype=torch.bfloat16, device="cpu")
        assert latents.shape == (1, 16, 5, 30, 52)
        assert latents.dtype == torch.bfloat16

    def test_passthrough_casts_dtype(self):
        pre = torch.ones(1, 4, 32, 32, dtype=torch.float16)
        out = prepare_latents((1, 4, 32, 32), dtype=torch.float32, device="cpu", latents=pre)
        assert out.dtype == torch.float32
        assert out.shape == (1, 4, 32, 32)

    def test_generator_list_length_mismatch_raises(self):
        try:
            prepare_latents(
                (2, 4, 64, 64), dtype=torch.float32, device="cpu", generator=[torch.Generator()]
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "does not match batch size" in str(e)

    def test_valid_generator_list(self):
        gens = [torch.Generator(), torch.Generator()]
        latents = prepare_latents((2, 4, 64, 64), dtype=torch.float32, device="cpu", generator=gens)
        assert latents.shape == (2, 4, 64, 64)

    def test_single_generator(self):
        gen = torch.Generator().manual_seed(42)
        latents = prepare_latents((1, 4, 32, 32), dtype=torch.float32, device="cpu", generator=gen)
        assert latents.shape == (1, 4, 32, 32)
