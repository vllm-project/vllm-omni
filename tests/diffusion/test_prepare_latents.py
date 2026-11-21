# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.utils import prepare_latents

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
        with pytest.raises(ValueError, match="does not match batch size"):
            prepare_latents((2, 4, 64, 64), dtype=torch.float32, device="cpu", generator=[torch.Generator()])

    def test_valid_generator_list(self):
        gens = [torch.Generator(), torch.Generator()]
        latents = prepare_latents((2, 4, 64, 64), dtype=torch.float32, device="cpu", generator=gens)
        assert latents.shape == (2, 4, 64, 64)

    def test_single_generator(self):
        gen = torch.Generator().manual_seed(42)
        latents = prepare_latents((1, 4, 32, 32), dtype=torch.float32, device="cpu", generator=gen)
        assert latents.shape == (1, 4, 32, 32)

    def test_same_seed_reproduces_latents(self):
        """The generator must be wired through to randn_tensor."""
        first = prepare_latents(
            (2, 4, 16, 16), dtype=torch.float32, device="cpu", generator=torch.Generator().manual_seed(7)
        )
        second = prepare_latents(
            (2, 4, 16, 16), dtype=torch.float32, device="cpu", generator=torch.Generator().manual_seed(7)
        )
        assert torch.equal(first, second)
