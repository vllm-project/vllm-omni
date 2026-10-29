# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Each batched HunyuanImage3 image samples from its own VAE seed."""

import pytest
import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3ForConditionalGeneration,
    _vae_generators_for_images,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _draw(generator: torch.Generator) -> torch.Tensor:
    return torch.randn(8, generator=generator)


def test_each_image_matches_a_standalone_seed():
    seeds = torch.tensor([11, 22, 11])
    generators = _vae_generators_for_images(seeds, 3, torch.device("cpu"))
    draws = [_draw(generator) for generator in generators]

    def alone(seed: int) -> torch.Tensor:
        return _draw(torch.Generator().manual_seed(seed))

    assert torch.equal(draws[0], alone(11))
    assert torch.equal(draws[1], alone(22))
    # Same seed as the first image, but not a continuation of its stream.
    assert torch.equal(draws[2], alone(11))


def test_embed_multimodal_uses_one_fresh_generator_per_image():
    recorded: list[int | None] = []

    def vae_encode(image, cfg_factor, generator=None):
        recorded.append(None if generator is None else generator.initial_seed())
        if generator is not None:
            torch.randn(1, generator=generator)
        return (torch.zeros(1),), torch.zeros(1, 4, 2, 2)

    host = HunyuanImage3ForConditionalGeneration.__new__(HunyuanImage3ForConditionalGeneration)
    host._parse_and_validate_image_input = lambda **kwargs: {
        "pixel_values": {
            "vit_pixel_values": torch.zeros(2, 1),
            "vit_pixel_attention_mask": None,
            "vit_spatial_shapes": None,
            "vae_pixel_values": [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)],
            "vae_generator_seed": torch.tensor([11, 22]),
        }
    }
    host._vit_encode = lambda *args, **kwargs: torch.zeros(2, 2, 4)
    host._vae_encode = vae_encode
    host.time_embed = lambda timestep: torch.zeros(1, 4)
    host.patch_embed = lambda latents, timestep_embed: (
        torch.zeros(1, 2, 4),
        None,
        None,
    )
    host._mrope_joint_img_sep_token_id = 1

    class _Model:
        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 4)

    host.model = _Model()

    embeddings = HunyuanImage3ForConditionalGeneration.embed_multimodal(host)
    assert len(embeddings) == 2
    assert recorded == [11, 22]
