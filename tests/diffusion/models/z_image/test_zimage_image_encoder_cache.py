# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import types

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    ZImagePipeline,
    _image_encoder_cache_key,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _LatentDistribution:
    def __init__(self):
        self.sample_calls = 0

    def sample(self, _generator):
        self.sample_calls += 1
        return torch.full((1, 4, 2, 2), float(self.sample_calls))


class _FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(shift_factor=0.0, scaling_factor=1.0)
        self.encode_calls = 0
        self.output = types.SimpleNamespace(latent_dist=_LatentDistribution())

    def encode(self, _image):
        self.encode_calls += 1
        return self.output


def _pipeline_with_fake_vae():
    pipeline = object.__new__(ZImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae = _FakeVae()
    pipeline.vae_scale_factor = 1
    pipeline.scheduler = types.SimpleNamespace(scale_noise=lambda image_latents, _timestep, _noise: image_latents)
    pipeline._prepared_image_cache = {}
    return pipeline


def test_image_encoder_cache_key_tracks_pixels_and_target_shape():
    device = torch.device("cpu")
    first = Image.new("RGB", (4, 4), (1, 2, 3))
    same_pixels = first.convert("RGBA")
    different = Image.new("RGB", (4, 4), (3, 2, 1))

    key = _image_encoder_cache_key(first, 16, 16, torch.bfloat16, device)
    assert key == _image_encoder_cache_key(same_pixels, 16, 16, torch.bfloat16, device)
    assert key != _image_encoder_cache_key(different, 16, 16, torch.bfloat16, device)
    assert key != _image_encoder_cache_key(first, 32, 16, torch.bfloat16, device)


def test_cached_prepared_image_still_encodes_and_resamples_each_request():
    pipeline = _pipeline_with_fake_vae()
    image = torch.zeros((1, 3, 2, 2))
    cache_key = (b"same-image",)
    cached = pipeline._get_or_cache_prepared_image(image, cache_key)

    first = pipeline.prepare_latents(
        1,
        4,
        2,
        2,
        torch.float32,
        torch.device("cpu"),
        None,
        image=cached,
        timestep=torch.zeros(1),
    )
    cached_again = pipeline._get_or_cache_prepared_image(None, cache_key)
    second = pipeline.prepare_latents(
        1,
        4,
        2,
        2,
        torch.float32,
        torch.device("cpu"),
        None,
        image=cached_again,
        timestep=torch.zeros(1),
    )

    assert cached_again is cached
    assert pipeline.vae.encode_calls == 2
    assert pipeline.vae.output.latent_dist.sample_calls == 2
    assert torch.equal(first, torch.ones_like(first))
    assert torch.equal(second, torch.full_like(second, 2.0))


def test_prepared_image_cache_is_bounded_and_lru():
    pipeline = _pipeline_with_fake_vae()
    image = torch.zeros((1, 3, 2, 2))

    first = pipeline._get_or_cache_prepared_image(image, ("first",))
    second = pipeline._get_or_cache_prepared_image(torch.ones_like(image), ("second",))

    assert first is not second
    assert list(pipeline._prepared_image_cache) == [("second",)]
