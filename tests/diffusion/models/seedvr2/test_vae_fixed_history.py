# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepared decoder history retains addresses and request-local semantics."""

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2.vae import Decoder3d, SeedVR2VAE, TemporalContext

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def decode_chunks(decoder, latent, context):
    outputs = []
    context.first = True
    for start in range(0, latent.shape[2], 2):
        outputs.append(decoder(latent[:, :, start : start + 2], context))
        context.first = False
    return torch.cat(outputs, dim=2)


@pytest.mark.parametrize("frames", [2, 3, 4, 5, 7])
def test_fixed_addresses_first_continuation_and_tail(frames):
    torch.manual_seed(177)
    decoder = Decoder3d((8, 8, 8, 8), groups=4, latent_channels=4).eval()
    latent = torch.randn(1, 4, frames, 4, 6)
    prepared = TemporalContext(cache_temporal=True)
    with torch.inference_mode():
        decoder(torch.zeros_like(latent[:, :, :2]), prepared)
        prepared.fixed_history = True
        addresses = {module: tensor.data_ptr() for module, tensor in prepared.history.items()}
        first = None
        for scale in [1.0, 2.0, 1.0]:
            value = latent * scale
            expected = decode_chunks(decoder, value, TemporalContext(cache_temporal=True))
            actual = decode_chunks(decoder, value, prepared)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert actual.shape[2] == 4 * frames - 3
            assert addresses == {module: tensor.data_ptr() for module, tensor in prepared.history.items()}
            if first is None:
                first = actual.clone()
            elif scale == 1:
                torch.testing.assert_close(actual, first, atol=0, rtol=0)


def test_first_phase_replaces_interrupted_request_history():
    torch.manual_seed(178)
    decoder = Decoder3d((8, 8, 8, 8), groups=4, latent_channels=4).eval()
    abandoned = torch.randn(1, 4, 5, 4, 6)
    next_request = torch.randn_like(abandoned)
    prepared = TemporalContext(cache_temporal=True)
    with torch.inference_mode():
        decoder(abandoned[:, :, :2], prepared)
        prepared.fixed_history = True
        prepared.first = False
        decoder(abandoned[:, :, 2:4], prepared)  # request is abandoned before its tail
        actual = decode_chunks(decoder, next_request, prepared)
        expected = decode_chunks(decoder, next_request, TemporalContext(cache_temporal=True))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("frames", [2, 5])
def test_prepared_history_matches_original_whole_clip_decode(frames):
    torch.manual_seed(179)
    model = SeedVR2VAE(channels=(8, 8, 8, 8), groups=4, latent_channels=4).eval()
    model.use_tiling = True
    latent = torch.randn(1, 4, frames, 4, 6)
    context = model._context(latent.shape[-2])
    context.cache_temporal = True
    with torch.inference_mode():
        model.decoder(torch.zeros_like(latent[:, :, :2]), context)
        context.fixed_history = True
        actual = decode_chunks(model.decoder, latent, context)
        # For T=2 the original decode never enables temporal caching at all.
        expected = model._decode_eager(latent)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
