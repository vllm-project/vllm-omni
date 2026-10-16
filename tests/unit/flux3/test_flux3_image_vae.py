# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Flux3ImageVAE (packed image autoencoder).

These tests run on CPU with random weights — no checkpoint required — and
verify the shape contracts, the internal BatchNorm + pixel-shuffle packing, and
the pixel-latent passthrough used by the pipeline's ``output_type="latent"``
path.
"""

import pytest
import torch

from vllm_omni.diffusion.models.flux3.flux3_image_vae import (
    AutoEncoder,
    AutoEncoderParams,
    Flux3ImageVAE,
)


@pytest.fixture(scope="module")
def vae() -> Flux3ImageVAE:
    torch.manual_seed(0)
    return Flux3ImageVAE(device="cpu")


def _image(h: int = 32, w: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, h, w, generator=g)


# ---------------------------------------------------------------------------
# Shape contracts
# ---------------------------------------------------------------------------


class TestImageVAEShapes:
    def test_config(self, vae):
        assert vae.config.block_out_channels == [128, 256, 512, 512]
        assert vae.config.latent_channels == 32
        assert vae.config.ps == [2, 2]
        # scale-factor math the pipeline relies on: 2**(len(block_out)-1) == 8
        assert 2 ** (len(vae.config.block_out_channels) - 1) == 8

    def test_encode_shape(self, vae):
        # 8x conv downsample + 2x pixel-shuffle pack = 16x, z_channels*4 = 128
        z = vae.encode(_image(32, 32))
        assert z.shape == (1, 128, 2, 2), f"got {tuple(z.shape)}"

    def test_encode_shape_larger(self, vae):
        z = vae.encode(_image(256, 256))
        assert z.shape == (1, 128, 16, 16), f"got {tuple(z.shape)}"

    def test_decode_roundtrip_shape(self, vae):
        z = vae.encode(_image(32, 32))
        rec = vae.decode(z)
        assert rec.shape == (1, 3, 32, 32), f"got {tuple(rec.shape)}"

    def test_to_pixel_latents_shape(self, vae):
        # (B, C*4, h, w) -> (B, C, 2h, 2w) without conv decode
        z = vae.encode(_image(32, 32))
        lat = vae.to_pixel_latents(z)
        assert lat.shape == (1, 32, 4, 4), f"got {tuple(lat.shape)}"

    def test_encode_rejects_wrong_ndim(self, vae):
        with pytest.raises(ValueError):
            vae.encode(torch.randn(3, 32, 32))


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------


class TestImageVAENumerics:
    def test_encode_finite(self, vae):
        assert torch.isfinite(vae.encode(_image())).all()

    def test_decode_finite(self, vae):
        assert torch.isfinite(vae.decode(vae.encode(_image()))).all()

    def test_encode_deterministic(self, vae):
        img = _image(seed=7)
        assert torch.equal(vae.encode(img), vae.encode(img))

    def test_to_pixel_latents_inverts_pack_and_norm(self, vae):
        # inv_normalize + unpack should invert normalize + pack exactly (the
        # decoder is not involved), so encode's internal packed->pixel roundtrip
        # recovers the encoder mean up to BN precision.
        img = _image()
        z = vae.encode(img)
        pixel = vae.to_pixel_latents(z)
        # re-pack + re-normalise must return the packed latents.
        repacked = vae._model.normalize(vae._model.pack(pixel))
        assert torch.allclose(repacked, z, atol=1e-4, rtol=1e-4)

    def test_wrapper_matches_raw_autoencoder(self, vae):
        # The wrapper must be a thin pass-through over the ported AutoEncoder.
        raw = AutoEncoder(AutoEncoderParams()).eval()
        raw.load_state_dict(vae._model.state_dict(), strict=True)
        img = _image()
        assert torch.equal(vae.encode(img), raw.encode(img))
