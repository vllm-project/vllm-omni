# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Plan-B LatentForm table (pure tensor functions)."""

import pytest
import torch

from vllm_omni.diffusion.pid.latent_forms import (
    LATENT_FORMS,
    _identity,
    _patchify_and_normalize,
    _unpack_packed_2x2,
    lookup_latent_form,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_VSF = 8  # common vae_scale_factor


# -- _unpack_packed_2x2 (Flux / QwenImage family) ---------------------------


def _make_packed_latent(batch=1, height=512, width=512, channels=16):
    """Pack a [B, C, H, W] grid into Flux-style 2x2 packed token form.

    Strict inverse of ``_unpack_packed_2x2``: tokens [B, T, 4C].
    """
    b, c = batch, channels
    h = 2 * (height // (_VSF * 2))
    w = 2 * (width // (_VSF * 2))
    grid = torch.arange(b * c * h * w, dtype=torch.float32).reshape(b, c, h, w)
    tokens = grid.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), 4 * c)
    return tokens, grid


def test_unpack_packed_2x2_roundtrip():
    tokens, grid = _make_packed_latent()
    x0, pid_h, pid_w = _unpack_packed_2x2(tokens, 512, 512, _VSF)
    assert x0.shape == (1, 16, 64, 64)
    assert (pid_h, pid_w) == (512, 512)
    assert torch.equal(x0, grid)


def test_unpack_packed_2x2_accepts_grid_input():
    """4D grid latents pass through (defensive compatibility)."""
    grid = torch.zeros(2, 16, 64, 64)
    x0, pid_h, pid_w = _unpack_packed_2x2(grid, None, None, _VSF)
    assert x0.shape == grid.shape
    assert (pid_h, pid_w) == (512, 512)


def test_unpack_packed_2x2_squeezes_5d():
    latent = torch.zeros(1, 1024, 16).unsqueeze(2)  # not canonical; use grid form
    grid = torch.zeros(1, 16, 64, 64).unsqueeze(2)
    latent = grid
    x0, pid_h, pid_w = _unpack_packed_2x2(latent, None, None, _VSF)
    assert x0.dim() == 4


def test_unpack_packed_2x2_bad_dims_raise():
    with pytest.raises(ValueError, match="3D token form"):
        _unpack_packed_2x2(torch.zeros(1024, 16), 512, 512, _VSF)


def test_unpack_packed_2x2_requires_size():
    with pytest.raises(ValueError, match="non-None"):
        _unpack_packed_2x2(torch.zeros(1, 1024, 16), None, 512, _VSF)


def test_unpack_packed_2x2_token_mismatch_raises():
    tokens, _ = _make_packed_latent(height=512, width=512)
    with pytest.raises(ValueError, match="token count"):
        _unpack_packed_2x2(tokens, 1024, 1024, _VSF)


# -- _identity (ZImage / SD3 / SDXL family) ----------------------------------


def test_identity_passthrough():
    grid = torch.randn(2, 16, 64, 96)
    x0, pid_h, pid_w = _identity(grid, None, None, _VSF)
    assert x0 is grid
    assert (pid_h, pid_w) == (64 * _VSF, 96 * _VSF)


def test_identity_bad_dims_raise():
    with pytest.raises(ValueError, match="4D grid"):
        _identity(torch.zeros(1, 1024, 16), None, None, _VSF)


# -- _patchify_and_normalize (Flux2 / Flux2Klein family) ----------------------

_FLUX2_VAE_CH = 32  # VAE-ready grid channels (8x compression)


class _FakeBn:
    def __init__(self, num_channels: int = 128):
        torch.manual_seed(0)
        self.running_mean = torch.randn(num_channels)
        self.running_var = torch.rand(num_channels) + 0.5


class _FakeVaeConfig:
    batch_norm_eps = 1e-4


class _FakeVae:
    def __init__(self, num_channels: int = 128):
        self.bn = _FakeBn(num_channels)
        self.config = _FakeVaeConfig()


class _FakeFlux2Pipeline:
    def __init__(self):
        self.vae = _FakeVae()


def _fake_flux2_pipeline():
    return _FakeFlux2Pipeline()


def _denorm_and_unpatchify(x0: torch.Tensor, vae) -> torch.Tensor:
    """Exact inverse of _patchify_and_normalize (matches Flux2Pipeline)."""
    eps = vae.config.batch_norm_eps
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + eps)
    x = x0 * std + mean
    b, c, h, w = x.shape
    return x.reshape(b, c // (2 * 2), 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def test_patchify_and_normalize_shape_and_size():
    # 512x512 -> VAE-ready grid [1, 32, 64, 64] -> PiD x0 [1, 128, 32, 32]
    grid = torch.randn(1, _FLUX2_VAE_CH, 64, 64)
    x0, pid_h, pid_w = _patchify_and_normalize(grid, 512, 512, _VSF, pipeline=_fake_flux2_pipeline())
    assert x0.shape == (1, 128, 32, 32)
    # LDM pixel size: VAE-ready grid * vae_scale_factor (8x)
    assert (pid_h, pid_w) == (512, 512)


def test_patchify_and_normalize_roundtrip():
    """denorm(unpatchify(x0)) must recover the pipeline's original latent."""
    grid = torch.randn(2, _FLUX2_VAE_CH, 64, 48, dtype=torch.float32)
    vae = _FakeVae()
    x0, _, _ = _patchify_and_normalize(grid, None, None, _VSF, pipeline=_FakeFlux2Pipeline())
    recovered = _denorm_and_unpatchify(x0, vae)
    # fp32 roundtrip: (x - m)/s then *s + m has ~1e-7 relative rounding.
    assert torch.allclose(recovered, grid, atol=1e-5)


def test_patchify_and_normalize_bn_math():
    """x0 == (patchify(grid) - mean) / sqrt(var + eps), computed in fp32."""
    grid = torch.randn(1, _FLUX2_VAE_CH, 32, 32, dtype=torch.float32)
    vae = _FakeVae()
    eps = vae.config.batch_norm_eps
    mean = vae.bn.running_mean
    std = torch.sqrt(vae.bn.running_var + eps)

    x0, _, _ = _patchify_and_normalize(grid, None, None, _VSF, pipeline=_FakeFlux2Pipeline())

    # patchify channel order: c*4 + ph*2 + pw
    b, c, z_h, z_w = grid.shape
    packed = grid.view(b, c, z_h // 2, 2, z_w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, z_h // 2, z_w // 2)
    expected = (packed - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    assert torch.allclose(x0, expected, atol=1e-6)


def test_patchify_and_normalize_missing_pipeline_raises():
    grid = torch.randn(1, _FLUX2_VAE_CH, 64, 64)
    with pytest.raises(ValueError, match="requires the pipeline"):
        _patchify_and_normalize(grid, None, None, _VSF, pipeline=None)


def test_patchify_and_normalize_pipeline_without_vae_bn_raises():
    class _NoBn:
        pass

    pipe = _NoBn()
    grid = torch.randn(1, _FLUX2_VAE_CH, 64, 64)
    with pytest.raises(ValueError, match="requires the pipeline"):
        _patchify_and_normalize(grid, None, None, _VSF, pipeline=pipe)


def test_patchify_and_normalize_bad_dims_raise():
    with pytest.raises(ValueError, match="4D VAE-ready grid"):
        _patchify_and_normalize(
            torch.zeros(1, 128, 32, 32).unsqueeze(2), None, None, _VSF, pipeline=_fake_flux2_pipeline()
        )


def test_patchify_and_normalize_odd_spatial_raises():
    grid = torch.randn(1, _FLUX2_VAE_CH, 63, 64)
    with pytest.raises(ValueError, match="even"):
        _patchify_and_normalize(grid, None, None, _VSF, pipeline=_fake_flux2_pipeline())


# -- registry lookup ----------------------------------------------------------


def test_registered_families_cover_expected_backbones():
    assert LATENT_FORMS["FluxPipeline"].backbone == "flux"
    assert LATENT_FORMS["QwenImagePipeline"].backbone == "qwenimage"
    assert LATENT_FORMS["ZImagePipeline"].backbone == "flux"
    assert LATENT_FORMS["Flux2Pipeline"].backbone == "flux2"
    assert LATENT_FORMS["Flux2KleinPipeline"].backbone == "flux2"


def test_flux2_family_uses_patchify_and_normalize():
    """Flux2 家族共享同一 VAE-ready -> BN patchify 转换。"""
    assert LATENT_FORMS["Flux2Pipeline"].to_x0 is _patchify_and_normalize
    assert LATENT_FORMS["Flux2KleinPipeline"].to_x0 is _patchify_and_normalize


class _FakePipeline:
    pass


class _FakeKlein(_FakePipeline):
    pass


def test_lookup_latent_form_resolves_class_name():
    assert lookup_latent_form(_FakePipeline()) is None


def test_lookup_latent_form_follows_mro():
    """Unregistered subclasses resolve via MRO (family inheritance)."""
    from vllm_omni.diffusion.pid.latent_forms import LatentForm

    form = LatentForm("flux", _identity)
    LATENT_FORMS["_FakePipeline"] = form
    try:
        assert lookup_latent_form(_FakeKlein()) is form
    finally:
        del LATENT_FORMS["_FakePipeline"]
