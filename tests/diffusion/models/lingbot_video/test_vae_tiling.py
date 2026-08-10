# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.lingbot_video.vae_tiling import (
    LingBotVAETileGeometry,
    configure_lingbot_vae_tiling,
    normalize_lingbot_vae_tiling,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _default_geometry() -> LingBotVAETileGeometry:
    return LingBotVAETileGeometry(
        tile_sample_min_height=256,
        tile_sample_min_width=320,
        tile_sample_stride_height=192,
        tile_sample_stride_width=256,
    )


def test_default_profile_preserves_vae_geometry():
    geometry = _default_geometry()

    assert normalize_lingbot_vae_tiling({}, base_geometry=geometry) == geometry


def test_base_profile_overrides_selected_geometry_fields():
    geometry = normalize_lingbot_vae_tiling(
        {
            "lingbot_vae_tiling": {
                "base": {
                    "tile_sample_min_height": 384,
                    "tile_sample_stride_height": 256,
                }
            }
        },
        base_geometry=_default_geometry(),
    )

    assert geometry.as_enable_kwargs() == {
        "tile_sample_min_height": 384,
        "tile_sample_min_width": 320,
        "tile_sample_stride_height": 256,
        "tile_sample_stride_width": 256,
    }


@pytest.mark.parametrize(
    "model_config, match",
    [
        ({"lingbot_vae_tiling": {"unknown": {}}}, "Unsupported LingBot VAE tiling profiles"),
        ({"lingbot_vae_tiling": {"base": {"tile_sample_min_height": 250}}}, "divisible"),
        (
            {"lingbot_vae_tiling": {"base": {"tile_sample_stride_width": 384}}},
            "stride cannot exceed tile width",
        ),
    ],
)
def test_invalid_base_profiles_are_rejected(model_config, match):
    with pytest.raises(ValueError, match=match):
        normalize_lingbot_vae_tiling(model_config, base_geometry=_default_geometry())


class _FakeVAE:
    def __init__(self):
        self.enable_kwargs = None
        self.disable_calls = 0

    def enable_tiling(self, **kwargs):
        self.enable_kwargs = kwargs

    def disable_tiling(self):
        self.disable_calls += 1


def test_configure_enables_serial_tiling_with_normalized_geometry():
    vae = _FakeVAE()
    geometry = _default_geometry()

    configure_lingbot_vae_tiling(vae, enabled=True, geometry=geometry)

    assert vae.enable_kwargs == geometry.as_enable_kwargs()
    assert vae.disable_calls == 0


def test_configure_disables_serial_tiling():
    vae = _FakeVAE()

    configure_lingbot_vae_tiling(vae, enabled=False, geometry=_default_geometry())

    assert vae.enable_kwargs is None
    assert vae.disable_calls == 1


def test_configure_requires_enable_tiling_support():
    with pytest.raises(RuntimeError, match="requires a VAE with enable_tiling"):
        configure_lingbot_vae_tiling(object(), enabled=True, geometry=_default_geometry())
