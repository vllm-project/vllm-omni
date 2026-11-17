# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.utils import load_vae_scale_factor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_vae_config(tmp_path, data):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_standard_block_out_channels(tmp_path):
    path = _make_vae_config(tmp_path, {"block_out_channels": [128, 256, 512, 512]})
    assert load_vae_scale_factor(path) == 8


def test_temporal_downsample(tmp_path):
    """Video VAEs read ``temporal_downsample`` with exponent offset 0."""
    path = _make_vae_config(tmp_path, {"temporal_downsample": [True, True, True]})
    assert load_vae_scale_factor(path, config_key="temporal_downsample", exponent_offset=0) == 8


def test_exponent_offset_zero(tmp_path):
    path = _make_vae_config(tmp_path, {"block_out_channels": [128, 256, 512, 512]})
    assert load_vae_scale_factor(path, exponent_offset=0) == 16


def test_missing_config_raises(tmp_path):
    """A missing config file must fail loudly, not silently assume a default."""
    with pytest.raises(FileNotFoundError, match="VAE config not found"):
        load_vae_scale_factor(str(tmp_path))


def test_corrupt_config_raises_with_path(tmp_path):
    """An unparsable config must raise with the offending path."""
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match="Failed to parse VAE config"):
        load_vae_scale_factor(str(tmp_path))


def test_missing_key_returns_default(tmp_path):
    path = _make_vae_config(tmp_path, {"other_key": "value"})
    assert load_vae_scale_factor(path) == 8
    assert load_vae_scale_factor(path, default=16) == 16


def test_empty_block_out_channels_returns_default(tmp_path):
    """An empty channel list must not produce a fractional scale factor."""
    path = _make_vae_config(tmp_path, {"block_out_channels": []})
    assert load_vae_scale_factor(path) == 8


def test_preloaded_config_skips_file_read(tmp_path):
    """Callers holding the parsed config can pass it to avoid a second read."""
    assert load_vae_scale_factor(str(tmp_path), vae_config={"block_out_channels": [32]}) == 1


def test_migrated_factories_pass_scale_factor_to_processor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Pin the refactoring at the call-site level: the helper's result must
    reach the image processor through the migrated factories (including the
    factories' ``* 2`` multiplier)."""
    model_path = tmp_path / "model"
    vae_path = model_path / "vae"
    vae_path.mkdir(parents=True)
    # One block: 2 ** (1 - 1) = 1; the factories double it to 2.
    (vae_path / "config.json").write_text(json.dumps({"block_out_channels": [32]}))

    captured: dict[str, object] = {}

    class FakeProcessor:
        def __init__(self, *, vae_scale_factor: int, **kwargs) -> None:
            captured["vae_scale_factor"] = vae_scale_factor

        def postprocess(self, images):
            return images

    import vllm_omni.diffusion.models.flux2.pipeline_flux2 as flux2
    import vllm_omni.diffusion.models.z_image.pipeline_z_image as z_image

    monkeypatch.setattr(flux2, "Flux2ImageProcessor", FakeProcessor)
    monkeypatch.setattr(z_image, "VaeImageProcessor", FakeProcessor)

    flux2.get_flux2_post_process_func(SimpleNamespace(model=str(model_path), output_type="pil"))
    assert captured["vae_scale_factor"] == 2

    z_image.get_post_process_func(SimpleNamespace(model=str(model_path)))
    assert captured["vae_scale_factor"] == 2


def test_migrated_factory_raises_on_missing_vae_config(tmp_path):
    """The restored fail-fast contract must hold at the call-site level too."""
    import vllm_omni.diffusion.models.z_image.pipeline_z_image as z_image

    with pytest.raises(FileNotFoundError, match="VAE config not found"):
        z_image.get_post_process_func(SimpleNamespace(model=str(tmp_path)))
