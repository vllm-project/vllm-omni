# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib

import pytest
import torch

from vllm_omni.diffusion import registry
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _vae():
    vae = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.model = torch.nn.Module()
    vae.model.vae_ratio = 16
    vae.model.tile_size = 256
    vae.model.decoder_tile_size = 128
    vae.model.tile_overlap_min = 64
    vae.model.decoder_tile_overlap_min = 96
    vae.model.temporal_tile_size = 8
    vae.use_tiling = False
    return vae


@pytest.mark.parametrize("tile_size", [128, 256, 384, 512])
def test_tile_size_updates_both_grids_only(tile_size):
    vae = _vae()
    before = vars(vae.model).copy()
    vae.set_tiling_config(tile_size=tile_size)
    assert vars(vae.model) == {**before, "tile_size": tile_size, "decoder_tile_size": tile_size}
    assert vae.use_tiling is False


@pytest.mark.parametrize("tile_size", [0, -16, True, 384.0, "384", 65, 80, 96])
def test_invalid_tile_size_does_not_mutate_model(tile_size):
    vae = _vae()
    before = vars(vae.model).copy()
    with pytest.raises(ValueError, match="vae_tile_size"):
        vae.set_tiling_config(tile_size=tile_size)
    assert vars(vae.model) == before


def test_tile_size_checks_encoder_overlap_and_checkpoint_ratio():
    vae = _vae()
    vae.model.tile_overlap_min = 128
    with pytest.raises(ValueError, match="overlap"):
        vae.set_tiling_config(tile_size=128)
    vae.model.vae_ratio = 32
    with pytest.raises(ValueError, match="compression ratio"):
        vae.set_tiling_config(tile_size=144)


@pytest.mark.parametrize("overlap", [0, 16, 32, 64, 112])
def test_overlap_updates_both_minimums_only(overlap):
    vae = _vae()
    before = vars(vae.model).copy()
    vae.set_tiling_config(tile_overlap=overlap)
    assert vars(vae.model) == {**before, "tile_overlap_min": overlap, "decoder_tile_overlap_min": overlap}
    assert vae.use_tiling is False


@pytest.mark.parametrize("overlap", [-16, True, 32.0, "32", 1, 128, 256])
def test_invalid_overlap_does_not_mutate_model(overlap):
    vae = _vae()
    before = vars(vae.model).copy()
    with pytest.raises(ValueError, match="vae_tile_overlap"):
        vae.set_tiling_config(tile_overlap=overlap)
    assert vars(vae.model) == before


@pytest.mark.parametrize("tile_size,overlap", [(64, 32), (384, 256)])
def test_combined_geometry_validates_final_values(tile_size, overlap):
    vae = _vae()
    vae.set_tiling_config(tile_size=tile_size, tile_overlap=overlap)
    assert vae.model.tile_size == vae.model.decoder_tile_size == tile_size
    assert vae.model.tile_overlap_min == vae.model.decoder_tile_overlap_min == overlap


@pytest.mark.parametrize("tile_size,overlap", [(64, 64), (128, 256), (65, 32), (256, 17), (64, -16)])
def test_invalid_combination_is_atomic(tile_size, overlap):
    vae = _vae()
    before = vars(vae.model).copy()
    with pytest.raises(ValueError):
        vae.set_tiling_config(tile_size=tile_size, tile_overlap=overlap)
    assert vars(vae.model) == before


def test_overlap_checks_encoder_size_and_checkpoint_ratio():
    vae = _vae()
    vae.model.tile_size = 96
    with pytest.raises(ValueError, match="overlap"):
        vae.set_tiling_config(tile_overlap=96)
    vae.model.vae_ratio = 32
    with pytest.raises(ValueError, match="compression ratio"):
        vae.set_tiling_config(tile_overlap=16)


@pytest.mark.parametrize("tile_size,overlap", [(None, None), (384, None), (None, 0), (None, 32), (64, 32), (384, 256)])
@pytest.mark.parametrize("supports_tiling", [False, True])
def test_registry_applies_geometry_or_rejects_unsupported_vae(monkeypatch, tile_size, overlap, supports_tiling):
    vae = _vae() if supports_tiling else torch.nn.Module()
    transformer = torch.nn.Linear(2, 2)
    transformer_state = {name: value.clone() for name, value in transformer.state_dict().items()}
    # Isolate tile-size setup from distributed groups and tile-batch scheduling.
    monkeypatch.setattr(vae, "set_parallel_size", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(vae, "set_tile_batch_size", lambda value: None, raising=False)

    class Pipeline(torch.nn.Module):
        def __init__(self, od_config):
            super().__init__()
            self.vae = vae
            self.transformer = transformer

    if tile_size is None and overlap is None and supports_tiling:

        def unexpected_override(**kwargs):
            pytest.fail("Unset geometry must not invoke the VAE setter")

        monkeypatch.setattr(vae, "set_tiling_config", unexpected_override)

    monkeypatch.setattr(registry.DiffusionModelRegistry, "_try_load_model_cls", lambda name: Pipeline)
    monkeypatch.setattr(registry, "_prepare_diffusion_quant_config", lambda *args: None)
    monkeypatch.setattr(registry, "set_current_diffusion_config", lambda config: contextlib.nullcontext())
    monkeypatch.setattr(registry, "_apply_sequence_parallel_if_enabled", lambda *args: None)
    monkeypatch.setattr(registry, "_apply_wan_vae_fastpath_if_enabled", lambda *args: None)
    config = OmniDiffusionConfig(vae_tile_size=tile_size, vae_tile_overlap=overlap)
    if (tile_size is not None or overlap is not None) and not supports_tiling:
        with pytest.raises(ValueError, match="does not support vae_tile_size"):
            registry.initialize_model(config)
    else:
        model = registry.initialize_model(config)
        assert model.vae is vae
        assert model.transformer is transformer
        for name, value in transformer.state_dict().items():
            assert torch.equal(value, transformer_state[name])
        if supports_tiling:
            assert vae.model.tile_size == (256 if tile_size is None else tile_size)
            assert vae.model.decoder_tile_size == (128 if tile_size is None else tile_size)
            assert vae.model.tile_overlap_min == (64 if overlap is None else overlap)
            assert vae.model.decoder_tile_overlap_min == (96 if overlap is None else overlap)
