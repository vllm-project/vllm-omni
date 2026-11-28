# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def create_vae(monkeypatch):
    remote = nn.Module()
    remote.model = nn.Module()
    remote.model.encoder = nn.Identity()
    remote.model.quant_conv = nn.Identity()
    remote.model.tile_size = 256
    remote.model.tile_overlap_min = 64
    remote.model.vae_ratio = 16
    remote.model.encoder_parallel = True
    monkeypatch.setattr(vae_module, "_load_component_config", lambda _: {})
    monkeypatch.setattr(vae_module, "_load_video_vae_encoder", lambda *_: remote)
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda *_, **__: remote)
    return vae_module.MiniMaxH3VideoVAE


@pytest.mark.parametrize(
    ("config", "batch_size", "geometry"),
    [
        ({}, 1, None),
        ({"video_vae_encode": {}}, 1, None),
        ({"video_vae_encode": {"tile_batch_size": 4, "tile_size": 288, "tile_min_overlap": 16}}, 4, (288, 16)),
        ({"video_vae_encode": {"tile_min_overlap": 32}}, 1, (256, 32)),
    ],
)
def test_video_vae_encode_config(create_vae, config, batch_size, geometry):
    vae = create_vae("unused", device=torch.device("cpu"), encode_only=True, additional_config=config)

    assert vae.encode_tile_batch_size == batch_size
    assert vae.encode_tile_geometry == geometry
    assert (vae.model.tile_size, vae.model.tile_overlap_min) == (256, 64)


def test_non_object_video_vae_encode_config_only_rejected_for_encoder(create_vae):
    config = {"video_vae_encode": False}
    with pytest.raises(ValueError, match="video_vae_encode must be an object"):
        create_vae("unused", device=torch.device("cpu"), encode_only=True, additional_config=config)

    vae = create_vae("unused", device=torch.device("cpu"), decode_only=True, additional_config=config)
    assert vae.encode_tile_batch_size == 1
    assert vae.encode_tile_geometry is None
