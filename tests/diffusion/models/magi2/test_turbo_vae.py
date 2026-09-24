# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.magi2.turbo_vae import (
    Magi2TurboVAEDecoder,
    extract_turbo_decoder_state_dict,
    turbo_unpatchify,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_turbo_unpatchify_matches_wan_channel_order():
    # Four patch channels for one output channel. MAGI's order interleaves
    # width before height when reconstructing each 2x2 patch.
    patched = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1)
    output = turbo_unpatchify(patched, patch_size=2)
    expected = torch.tensor([[[[[0.0, 2.0], [1.0, 3.0]]]]])
    torch.testing.assert_close(output, expected)


def test_extract_turbo_decoder_prefers_ema_and_drops_training_heads():
    decoder_weight = torch.ones(1)
    checkpoint = {
        "state_dict": {"module.decoder.old.weight": torch.zeros(1)},
        "ema_state_dict": {
            "module.decoder.conv_in.conv.weight": decoder_weight,
            "module.aligned_feature_projection_heads.0.weight": torch.empty(1),
        },
    }
    assert extract_turbo_decoder_state_dict(checkpoint) == {"decoder.conv_in.conv.weight": decoder_weight}


def test_turbo_temporal_tiles_preserve_chunk_overlap_contract():
    class RepeatDecoder(nn.Module):
        def forward(self, value: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
            del is_first_chunk
            return value.repeat_interleave(4, dim=2)

    decoder = object.__new__(Magi2TurboVAEDecoder)
    nn.Module.__init__(decoder)
    decoder.z_dim = 1
    decoder.first_chunk_size = 3
    decoder.step_size = 2
    decoder.temporal_compression_ratio = 4
    decoder.decoder = RepeatDecoder()
    decoder.use_tiling = False
    decoder.register_buffer("latent_mean", torch.zeros(1), persistent=False)
    decoder.register_buffer("latent_std", torch.ones(1), persistent=False)

    latent = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1, 1)
    actual = decoder.decode(latent)
    expected = latent.repeat_interleave(4, dim=2)

    torch.testing.assert_close(actual, expected)

    decoder.latent_mean = torch.zeros(1, device="meta")
    decoder.latent_std = torch.ones(1, device="meta")
    prepared, _ = decoder._prepare_latent(latent)
    assert prepared.device.type == "meta"
