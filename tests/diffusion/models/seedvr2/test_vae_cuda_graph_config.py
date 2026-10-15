# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporal profiles, independent eager override, and storage retirement."""

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import validate_seedvr2_config
from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
from vllm_omni.diffusion.models.seedvr2.vae_cuda_graph import (
    CONFIG_KEY,
    VAEDecoderGraphConfig,
    temporal_phases,
)

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def test_temporal_graph_configuration_and_eager_override():
    profile: dict[str, object] = {"profiles": [[5, 60, 106], [2, 60, 106], [3, 44, 80]]}
    extras = {CONFIG_KEY: profile}
    config = VAEDecoderGraphConfig.from_extras(extras, enforce_eager=False)
    assert config.profiles == ((2, 60, 106), (3, 44, 80), (5, 60, 106))
    assert VAEDecoderGraphConfig.from_extras(extras, enforce_eager=True) is None
    assert VAEDecoderGraphConfig.from_extras({}, enforce_eager=False) is None
    assert config.fuse_normalization is False
    profile["fuse_normalization"] = True
    assert VAEDecoderGraphConfig.from_extras(extras, enforce_eager=False).fuse_normalization


@pytest.mark.parametrize(
    "value",
    [
        True,
        {},
        [],
        {"profiles": []},
        {"profiles": [[1, 4, 6]]},
        {"profiles": [[2, 4]]},
        {"profiles": [[True, 4, 6]]},
        {"profiles": [[2, 0, 6]]},
        {"profiles": [[2, 3, 6]]},
        {"profiles": [[2, 4, 5]]},
        {"profiles": [[2, 4, 6], [2, 4, 6]]},
        {"profiles": [[i, 4, 6] for i in range(2, 7)]},
        {"profiles": [[2, 4, 6]], "max_pool_bytes": 0},
        {"profiles": [[2, 4, 6]], "eager_headroom_bytes": True},
        {"profiles": [[2, 4, 6]], "recapture_on_miss": True},
        {"profiles": [[2, 4, 6]], "fuse_normalization": 1},
        {"profiles": [[2, 4, 6]], "fuse_normalization": "true"},
    ],
)
def test_invalid_vae_graph_profiles_are_startup_errors(value):
    with pytest.raises(ValueError):
        VAEDecoderGraphConfig.from_extras({CONFIG_KEY: value}, enforce_eager=False)


@pytest.mark.parametrize(
    "frames,phases",
    [
        (2, ("first",)),
        (3, ("first", "tail")),
        (4, ("first", "continuation")),
        (5, ("first", "continuation", "tail")),
        (7, ("first", "continuation", "continuation", "tail")),
    ],
)
def test_original_chunk_boundaries(frames, phases):
    assert temporal_phases(frames) == phases


@pytest.mark.parametrize("operation", ["move", "reload", "parallel"])
def test_vae_retires_graph_before_storage_or_group_change(operation):
    model = SeedVR2VAE(channels=(8, 8, 8, 8), groups=4, latent_channels=4)
    original_dtype = model.decoder.conv_in.weight.dtype
    observed = []

    class Graphs:
        def close(self):
            observed.append(model.decoder.conv_in.weight.dtype)

    model.decoder_graphs = Graphs()
    model.fused_spatial_normalization = True
    if operation == "move":
        model.to(dtype=torch.float64)
    elif operation == "reload":
        model.load_state_dict(model.state_dict())
    else:
        model.set_parallel_size(1)
    assert observed == [original_dtype] and model.decoder_graphs is None
    assert model.fused_spatial_normalization is False


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("degree", [2, 4])
def test_vae_graph_is_independent_and_can_combine_with_nadit(combined, degree):
    extras = {CONFIG_KEY: {"profiles": [[2, 60, 106], [3, 44, 80], [5, 32, 56]]}}
    if combined:
        extras["seedvr2_nadit_cuda_graph"] = {"profiles": [[2, 60, 106]]}
    config = OmniDiffusionConfig(
        model="SeedVR2-3B",
        dtype=torch.float16,
        enforce_eager=False,
        vae_use_tiling=True,
        parallel_config=DiffusionParallelConfig(ulysses_degree=degree, vae_patch_parallel_size=degree),
        extras=extras,
    )
    validate_seedvr2_config(config)
    config.parallel_config.vae_patch_parallel_size = 1
    with pytest.raises(ValueError, match="VAE height sharding"):
        validate_seedvr2_config(config)
    config.enforce_eager = True
    validate_seedvr2_config(config)
